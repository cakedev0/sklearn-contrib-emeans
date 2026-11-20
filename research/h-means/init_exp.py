"""Initializations in O(n sqrt(k)) for e-means, against random and k-means||.

Each init gives k centers. They are then used by the exact `EMeans` and by the
adaptive two-level prototype, for several numbers of epochs.
"""

import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp
import hkm


def coarse_groups(X, g, seed):
    """One epoch of e-means into g groups: O(n g)."""
    _, labels, _ = hkm.emeans_fit(
        X, hkm.random_init(X, g, seed), 1, hkm.FlatAssigner(), 0.1, seed
    )
    return labels


def sample_groups(labels, g, quotas, gen):
    """Up to quotas[j] random rows of each group, padded: (g, M) rows and mask."""
    n = labels.shape[0]
    perm = torch.randperm(n, generator=gen)
    order = perm[torch.argsort(labels[perm], stable=True)]
    sizes = torch.bincount(labels, minlength=g)
    starts = torch.cumsum(sizes, 0) - sizes
    m = torch.minimum(sizes, quotas)
    M = int(m.max())
    pos = torch.arange(M)[None, :]
    mask = pos < m[:, None]
    idx = (starts[:, None] + pos).clamp_max(n - 1)
    return order[idx], mask


def allocate_weights(weights, sizes, k):
    """k centers split in proportion to weights, at most sizes[j] in group j."""
    ks = torch.zeros_like(sizes)
    left = k
    while left > 0:
        room = sizes - ks
        w = weights * (room > 0)
        quota = w * left / w.sum()
        add = torch.minimum(quota.floor().long(), room)
        if int(add.sum()) == 0:
            order = torch.argsort(quota, descending=True)[:left]
            add = torch.zeros_like(sizes)
            add[order] = 1
            add = torch.minimum(add, room)
        ks += add
        left = k - int(ks.sum())
    return ks


def group_alloc(X, labels, g, k, alloc):
    """Number of centers of each group.

    "size": proportional to the size n_j of the group. "inertia": to its
    inertia around its mean, n_j s_j^2. "half": to n_j s_j.
    """
    sizes = torch.bincount(labels, minlength=g)
    if alloc == "size":
        return hkm.allocate(sizes, k)
    d = X.shape[1]
    sums = torch.zeros((g, d), dtype=torch.float64).index_add_(0, labels, X.double())
    sq = torch.zeros(g, dtype=torch.float64).index_add_(
        0, labels, (X.double() ** 2).sum(1)
    )
    inertia = (sq - (sums**2).sum(1) / sizes.clamp_min(1)).clamp_min(0)
    if alloc == "half":
        inertia = (inertia * sizes).sqrt()
    return allocate_weights(inertia * (sizes > 0), sizes, k)


def init_stratified(X, k, g, seed, alloc="size"):
    """Random rows inside each coarse group."""
    gen = torch.Generator().manual_seed(seed + 31)
    labels = coarse_groups(X, g, seed)
    ks = group_alloc(X, labels, g, k, alloc)
    rows, mask = sample_groups(labels, g, ks, gen)
    return X[rows[mask]].clone(), 0


def init_subsample(X, k, g, seed, percent, T=2):
    """Two-level e-means on a random subsample."""
    gen = torch.Generator().manual_seed(seed + 57)
    m = max(k, X.shape[0] * percent // 100)
    Xs = X[torch.randperm(X.shape[0], generator=gen)[:m]]
    C, _ = fit_two_level(Xs, hkm.random_init(Xs, k, seed), T, g, seed)
    return C, 0


def init_group_kpp(X, k, g, seed, c=16, trials=1, alloc="size"):
    """k-means++ inside each coarse group, on c candidates per center.

    All the groups advance together: one batched product per new center, so
    about sqrt(k) steps of cost c k d each.
    """
    gen = torch.Generator().manual_seed(seed + 31)
    labels = coarse_groups(X, g, seed)
    ks = group_alloc(X, labels, g, k, alloc)
    rows, mask = sample_groups(labels, g, torch.maximum(c * ks, ks), gen)
    P = X[rows]  # (g, M, d)
    P[~mask] = 0
    pn = (P * P).sum(2)
    M = P.shape[1]
    kmax = int(ks.max())
    mind = torch.full((g, M), float("inf"))
    mind[~mask] = 0
    chosen = torch.zeros((g, kmax), dtype=torch.long)
    ar = torch.arange(g)
    # First center of each group: its first sampled row (the rows are shuffled).
    cur = torch.zeros(g, dtype=torch.long)
    dots = 0
    for j in range(kmax):
        chosen[:, j] = cur
        c_ = P[ar, cur]  # (g, d)
        d2 = pn - 2 * torch.bmm(P, c_[:, :, None])[:, :, 0] + (c_ * c_).sum(1)[:, None]
        dots += int(mask.sum())
        mind = torch.minimum(mind, d2.clamp_min_(0))
        mind[~mask] = 0
        if j + 1 == kmax:
            break
        w = mind.clone()
        dead = w.sum(1) <= 0  # only duplicates left, or group done
        w[dead] = mask[dead].float()
        w[w.sum(1) <= 0] = 1
        if trials == 1:
            cur = torch.multinomial(w, 1, generator=gen)[:, 0]
        else:
            cand = torch.multinomial(w, trials, replacement=True, generator=gen)
            Cc = P[ar[:, None], cand]  # (g, L, d)
            d2c = (
                pn[:, None, :]
                - 2 * torch.bmm(Cc, P.transpose(1, 2))
                + (Cc * Cc).sum(2)[:, :, None]
            )
            dots += trials * int(mask.sum())
            pot = torch.minimum(mind[:, None, :], d2c.clamp_min_(0)).sum(2)
            cur = cand[ar, pot.argmin(1)]
    keep = torch.arange(kmax)[None, :] < ks[:, None]
    picked = rows[ar[:, None], chosen][keep]
    return X[picked].clone(), dots


def init_h2(X, k, g, seed, top_n=None):
    """One epoch of the frozen two-level e-means."""
    C, _, _, _, _ = hkm.h2_fit(X, k, 1, g, seed, top_T=1, top_n=top_n)
    return C, 0


def make_init(name, X, k, g, seed):
    if name == "random":
        return hkm.random_init(X, k, seed), 0
    if name.startswith("stratified"):
        return init_stratified(X, k, g, seed, (name.split("-") + ["size"])[1])
    if name.startswith("sub-"):
        return init_subsample(X, k, g, seed, int(name.split("-")[1]))
    if name == "h2":
        return init_h2(X, k, g, seed)
    if name.startswith("kpp"):
        # kpp[-c<int>][-t<int>][-inertia]
        kw = {}
        for part in name.split("-")[1:]:
            if part[0] == "c":
                kw["c"] = int(part[1:])
            elif part[0] == "t":
                kw["trials"] = int(part[1:])
            elif part == "inertia":
                kw["alloc"] = "inertia"
        return init_group_kpp(X, k, g, seed, **kw)
    raise ValueError(name)


def fit_two_level(X, C0, T, g, seed):
    G, member = hkm.coarse_from_centers(C0, g, seed)
    ivf = hkm.IVFAssigner(G, member, 16, "lloyd", 1)
    ivf.p_final = 16
    ivf.ratio2 = 1.15**2
    C, labels, _ = hkm.emeans_fit(X, C0.clone(), T, ivf, 0.1, seed)
    return C, labels


def fit_exact(X, C0, T, seed):
    from sklearn_emeans import EMeans

    est = EMeans(n_clusters=C0.shape[0], init=C0, n_epochs=T, random_state=seed)
    est.fit(X.numpy())
    return torch.from_numpy(est.cluster_centers_), torch.from_numpy(est.labels_).long()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--inits", required=True)
    ap.add_argument("--epochs", default="1,2,5")
    ap.add_argument("--fits", default="two-level,exact")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    hkm.use_kernels(args.threads)
    X = torch.from_numpy(exp.load(args.data, args.n))
    n, k = X.shape[0], args.k
    g = max(2, round(math.sqrt(k)))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(4242))
    eval_rows = perm[: min(n, 1_000_000 if k <= 20000 else 200_000)]
    hkm.run(X[:20000], {"method": "ivf", "k": 64, "T": 1}, torch.arange(1000))
    for seed in (int(s) for s in args.seeds.split(",")):
        for name in args.inits.split(","):
            base = dict(data=args.data, n=n, k=k, init=name, seed=seed)
            if name == "k-means||":
                from sklearn_emeans import EMeans

                # Its cost: the fit time minus the fit from the same centers.
                for T in (int(t) for t in args.epochs.split(",")):
                    est = EMeans(
                        n_clusters=k, init="k-means||", n_epochs=T, random_state=seed
                    )
                    t0 = time.perf_counter()
                    est.fit(X.numpy())
                    t = time.perf_counter() - t0
                    res = dict(base, fit="exact", T=T, t_total=t)
                    res.update(
                        hkm.evaluate(
                            X,
                            torch.from_numpy(est.cluster_centers_),
                            torch.from_numpy(est.labels_).long(),
                            eval_rows,
                        )
                    )
                    emit(args.out, res)
                continue
            t0 = time.perf_counter()
            C0, dots = make_init(name, X, k, g, seed)
            t_init = time.perf_counter() - t0
            init_inertia = hkm.evaluate(X, C0, None, eval_rows)["inertia_exact"]
            for fit in args.fits.split(","):
                for T in (int(t) for t in args.epochs.split(",")):
                    t0 = time.perf_counter()
                    if fit == "exact":
                        C, labels = fit_exact(X, C0, T, seed)
                    else:
                        C, labels = fit_two_level(X, C0, T, g, seed)
                    res = dict(
                        base,
                        fit=fit,
                        T=T,
                        t_init=t_init,
                        t_fit=time.perf_counter() - t0,
                        init_inertia=init_inertia,
                        init_dots=dots / (n * k),
                    )
                    res.update(hkm.evaluate(X, C, labels, eval_rows))
                    emit(args.out, res)


def emit(out, res):
    with open(out, "a") as f:
        f.write(json.dumps(res) + "\n")
    keys = (
        "init",
        "fit",
        "T",
        "t_init",
        "t_fit",
        "t_total",
        "init_inertia",
        "inertia_exact",
        "inertia_labels",
        "size_cv",
    )
    print(
        {
            key: (round(v, 5) if isinstance(v, float) else v)
            for key, v in res.items()
            if key in keys
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
