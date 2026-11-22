"""Initializations in O(n sqrt(k)) with `EMeans(assignment="two-level")`.

Same question as `init_exp.py`, with the released estimator instead of the
prototype, inits written with NumPy, and more datasets. Each run is an init
followed by a two-level fit. The inertia is both the one of the fit
(`inertia_`, two-level labels) and the exact one (`-score(X)`).
"""

import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from threadpoolctl import threadpool_limits

from sklearn_emeans import EMeans

DATA = {
    "sift": "~/work/hkm/data/sift1m.npy",
    "glove": "~/work/hkm/data/glove200.npy",
    "clip": "~/work/hkm/data/wolt-clip512.npy",
    "rcv1-lsa": "~/work/emeans/init-sweep/raw/rcv1-lsa-128.npy",
    "covtype": "~/work/emeans/init-sweep/raw/covtype-54.npy",
    "kddcup99": "~/work/emeans/init-sweep/raw/kddcup99-num.npy",
}
N_SYNTH = 1_000_000
CHUNK = 16384


def load(name, k, n=None):
    if name.startswith("deep"):
        n_all = 100_000_000 if name == "deep100M" else 10_000_000
        path = os.path.expanduser(f"~/work/hkm/data/{name}.fbin")
        X = np.memmap(path, dtype=np.float32, mode="r", offset=8, shape=(n_all, 96))
    elif name.startswith(("kblobs-", "zipfblobs-")):
        # Exactly k blobs, of the same size or with sizes in 1 / rank.
        from sklearn.datasets import make_blobs

        d = int(name.split("-")[1])
        sizes = N_SYNTH
        centers = k
        if name.startswith("zipfblobs"):
            w = 1 / np.arange(1, k + 1)
            sizes = np.maximum((N_SYNTH * w / w.sum()).astype(int), 1)
            centers = None
        X = make_blobs(sizes, n_features=d, centers=centers, random_state=0)[0]
    else:
        X = np.load(os.path.expanduser(DATA[name]), mmap_mode="r")
    if n is not None:
        X = X[:n]
    return np.ascontiguousarray(X, dtype=np.float32)


def coarse_groups(X, g, seed):
    """One epoch of exact e-means into g groups: O(n g)."""
    est = EMeans(n_clusters=g, n_epochs=1, random_state=seed).fit(X)
    return est.labels_, est.cluster_centers_


def group_inertia(X, labels, centers, n_threads):
    """Inertia of each group around its center: O(n d)."""
    g = centers.shape[0]

    def work(start):
        s = slice(start, start + CHUNK)
        diff = X[s] - centers[labels[s]]
        return np.bincount(
            labels[s], weights=np.einsum("ij,ij->i", diff, diff), minlength=g
        )

    with ThreadPoolExecutor(n_threads) as pool:
        return sum(pool.map(work, range(0, X.shape[0], CHUNK)))


def allocate(weights, sizes, k):
    """k centers split in proportion to weights.

    Each non-empty group gets at least one center and at most its size.
    """
    ks = (sizes > 0).astype(np.int64)
    weights = np.where(sizes > 0, weights, 0).astype(np.float64)
    left = k - ks.sum()
    while left > 0:
        room = sizes - ks
        w = weights * (room > 0)
        if w.sum() <= 0:
            w = (room > 0).astype(np.float64)
        quota = w * left / w.sum()
        add = np.minimum(np.floor(quota).astype(np.int64), room)
        if add.sum() == 0:
            add = np.zeros_like(ks)
            add[np.argsort(-quota, kind="stable")[:left]] = 1
            add = np.minimum(add, room)
        ks += add
        left = k - ks.sum()
    return ks


def shuffled_by_group(labels, g, rng):
    """Rows sorted by group, in random order inside a group."""
    perm = rng.permutation(labels.shape[0])
    key = labels[perm].astype(np.uint16 if g <= 65536 else np.int64)
    order = perm[np.argsort(key, kind="stable")]
    sizes = np.bincount(labels, minlength=g)
    return order, np.cumsum(sizes) - sizes, sizes


def kpp_rows(P, kj, rng, trials=1):
    """k-means++ of kj rows among the candidates P. The first row is taken.

    With trials > 1, each step draws that many rows and keeps the one that
    lowers the potential the most (greedy k-means++).
    """
    m = P.shape[0]
    if kj >= m:
        return np.arange(m)
    pn = np.einsum("ij,ij->i", P, P)
    chosen = np.empty(kj, dtype=np.int64)
    chosen[0] = 0
    mind = np.maximum(pn - 2 * (P @ P[0]) + pn[0], 0)
    mind[0] = 0
    for j in range(1, kj):
        cum = np.cumsum(mind, dtype=np.float64)
        if cum[-1] <= 0:  # only duplicates of the chosen rows are left
            free = np.setdiff1d(np.arange(m), chosen[:j])
            cand = rng.choice(free, 1)
        else:
            cand = np.searchsorted(cum, rng.random(trials) * cum[-1])
            np.minimum(cand, m - 1, out=cand)
        d2 = pn[None, :] - 2 * (P[cand] @ P.T) + pn[cand][:, None]
        np.maximum(d2, 0, out=d2)
        np.minimum(d2, mind[None, :], out=d2)
        best = int(np.argmin(d2.sum(axis=1, dtype=np.float64)))
        chosen[j] = cand[best]
        mind = d2[best]
        mind[chosen[j]] = 0
    return chosen


def init_groups(X, k, g, seed, alloc, kpp, n_threads, c=16, trials=1):
    """Coarse groups, then random rows or k-means++ inside each group.

    alloc is "size" (centers in proportion to the size of the group) or
    "inertia" (to its inertia). With kpp, k-means++ runs in each group on c
    random candidates per center, with the given number of trials per step.
    """
    rng = np.random.default_rng(seed + 31)
    labels, centers = coarse_groups(X, g, seed)
    sizes = np.bincount(labels, minlength=g)
    if alloc == "size":
        weights = sizes
    else:
        weights = group_inertia(X, labels, centers, n_threads)
    ks = allocate(weights, sizes, k)
    order, starts, sizes = shuffled_by_group(labels, g, rng)
    if not kpp:
        rows = np.concatenate([order[s : s + kj] for s, kj in zip(starts, ks)])
        return X[rows]
    rngs = rng.spawn(g)

    def work(j):
        cand = order[starts[j] : starts[j] + min(sizes[j], c * ks[j])]
        return cand[kpp_rows(X[cand], ks[j], rngs[j], trials)]

    with threadpool_limits(1, user_api="blas"), ThreadPoolExecutor(n_threads) as pool:
        # The largest groups first, they take the longest.
        rows = np.concatenate(list(pool.map(work, np.argsort(-ks, kind="stable"))))
    return X[rows]


def init_subsample(X, k, seed, percent, n_epochs=2):
    """Two-level e-means on a random subsample."""
    rng = np.random.default_rng(seed + 57)
    n = X.shape[0]
    m = min(n, max(n * percent // 100, 10 * k))
    Xs = X[np.sort(rng.choice(n, m, replace=False))]
    est = EMeans(
        n_clusters=k, n_epochs=n_epochs, assignment="two-level", random_state=seed
    )
    return est.fit(Xs).cluster_centers_


def make_init(name, X, k, g, seed, n_threads):
    if name in ("random", "k-means||"):
        return name
    if name.startswith("sub-"):
        return init_subsample(X, k, seed, int(name.split("-")[1]))
    # stratified[-inertia] or kpp[-inertia][-t<trials>]
    parts = name.split("-")
    alloc = "inertia" if "inertia" in parts else "size"
    trials = int(parts[-1][1:]) if parts[-1][0] == "t" else 1
    return init_groups(
        X, k, g, seed, alloc, parts[0] == "kpp", n_threads, trials=trials
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--inits", required=True)
    ap.add_argument("--epochs", default="5")
    ap.add_argument("--random-epochs", default="", help="more runs of random")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--no-exact", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    X = load(args.data, args.k, args.n)
    n, d = X.shape
    k = args.k
    g = round(math.sqrt(k))
    print(f"{args.data}: {X.shape}, k={k}, g={g}", flush=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["data"], r["n"], r["k"], r["init"], r["epochs"], r["seed"]))
    runs = [(i, int(e)) for e in args.epochs.split(",") for i in args.inits.split(",")]
    runs += [("random", int(e)) for e in args.random_epochs.split(",") if e]
    # Warm-up of the threads and of the page cache.
    EMeans(n_clusters=1024, n_epochs=1, assignment="two-level", random_state=0).fit(
        X[: min(n, 200_000)]
    )
    for seed in [int(s) for s in args.seeds.split(",")]:
        for init, epochs in runs:
            if (args.data, n, k, init, epochs, seed) in done:
                continue
            t0 = time.perf_counter()
            C0 = make_init(init, X, k, g, seed, args.threads)
            t1 = time.perf_counter()
            est = EMeans(
                n_clusters=k,
                init=C0,
                n_epochs=epochs,
                assignment="two-level",
                random_state=seed,
            ).fit(X)
            t2 = time.perf_counter()
            res = dict(
                data=args.data,
                n=n,
                d=d,
                k=k,
                g=g,
                init=init,
                epochs=epochs,
                seed=seed,
                threads=args.threads,
                t_init=t1 - t0,
                t_fit=t2 - t1,
                inertia_labels=float(est.inertia_),
            )
            if not args.no_exact:
                res["inertia_exact"] = float(-est.score(X))
            with open(args.out, "a") as f:
                f.write(json.dumps(res) + "\n")
            print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
