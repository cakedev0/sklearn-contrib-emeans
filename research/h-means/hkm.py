"""Research prototype: e-means with sub-quadratic assignment.

Everything is PyTorch, so that the same code runs on a CPU and on a GPU.

Methods
-------
flat   e-means, exact assignment (same steps as sklearn_emeans._array_api).
h2     frozen two-level: e-means into g groups, then e-means in each group.
ivf    e-means on the k fine centers, where a point is only compared to the
       centers of its p nearest groups of centers. The groups (coarse centers
       + membership of the fine centers) are maintained during the fit.
graph  e-means on the k fine centers, where a point is only compared to the
       m nearest centers of the center it had at its previous visit.

`WORK` counts the point-center products (in units of one d-dimensional dot
product), a hardware-independent cost.
"""

import math
import time

import torch
import torch.nn.functional as F

# Constants of sklearn_emeans._emeans.
MIN_STEP_SIZE_FACTOR = 0.1
REVIVAL_PERIOD = 5
INIT_WEIGHT = 2.0**-16
SPLIT_EPSILON = 1e-4
BATCH_LAW = (0.035134, 0.134749, 0.876473, 0.813344)
STEP_LAW = (0.8244, -0.4617, 0.9003, -0.4498, -0.1057)

WORK = {"dots": 0}
KERN = {"mod": None, "n_threads": 1}


def use_kernels(n_threads):
    """Use the Cython kernels for CPU tensors."""
    import hkm_kernels
    import numpy as np
    from threadpoolctl import ThreadpoolController

    # One BLAS thread in the OpenBLAS of SciPy, called from OpenMP threads.
    for lib in ThreadpoolController().select(internal_api="openblas").lib_controllers:
        lib.set_num_threads(1)
    KERN.update(mod=hkm_kernels, n_threads=n_threads, np=np)


def _cpu(X):
    return KERN["mod"] is not None and X.device.type == "cpu"


MAX_SCORES = {"cpu": 2**25, "cuda": 2**28}


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def round_power_of_two(value):
    if value <= 1:
        return 1
    low = 2 ** math.floor(math.log2(value))
    return low if value / low < 2 * low / value else 2 * low


def resolve_batch_size(n, k, T):
    c, a_k, a_n, a_t = BATCH_LAW
    b = round_power_of_two(c * k**a_k * n**a_n * T**a_t)
    return max(1, min(b, round_power_of_two(n / 2), n))


def resolve_step_size(n, k, T, b):
    c, a_k, a_b, a_n, a_t = STEP_LAW
    return min(max(c * k**a_k * b**a_b * n**a_n * T**a_t, 1e-4), 1.0)


def epochs_n_rows(n, T):
    n_full = math.floor(T)
    rem = round((T - n_full) * n)
    if n_full == 0:
        return [max(rem, 1)]
    return [n] * (n_full - 1) + ([rem] if rem else []) + [n]


def scheduled_step_size(s0, step, n_steps):
    cosine = 0.5 * (1.0 + math.cos(math.pi * step / n_steps))
    return s0 * (MIN_STEP_SIZE_FACTOR + (1.0 - MIN_STEP_SIZE_FACTOR) * cosine)


def half_norms(C):
    return 0.5 * (C * C).sum(1)


def assign_flat(X, C, H=None, return_scores=False):
    """Exact nearest center, by chunks of rows."""
    if H is None:
        H = half_norms(C)
    n, k = X.shape[0], C.shape[0]
    chunk = max(1, MAX_SCORES[X.device.type] // k)
    WORK["dots"] += n * k
    if _cpu(X):
        labels = torch.empty(n, dtype=torch.long)
        best = torch.empty(n, dtype=X.dtype)
        KERN["mod"].flat_assign(
            X.numpy(),
            C.numpy(),
            H.numpy(),
            labels.numpy(),
            best.numpy(),
            KERN["n_threads"],
        )
        return (labels, best) if return_scores else labels
    labels, best = [], []
    for start in range(0, n, chunk):
        m, i = F.linear(X[start : start + chunk], C, -H).max(1)
        labels.append(i)
        best.append(m)
    labels = labels[0] if len(labels) == 1 else torch.cat(labels)
    if return_scores:
        return labels, (best[0] if len(best) == 1 else torch.cat(best))
    return labels


class FlatAssigner:
    name = "flat"

    def refresh(self, C, H, w, step, force=False):
        self.C, self.H = C, H

    def teleport(self, dead, donors, step):
        pass

    def assign(self, Xb, rows, step):
        return assign_flat(Xb, self.C, self.H)


class IVFAssigner:
    """Assign through groups of centers.

    coarse: 'lloyd' (coarse centers = weighted means of their fine centers,
    then the fine centers go to their nearest coarse center), 'reassign' (fixed
    coarse centers, membership refreshed) or 'frozen' (nothing refreshed).
    """

    name = "ivf"

    def __init__(self, G, member, p, coarse="lloyd", period=1):
        self.G = G.clone()
        self.member = member.clone()
        self.p = p
        self.p_final = p
        self.late_step = 10**18
        self.ratio2 = 0.0
        self.coarse = coarse
        self.period = period
        self.g = G.shape[0]
        self.n_refresh = 0

    def _build_lists(self):
        self.order = torch.argsort(self.member, stable=True)
        counts = torch.bincount(self.member, minlength=self.g)
        self.counts = counts.tolist()
        self.starts = (torch.cumsum(counts, 0) - counts).tolist()
        self.counts_t = counts
        self.starts_t = torch.cumsum(counts, 0) - counts
        self.HG = half_norms(self.G)
        # An empty group is never probed.
        self.HG[counts == 0] = torch.inf

    def refresh(self, C, H, w, step, force=False):
        self.C, self.H = C, H
        if self.n_refresh == 0:
            self._build_lists()
        elif self.coarse != "frozen" and (force or step % self.period == 0):
            if self.coarse == "lloyd":
                wf = w.to(C.dtype).clamp_min(1e-12)
                sums = torch.zeros_like(self.G).index_add_(
                    0, self.member, C * wf[:, None]
                )
                tot = torch.zeros(self.g, dtype=C.dtype, device=C.device).index_add_(
                    0, self.member, wf
                )
                ok = tot > 0
                self.G[ok] = sums[ok] / tot[ok, None]
            self.member = assign_flat(C, self.G)
            self._build_lists()
        self.n_refresh += 1
        self.Cs = C[self.order]
        self.Hs = H[self.order]

    def teleport(self, dead, donors, step):
        self.member[dead] = self.member[donors]

    def search(self, Xb, p, topm=1, ratio2=0.0):
        """Best `topm` centers among those of the p nearest groups.

        Returns labels of shape (B,) if topm == 1 else (B, topm), -1 padded.
        """
        B = Xb.shape[0]
        g = self.g
        p = min(p, g)
        WORK["dots"] += B * g
        if _cpu(Xb):
            args = (
                Xb.numpy(),
                self.G.numpy(),
                self.HG.numpy(),
                self.Cs.numpy(),
                self.Hs.numpy(),
                self.starts_t.numpy(),
                self.counts_t.numpy(),
                self.order.numpy(),
                p,
                max(self.counts),
            )
            if topm == 1:
                lab = torch.empty(B, dtype=torch.long)
                fn = getattr(KERN["mod"], KERN.get("ivf", "ivf_assign"))
                WORK["dots"] += fn(*args, lab.numpy(), KERN["n_threads"], ratio2)
            else:
                lab = torch.empty((B, topm), dtype=torch.long)
                WORK["dots"] += KERN["mod"].ivf_topm(
                    *args, lab.numpy(), KERN["n_threads"]
                )
            return lab
        chunk = max(1, MAX_SCORES[Xb.device.type] // g)
        tops = [
            F.linear(Xb[s : s + chunk], self.G, -self.HG).topk(p, dim=1)
            for s in range(0, B, chunk)
        ]
        probes = torch.cat([t.indices for t in tops])
        if ratio2 > 0 and topm == 1:
            gd = (Xb * Xb).sum(1, keepdim=True) - 2 * torch.cat(
                [t.values for t in tops]
            )
            pruned = gd > ratio2 * gd[:, :1]
            pruned[:, 0] = False
            probes = torch.where(pruned, g, probes)
        pair_group = probes.reshape(-1)
        perm = torch.argsort(pair_group)
        pts = torch.div(perm, p, rounding_mode="floor")
        bounds = torch.searchsorted(
            pair_group[perm], torch.arange(g + 1, device=Xb.device)
        ).tolist()
        best = torch.full((B * p, topm), -torch.inf, dtype=Xb.dtype, device=Xb.device)
        lab = torch.full((B * p, topm), -1, dtype=torch.long, device=Xb.device)
        dots = 0
        for j in range(g):
            a, b = bounds[j], bounds[j + 1]
            cnt = self.counts[j]
            if a == b or cnt == 0:
                continue
            st = self.starts[j]
            sl = perm[a:b]
            scores = F.linear(
                Xb[pts[a:b]], self.Cs[st : st + cnt], -self.Hs[st : st + cnt]
            )
            dots += (b - a) * cnt
            if topm == 1:
                m, i = scores.max(1)
                best[sl, 0] = m
                lab[sl, 0] = self.order[st + i]
            else:
                t = min(topm, cnt)
                m, i = scores.topk(t, dim=1)
                best[sl, :t] = m
                lab[sl, :t] = self.order[st + i]
        WORK["dots"] += dots
        best = best.view(B, p * topm)
        lab = lab.view(B, p * topm)
        if topm == 1:
            return lab.gather(1, best.argmax(1, keepdim=True))[:, 0]
        m, i = best.topk(topm, dim=1)
        return lab.gather(1, i)

    def assign(self, Xb, rows, step, final=False):
        late = final or step >= self.late_step
        return self.search(Xb, self.p_final if late else self.p, ratio2=self.ratio2)


class GraphAssigner:
    """Compare a point to the m nearest centers of its previous center.

    Points without a valid previous center (first visit, or their center was
    moved by a revival since) are assigned through `ivf`.
    """

    name = "graph"

    def __init__(
        self,
        n,
        m,
        ivf,
        period=1,
        p_graph=None,
        exact_graph=False,
        labels=None,
        device="cpu",
        hops=1,
    ):
        self.hops = hops
        self.m = m
        self.ivf = ivf
        self.period = period
        self.p_graph = p_graph if p_graph is not None else max(2 * ivf.p, 4)
        self.exact_graph = exact_graph
        self.prev = torch.full((n,), -1, dtype=torch.long, device=device)
        self.prev_step = torch.full((n,), -1, dtype=torch.long, device=device)
        if labels is not None:
            self.prev.copy_(labels)
            self.prev_step.fill_(0)
        self.tele_step = None
        self.n_refresh = 0
        self.n_fallback = 0

    def _build_graph(self):
        C, H = self.C, self.H
        k = C.shape[0]
        m = min(self.m, k)
        if self.exact_graph:
            WORK["dots"] += k * k
            chunk = max(1, MAX_SCORES[C.device.type] // k)
            self.NB = torch.cat(
                [
                    F.linear(C[s : s + chunk], C, -H).topk(m, dim=1).indices
                    for s in range(0, k, chunk)
                ]
            )
        else:
            self.NB = self.ivf.search(C, self.p_graph, topm=m)
        # The center itself first, whatever the search returned. Missing
        # neighbors (-1) are replaced by the center itself.
        own = torch.arange(k, device=C.device)
        self.NB = torch.where(self.NB < 0, own[:, None], self.NB)
        has_own = (self.NB == own[:, None]).any(1)
        self.NB[~has_own, -1] = own[~has_own]

    def refresh(self, C, H, w, step, force=False):
        self.C, self.H = C, H
        if self.tele_step is None:
            self.tele_step = torch.full(
                (C.shape[0],), -1, dtype=torch.long, device=C.device
            )
        self.ivf.refresh(C, H, w, step, force)
        if self.n_refresh == 0 or force or step % self.period == 0:
            self._build_graph()
        self.n_refresh += 1

    def teleport(self, dead, donors, step):
        self.ivf.teleport(dead, donors, step)
        self.tele_step[dead] = step

    def assign(self, Xb, rows, step):
        prev = self.prev[rows]
        valid = prev >= 0
        valid &= self.prev_step[rows] >= self.tele_step[prev.clamp_min(0)]
        labels = torch.empty_like(prev)
        idx = torch.nonzero(valid)[:, 0]
        if idx.numel() < prev.numel():
            bad = torch.nonzero(~valid)[:, 0]
            self.n_fallback += bad.numel()
            labels[bad] = self.ivf.search(Xb[bad], self.ivf.p)
        if idx.numel() and _cpu(Xb):
            out = torch.empty_like(prev)
            WORK["dots"] += KERN["mod"].graph_assign(
                Xb.numpy(),
                torch.where(valid, prev, -1).numpy(),
                self.C.numpy(),
                self.H.numpy(),
                self.NB.numpy(),
                out.numpy(),
                self.hops,
                KERN["n_threads"],
            )
            labels[idx] = out[idx]
        elif idx.numel():
            m = self.NB.shape[1]
            d = Xb.shape[1]
            chunk = max(1, (2**27 if Xb.device.type == "cuda" else 2**24) // (m * d))
            cur = prev[idx]
            best = torch.full(
                (idx.numel(),), -torch.inf, dtype=Xb.dtype, device=Xb.device
            )
            active = torch.arange(idx.numel(), device=Xb.device)
            for h in range(self.hops):
                WORK["dots"] += active.numel() * m
                moved = []
                for s in range(0, active.numel(), chunk):
                    aa = active[s : s + chunk]
                    cand = self.NB[cur[aa]]
                    scores = torch.bmm(self.C[cand], Xb[idx[aa]].unsqueeze(2)).squeeze(
                        2
                    )
                    scores -= self.H[cand]
                    mx, am = scores.max(1)
                    new = cand.gather(1, am[:, None])[:, 0]
                    better = (mx > best[aa]) & (new != cur[aa])
                    best[aa] = torch.maximum(mx, best[aa])
                    cur[aa] = torch.where(better, new, cur[aa])
                    moved.append(aa[better])
                active = torch.cat(moved)
                if active.numel() == 0:
                    break
            labels[idx] = cur
        self.prev[rows] = labels
        self.prev_step[rows] = step
        return labels


def revive(C, w, threshold, gen):
    """Split heavy centers to replace the dead ones, as in sklearn_emeans."""
    dead = w < threshold * w.mean()
    donors = ~dead & (w > 0)
    n_rev = min(int(dead.sum()), int(donors.sum()))
    if n_rev == 0:
        return None
    u = torch.rand(w.shape[0], generator=gen, dtype=torch.float64).to(w.device)
    u.clamp_min_(1e-300)
    keys = torch.where(
        donors, torch.log(w.clamp_min(1e-300)) - torch.log(-torch.log(u)), -torch.inf
    )
    donor_idx = keys.topk(n_rev).indices
    dead_idx = torch.nonzero(dead)[:n_rev, 0]
    offset = torch.full((C.shape[1],), SPLIT_EPSILON, dtype=C.dtype, device=C.device)
    offset[1::2] *= -1
    donor_centers = C[donor_idx]
    C[dead_idx] = donor_centers * (1 + offset)
    C[donor_idx] = donor_centers * (1 - offset)
    w[donor_idx] /= 2
    w[dead_idx] = w[donor_idx]
    return dead_idx, donor_idx


def emeans_fit(
    X, C0, T, assigner, revival=0.1, seed=0, final_labels=True, diag=None, law_k=None
):
    """E-means steps from the centers C0 with the given assigner.

    Returns the final centers (means of the last epoch), the labels of all the
    rows from a last pass with the assigner (or None), and a dict of stats.
    """
    n, d = X.shape
    k = C0.shape[0]
    dev = X.device
    gen = torch.Generator().manual_seed(seed)
    C = C0.clone()
    b = resolve_batch_size(n, law_k or k, T)
    s0 = resolve_step_size(n, law_k or k, T, b)
    rows_per_epoch = epochs_n_rows(n, T)
    n_steps = sum(math.ceil(r / b) for r in rows_per_epoch)
    w = torch.full((k,), INIT_WEIGHT, dtype=torch.float64, device=dev)
    epoch_sums = torch.zeros((k, d), dtype=torch.float64, device=dev)
    epoch_counts = torch.zeros(k, dtype=torch.float64, device=dev)
    ones = torch.ones(1, dtype=X.dtype, device=dev)
    n_revived = 0
    t_diag = 0.0
    tm = {"gather": 0.0, "assign": 0.0, "acc": 0.0, "update": 0.0, "refresh": 0.0}
    if _cpu(X):
        Xbuf = torch.empty((b, d), dtype=X.dtype)
        sums_buf = torch.empty((k, d), dtype=X.dtype)
        counts_buf = torch.empty(k, dtype=X.dtype)
        order_buf = KERN["np"].empty(b, dtype="int64")
        bounds_buf = KERN["np"].empty(k + 1, dtype="int64")
    step = 0
    assigner.refresh(C, half_norms(C), w, 0, force=True)
    for epoch, n_rows in enumerate(rows_per_epoch):
        last = epoch == len(rows_per_epoch) - 1
        perm = torch.randperm(n, generator=gen).to(dev)
        for s in range(math.ceil(n_rows / b)):
            rows = perm[s * b : min((s + 1) * b, n_rows)]
            t1 = time.perf_counter()
            if _cpu(X):
                # Reused buffers: fresh ones are page-faulted at each step.
                Xb = Xbuf[: rows.shape[0]]
                KERN["mod"].gather_rows(
                    X.numpy(), rows.numpy(), Xb.numpy(), KERN["n_threads"]
                )
            else:
                Xb = X[rows]
            t2 = time.perf_counter()
            labels = assigner.assign(Xb, rows, step)
            t3 = time.perf_counter()
            if diag is not None:
                sync(dev)
                t0 = time.perf_counter()
                work = WORK["dots"]
                q = min(2048, Xb.shape[0])
                exact, best = assign_flat(Xb[:q], C, return_scores=True)
                got = (Xb[:q] * C[labels[:q]]).sum(1) - half_norms(C)[labels[:q]]
                xn = (Xb[:q] * Xb[:q]).sum(1)
                diag.append(
                    {
                        "step": step,
                        "recall": float((exact == labels[:q]).float().mean()),
                        "excess": float(
                            ((xn - 2 * got).sum() / (xn - 2 * best).sum()) - 1
                        ),
                    }
                )
                WORK["dots"] = work
                sync(dev)
                t_diag += time.perf_counter() - t0
            if _cpu(X):
                sums, counts = sums_buf, counts_buf
                KERN["mod"].accumulate(
                    Xb.numpy(),
                    labels.numpy(),
                    sums.numpy(),
                    counts.numpy(),
                    order_buf,
                    bounds_buf,
                    KERN["n_threads"],
                )
            else:
                sums = torch.zeros((k, d), dtype=X.dtype, device=dev).index_add_(
                    0, labels, Xb
                )
                counts = torch.zeros(k, dtype=X.dtype, device=dev).index_add_(
                    0, labels, ones.expand(labels.shape[0])
                )
            t4 = time.perf_counter()
            if last:
                epoch_sums += sums
                epoch_counts += counts
            alpha = scheduled_step_size(s0, step, n_steps)
            assigned = counts > 0
            retained = (1.0 - alpha) * w
            w = retained + alpha * counts.to(torch.float64)
            safe = torch.where(assigned, w, torch.ones_like(w))
            coef_c = torch.where(assigned, retained / safe, 1.0).to(C.dtype)
            coef_s = torch.where(assigned, alpha / safe, 0.0).to(C.dtype)
            C.mul_(coef_c[:, None]).addcmul_(sums, coef_s[:, None])
            step += 1
            force = False
            t5 = time.perf_counter()
            if revival > 0 and d > 2 and step % REVIVAL_PERIOD == 0:
                res = revive(C, w, revival, gen)
                if res is not None:
                    n_revived += res[0].numel()
                    assigner.teleport(res[0], res[1], step)
                    force = True
            assigner.refresh(C, half_norms(C), w, step, force)
            t6 = time.perf_counter()
            tm["gather"] += t2 - t1
            tm["assign"] += t3 - t2
            tm["acc"] += t4 - t3
            tm["update"] += t5 - t4
            tm["refresh"] += t6 - t5
    ok = epoch_counts > 0
    C[ok] = (epoch_sums[ok] / epoch_counts[ok, None]).to(C.dtype)
    labels = None
    if final_labels:
        assigner.refresh(C, half_norms(C), w, step, force=True)
        chunk = 2**18
        final_kw = {"final": True} if assigner.name == "ivf" else {}
        allrows = torch.arange(n, device=dev)
        labels = torch.cat(
            [
                assigner.assign(
                    X[s : s + chunk], allrows[s : s + chunk], step, **final_kw
                )
                for s in range(0, n, chunk)
            ]
        )
    stats = {
        "batch_size": b,
        "n_steps": n_steps,
        "n_revived": n_revived,
        "t_diag": t_diag,
        "tm": {a: round(v, 3) for a, v in tm.items()},
    }
    return C, labels, stats


def random_init(X, k, seed):
    gen = torch.Generator().manual_seed(seed + 12345)
    idx = torch.randperm(X.shape[0], generator=gen)[:k].to(X.device)
    return X[idx].clone()


def allocate(sizes, k):
    """Number of sub-clusters of each group, proportional to its size."""
    sizes = sizes.double().cpu()
    quota = sizes * k / sizes.sum()
    base = torch.minimum(quota.floor().clamp_min(1), sizes).long()
    base[sizes == 0] = 0
    diff = k - int(base.sum())
    frac = quota - base
    if diff > 0:
        room = sizes.long() - base
        order = torch.argsort(frac, descending=True)
        while diff > 0:
            for j in order.tolist():
                if diff == 0:
                    break
                if room[j] > 0:
                    base[j] += 1
                    room[j] -= 1
                    diff -= 1
    elif diff < 0:
        order = torch.argsort(frac)
        while diff < 0:
            for j in order.tolist():
                if diff == 0:
                    break
                if base[j] > 1:
                    base[j] -= 1
                    diff += 1
    return base


def h2_fit(X, k, T, g, seed=0, revival=0.1, alloc="prop", top_T=None, top_n=None):
    """Frozen two-level e-means. Returns centers, labels, top centers, the
    group of each center, and stats."""
    n, d = X.shape
    dev = X.device
    top_T = T if top_T is None else top_T
    if top_n is not None and top_n < n:
        gen = torch.Generator().manual_seed(seed + 777)
        Xtop = X[torch.randperm(n, generator=gen)[:top_n].to(dev)]
    else:
        Xtop = X
    G, top_labels, _ = emeans_fit(
        Xtop,
        random_init(Xtop, g, seed),
        top_T,
        FlatAssigner(),
        revival,
        seed,
        final_labels=Xtop is X,
    )
    if top_labels is None:
        top_labels = assign_flat(X, G)
    sizes = torch.bincount(top_labels, minlength=g)
    if alloc == "prop":
        ks = allocate(sizes, k)
    else:
        ks = allocate(torch.ones(g), k)
        ks = torch.minimum(ks, sizes.cpu())
        ks = ks + (allocate(sizes, k - int(ks.sum())) if int(ks.sum()) < k else 0) * 0
    order = torch.argsort(top_labels, stable=True)
    starts = (torch.cumsum(sizes, 0) - sizes).tolist()
    sizes_l = sizes.tolist()
    C = torch.empty((int(ks.sum()), d), dtype=X.dtype, device=dev)
    member = torch.empty(C.shape[0], dtype=torch.long, device=dev)
    labels = torch.empty(n, dtype=torch.long, device=dev)
    off = 0
    n_steps = 0
    for j in range(g):
        kj = int(ks[j])
        if kj == 0:
            continue
        idx = order[starts[j] : starts[j] + sizes_l[j]]
        Xj = X[idx]
        if kj == 1:
            Cj = Xj.mean(0, keepdim=True)
            lj = torch.zeros(Xj.shape[0], dtype=torch.long, device=dev)
        else:
            Cj, lj, st = emeans_fit(
                Xj,
                random_init(Xj, kj, seed + j),
                T,
                FlatAssigner(),
                revival,
                seed + j,
            )
            n_steps += st["n_steps"]
        C[off : off + kj] = Cj
        member[off : off + kj] = j
        labels[idx] = lj + off
        off += kj
    return C, labels, G, member, {"n_steps": n_steps, "max_ks": int(ks.max())}


def coarse_from_centers(C, g, seed):
    """Groups of centers: e-means of the k centers into g groups."""
    G, member, _ = emeans_fit(
        C, random_init(C, g, seed + 99), 10, FlatAssigner(), 0.1, seed + 99
    )
    return G, member


def evaluate(X, C, labels, eval_rows):
    """Exact inertia of the centers and inertia of the labels, on eval rows."""
    work = WORK["dots"]
    Xe = X[eval_rows]
    xn = (Xe.double() * Xe.double()).sum(1)
    H = half_norms(C)
    exact, best = assign_flat(Xe, C, H, return_scores=True)
    out = {"inertia_exact": float((xn - 2 * best.double()).sum() / len(eval_rows))}
    if labels is not None:
        le = labels[eval_rows]
        got = (Xe * C[le]).sum(1) - H[le]
        out["inertia_labels"] = float((xn - 2 * got.double()).sum() / len(eval_rows))
        out["recall"] = float((le == exact).float().mean())
        sizes = torch.bincount(labels, minlength=C.shape[0]).double()
        mean = sizes.mean()
        out["size_cv"] = float(sizes.std() / mean)
        out["size_min"] = int(sizes.min())
        out["size_max"] = int(sizes.max())
        out["frac_small"] = float((sizes < 0.25 * mean).double().mean())
        out["frac_large"] = float((sizes > 4 * mean).double().mean())
        q = torch.quantile(
            sizes,
            torch.tensor([0.01, 0.5, 0.99], dtype=sizes.dtype, device=sizes.device),
        )
        out["size_q01"], out["size_q50"], out["size_q99"] = (float(v) for v in q)
    WORK["dots"] = work
    return out


def run(X, cfg, eval_rows):
    """Run one configuration (a dict) and return its results."""
    cfg = dict(cfg)
    method = cfg["method"]
    k, T, seed = cfg["k"], cfg.get("T", 5), cfg.get("seed", 0)
    revival = cfg.get("revival", 0.1)
    n = X.shape[0]
    dev = X.device
    g = cfg.get("g") or max(2, round(math.sqrt(k) * cfg.get("gf", 1.0)))
    WORK["dots"] = 0
    diag = [] if cfg.get("diag") else None
    extra = {}
    sync(dev)
    t0 = time.perf_counter()
    t_init = 0.0
    if method == "flat":
        C, labels, st = emeans_fit(
            X, random_init(X, k, seed), T, FlatAssigner(), revival, seed, diag=diag
        )
    elif method == "h2":
        C, labels, G, member, st = h2_fit(
            X,
            k,
            T,
            g,
            seed,
            revival,
            top_T=cfg.get("top_T"),
            top_n=cfg.get("top_n"),
        )
    elif method in ("ivf", "graph"):
        init = cfg.get("init", "random")
        init_labels = None
        if init == "random":
            C0 = random_init(X, k, seed)
            G, member = coarse_from_centers(C0, g, seed)
        else:
            C0, init_labels, G, member, st0 = h2_fit(
                X,
                k,
                cfg.get("init_T", 1),
                g,
                seed,
                revival,
                top_T=cfg.get("top_T"),
                top_n=cfg.get("top_n"),
            )
        sync(dev)
        t_init = time.perf_counter() - t0
        extra["work_init"] = WORK["dots"] / (n * k)
        ivf = IVFAssigner(
            G,
            member,
            cfg.get("p", 4),
            cfg.get("coarse", "lloyd"),
            cfg.get("period", 1),
        )
        ivf.p_final = cfg.get("p_final", ivf.p)
        ivf.ratio2 = cfg.get("ratio", 0.0) ** 2
        if cfg.get("late_epochs"):
            # p_final probes during the last `late_epochs` epochs too.
            b_ = resolve_batch_size(n, k, T)
            total = sum(math.ceil(r / b_) for r in epochs_n_rows(n, T))
            ivf.late_step = total - round(cfg["late_epochs"] * math.ceil(n / b_))
        if method == "ivf":
            assigner = ivf
        else:
            assigner = GraphAssigner(
                n,
                cfg.get("m", 32),
                ivf,
                cfg.get("graph_period", 1),
                cfg.get("p_graph"),
                cfg.get("exact_graph", False),
                labels=init_labels,
                device=dev,
                hops=cfg.get("hops", 1),
            )
        C, labels, st = emeans_fit(X, C0, T, assigner, revival, seed, diag=diag)
        extra["max_group"] = max(ivf.counts)
        if method == "graph":
            extra["frac_fallback"] = assigner.n_fallback / (n * (T + 1))
    else:
        raise ValueError(method)
    sync(dev)
    t_fit = time.perf_counter() - t0 - st.get("t_diag", 0.0)
    res = dict(cfg)
    res.update(
        n=n,
        d=X.shape[1],
        g=g,
        t_fit=t_fit,
        t_init=t_init,
        work=WORK["dots"] / (n * k),
        **extra,
    )
    res.update({key: v for key, v in st.items() if key != "t_diag"})
    res.update(evaluate(X, C, labels, eval_rows))
    if diag:
        res["diag"] = diag
    return res
