"""Design benchmark: the two-level assignment of one batch, alone, on a GPU.

    python bench_assign.py DATA.npy K [--methods exact,padded,loop,omeans,triton,cuda]

One batch of 2**20 rows is assigned to K centers through round(sqrt(K)) cells.
Each line is one way to do it: time of the coarse step (probes of each row)
and of the rest, mean number of probed cells, inertia of the labels against
the exact assignment and share of the rows that get their nearest center.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEV = "cuda"
BATCH = 2**20


def sync():
    torch.cuda.synchronize()


def timed(fn, reps=5):
    fn()
    sync()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        sync()
        times.append(time.perf_counter() - t0)
    return out, float(np.median(times)) * 1e3


def exact_assign(X, C, chunk=2**28):
    H = 0.5 * C.float().square().sum(1)
    labels = []
    step = max(1, chunk // C.shape[0])
    for s in range(0, X.shape[0], step):
        labels.append(F.linear(X[s : s + step], C, -H.to(C.dtype)).argmax(1))
    return torch.cat(labels)


def make_problem(X, k, seed=0):
    """Centers after one Lloyd iteration from random rows, and their cells."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    n, d = X.shape
    C = X[torch.randperm(n, generator=gen, device=DEV)[:k]].clone()
    labels = exact_assign(X, C)
    sums = torch.zeros_like(C).index_add_(0, labels, X)
    cnt = torch.bincount(labels, minlength=k).float()
    C = torch.where((cnt > 0)[:, None], sums / cnt.clamp_min(1)[:, None], C)
    g = max(2, round(math.sqrt(k)))
    G = C[torch.randperm(k, generator=gen, device=DEV)[:g]].clone()
    for _ in range(6):
        member = exact_assign(C, G)
        sums = torch.zeros_like(G).index_add_(0, member, C)
        cnt = torch.bincount(member, minlength=g).float()
        G = torch.where((cnt > 0)[:, None], sums / cnt.clamp_min(1)[:, None], G)
    member = exact_assign(C, G)
    return C, G, member


def inertia(X, C, labels):
    return float((X - C[labels.long()]).double().square().sum(1).mean())


# ---------------------------------------------------------------- option 1


class Padded:
    """No custom kernel: cells padded to the same size, batched matmul."""

    def __init__(self, C, G, member, dtype):
        g, d = G.shape
        self.g, self.dtype = g, dtype
        self.G = G
        self.HG = 0.5 * G.square().sum(1)
        size = torch.bincount(member, minlength=g)
        self.HG[size == 0] = torch.inf
        self.order = torch.argsort(member, stable=True)
        off = torch.cumsum(size, 0) - size
        self.Lmax = int(size.max())
        pos = torch.arange(C.shape[0], device=DEV) - off[member[self.order]]
        self.Cp = torch.zeros((g, self.Lmax, d), dtype=dtype, device=DEV)
        self.Cp[member[self.order], pos] = C[self.order].to(dtype)
        self.Hp = torch.full((g, self.Lmax), torch.inf, dtype=dtype, device=DEV)
        self.Hp[member[self.order], pos] = (
            0.5 * self.Cp[member[self.order], pos].float().square().sum(1)
        ).to(dtype)
        self.lab = torch.zeros((g, self.Lmax), dtype=torch.long, device=DEV)
        self.lab[member[self.order], pos] = self.order

    def coarse(self, Xb, p, ratio2):
        g = self.g
        chunk = max(1, 2**24 // g)
        tops = [
            F.linear(Xb[s : s + chunk], self.G, -self.HG).topk(p, dim=1)
            for s in range(0, Xb.shape[0], chunk)
        ]
        probes = torch.cat([t.indices for t in tops])
        if ratio2 > 0:
            gd = Xb.square().sum(1, keepdim=True) - 2 * torch.cat(
                [t.values for t in tops]
            )
            pruned = gd > ratio2 * gd[:, :1].clamp_min(0)
            pruned[:, 0] = False
            probes = torch.where(pruned, g, probes)
        return probes

    def fine(self, Xb, probes, max_elements=2**27):
        R, p = probes.shape
        g = self.g
        pair_cell = probes.reshape(-1)
        cnt = torch.bincount(pair_cell, minlength=g + 1)[:g]
        n_valid = int(cnt.sum())
        perm = torch.argsort(pair_cell)[:n_valid]
        cell = pair_cell[perm]
        starts = torch.cumsum(cnt, 0) - cnt
        pos = torch.arange(n_valid, device=DEV) - starts[cell]
        Mmax = int(cnt.max())
        Xe = torch.cat([Xb, torch.zeros_like(Xb[:1])]).to(self.dtype)
        best = torch.full((R * p,), torch.inf, dtype=torch.float32, device=DEV)
        lab = torch.zeros(R * p, dtype=torch.long, device=DEV)
        n_cells = max(1, max_elements // (Mmax * self.Lmax))
        bounds = torch.cat([starts, cnt.sum()[None]]).tolist()
        self.padding = g * Mmax * self.Lmax / float((cnt * self.size_of()).sum())
        for a in range(0, g, n_cells):
            b = min(g, a + n_cells)
            sl = slice(bounds[a], bounds[b])
            rows = torch.full((b - a, Mmax), R, dtype=torch.long, device=DEV)
            rows[cell[sl] - a, pos[sl]] = torch.div(perm[sl], p, rounding_mode="floor")
            scores = torch.baddbmm(
                self.Hp[a:b, None, :], Xe[rows], self.Cp[a:b].transpose(1, 2), alpha=-1
            )
            m, i = scores.min(2)
            best[perm[sl]] = m[cell[sl] - a, pos[sl]].float()
            lab[perm[sl]] = self.lab[cell[sl], i[cell[sl] - a, pos[sl]]]
        return lab.view(R, p).gather(1, best.view(R, p).argmin(1, keepdim=True))[:, 0]

    def size_of(self):
        return torch.isfinite(self.Hp.float()).sum(1)


class Loop:
    """One GEMM per cell, in a Python loop (the prototype of hkm.py)."""

    def __init__(self, C, G, member, dtype):
        self.dtype = dtype
        self.order = torch.argsort(member, stable=True)
        size = torch.bincount(member, minlength=G.shape[0])
        self.counts = size.tolist()
        self.starts = (torch.cumsum(size, 0) - size).tolist()
        self.Cs = C[self.order].to(dtype)
        self.Hs = (0.5 * self.Cs.float().square().sum(1)).to(dtype)

    def fine(self, Xb, probes):
        R, p = probes.shape
        g = len(self.counts)
        Xb = Xb.to(self.dtype)
        pair_cell = probes.reshape(-1)
        perm = torch.argsort(pair_cell)
        pts = torch.div(perm, p, rounding_mode="floor")
        bounds = torch.searchsorted(
            pair_cell[perm], torch.arange(g + 1, device=DEV)
        ).tolist()
        best = torch.full((R * p,), -torch.inf, dtype=torch.float32, device=DEV)
        lab = torch.zeros(R * p, dtype=torch.long, device=DEV)
        for j in range(g):
            a, b = bounds[j], bounds[j + 1]
            cnt, st = self.counts[j], self.starts[j]
            if a == b or cnt == 0:
                continue
            m, i = F.linear(
                Xb[pts[a:b]], self.Cs[st : st + cnt], -self.Hs[st : st + cnt]
            ).max(1)
            best[perm[a:b]] = m.float()
            lab[perm[a:b]] = self.order[st + i]
        return lab.view(R, p).gather(1, best.view(R, p).argmax(1, keepdim=True))[:, 0]


# -------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("k", type=int)
    parser.add_argument("--methods", default="exact,padded,loop,omeans,triton,cuda")
    parser.add_argument("--out", default=None)
    parser.add_argument("--ratio", type=float, default=1.15)
    args = parser.parse_args()
    methods = args.methods.split(",")
    torch.backends.cuda.matmul.allow_tf32 = False

    X = torch.from_numpy(np.load(args.data, mmap_mode="r")[:]).to(DEV)
    n, d = X.shape
    k = args.k
    C, G, member = make_problem(X, k)
    g = G.shape[0]
    Xb = X[:BATCH].contiguous()
    size = torch.bincount(member, minlength=g)
    print(
        f"n={n} d={d} k={k} g={g} cell sizes: mean {k / g:.0f} max {int(size.max())}"
        f" empty {int((size == 0).sum())}",
        flush=True,
    )
    ratio2 = args.ratio**2
    ref_labels, t_exact = timed(
        lambda: exact_assign(Xb, C), reps=2 if k < 500_000 else 1
    )
    ref_inertia = inertia(Xb, C, ref_labels)
    results = []

    def report(name, config, t_coarse, t_total, probes, labels, **extra):
        res = dict(
            method=name,
            config=config,
            k=k,
            coarse_ms=t_coarse,
            total_ms=t_total,
            probes=probes,
            loss_pct=100 * (inertia(Xb, C, labels) / ref_inertia - 1),
            recall=float((labels.long() == ref_labels).float().mean()),
            **extra,
        )
        results.append(res)
        print(
            f"{name:8s} {config:34s} coarse {t_coarse:7.1f} ms  total {t_total:7.1f} ms"
            f"  probes {probes:5.2f}  loss {res['loss_pct']:+.3f}%"
            f"  recall {res['recall']:.4f}  {extra or ''}",
            flush=True,
        )

    # (label, max probes, ratio²): ratio² = 0 is a fixed number of probes.
    rules = [("p=4", 4, 0.0), ("p=8", 8, 0.0), (f"adaptive {args.ratio}", 16, ratio2)]

    if "exact" in methods:
        report("exact", "float32", 0.0, t_exact, g, ref_labels)
        X16, C16 = Xb.half(), C.half()
        labels, t = timed(lambda: exact_assign(X16, C16), reps=2)
        report("exact", "float16", 0.0, t, g, labels)

    if "padded" in methods or "loop" in methods:
        for dtype in (torch.float32, torch.float16):
            padded = Padded(C, G, member, dtype)
            loop = Loop(C, G, member, dtype)
            for label, p, r2 in rules:
                probes, t_coarse = timed(lambda: padded.coarse(Xb, p, r2), reps=3)
                n_probes = float((probes < g).float().sum(1).mean())
                config = f"{label} {str(dtype)[6:]}"
                for name, obj in (("padded", padded), ("loop", loop)):
                    if name not in methods:
                        continue
                    try:
                        labels, t_fine = timed(lambda: obj.fine(Xb, probes), reps=3)
                    except torch.OutOfMemoryError as exc:
                        print(name, config, "OOM", str(exc)[:80])
                        continue
                    extra = (
                        dict(padding=round(padded.padding, 2))
                        if name == "padded"
                        else {}
                    )
                    report(
                        name,
                        config,
                        t_coarse,
                        t_coarse + t_fine,
                        n_probes,
                        labels,
                        **extra,
                    )
            del padded, loop
            torch.cuda.empty_cache()

    if "omeans" in methods:
        import triton
        from faiss.contrib.o_means import triton_two_level as om

        order = torch.argsort(member, stable=True)
        GP = max(16, triton.next_power_of_2(g))
        g16 = torch.zeros((GP, d), dtype=torch.float16, device=DEV)
        g16[:g] = G
        gn = torch.full((GP,), float("inf"), device=DEV)
        gn[:g] = torch.where(size > 0, g16[:g].float().square().sum(1), om.EMPTY_NORM)
        off = torch.zeros(g + 1, dtype=torch.int32, device=DEV)
        off[1:] = torch.cumsum(size, 0)
        cells = om.Cells(
            G=g,
            GP=GP,
            g32=G,
            g16=g16,
            gn=gn,
            off=off,
            size=size,
            child=torch.full((k + 1,), -1, dtype=torch.int32, device=DEV),
        )
        c16 = C[order].half().contiguous()
        cn = c16.float().square().sum(1)
        x16 = Xb.half().contiguous()
        for p in (4, 8, 16):
            try:
                ws = om.TwoLevelWorkspace(BATCH, g, p, DEV)
                labels, t = timed(
                    lambda: order[
                        om.two_level_assign(x16, c16, cn, cells, ws, p).long()
                    ]
                )
                report("omeans", f"p={p} float16", float("nan"), t, p, labels)
            except Exception as exc:
                print("omeans", f"p={p}", "FAILED", repr(exc)[:300], flush=True)
            ws = None
        del cells, c16, x16
        torch.cuda.empty_cache()

    if "triton" in methods:
        import triton_proto as tp

        variants = [
            ("float32 tf32", torch.float32, "tf32"),
            ("float32 ieee", torch.float32, "ieee"),
            ("float16", torch.float16, "tf32"),
        ]
        for vname, dtype, precision in variants:
            cells = tp.Cells(C, G, member, dtype)
            Xv = Xb.to(dtype).contiguous()
            for label, p, r2 in rules:
                r2 = r2 or 1e30
                for dynamic in (True, False):
                    if not dynamic and r2 == 1e30:
                        continue
                    ws = tp.Workspace(BATCH, cells, p, DEV)
                    config = f"{label} {vname}{'' if dynamic else ' static'}"
                    try:
                        R, t_coarse = timed(
                            lambda: tp.coarse(
                                Xv, cells, ws, r2, precision=precision, dynamic=dynamic
                            )
                        )
                        labels, t_fine = timed(
                            lambda: tp.fine(Xv, cells, ws, R, precision=precision)
                        )
                        _, t_total = timed(
                            lambda: tp.assign(
                                Xv, cells, ws, r2, precision=precision, dynamic=dynamic
                            )
                        )
                    except Exception as exc:
                        print("triton", config, "FAILED", repr(exc)[:300], flush=True)
                        continue
                    n_probes = float((ws.probes >= 0).float().sum(1).mean())
                    report(
                        "triton",
                        config,
                        t_coarse,
                        t_total,
                        n_probes,
                        labels,
                        fine_ms=round(t_fine, 1),
                    )
            del cells, Xv, ws
            torch.cuda.empty_cache()

    if "cuda" in methods:
        import cupy_proto as cu
        import triton_proto as tp

        cells = tp.Cells(C, G, member, torch.float32)
        for label, p, r2 in rules:
            r2 = r2 or 1e30
            ws = tp.Workspace(BATCH, cells, p, DEV)
            R, t_coarse = timed(lambda: cu.coarse(Xb, cells, ws, r2))
            labels, t_fine = timed(lambda: cu.fine(Xb, cells, ws, R), reps=3)
            n_probes = float((ws.probes >= 0).float().sum(1).mean())
            report(
                "cuda",
                f"{label} float32",
                t_coarse,
                t_coarse + t_fine,
                n_probes,
                labels,
            )

    if args.out:
        with open(args.out, "a") as f:
            for res in results:
                f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
