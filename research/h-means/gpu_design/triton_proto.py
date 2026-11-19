"""Prototype Triton kernels of the two-level assignment (design benchmark).

Same layout as the kernels of o-means (`contrib/o_means/triton_two_level.py`
of the owlgebra-ai/faiss fork, MIT license, Copyright (c) Meta Platforms, Inc.
and affiliates), from which the packed keys of the coarse kernel and the
structure of the fine kernel are taken. What differs:

- the probes are adaptive (ratio rule, at most P cells), so the pairs are
  counted first and placed with a prefix sum instead of fixed-size buckets;
- the rows and the centers are read in float32 (TF32 or IEEE products) or in
  float16, without padded copies;
- the blocks are sized from the number of cells, to stay under the shared
  memory limit of the GPU.
"""

import os

import torch
import triton
import triton.language as tl

INT_MAX = 2147483647


@triton.jit
def _pack_keys(score, gid, GP: tl.constexpr):
    # Order-preserving int32 image of the float32 scores, with the cell in
    # the low bits: one probe is one min.
    bits = score.to(tl.int32, bitcast=True)
    key = tl.where(bits < 0, bits ^ 2147483647, bits)
    return (key & (-GP)) | gid[None, :]


@triton.jit(do_not_specialize=["R"])
def coarse_kernel(
    x_ptr,
    idx_ptr,
    g_ptr,
    gn_ptr,
    probes_ptr,
    cnt_ptr,
    R,
    D,
    sx,
    sg,
    ratio2,
    HAS_IDX: tl.constexpr,
    P: tl.constexpr,
    GP: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PRECISION: tl.constexpr,
    DYNAMIC: tl.constexpr,
):
    pid = tl.program_id(0)
    r = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    rmask = r < R
    if HAS_IDX:
        rows = tl.load(idx_ptr + r, mask=rmask, other=0).to(tl.int64)
    else:
        rows = r.to(tl.int64)
    xbase = x_ptr + rows[:, None] * sx
    gid = tl.arange(0, GP)
    gbase = g_ptr + gid[:, None].to(tl.int64) * sg
    acc = tl.zeros((BLOCK_B, GP), tl.float32)
    xn = tl.zeros((BLOCK_B,), tl.float32)
    for d0 in range(0, D, BLOCK_D):
        col = d0 + tl.arange(0, BLOCK_D)
        cmask = col < D
        xt = tl.load(
            xbase + col[None, :], mask=rmask[:, None] & cmask[None, :], other=0.0
        )
        gt = tl.load(gbase + col[None, :], mask=cmask[None, :], other=0.0)
        acc = tl.dot(
            xt, tl.trans(gt), acc=acc, input_precision=PRECISION, out_dtype=tl.float32
        )
        xf = xt.to(tl.float32)
        xn += tl.sum(xf * xf, axis=1)
    # ||g||² - 2 x.g: +inf for the empty cells and the padding.
    score = tl.load(gn_ptr + gid)[None, :] - 2.0 * acc
    dmin = tl.min(score, axis=1)
    nearest = tl.maximum(xn + dmin, 0.0)
    threshold = tl.maximum(ratio2 * nearest - xn, dmin)
    key = _pack_keys(score, gid, GP)
    key = tl.where(score <= threshold[:, None], key, 2147483647)
    one = tl.full((BLOCK_B,), 1, tl.int32)
    base = probes_ptr + r.to(tl.int64) * P
    if DYNAMIC:
        s = 0
        kmin = tl.min(key, axis=1)
        valid = (kmin != 2147483647) & rmask
        while (s < P) and (tl.sum(valid.to(tl.int32), axis=0) > 0):
            bj = kmin & (GP - 1)
            key = tl.where(key == kmin[:, None], 2147483647, key)
            tl.store(base + s, bj, mask=valid)
            tl.atomic_add(cnt_ptr + bj, one, mask=valid, sem="relaxed")
            s += 1
            kmin = tl.min(key, axis=1)
            valid = (kmin != 2147483647) & rmask
    else:
        for s in tl.static_range(P):
            kmin = tl.min(key, axis=1)
            valid = (kmin != 2147483647) & rmask
            bj = kmin & (GP - 1)
            key = tl.where(key == kmin[:, None], 2147483647, key)
            tl.store(base + s, tl.where(valid, bj, -1), mask=rmask)
            tl.atomic_add(cnt_ptr + bj, one, mask=valid, sem="relaxed")


@triton.jit(do_not_specialize=["R"])
def scatter_kernel(
    probes_ptr, cursor_ptr, pairs_ptr, R, P: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    r = pid * BLOCK + tl.arange(0, BLOCK)
    rmask = r < R
    one = tl.full((BLOCK,), 1, tl.int32)
    for s in tl.static_range(P):
        cell = tl.load(probes_ptr + r.to(tl.int64) * P + s, mask=rmask, other=-1)
        valid = cell >= 0
        cell = tl.where(valid, cell, 0)
        pos = tl.atomic_add(cursor_ptr + cell, one, mask=valid, sem="relaxed")
        tl.store(pairs_ptr + pos, r * P + s, mask=valid)


@triton.jit
def fine_kernel(
    x_ptr,
    idx_ptr,
    c_ptr,
    cn_ptr,
    order_ptr,
    off_ptr,
    pair_off_ptr,
    tile_cell_ptr,
    tile_start_ptr,
    pairs_ptr,
    best_ptr,
    lab_ptr,
    D,
    sx,
    sc,
    HAS_IDX: tl.constexpr,
    P: tl.constexpr,
    BT: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PRECISION: tl.constexpr,
):
    t = tl.program_id(0)
    c = tl.load(tile_cell_ptr + t)
    q0 = tl.load(tile_start_ptr + t)
    q_end = tl.load(pair_off_ptr + c + 1)
    q = q0 + tl.arange(0, BT)
    qm = q < q_end
    pair = tl.load(pairs_ptr + q, mask=qm, other=0)
    rl = pair // P
    if HAS_IDX:
        rows = tl.load(idx_ptr + rl, mask=qm, other=0).to(tl.int64)
    else:
        rows = rl.to(tl.int64)
    xbase = x_ptr + rows[:, None] * sx
    k_lo = tl.load(off_ptr + c)
    k_hi = tl.load(off_ptr + c + 1)
    best = tl.full((BT,), float("inf"), tl.float32)
    lab = tl.zeros((BT,), tl.int32)
    for k0 in range(k_lo, k_hi, BLOCK_K):
        cid = k0 + tl.arange(0, BLOCK_K)
        cm = cid < k_hi
        cbase = c_ptr + cid[:, None].to(tl.int64) * sc
        acc = tl.zeros((BT, BLOCK_K), tl.float32)
        for d0 in range(0, D, BLOCK_D):
            col = d0 + tl.arange(0, BLOCK_D)
            cmask = col < D
            xt = tl.load(
                xbase + col[None, :], mask=qm[:, None] & cmask[None, :], other=0.0
            )
            ct = tl.load(
                cbase + col[None, :], mask=cm[:, None] & cmask[None, :], other=0.0
            )
            acc = tl.dot(
                xt,
                tl.trans(ct),
                acc=acc,
                input_precision=PRECISION,
                out_dtype=tl.float32,
            )
        score = tl.load(cn_ptr + cid, mask=cm, other=float("inf"))[None, :] - 2.0 * acc
        bs, bl = tl.min(
            score, axis=1, return_indices=True, return_indices_tie_break_left=True
        )
        better = bs < best
        best = tl.where(better, bs, best)
        lab = tl.where(better, bl.to(tl.int32) + k0, lab)
    lab = tl.load(order_ptr + lab, mask=qm, other=0)
    tl.store(best_ptr + pair, best, mask=qm)
    tl.store(lab_ptr + pair, lab, mask=qm)


@triton.jit(do_not_specialize=["R"])
def reduce_kernel(
    probes_ptr, best_ptr, lab_ptr, out_ptr, R, P: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    r = pid * BLOCK + tl.arange(0, BLOCK)
    m = r < R
    base = r.to(tl.int64) * P
    b = tl.load(best_ptr + base, mask=m, other=float("inf"))
    lab = tl.load(lab_ptr + base, mask=m, other=0)
    for s in tl.static_range(1, P):
        valid = tl.load(probes_ptr + base + s, mask=m, other=-1) >= 0
        bp = tl.load(best_ptr + base + s, mask=valid, other=float("inf"))
        lp = tl.load(lab_ptr + base + s, mask=valid, other=0)
        better = valid & ((bp < b) | ((bp == b) & (lp < lab)))
        b = tl.where(better, bp, b)
        lab = tl.where(better, lp, lab)
    tl.store(out_ptr + r, lab, mask=m)


class Cells:
    """Cells of a set of centers, in the layout of the kernels."""

    def __init__(self, C, G, member, dtype):
        dev = C.device
        g = G.shape[0]
        self.g = g
        self.GP = max(16, triton.next_power_of_2(g))
        self.order = torch.argsort(member, stable=True).to(torch.int32)
        size = torch.bincount(member, minlength=g)
        self.size = size
        self.off = torch.zeros(g + 1, dtype=torch.int32, device=dev)
        self.off[1:] = torch.cumsum(size, 0)
        self.Cs = C[self.order.long()].to(dtype).contiguous()
        self.cn = self.Cs.float().square().sum(1)
        self.G = torch.zeros((self.GP, C.shape[1]), dtype=dtype, device=dev)
        self.G[:g] = G
        self.gn = torch.full((self.GP,), float("inf"), device=dev)
        self.gn[:g] = torch.where(
            size > 0, self.G[:g].float().square().sum(1), float("inf")
        )


class Workspace:
    def __init__(self, rows, cells, P, device):
        self.P = P
        self.probes = torch.empty((rows, P), dtype=torch.int32, device=device)
        self.cnt = torch.zeros(cells.GP, dtype=torch.int32, device=device)
        self.pairs = torch.empty(rows * P, dtype=torch.int32, device=device)
        self.best = torch.empty(rows * P, dtype=torch.float32, device=device)
        self.lab = torch.empty(rows * P, dtype=torch.int32, device=device)
        self.labels = torch.empty(rows, dtype=torch.int32, device=device)


# Software pipelining keeps `num_stages - 1` copies of the operands of the
# product in shared memory, and the cells are one operand of the coarse
# kernel: GP * 32 * itemsize bytes per stage. One stage fits on all GPUs.
COARSE_STAGES = int(os.environ.get("COARSE_STAGES", "1"))
COARSE_BLOCK = int(os.environ.get("COARSE_BLOCK", "32"))


def coarse(
    X,
    cells,
    ws,
    ratio2,
    idx=None,
    precision="tf32",
    dynamic=True,
    block_b=None,
    num_warps=8,
    num_stages=None,
):
    R = X.shape[0] if idx is None else idx.shape[0]
    P, GP = ws.P, cells.GP
    if block_b is None:
        block_b = COARSE_BLOCK
    ws.cnt.zero_()
    if dynamic:
        ws.probes[:R].fill_(-1)
    coarse_kernel[(triton.cdiv(R, block_b),)](
        X,
        X if idx is None else idx,
        cells.G,
        cells.gn,
        ws.probes,
        ws.cnt,
        R,
        X.shape[1],
        X.stride(0),
        cells.G.stride(0),
        ratio2,
        HAS_IDX=idx is not None,
        P=P,
        GP=GP,
        BLOCK_B=block_b,
        BLOCK_D=32,
        PRECISION=precision,
        DYNAMIC=dynamic,
        num_warps=num_warps,
        num_stages=COARSE_STAGES if num_stages is None else num_stages,
    )
    return R


def fine(
    X,
    cells,
    ws,
    R,
    idx=None,
    precision="tf32",
    block_t=64,
    block_k=64,
    num_warps=4,
    num_stages=2,
):
    P, g = ws.P, cells.g
    dev = X.device
    cnt = ws.cnt[:g]
    pair_off = torch.zeros(g + 1, dtype=torch.int32, device=dev)
    pair_off[1:] = torch.cumsum(cnt, 0)
    cursor = pair_off[:g].clone()
    scatter_kernel[(triton.cdiv(R, 256),)](
        ws.probes, cursor, ws.pairs, R, P=P, BLOCK=256, num_warps=4
    )
    n_tiles_cell = (cnt + block_t - 1) // block_t
    tile_cell = torch.repeat_interleave(
        torch.arange(g, dtype=torch.int32, device=dev), n_tiles_cell
    )
    first_tile = torch.cumsum(n_tiles_cell, 0) - n_tiles_cell
    tc = tile_cell.long()
    tile_start = (
        pair_off[:g][tc]
        + (torch.arange(tile_cell.shape[0], device=dev) - first_tile[tc]) * block_t
    ).to(torch.int32)
    n_tiles = tile_cell.shape[0]
    fine_kernel[(n_tiles,)](
        X,
        X if idx is None else idx,
        cells.Cs,
        cells.cn,
        cells.order,
        cells.off,
        pair_off,
        tile_cell,
        tile_start,
        ws.pairs,
        ws.best,
        ws.lab,
        X.shape[1],
        X.stride(0),
        cells.Cs.stride(0),
        HAS_IDX=idx is not None,
        P=P,
        BT=block_t,
        BLOCK_K=block_k,
        BLOCK_D=32,
        PRECISION=precision,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    reduce_kernel[(triton.cdiv(R, 256),)](
        ws.probes, ws.best, ws.lab, ws.labels, R, P=P, BLOCK=256, num_warps=4
    )
    return ws.labels[:R]


def assign(X, cells, ws, ratio2, idx=None, precision="tf32", dynamic=True):
    R = coarse(X, cells, ws, ratio2, idx, precision, dynamic)
    return fine(X, cells, ws, R, idx, precision)
