"""Prototype CUDA C kernels (CuPy RawKernel) of the two-level assignment.

Called on PyTorch tensors through `__cuda_array_interface__`. Float32 only:
CUDA C has no access to the tensor cores, so the fine kernel is a tiled
product in shared memory with scalar FMAs. The coarse scores come from one
GEMM of the library (`torch.addmm`), then a kernel selects the probes.
"""

import cupy as cp
import torch

BT = 32  # pairs per tile
BK = 32  # centers per block

_SOURCE = r"""
#define INFINITY (__int_as_float(0x7f800000))
#define BT 32
#define BK 32

extern "C" __global__ void select_probes(
    const float* scores, const float* xn, int R, int g, int P, float ratio2,
    int* probes, int* cnt)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= R) return;
    const float* s = scores + (long long) r * g;
    float dmin = INFINITY;
    for (int c = 0; c < g; c++) dmin = fminf(dmin, s[c]);
    float nearest = fmaxf(xn[r] + dmin, 0.f);
    float thr = fmaxf(ratio2 * nearest - xn[r], dmin);
    float bd[16];
    int bc[16];
    int n = 0;
    for (int c = 0; c < g; c++) {
        float v = s[c];
        if (v <= thr && (n < P || v < bd[n - 1])) {
            int q = n < P ? n : P - 1;
            while (q > 0 && bd[q - 1] > v) {
                bd[q] = bd[q - 1];
                bc[q] = bc[q - 1];
                q--;
            }
            bd[q] = v;
            bc[q] = c;
            if (n < P) n++;
        }
    }
    for (int q = 0; q < P; q++) {
        probes[(long long) r * P + q] = q < n ? bc[q] : -1;
        if (q < n) atomicAdd(&cnt[bc[q]], 1);
    }
}

extern "C" __global__ void scatter_pairs(
    const int* probes, int* cursor, int* pairs, int R, int P)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= R) return;
    for (int q = 0; q < P; q++) {
        int c = probes[(long long) r * P + q];
        if (c < 0) break;
        pairs[atomicAdd(&cursor[c], 1)] = r * P + q;
    }
}

// One block per tile of BT pairs of a cell, one thread per (pair, center of
// a block of BK centers). Shared memory: the rows of the tile, the block of
// centers (transposed, to avoid bank conflicts) and their scores.
extern "C" __global__ void fine(
    const float* X, const long long* idx, const float* Cs, const float* cn,
    const int* order, const int* off, const int* pair_off,
    const int* tile_cell, const int* tile_start, const int* pairs,
    float* best, int* lab, int D, int P, int has_idx)
{
    extern __shared__ float smem[];
    float* xs = smem;                    // [BT][D]
    float* cs = xs + BT * D;             // [D][BK]
    float* sc = cs + BK * D;             // [BT][BK]
    int* prs = (int*) (sc + BT * BK);    // [BT]
    int j = threadIdx.x, i = threadIdx.y;
    int tid = i * BK + j;
    int t = blockIdx.x;
    int c = tile_cell[t];
    int q0 = tile_start[t];
    int n_pairs = min(BT, pair_off[c + 1] - q0);
    if (tid < BT) prs[tid] = tid < n_pairs ? pairs[q0 + tid] : -1;
    __syncthreads();
    for (int e = tid; e < BT * D; e += BT * BK) {
        int row = e / D;
        int pr = prs[row];
        float v = 0.f;
        if (pr >= 0) {
            long long x_row = pr / P;
            if (has_idx) x_row = idx[x_row];
            v = X[x_row * D + (e - row * D)];
        }
        xs[e] = v;
    }
    int k_lo = off[c], k_hi = off[c + 1];
    float best_i = INFINITY;
    int lab_i = 0;
    for (int k0 = k_lo; k0 < k_hi; k0 += BK) {
        for (int e = tid; e < BK * D; e += BT * BK) {
            int cj = e / D, d = e - cj * D;
            cs[d * BK + cj] = k0 + cj < k_hi ? Cs[(long long) (k0 + cj) * D + d] : 0.f;
        }
        __syncthreads();
        float acc = 0.f;
        const float* xr = xs + i * D;
        for (int d = 0; d < D; d++) acc += xr[d] * cs[d * BK + j];
        sc[i * BK + j] = k0 + j < k_hi ? cn[k0 + j] - 2.f * acc : INFINITY;
        __syncthreads();
        if (j == 0) {
            for (int jj = 0; jj < BK; jj++) {
                float v = sc[i * BK + jj];
                if (v < best_i) {
                    best_i = v;
                    lab_i = k0 + jj;
                }
            }
        }
        __syncthreads();
    }
    if (j == 0 && i < n_pairs) {
        best[prs[i]] = best_i;
        lab[prs[i]] = order[lab_i];
    }
}

extern "C" __global__ void reduce_probes(
    const int* probes, const float* best, const int* lab, int* out, int R, int P)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= R) return;
    long long base = (long long) r * P;
    float b = best[base];
    int l = lab[base];
    for (int q = 1; q < P; q++) {
        if (probes[base + q] < 0) break;
        float bp = best[base + q];
        int lp = lab[base + q];
        if (bp < b || (bp == b && lp < l)) {
            b = bp;
            l = lp;
        }
    }
    out[r] = l;
}
"""

_module = cp.RawModule(code=_SOURCE)
_select = _module.get_function("select_probes")
_scatter = _module.get_function("scatter_pairs")
_fine = _module.get_function("fine")
_reduce = _module.get_function("reduce_probes")


def _cp(t):
    return cp.asarray(t)


def coarse(X, cells, ws, ratio2, idx=None, chunk=65536):
    """cells.G and cells.gn are float32, not padded (g rows)."""
    Xb = X if idx is None else X[idx]
    R = Xb.shape[0]
    g, P = cells.g, ws.P
    ws.cnt.zero_()
    cnt = _cp(ws.cnt)
    probes = _cp(ws.probes)
    G = cells.G[:g]
    gn = cells.gn[:g]
    for start in range(0, R, chunk):
        rows = Xb[start : start + chunk]
        scores = torch.addmm(gn[None, :], rows, G.t(), alpha=-2.0)
        xn = rows.square().sum(1)
        n = rows.shape[0]
        _select(
            ((n + 127) // 128,),
            (128,),
            (
                _cp(scores),
                _cp(xn),
                cp.int32(n),
                cp.int32(g),
                cp.int32(P),
                cp.float32(ratio2),
                probes[start:],
                cnt,
            ),
        )
    return R


def fine(X, cells, ws, R, idx=None):
    P, g = ws.P, cells.g
    dev = X.device
    D = X.shape[1]
    cnt = ws.cnt[:g]
    pair_off = torch.zeros(g + 1, dtype=torch.int32, device=dev)
    pair_off[1:] = torch.cumsum(cnt, 0)
    cursor = pair_off[:g].clone()
    grid = ((R + 255) // 256,)
    _scatter(
        grid,
        (256,),
        (_cp(ws.probes), _cp(cursor), _cp(ws.pairs), cp.int32(R), cp.int32(P)),
    )
    n_tiles_cell = (cnt + BT - 1) // BT
    tile_cell = torch.repeat_interleave(
        torch.arange(g, dtype=torch.int32, device=dev), n_tiles_cell
    )
    first_tile = torch.cumsum(n_tiles_cell, 0) - n_tiles_cell
    tc = tile_cell.long()
    tile_start = (
        pair_off[:g][tc]
        + (torch.arange(tile_cell.shape[0], device=dev) - first_tile[tc]) * BT
    ).to(torch.int32)
    n_tiles = tile_cell.shape[0]
    _fine(
        (n_tiles,),
        (BK, BT),
        (
            _cp(X),
            _cp(X if idx is None else idx),
            _cp(cells.Cs),
            _cp(cells.cn),
            _cp(cells.order),
            _cp(cells.off),
            _cp(pair_off),
            _cp(tile_cell),
            _cp(tile_start),
            _cp(ws.pairs),
            _cp(ws.best),
            _cp(ws.lab),
            cp.int32(D),
            cp.int32(P),
            cp.int32(idx is not None),
        ),
        shared_mem=4 * (BT * D + BK * D + BT * BK + BT),
    )
    _reduce(
        grid,
        (256,),
        (
            _cp(ws.probes),
            _cp(ws.best),
            _cp(ws.lab),
            _cp(ws.labels),
            cp.int32(R),
            cp.int32(P),
        ),
    )
    return ws.labels[:R]
