# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""CPU kernels of the prototype: OpenMP over chunks of rows, one BLAS thread."""

from cython.parallel import prange, parallel
import numpy as np
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libc.math cimport INFINITY
from scipy.linalg.cython_blas cimport sgemm, sdot

cdef int CHUNK = 256

cdef extern from *:
    """
    /* out[q] = <x, C[q]> for the L rows of the contiguous block C. No BLAS:
       the calls are short, and OpenBLAS takes a lock per call. */
    static inline void dots_block(const float *x, const float *C, long L,
                                  long d, float *out) {
        long q = 0;
        for (; q + 4 <= L; q += 4) {
            const float *c0 = C + q * d, *c1 = c0 + d, *c2 = c1 + d, *c3 = c2 + d;
            float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            #pragma omp simd reduction(+:a0,a1,a2,a3)
            for (long t = 0; t < d; t++) {
                float xv = x[t];
                a0 += xv * c0[t]; a1 += xv * c1[t];
                a2 += xv * c2[t]; a3 += xv * c3[t];
            }
            out[q] = a0; out[q + 1] = a1; out[q + 2] = a2; out[q + 3] = a3;
        }
        for (; q < L; q++) {
            const float *c0 = C + q * d;
            float a0 = 0;
            #pragma omp simd reduction(+:a0)
            for (long t = 0; t < d; t++) a0 += x[t] * c0[t];
            out[q] = a0;
        }
    }
    """
    void dots_block(const float *x, const float *C, long L, long d,
                    float *out) noexcept nogil


def flat_assign(float[:, ::1] X, float[:, ::1] C, float[::1] H,
                long[::1] labels, float[::1] best, int n_threads):
    """Exact nearest center: one GEMM per chunk of rows."""
    cdef:
        Py_ssize_t n = X.shape[0], c, r, j, start
        int d = X.shape[1], k = C.shape[0], rows
        Py_ssize_t n_chunks = (n + CHUNK - 1) // CHUNK
        float *S
        float one = 1.0, zero = 0.0, v, bv
        Py_ssize_t bj
        char *T = 'T'
        char *N = 'N'
    with nogil, parallel(num_threads=n_threads):
        S = <float *> malloc(CHUNK * k * sizeof(float))
        for c in prange(n_chunks, schedule='dynamic'):
            start = c * CHUNK
            rows = <int> min(CHUNK, n - start)
            sgemm(T, N, &k, &rows, &d, &one, &C[0, 0], &d, &X[start, 0], &d,
                  &zero, S, &k)
            for r in range(rows):
                bv = -INFINITY
                bj = 0
                for j in range(k):
                    v = S[r * k + j] - H[j]
                    if v > bv:
                        bv = v
                        bj = j
                labels[start + r] = bj
                best[start + r] = bv
        free(S)


_BUFFERS = {}


def _buffer(name, size, dtype):
    """Grow-only work buffers: fresh ones are page-faulted at each call."""
    buf = _BUFFERS.get(name)
    if buf is None or buf.shape[0] < size:
        buf = np.empty(max(size, 1), dtype=dtype)
        _BUFFERS[name] = buf
    return buf


def ivf_assign(float[:, ::1] X, float[:, ::1] G, float[::1] HG,
               float[:, ::1] Cs, float[::1] Hs, long[::1] starts,
               long[::1] counts, long[::1] order, int p, int max_count,
               long[::1] labels, int n_threads, float ratio2=0.0):
    """Best center among those of the p nearest groups.

    If ratio2 > 0, a group other than the nearest one is only probed when its
    squared distance to the row is below ratio2 times that of the nearest.

    The (row, group) pairs are bucketed by group, so that the fine search is
    one GEMM per chunk of the rows probing a group. Returns the number of
    row-center products of the fine search."""
    cdef:
        Py_ssize_t n = X.shape[0], c, r, j, i, start, st, q, gj, t, a, e, pos
        Py_ssize_t blk, lo, hi, acc, tmp
        int d = X.shape[1], g = G.shape[0], rows, L, inc = 1
        Py_ssize_t n_chunks = (n + CHUNK - 1) // CHUNK
        Py_ssize_t n_pairs = n * p, n_tasks
        int nb = n_threads
        float *S
        float *buf
        float one = 1.0, zero = 0.0, v, bv, gv, xn = 0, d0 = 0
        Py_ssize_t bj
        char *T = 'T'
        char *N = 'N'
        long dots = 0
        int[::1] probes = _buffer("probes", n_pairs, np.intc)
        long[::1] bounds = np.zeros(g + 1, dtype=np.int64)
        long[:, ::1] cnt = np.zeros((nb, g), dtype=np.int64)
        long[::1] pair_slot = _buffer("pair_slot", n_pairs, np.int64)
        float[::1] pair_best = _buffer("pair_best", n_pairs, np.float32)
        long[::1] pair_lab = _buffer("pair_lab", n_pairs, np.int64)
        long[::1] task_cell, task_a
    # 1. the p nearest groups of each row
    with nogil, parallel(num_threads=n_threads):
        S = <float *> malloc(CHUNK * g * sizeof(float))
        for c in prange(n_chunks, schedule='dynamic'):
            start = c * CHUNK
            rows = <int> min(CHUNK, n - start)
            sgemm(T, N, &g, &rows, &d, &one, &G[0, 0], &d, &X[start, 0], &d,
                  &zero, S, &g)
            for r in range(rows):
                for j in range(g):
                    S[r * g + j] = S[r * g + j] - HG[j]
                if ratio2 > 0:
                    xn = sdot(&d, &X[start + r, 0], &inc, &X[start + r, 0], &inc)
                for q in range(p):
                    gv = -INFINITY
                    gj = 0
                    for j in range(g):
                        if S[r * g + j] > gv:
                            gv = S[r * g + j]
                            gj = j
                    S[r * g + gj] = -INFINITY
                    if q == 0:
                        d0 = xn - 2 * gv
                    elif ratio2 > 0 and xn - 2 * gv > ratio2 * d0:
                        # The groups come by increasing distance: the next
                        # ones fail too.
                        for i in range(q, p):
                            probes[(start + r) * p + i] = -1
                        break
                    if counts[gj] == 0:
                        gj = -1
                    probes[(start + r) * p + q] = <int> gj
        free(S)
    # 2. buckets of pairs per group: counting sort, in parallel over blocks
    for blk in prange(nb, schedule='static', nogil=True, num_threads=nb):
        lo = blk * n_pairs // nb
        hi = (blk + 1) * n_pairs // nb
        for i in range(lo, hi):
            if probes[i] >= 0:
                cnt[blk, probes[i]] += 1
    acc = 0
    for j in range(g):
        bounds[j] = acc
        for blk in range(nb):
            tmp = cnt[blk, j]
            cnt[blk, j] = acc
            acc += tmp
    bounds[g] = acc
    for blk in prange(nb, schedule='static', nogil=True, num_threads=nb):
        lo = blk * n_pairs // nb
        hi = (blk + 1) * n_pairs // nb
        for i in range(lo, hi):
            j = probes[i]
            if j >= 0:
                pair_slot[cnt[blk, j]] = i
                cnt[blk, j] += 1
    n_tasks = 0
    for j in range(g):
        n_tasks += (bounds[j + 1] - bounds[j] + CHUNK - 1) // CHUNK
    task_cell = np.empty(max(n_tasks, 1), dtype=np.int64)
    task_a = np.empty(max(n_tasks, 1), dtype=np.int64)
    t = 0
    for j in range(g):
        for a in range(bounds[j], bounds[j + 1], CHUNK):
            task_cell[t] = j
            task_a[t] = a
            t += 1
    # 3. fine search: one GEMM per chunk of the pairs of a group
    with nogil, parallel(num_threads=n_threads):
        S = <float *> malloc(CHUNK * max(max_count, 1) * sizeof(float))
        buf = <float *> malloc(CHUNK * d * sizeof(float))
        for t in prange(n_tasks, schedule='dynamic'):
            j = task_cell[t]
            a = task_a[t]
            e = min(a + CHUNK, bounds[j + 1])
            rows = <int> (e - a)
            L = <int> counts[j]
            st = starts[j]
            for i in range(rows):
                memcpy(buf + i * d, &X[pair_slot[a + i] // p, 0],
                       d * sizeof(float))
            sgemm(T, N, &L, &rows, &d, &one, &Cs[st, 0], &d, buf, &d,
                  &zero, S, &L)
            dots += rows * L
            for i in range(rows):
                bv = -INFINITY
                bj = 0
                for q in range(L):
                    v = S[i * L + q] - Hs[st + q]
                    if v > bv:
                        bv = v
                        bj = q
                pos = pair_slot[a + i]
                pair_best[pos] = bv
                pair_lab[pos] = order[st + bj]
        free(S)
        free(buf)
    # 4. best of the probed groups
    for r in prange(n, schedule='static', nogil=True, num_threads=n_threads):
        bv = -INFINITY
        bj = 0
        for q in range(p):
            if probes[r * p + q] >= 0 and pair_best[r * p + q] > bv:
                bv = pair_best[r * p + q]
                bj = pair_lab[r * p + q]
        labels[r] = bj
    return dots


def ivf_topm(float[:, ::1] X, float[:, ::1] G, float[::1] HG,
             float[:, ::1] Cs, float[::1] Hs, long[::1] starts,
             long[::1] counts, long[::1] order, int p, int max_count,
             long[:, ::1] out, int n_threads):
    """The `out.shape[1]` best centers among those of the p nearest groups,
    -1 padded."""
    cdef:
        Py_ssize_t n = X.shape[0], c, r, j, i, start, st, q, gj, pos
        int d = X.shape[1], g = G.shape[0], rows, L, inc = 1
        int m = out.shape[1]
        Py_ssize_t n_chunks = (n + CHUNK - 1) // CHUNK
        float *S
        float *buf
        float *tv
        float one = 1.0, zero = 0.0, v, gv
        char *T = 'T'
        char *N = 'N'
        long dots = 0
    with nogil, parallel(num_threads=n_threads):
        S = <float *> malloc(CHUNK * g * sizeof(float))
        buf = <float *> malloc(max(max_count, 1) * sizeof(float))
        tv = <float *> malloc(m * sizeof(float))
        for c in prange(n_chunks, schedule='dynamic'):
            start = c * CHUNK
            rows = <int> min(CHUNK, n - start)
            sgemm(T, N, &g, &rows, &d, &one, &G[0, 0], &d, &X[start, 0], &d,
                  &zero, S, &g)
            for r in range(rows):
                for j in range(g):
                    S[r * g + j] = S[r * g + j] - HG[j]
                for i in range(m):
                    tv[i] = -INFINITY
                    out[start + r, i] = -1
                for q in range(p):  # no-cython-lint
                    gv = -INFINITY
                    gj = 0
                    for j in range(g):
                        if S[r * g + j] > gv:
                            gv = S[r * g + j]
                            gj = j
                    S[r * g + gj] = -INFINITY
                    L = <int> counts[gj]
                    if L == 0:
                        continue
                    st = starts[gj]
                    # Not sgemv: OpenBLAS takes a global lock to allocate its
                    # buffer when d is large.
                    for i in range(L):
                        buf[i] = sdot(&d, &Cs[st + i, 0], &inc,
                                      &X[start + r, 0], &inc)
                    dots += L
                    for i in range(L):
                        v = buf[i] - Hs[st + i]
                        if v > tv[m - 1]:
                            # insertion in the sorted top-m
                            pos = m - 1
                            while pos > 0 and tv[pos - 1] < v:
                                tv[pos] = tv[pos - 1]
                                out[start + r, pos] = out[start + r, pos - 1]
                                pos = pos - 1
                            tv[pos] = v
                            out[start + r, pos] = order[st + i]
        free(S)
        free(buf)
        free(tv)
    return dots


def graph_assign(float[:, ::1] X, long[::1] prev, float[:, ::1] C,
                 float[::1] H, long[:, ::1] NB, long[::1] labels, int hops,
                 int n_threads):
    """Best center among the neighbors of the previous center, then among the
    neighbors of that one, `hops` times at most. Rows with prev < 0 are
    skipped (label -1). Returns the number of products."""
    cdef:
        Py_ssize_t n = X.shape[0], r, i, cj, bj, pc, h
        int d = X.shape[1], m = NB.shape[1], inc = 1
        float v, bv
        long dots = 0
    for r in prange(n, schedule='static', nogil=True, num_threads=n_threads):
        pc = prev[r]
        if pc < 0:
            labels[r] = -1
            continue
        bj = pc
        bv = sdot(&d, &X[r, 0], &inc, &C[pc, 0], &inc) - H[pc]
        dots += 1
        for h in range(hops):  # no-cython-lint
            pc = bj
            for i in range(m):
                cj = NB[pc, i]
                if cj == pc:
                    continue
                v = sdot(&d, &X[r, 0], &inc, &C[cj, 0], &inc) - H[cj]
                if v > bv:
                    bv = v
                    bj = cj
            dots += m - 1
            if bj == pc:
                break
        labels[r] = bj
    return dots


def accumulate(float[:, ::1] X, long[::1] labels, float[:, ::1] sums,
               float[::1] counts, long[::1] order, long[::1] bounds,
               int n_threads):
    """Sums and numbers of the rows of each center. `order` (n,) and `bounds`
    (k + 1,) are work arrays."""
    cdef:
        Py_ssize_t n = X.shape[0], k = sums.shape[0], r, j, i, t
        int d = X.shape[1]
        float *row
    for j in range(k + 1):
        bounds[j] = 0
    for r in range(n):
        bounds[labels[r] + 1] += 1
    for j in range(k):
        counts[j] = <float> bounds[j + 1]
        bounds[j + 1] += bounds[j]
    # bounds[j] = start of center j; fill with a moving cursor kept in counts
    for r in range(n):
        j = labels[r]
        order[bounds[j]] = r
        bounds[j] += 1
    # bounds[j] is now the end of center j
    for j in prange(k, schedule='dynamic', chunksize=16, nogil=True,
                    num_threads=n_threads):
        for t in range(d):
            sums[j, t] = 0
        for i in range(bounds[j] - <Py_ssize_t> counts[j], bounds[j]):
            row = &X[order[i], 0]
            for t in range(d):
                sums[j, t] += row[t]


def gather_rows(float[:, ::1] X, long[::1] rows, float[:, ::1] out,
                int n_threads):
    """out[i] = X[rows[i]], in parallel."""
    cdef:
        Py_ssize_t n = rows.shape[0], i
        Py_ssize_t d = X.shape[1]
    for i in prange(n, schedule='static', nogil=True, num_threads=n_threads):
        memcpy(&out[i, 0], &X[rows[i], 0], d * sizeof(float))


def ivf_assign_rw(float[:, ::1] X, float[:, ::1] G, float[::1] HG,
                  float[:, ::1] Cs, float[::1] Hs, long[::1] starts,
                  long[::1] counts, long[::1] order, int p, int max_count,
                  long[::1] labels, int n_threads, float ratio2=0.0):
    """Row-wise version of ivf_assign, without BLAS: the row stays in L1 and
    the blocks of centers in L2/L3."""
    cdef:
        Py_ssize_t n = X.shape[0], r, j, i, st, q, gj, bj, L
        Py_ssize_t d = X.shape[1], g = G.shape[0]
        float *S
        float *buf
        float v, bv, gv, xn, d0
        long dots = 0
    with nogil, parallel(num_threads=n_threads):
        S = <float *> malloc(g * sizeof(float))
        buf = <float *> malloc(max(max_count, 1) * sizeof(float))
        for r in prange(n, schedule='static'):
            dots_block(&X[r, 0], &G[0, 0], g, d, S)
            for j in range(g):
                S[j] = S[j] - HG[j]
            xn = 0
            if ratio2 > 0:
                dots_block(&X[r, 0], &X[r, 0], 1, d, &xn)
            bv = -INFINITY
            bj = -1
            d0 = 0
            for q in range(p):
                gv = -INFINITY
                gj = 0
                for j in range(g):
                    if S[j] > gv:
                        gv = S[j]
                        gj = j
                S[gj] = -INFINITY
                if q == 0:
                    d0 = xn - 2 * gv
                elif ratio2 > 0 and xn - 2 * gv > ratio2 * d0:
                    break
                L = counts[gj]
                if L == 0:
                    continue
                st = starts[gj]
                dots_block(&X[r, 0], &Cs[st, 0], L, d, buf)
                dots += L
                for i in range(L):
                    v = buf[i] - Hs[st + i]
                    if v > bv:
                        bv = v
                        bj = st + i
            labels[r] = order[bj] if bj >= 0 else 0
        free(S)
        free(buf)
    return dots
