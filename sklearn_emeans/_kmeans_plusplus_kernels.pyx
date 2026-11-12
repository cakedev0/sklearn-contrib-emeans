# cython: language_level=3, boundscheck=False, wraparound=False
# cython: cdivision=True, initializedcheck=False

"""Cython kernels of the k-means++ initialization.

Adapted from `sklearn/cluster/_k_means_common.pyx` and
`sklearn/utils/_cython_blas.pyx` of scikit-learn (BSD-3-Clause), at commit
6e37063f7ab7ef7a7e257b9d44bc3ec6ee1b1bcd. Only the k-means++ kernels are kept,
and `_gemm` calls `scipy.linalg.cython_blas` directly.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from cython cimport floating
from cython.parallel cimport parallel, prange
from libc.math cimport sqrt
from libc.stdlib cimport calloc, free
from scipy.linalg.cython_blas cimport sgemm, dgemm

ctypedef Py_ssize_t intp_t


cdef enum BLAS_Order:
    RowMajor  # C contiguous
    ColMajor  # Fortran contiguous


cdef enum BLAS_Trans:
    NoTrans = 110  # correspond to 'n'
    Trans = 116    # correspond to 't'


# Number of samples per data chunk defined as a global constant.
CHUNK_SIZE = 256


cdef void _gemm(BLAS_Order order, BLAS_Trans ta, BLAS_Trans tb, int m, int n,
                int k, floating alpha, const floating *A, int lda, const floating *B,
                int ldb, floating beta, floating *C, int ldc) noexcept nogil:
    """C := alpha * op(A).op(B) + beta * C"""
    cdef:
        char ta_ = ta
        char tb_ = tb
    if order == BLAS_Order.RowMajor:
        if floating is float:
            sgemm(&tb_, &ta_, &n, &m, &k, &alpha, <float*>B,
                  &ldb, <float*>A, &lda, &beta, C, &ldc)
        else:
            dgemm(&tb_, &ta_, &n, &m, &k, &alpha, <double*>B,
                  &ldb, <double*>A, &lda, &beta, C, &ldc)
    else:
        if floating is float:
            sgemm(&ta_, &tb_, &m, &n, &k, &alpha, <float*>A,
                  &lda, <float*>B, &ldb, &beta, C, &ldc)
        else:
            dgemm(&ta_, &tb_, &m, &n, &k, &alpha, <double*>A,
                  &lda, <double*>B, &ldb, &beta, C, &ldc)


cdef floating _euclidean_dense_dense(
        const floating* a,  # IN
        const floating* b,  # IN
        int n_features,
        bint squared
) noexcept nogil:
    """Euclidean distance between a dense and b dense"""
    cdef:
        int i
        int n = n_features // 4
        int rem = n_features % 4
        floating result = 0

    # We manually unroll the loop for better cache optimization.
    for _ in range(n):
        result += (
            (a[0] - b[0]) * (a[0] - b[0]) +
            (a[1] - b[1]) * (a[1] - b[1]) +
            (a[2] - b[2]) * (a[2] - b[2]) +
            (a[3] - b[3]) * (a[3] - b[3])
        )
        a += 4
        b += 4

    for i in range(rem):
        result += (a[i] - b[i]) * (a[i] - b[i])

    return result if squared else sqrt(result)


cdef inline floating _recompute_threshold(floating dummy) noexcept nogil:
    """Relative threshold below which a squared distance is recomputed directly.

    ||x||² - 2 x.c + ||c||² has an absolute error of a few eps * (||x||² + ||c||²),
    which is large relative to the distance when x is close to c, in particular in
    float32. Distances below `threshold * (||x||² + ||c||²)` are recomputed, which
    bounds the relative error by about eps / threshold. In particular, a sample
    identical to an existing center gets a zero distance and is never sampled
    again. The threshold is lower in float64 because recomputing is costly when the
    data is far from the origin, where most distances are small relative to the
    norms.
    """
    if floating is float:
        return 1e-2
    else:
        return 1e-8


def _kmeans_plusplus_trials_dense(
        const floating[:, ::1] X,                       # IN (n_samples, n_features)
        const floating[::1] x_squared_norms,            # IN (n_samples,)
        const floating[::1] sample_weight,              # IN (n_samples,)
        const floating[:, ::1] candidates,              # IN (n_trials, n_features)
        const floating[::1] candidates_squared_norms,   # IN (n_trials,)
        const floating[::1] closest_dist_sq,            # IN (n_samples,)
        floating[:, ::1] trials_dist_sq,                # OUT (n_trials, n_samples)
        double[:, ::1] trials_chunk_potentials,         # OUT (n_trials, n_chunks)
        int n_threads):
    """Evaluate k-means++ candidate centers for dense input data.

    The potential of a sample is its squared distance to the closest center,
    weighted by `sample_weight`.

    Each candidate is evaluated as a possible new center: with the candidate
    `candidates[trial_idx]` added to the current centers, the closest center of
    sample `X[sample_idx]` is either its current one or the candidate, whichever
    is nearer. For each candidate, compute:

    - `trials_dist_sq[trial_idx, sample_idx]`: the squared distance from the
      sample to that closest center, i.e.
      `min(closest_dist_sq[sample_idx], ||X[sample_idx] - candidates[trial_idx]||²)`,
    - `trials_chunk_potentials[trial_idx]`: the sums of the corresponding
      potentials over each chunk of `CHUNK_SIZE` samples.

    Summing per chunk is the parallel part of the sum over all the samples:
    each chunk is summed by a single thread, without synchronization, and the
    caller finishes the sum over the chunks. The chunks have a fixed size,
    so the result doesn't depend on the number of threads.
    The caller uses these sums to:
    - pick the candidate with the smallest potential
      `trials_chunk_potentials[trial_idx].sum()`,
    - sample the next candidates using a fast two-stage process (see
      `_kmeans_plusplus_sample_candidates` for details).
    """
    cdef:
        int n_samples = X.shape[0]
        int n_features = X.shape[1]
        int n_trials = candidates.shape[0]
        int n_chunks = trials_chunk_potentials.shape[1]
        int chunk_size = CHUNK_SIZE
        # Number of elements between two rows of trials_dist_sq (the "leading
        # dimension" of BLAS)
        int trials_row_stride = trials_dist_sq.strides[0] // sizeof(floating)
        # Each thread processes a contiguous block of chunks with a single BLAS
        # call: many small concurrent BLAS calls scale badly with the number of
        # threads (at least with OpenBLAS).
        int n_blocks = min(n_threads, n_chunks)
        int block_idx, first_chunk, last_chunk, chunk_idx, start, end
        int trial_idx, sample_idx
        floating dist_sq, norms
        floating threshold = _recompute_threshold(<floating> 0)
        double potential

    for block_idx in prange(n_blocks, nogil=True, num_threads=n_blocks,
                            schedule='static'):
        # Samples X[start:end] of the chunks first_chunk to last_chunk - 1
        first_chunk = block_idx * n_chunks // n_blocks
        last_chunk = (block_idx + 1) * n_chunks // n_blocks
        start = first_chunk * chunk_size
        end = min(last_chunk * chunk_size, n_samples)

        # trials_dist_sq[:, start:end] = -2 candidates @ X[start:end].T
        _gemm(RowMajor, NoTrans, Trans, n_trials, end - start, n_features,
              -2.0, &candidates[0, 0], n_features, &X[start, 0], n_features,
              0.0, &trials_dist_sq[0, start], trials_row_stride)

        for chunk_idx in range(first_chunk, last_chunk):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, n_samples)
            for trial_idx in range(n_trials):
                # potential = (
                #     sample_weight * trials_dist_sq[trial_idx]
                # )[start:end].sum()
                potential = 0.0
                for sample_idx in range(start, end):
                    # dist_sq = ||x||² - 2 x.c + ||c||²
                    norms = (x_squared_norms[sample_idx]
                             + candidates_squared_norms[trial_idx])
                    dist_sq = trials_dist_sq[trial_idx, sample_idx] + norms
                    if dist_sq < threshold * norms:
                        # Small squared distances computed this way can be
                        # inaccurate because of catastrophic cancellation:
                        # recompute these with the numerically safer but
                        # slower sum of squared differences.
                        dist_sq = _euclidean_dense_dense(
                            &X[sample_idx, 0], &candidates[trial_idx, 0],
                            n_features, True)
                    # trials_dist_sq[trial_idx, sample_idx] = min(
                    #     closest_dist_sq[sample_idx], dist_sq)
                    if closest_dist_sq[sample_idx] < dist_sq:
                        dist_sq = closest_dist_sq[sample_idx]
                    trials_dist_sq[trial_idx, sample_idx] = dist_sq
                    potential = potential + sample_weight[sample_idx] * dist_sq
                trials_chunk_potentials[trial_idx, chunk_idx] = potential


cdef inline int _lower_bound(
        const int[::1] a, int lo, int hi, int value) noexcept nogil:
    """First index of the sorted a[lo:hi] whose value is >= value, or hi."""
    cdef int mid
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _kmeans_plusplus_trials_sparse(
        X,                                              # IN (n_samples, n_features)
        X_csc,                                          # IN (n_samples, n_features)
        const floating[::1] x_squared_norms,            # IN (n_samples,)
        const floating[::1] sample_weight,              # IN (n_samples,)
        const intp_t[::1] candidate_indices,            # IN (n_trials,)
        const floating[:, ::1] candidates,              # IN (n_trials, n_features)
        const floating[::1] candidates_squared_norms,   # IN (n_trials,)
        const floating[::1] closest_dist_sq,            # IN (n_samples,)
        floating[:, ::1] trials_dist_sq,                # OUT (n_trials, n_samples)
        double[:, ::1] trials_chunk_potentials,         # OUT (n_trials, n_chunks)
        int n_threads):
    """Evaluate k-means++ candidate centers for sparse input data.

    `X` is in CSR format and `X_csc` is the same matrix in CSC format, with
    sorted row indices. `candidates` are the rows `candidate_indices` of `X`, as a
    dense array.

    Same outputs, chunks and parallel split as `_kmeans_plusplus_trials_dense`
    (see it for details). Only the computation of the distances differs:
    `-2 candidates @ X.T` doesn't use BLAS, it reads the CSC columns of the
    non-zeros of each candidate. And the small distances are recomputed from a
    dense copy of the sample.
    """
    cdef:
        const floating[::1] X_data = X.data
        const int[::1] X_indices = X.indices
        const int[::1] X_indptr = X.indptr
        const floating[::1] Xc_data = X_csc.data
        const int[::1] Xc_indices = X_csc.indices
        const int[::1] Xc_indptr = X_csc.indptr

        int n_samples = X.shape[0]
        int n_features = X.shape[1]
        int n_trials = candidates.shape[0]
        int n_chunks = trials_chunk_potentials.shape[1]
        int chunk_size = CHUNK_SIZE
        int n_blocks = min(n_threads, n_chunks)
        int block_idx, first_chunk, last_chunk, chunk_idx, start, end
        int trial_idx, sample_idx, k
        int p, q, q_end, col
        intp_t candidate_idx
        floating dist_sq, norms, minus_2_c_k
        floating threshold = _recompute_threshold(<floating> 0)
        double potential
        # Dense copy of a sample, for the direct computation of small distances
        floating *x_dense

    with nogil, parallel(num_threads=n_blocks):
        x_dense = <floating*> calloc(n_features, sizeof(floating))

        for block_idx in prange(n_blocks, schedule='static'):
            # Samples X[start:end] of the chunks first_chunk to last_chunk - 1
            first_chunk = block_idx * n_chunks // n_blocks
            last_chunk = (block_idx + 1) * n_chunks // n_blocks
            start = first_chunk * chunk_size
            end = min(last_chunk * chunk_size, n_samples)

            # trials_dist_sq[:, start:end] = -2 candidates @ X[start:end].T. The
            # candidates are rows of X, so they are sparse too: only the columns
            # of their non-zeros are read, which is much less work than a pass
            # over all the non-zeros of X[start:end] per candidate.
            for trial_idx in range(n_trials):
                for sample_idx in range(start, end):
                    trials_dist_sq[trial_idx, sample_idx] = 0.0
                candidate_idx = candidate_indices[trial_idx]
                for p in range(X_indptr[candidate_idx], X_indptr[candidate_idx + 1]):
                    # trials_dist_sq[trial_idx, start:end] += (
                    #     -2 c_k * X[start:end, col])
                    col = X_indices[p]
                    minus_2_c_k = -2 * X_data[p]
                    q_end = Xc_indptr[col + 1]
                    q = _lower_bound(Xc_indices, Xc_indptr[col], q_end, start)
                    while q < q_end and Xc_indices[q] < end:
                        sample_idx = Xc_indices[q]
                        trials_dist_sq[trial_idx, sample_idx] = (
                            trials_dist_sq[trial_idx, sample_idx]
                            + minus_2_c_k * Xc_data[q])
                        q = q + 1

            for chunk_idx in range(first_chunk, last_chunk):
                start = chunk_idx * chunk_size
                end = min(start + chunk_size, n_samples)
                for trial_idx in range(n_trials):
                    # potential = (
                    #     sample_weight * trials_dist_sq[trial_idx]
                    # )[start:end].sum()
                    potential = 0.0
                    for sample_idx in range(start, end):
                        # dist_sq = ||x||² - 2 x.c + ||c||²
                        norms = (x_squared_norms[sample_idx]
                                 + candidates_squared_norms[trial_idx])
                        dist_sq = trials_dist_sq[trial_idx, sample_idx] + norms
                        if dist_sq < threshold * norms:
                            # Small squared distances computed this way can be
                            # inaccurate because of catastrophic cancellation:
                            # recompute these with the numerically safer but
                            # slower sum of squared differences with
                            # x_dense = X[sample_idx].toarray()
                            for k in range(X_indptr[sample_idx],
                                           X_indptr[sample_idx + 1]):
                                x_dense[X_indices[k]] = X_data[k]
                            dist_sq = _euclidean_dense_dense(
                                x_dense, &candidates[trial_idx, 0], n_features, True)
                            # Reset x_dense to zeros for the next sample
                            for k in range(X_indptr[sample_idx],
                                           X_indptr[sample_idx + 1]):
                                x_dense[X_indices[k]] = 0.0
                        # trials_dist_sq[trial_idx, sample_idx] = min(
                        #     closest_dist_sq[sample_idx], dist_sq)
                        if closest_dist_sq[sample_idx] < dist_sq:
                            dist_sq = closest_dist_sq[sample_idx]
                        trials_dist_sq[trial_idx, sample_idx] = dist_sq
                        potential = potential + sample_weight[sample_idx] * dist_sq
                    trials_chunk_potentials[trial_idx, chunk_idx] = potential

        free(x_dense)


def _kmeans_plusplus_sample_candidates(
        random_state,                           # IN
        const double[::1] chunk_potentials,     # IN (n_chunks,)
        const floating[::1] closest_dist_sq,    # IN (n_samples,)
        const floating[::1] sample_weight,      # IN (n_samples,)
        intp_t[::1] candidate_indices):         # OUT (n_trials,)
    """Sample k-means++ candidate centers.

    The probability of a sample is proportional to its potential, i.e. its weighted
    squared distance to the closest center
    `sample_weight[sample_idx] * closest_dist_sq[sample_idx]`.

    For each candidate, using a `random_potential` drawn uniformly in
    [0, total potential) from `random_state`:

    - first, a chunk is chosen with probability proportional to the chunk's
      own potential (the sum of the potentials of its samples), i.e.::

          chunk_idx = np.searchsorted(np.cumsum(chunk_potentials), random_potential)

    - then, a sample of that chunk is chosen with probability proportional to
      its potential, i.e.::

          candidate_idx = chunk_start + np.searchsorted(
              np.cumsum(sample_potentials[chunk_start:chunk_end]),
              random_potential - chunk_potentials[:chunk_idx].sum(),
          )

    This picks the same samples as a search in the cumulative sum of the
    potentials of all the samples (up to rounding errors), but only computes
    the cumulative sums of `n_chunks + CHUNK_SIZE` values. Unlike the numpy
    code above, a sample with a zero potential is never picked, even with
    rounding errors.
    """
    cdef:
        const double[::1] chunk_potentials_cumsum = np.cumsum(chunk_potentials)
        int n_samples = closest_dist_sq.shape[0]
        int n_chunks = chunk_potentials.shape[0]
        int n_trials = candidate_indices.shape[0]
        # Drawn uniformly in [0, total potential)
        const double[::1] random_potentials = (
            random_state.uniform(size=n_trials)
            * chunk_potentials_cumsum[n_chunks - 1])
        int chunk_size = CHUNK_SIZE
        int trial_idx, chunk_idx, lo, hi, start, end, sample_idx
        intp_t candidate_idx
        double random_potential, potential_cumsum, potential

    with nogil:
        for trial_idx in range(n_trials):
            random_potential = random_potentials[trial_idx]

            # Find the index of the first chunk whose cumulative potential
            # reaches random_potential.
            lo, hi = 0, n_chunks - 1
            while lo < hi:
                chunk_idx = (lo + hi) // 2
                if chunk_potentials_cumsum[chunk_idx] < random_potential:
                    lo = chunk_idx + 1
                else:
                    hi = chunk_idx
            chunk_idx = lo

            if chunk_idx > 0:
                random_potential -= chunk_potentials_cumsum[chunk_idx - 1]

            # Find the first sample of the chunk with a non-zero potential whose
            # cumulative potential reaches random_potential. Rounding errors can
            # make random_potential larger than the potential of the chunk: then
            # take its last sample with a non-zero potential.
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, n_samples)
            candidate_idx = start
            potential_cumsum = 0.0
            for sample_idx in range(start, end):
                potential = sample_weight[sample_idx] * closest_dist_sq[sample_idx]
                if potential > 0:
                    candidate_idx = sample_idx
                    potential_cumsum += potential
                    if potential_cumsum >= random_potential:
                        break
            candidate_indices[trial_idx] = candidate_idx
