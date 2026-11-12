# cython: language_level=3, boundscheck=False, wraparound=False
# cython: cdivision=True, initializedcheck=False

"""Cython kernels of EMeans.

A step on a mini-batch has two parallel phases:

1. Assignment: the batch is cut in chunks of rows. Each chunk is gathered into
   a contiguous buffer and assigned to its nearest center with one GEMM.
2. Update: the batch rows are counting-sorted by label, then each center
   computes the sum of its rows and applies the EMA update. The parallelism is
   over centers, so no thread-local (n_clusters, n_features) buffer is needed
   and the result doesn't depend on the number of threads.

`_labels_inertia` assigns all the samples with the same chunked GEMM, without
the gather.
"""

from cython cimport floating
from cython.parallel cimport parallel, prange
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset
from scipy.linalg.cython_blas cimport sgemm, dgemm

ctypedef Py_ssize_t intp_t


cdef extern from *:
    """
    #ifdef _OPENMP
    #include <omp.h>
    static int emeans_omp_max_threads(void) { return omp_get_max_threads(); }
    #else
    static int emeans_omp_max_threads(void) { return 1; }
    #endif
    """
    int emeans_omp_max_threads() noexcept nogil


# Number of rows assigned by one GEMM call.
cdef intp_t CHUNK_SIZE = 256


def _omp_max_threads():
    """omp_get_max_threads(), or 1 if built without OpenMP."""
    return emeans_omp_max_threads()


def _chunk_size():
    return CHUNK_SIZE


def _emeans_step(
    const floating[:, ::1] X,              # IN
    const intp_t[::1] batch_indices,       # IN
    floating[:, ::1] centers,              # INOUT
    double[::1] weight_ema,                # INOUT
    double step_size,
    double[:, ::1] epoch_sums,             # INOUT, or None
    double[::1] epoch_counts,              # INOUT, or None
    int n_threads,
):
    """One e-means step on the rows `X[batch_indices]`.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features), dtype=floating
        The observations to cluster.

    batch_indices : ndarray of shape (batch_size,), dtype=intp
        Indices of the rows of the batch.

    centers : ndarray of shape (n_clusters, n_features), dtype=floating
        EMA centers, updated inplace.

    weight_ema : ndarray of shape (n_clusters,), dtype=float64
        EMA of the number of rows assigned to each center, updated inplace.

    step_size : float
        EMA step alpha of this batch, in (0, 1].

    epoch_sums : ndarray of shape (n_clusters, n_features), dtype=float64
        If not None, the sums of the batch rows assigned to each center are
        added to it.

    epoch_counts : ndarray of shape (n_clusters,), dtype=float64
        If not None, the number of batch rows assigned to each center is added
        to it.

    n_threads : int
        The number of threads to be used by openmp. All the parallel regions
        use this exact number of threads, since changing the size of the
        OpenMP team between consecutive regions is slow.
    """
    cdef:
        intp_t n_batch = batch_indices.shape[0]
        intp_t n_clusters = centers.shape[0]
        intp_t n_features = centers.shape[1]
        intp_t n_chunks = (n_batch + CHUNK_SIZE - 1) // CHUNK_SIZE
        intp_t chunk_idx, start, n_rows, r, j
        double *epoch_sums_ptr = NULL
        double *epoch_counts_ptr = NULL
        int *labels
        intp_t *cluster_starts
        intp_t *batch_order
        floating *centers_squared_norms
        floating *X_chunk = NULL
        floating *distances = NULL
        double *sums = NULL

    if n_batch == 0:
        return

    if epoch_sums is not None:
        epoch_sums_ptr = &epoch_sums[0, 0]
        epoch_counts_ptr = &epoch_counts[0]

    labels = <int*> malloc(n_batch * sizeof(int))
    cluster_starts = <intp_t*> malloc((n_clusters + 1) * sizeof(intp_t))
    batch_order = <intp_t*> malloc(n_batch * sizeof(intp_t))
    centers_squared_norms = <floating*> malloc(n_clusters * sizeof(floating))

    with nogil:
        for j in prange(n_clusters, num_threads=n_threads, schedule='static'):
            centers_squared_norms[j] = _squared_norm(&centers[j, 0], n_features)

        # Phase 1: assignment, parallel over chunks of batch rows.
        with parallel(num_threads=n_threads):
            X_chunk = <floating*> malloc(CHUNK_SIZE * n_features * sizeof(floating))
            distances = <floating*> malloc(CHUNK_SIZE * n_clusters * sizeof(floating))

            for chunk_idx in prange(n_chunks, schedule='dynamic'):
                start = chunk_idx * CHUNK_SIZE
                n_rows = min(CHUNK_SIZE, n_batch - start)
                for r in range(n_rows):
                    memcpy(&X_chunk[r * n_features],
                           &X[batch_indices[start + r], 0],
                           n_features * sizeof(floating))
                _assign_rows(
                    X_chunk,
                    n_rows,
                    &centers[0, 0],
                    centers_squared_norms,
                    n_clusters,
                    n_features,
                    labels + start,
                    distances,
                    NULL,
                )

            free(X_chunk)
            free(distances)

        _sort_by_label(&batch_indices[0], labels, n_batch, n_clusters,
                       cluster_starts, batch_order)

        # Phase 2: EMA update, parallel over centers.
        with parallel(num_threads=n_threads):
            sums = <double*> malloc(n_features * sizeof(double))

            for j in prange(n_clusters, schedule='guided'):
                _update_center(
                    &X[0, 0],
                    batch_order + cluster_starts[j],
                    cluster_starts[j + 1] - cluster_starts[j],
                    n_features,
                    &centers[j, 0],
                    &weight_ema[j],
                    step_size,
                    NULL if epoch_sums_ptr == NULL
                    else epoch_sums_ptr + j * n_features,
                    NULL if epoch_counts_ptr == NULL else epoch_counts_ptr + j,
                    sums,
                )

            free(sums)

    free(labels)
    free(cluster_starts)
    free(batch_order)
    free(centers_squared_norms)


def _labels_inertia(
    const floating[:, ::1] X,        # IN
    const floating[:, ::1] centers,  # IN
    int[::1] labels,                 # OUT
    int n_threads,
):
    """Assign each sample to its nearest center.

    Fills `labels` and returns the inertia, the sum of squared distances of
    the samples to their nearest center. The result doesn't depend on the
    number of threads.
    """
    cdef:
        intp_t n_samples = X.shape[0]
        intp_t n_clusters = centers.shape[0]
        intp_t n_features = centers.shape[1]
        intp_t n_chunks = (n_samples + CHUNK_SIZE - 1) // CHUNK_SIZE
        intp_t chunk_idx, start, j
        double inertia = 0.0
        double *chunk_inertia
        floating *centers_squared_norms
        floating *distances = NULL

    if n_samples == 0:
        return 0.0

    n_threads = min(n_threads, n_chunks)
    chunk_inertia = <double*> malloc(n_chunks * sizeof(double))
    centers_squared_norms = <floating*> malloc(n_clusters * sizeof(floating))

    with nogil:
        for j in range(n_clusters):
            centers_squared_norms[j] = _squared_norm(&centers[j, 0], n_features)

        with parallel(num_threads=n_threads):
            distances = <floating*> malloc(CHUNK_SIZE * n_clusters * sizeof(floating))

            for chunk_idx in prange(n_chunks, schedule='static'):
                start = chunk_idx * CHUNK_SIZE
                _assign_rows(
                    &X[start, 0],
                    min(CHUNK_SIZE, n_samples - start),
                    &centers[0, 0],
                    centers_squared_norms,
                    n_clusters,
                    n_features,
                    &labels[start],
                    distances,
                    &chunk_inertia[chunk_idx],
                )

            free(distances)

        # Sum the inertias of the chunks in a fixed order.
        for chunk_idx in range(n_chunks):
            inertia += chunk_inertia[chunk_idx]

    free(chunk_inertia)
    free(centers_squared_norms)
    return inertia


cdef inline floating _squared_norm(
    const floating *x,
    intp_t n_features,
) noexcept nogil:
    cdef:
        floating result = 0.0
        intp_t k

    for k in range(n_features):
        result += x[k] * x[k]
    return result


cdef inline double _squared_distance(
    const floating *x,
    const floating *y,
    intp_t n_features,
) noexcept nogil:
    cdef:
        double result = 0.0
        double diff
        intp_t k

    for k in range(n_features):
        diff = x[k] - y[k]
        result += diff * diff
    return result


cdef void _assign_rows(
    const floating *X_rows,                 # IN (n_rows, n_features)
    intp_t n_rows,
    const floating *centers,                # IN (n_clusters, n_features)
    const floating *centers_squared_norms,  # IN (n_clusters,)
    intp_t n_clusters,
    intp_t n_features,
    int *labels,                            # OUT (n_rows,)
    floating *distances,                    # TMP (n_rows, n_clusters)
    double *inertia,                        # OUT, or NULL
) noexcept nogil:
    """Assign the contiguous rows `X_rows` to their nearest center."""
    cdef:
        intp_t r, j, label
        floating dist, min_dist
        double total = 0.0

    for r in range(n_rows):
        for j in range(n_clusters):
            distances[r * n_clusters + j] = centers_squared_norms[j]

    # ||x - c||² = ||x||² - 2 x.c + ||c||², and the argmin over c only needs
    # distances = ||c||² - 2 x.c
    _gemm_nt(n_rows, n_clusters, n_features, -2.0, X_rows, centers, 1.0,
             distances)

    for r in range(n_rows):
        min_dist = distances[r * n_clusters]
        label = 0
        for j in range(1, n_clusters):
            dist = distances[r * n_clusters + j]
            if dist < min_dist:
                min_dist = dist
                label = j
        labels[r] = <int> label
        if inertia != NULL:
            # Exact ||x - c||², without the cancellation of the GEMM trick.
            total += _squared_distance(&X_rows[r * n_features],
                                       &centers[label * n_features],
                                       n_features)

    if inertia != NULL:
        inertia[0] = total


cdef inline void _gemm_nt(
    intp_t m,
    intp_t n,
    intp_t k,
    floating alpha,
    const floating *A,  # IN (m, k), row-major
    const floating *B,  # IN (n, k), row-major
    floating beta,
    floating *C,        # INOUT (m, n), row-major
) noexcept nogil:
    """C = alpha A B^T + beta C, with row-major matrices.

    BLAS is column-major, where the row-major C is C^T = B A^T, with B and A
    seen as the column-major B^T and A^T.
    """
    cdef:
        char transa = b'T'
        char transb = b'N'
        int m_ = <int> n
        int n_ = <int> m
        int k_ = <int> k
        int ldb = <int> k
        int ldc = <int> n

    if floating is float:
        sgemm(&transa, &transb, &m_, &n_, &k_, &alpha, <float*> B, &ldb,
              <float*> A, &ldb, &beta, C, &ldc)
    else:
        dgemm(&transa, &transb, &m_, &n_, &k_, &alpha, <double*> B, &ldb,
              <double*> A, &ldb, &beta, C, &ldc)


cdef void _sort_by_label(
    const intp_t *batch_indices,  # IN (n_batch,)
    const int *labels,            # IN (n_batch,)
    intp_t n_batch,
    intp_t n_clusters,
    intp_t *cluster_starts,       # OUT (n_clusters + 1,)
    intp_t *batch_order,          # OUT (n_batch,)
) noexcept nogil:
    """Counting sort of the batch rows by label.

    The rows assigned to center j are
    `batch_order[cluster_starts[j]:cluster_starts[j + 1]]`, in batch order.
    """
    cdef intp_t p, j

    memset(cluster_starts, 0, (n_clusters + 1) * sizeof(intp_t))
    for p in range(n_batch):
        cluster_starts[labels[p] + 1] += 1
    for j in range(n_clusters):
        cluster_starts[j + 1] += cluster_starts[j]

    # Fill each segment, using cluster_starts[j] as the write cursor of center
    # j. It ends at the start of center j + 1, so shift it back afterwards.
    for p in range(n_batch):
        batch_order[cluster_starts[labels[p]]] = batch_indices[p]
        cluster_starts[labels[p]] += 1
    for j in range(n_clusters, 0, -1):
        cluster_starts[j] = cluster_starts[j - 1]
    cluster_starts[0] = 0


cdef void _update_center(
    const floating *X,      # IN (n_samples, n_features)
    const intp_t *members,  # IN (n_members,)
    intp_t n_members,
    intp_t n_features,
    floating *center,       # INOUT (n_features,)
    double *weight_ema,     # INOUT
    double step_size,
    double *epoch_sum,      # INOUT (n_features,), or NULL
    double *epoch_count,    # INOUT, or NULL
    double *sums,           # TMP (n_features,)
) noexcept nogil:
    """EMA update of one center from its batch rows `X[members]`.

    The center c_j = S_j / N_j is the ratio of two EMAs with step alpha, of the
    sum s_j and of the number n_j of its batch rows:
        S_j <- (1 - alpha) S_j + alpha s_j
        N_j <- (1 - alpha) N_j + alpha n_j
    so with N'_j the new N_j:
        c_j <- ((1 - alpha) N_j c_j + alpha s_j) / N'_j
    """
    cdef:
        intp_t m, k
        const floating *x
        double retained = (1.0 - step_size) * weight_ema[0]
        double new_weight, center_coef, sums_coef

    if n_members == 0:
        # Only N_j decays, c_j stays.
        weight_ema[0] = retained
        return

    # s_j = sum of the rows of center j
    memset(sums, 0, n_features * sizeof(double))
    for m in range(n_members):
        x = &X[members[m] * n_features]
        for k in range(n_features):
            sums[k] += x[k]

    if epoch_sum != NULL:
        epoch_count[0] += n_members
        for k in range(n_features):
            epoch_sum[k] += sums[k]

    new_weight = retained + step_size * n_members
    center_coef = retained / new_weight
    sums_coef = step_size / new_weight
    for k in range(n_features):
        center[k] = <floating> (center_coef * center[k] + sums_coef * sums[k])
    weight_ema[0] = new_weight
