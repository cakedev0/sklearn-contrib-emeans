# cython: language_level=3, boundscheck=False, wraparound=False
# cython: cdivision=True, initializedcheck=False

"""Cython kernels of EMeans.

A step on a mini-batch has two main phases:

1. Assignment: the batch is cut in chunks of rows. Each chunk is gathered into
   a contiguous buffer and assigned to its nearest center with one GEMM.
2. Update: the batch rows are counting-sorted by label, then each center
   computes the sum of its rows and applies the EMA update. The parallelism is
   over centers, so no thread-local (n_clusters, n_features) buffer is needed
   and the result doesn't depend on the number of threads.

All the steps of an epoch run in a single OpenMP parallel region, where the
threads synchronize between phases with atomic counters instead of OpenMP
constructs, see `_run_steps`.

`_labels_inertia` assigns all the samples with the same chunked GEMM, without
the gather.

The `_sparse` variants take a CSR matrix. Without a sparse BLAS, the assignment
computes the dot products row by row against the transposed centers, so that
each nonzero x_k adds -2 x_k c[:, k] to the distances of its row with one
contiguous loop over the centers.

`_update_min_distances` and `_update_min_distances_sparse` use the same
assignment to update the distances of the samples to their nearest center, for
the k-means|| initialization.
"""

from cython cimport floating
from cython.parallel cimport parallel, prange
from cpython.exc cimport PyErr_CheckSignals
from cpython.ref cimport PyObject
from libc.stdlib cimport calloc, malloc, free
from libc.string cimport memcpy, memset
from scipy.linalg.cython_blas cimport sgemm, dgemm

ctypedef Py_ssize_t intp_t


cdef extern from *:
    """
    #ifdef _OPENMP
    #include <omp.h>
    static int emeans_omp_max_threads(void) { return omp_get_max_threads(); }
    static int emeans_omp_thread_num(void) { return omp_get_thread_num(); }
    static int emeans_omp_num_threads(void) { return omp_get_num_threads(); }
    #else
    static int emeans_omp_max_threads(void) { return 1; }
    static int emeans_omp_thread_num(void) { return 0; }
    static int emeans_omp_num_threads(void) { return 1; }
    #endif

    /* Atomic counters shared by the threads of `_run_steps`. */
    #if defined(_MSC_VER)
    #include <windows.h>
    #if defined(_WIN64)
    #define emeans_fetch_add(p, v) \\
        ((Py_ssize_t) InterlockedExchangeAdd64((volatile LONG64*) (p), (v)))
    #define emeans_load(p) ((Py_ssize_t) ReadAcquire64((const volatile LONG64*) (p)))
    #else
    #define emeans_fetch_add(p, v) \\
        ((Py_ssize_t) InterlockedExchangeAdd((volatile LONG*) (p), (v)))
    #define emeans_load(p) ((Py_ssize_t) ReadAcquire((const volatile LONG*) (p)))
    #endif
    #define emeans_relax() YieldProcessor()
    #define emeans_yield() SwitchToThread()
    #else
    #include <sched.h>
    #define emeans_fetch_add(p, v) __atomic_fetch_add((p), (v), __ATOMIC_SEQ_CST)
    #define emeans_load(p) __atomic_load_n((p), __ATOMIC_SEQ_CST)
    #if defined(__x86_64__) || defined(__i386__)
    #define emeans_relax() __builtin_ia32_pause()
    #else
    #define emeans_relax() ((void) 0)
    #endif
    #define emeans_yield() sched_yield()
    #endif

    /* Wait until *counter >= target. After a short spin, yield the CPU at
       each check: when two threads of the team share a core, the one that
       waits must let the one that works run. */
    static void emeans_wait(Py_ssize_t *counter, Py_ssize_t target) {
        int n_spins = 0;
        while (emeans_load(counter) < target) {
            if (n_spins < 100) {
                n_spins++;
                emeans_relax();
            } else {
                emeans_yield();
            }
        }
    }

    /* Called by each thread of a parallel region once it has allocated its
       buffers, with a non-zero `failed` if an allocation failed. Waits for
       all the threads of the team, then returns whether all the allocations
       succeeded: no thread starts to work when one of them lacks a buffer. */
    static int emeans_buffers_ready(Py_ssize_t *n_ready, Py_ssize_t *n_failed,
                                    int failed) {
        if (failed) {
            emeans_fetch_add(n_failed, 1);
        }
        emeans_fetch_add(n_ready, 1);
        emeans_wait(n_ready, emeans_omp_num_threads());
        return emeans_load(n_failed) == 0;
    }
    """
    int emeans_omp_max_threads() noexcept nogil
    int emeans_omp_thread_num() noexcept nogil
    intp_t emeans_fetch_add(intp_t *counter, intp_t value) noexcept nogil
    intp_t emeans_load(intp_t *counter) noexcept nogil
    void emeans_wait(intp_t *counter, intp_t target) noexcept nogil
    bint emeans_buffers_ready(
        intp_t *n_ready, intp_t *n_failed, bint failed
    ) noexcept nogil


# Number of rows assigned by one GEMM call.
cdef intp_t CHUNK_SIZE = 256

# Number of columns transposed together by `_transpose`.
cdef intp_t TRANSPOSE_BLOCK_SIZE = 64

# Atomic counters of a step in `_run_steps`: for each phase, the work claimed
# so far, then the work done so far.
cdef enum:
    NORMS = 0
    TRANSPOSE = 2
    ASSIGN = 4
    SORT = 6
    UPDATE = 8
    REVIVE = 10
    N_COUNTERS = 12


def _omp_max_threads():
    """omp_get_max_threads(), or 1 if built without OpenMP."""
    return emeans_omp_max_threads()


def _chunk_size():
    return CHUNK_SIZE


def _emeans_steps(
    const floating[:, ::1] X,              # IN
    const intp_t[::1] rows,                # IN
    intp_t batch_size,
    floating[:, ::1] centers,              # INOUT
    double[::1] weight_ema,                # INOUT
    const double[::1] step_sizes,          # IN
    double[:, ::1] epoch_sums,             # INOUT, or None
    double[::1] epoch_counts,              # INOUT, or None
    intp_t first_step,
    intp_t revival_period,
    revive,
    int n_threads,
):
    """Consecutive e-means steps on the rows `X[rows]`.

    Step `s` is on the batch `rows[s * batch_size:(s + 1) * batch_size]`.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features), dtype=floating
        The observations to cluster.

    rows : ndarray of shape (n_rows,), dtype=intp
        Indices of the rows of all the batches, with
        `(n_steps - 1) * batch_size < n_rows <= n_steps * batch_size`.

    batch_size : int
        Number of rows of each batch, but for the last one, which can be
        smaller.

    centers : ndarray of shape (n_clusters, n_features), dtype=floating
        EMA centers, updated inplace.

    weight_ema : ndarray of shape (n_clusters,), dtype=float64
        EMA of the number of rows assigned to each center, updated inplace.

    step_sizes : ndarray of shape (n_steps,), dtype=float64
        EMA step alpha of each batch, in (0, 1].

    epoch_sums : ndarray of shape (n_clusters, n_features), dtype=float64
        If not None, the sums of the batch rows assigned to each center are
        added to it.

    epoch_counts : ndarray of shape (n_clusters,), dtype=float64
        If not None, the number of batch rows assigned to each center is added
        to it.

    first_step : int
        Number of steps done before this call, which sets when `revive` is
        called.

    revival_period : int
        `revive` is called after each step whose 1-based index, counted from
        the first step of the fit, is a multiple of `revival_period`. 0 to
        never call it.

    revive : callable
        Called with no argument and with the GIL, possibly from another
        thread. It can change `centers` and `weight_ema` inplace.

    n_threads : int
        The number of threads to be used by openmp.
    """
    if rows.shape[0] == 0:
        return

    _run_steps(
        &X[0, 0],
        <const floating*> NULL,
        NULL,
        NULL,
        &rows[0],
        rows.shape[0],
        batch_size,
        &centers[0, 0],
        centers.shape[0],
        centers.shape[1],
        &weight_ema[0],
        &step_sizes[0],
        step_sizes.shape[0],
        NULL if epoch_sums is None else &epoch_sums[0, 0],
        NULL if epoch_counts is None else &epoch_counts[0],
        first_step,
        revival_period,
        revive,
        n_threads,
    )


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
        intp_t n_ready = 0
        intp_t n_failed = 0
        double inertia = 0.0
        double *chunk_inertia
        floating *centers_squared_norms
        floating *distances = NULL

    if n_samples == 0:
        return 0.0

    n_threads = min(n_threads, n_chunks)
    chunk_inertia = <double*> malloc(n_chunks * sizeof(double))
    centers_squared_norms = <floating*> malloc(n_clusters * sizeof(floating))
    if chunk_inertia == NULL or centers_squared_norms == NULL:
        free(chunk_inertia)
        free(centers_squared_norms)
        raise MemoryError()

    with nogil:
        for j in range(n_clusters):
            centers_squared_norms[j] = _squared_norm(&centers[j, 0], n_features)

        with parallel(num_threads=n_threads):
            distances = <floating*> malloc(CHUNK_SIZE * n_clusters * sizeof(floating))

            if emeans_buffers_ready(&n_ready, &n_failed, distances == NULL):
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

        if not n_failed:
            # Sum the inertias of the chunks in a fixed order.
            for chunk_idx in range(n_chunks):
                inertia += chunk_inertia[chunk_idx]

    free(chunk_inertia)
    free(centers_squared_norms)
    if n_failed:
        raise MemoryError()
    return inertia


def _emeans_steps_sparse(
    X,                                     # IN
    const intp_t[::1] rows,                # IN
    intp_t batch_size,
    floating[:, ::1] centers,              # INOUT
    double[::1] weight_ema,                # INOUT
    const double[::1] step_sizes,          # IN
    double[:, ::1] epoch_sums,             # INOUT, or None
    double[::1] epoch_counts,              # INOUT, or None
    intp_t first_step,
    intp_t revival_period,
    revive,
    int n_threads,
):
    """Consecutive e-means steps on the CSR rows `X[rows]`.

    Same as `_emeans_steps`, with `X` a CSR matrix with int32 indices.
    """
    cdef:
        const floating[::1] X_data = X.data
        const int[::1] X_indices = X.indices
        const int[::1] X_indptr = X.indptr

    if rows.shape[0] == 0:
        return

    _run_steps(
        <const floating*> NULL,
        # X.data is empty when X has no nonzero.
        &X_data[0] if X_data.shape[0] > 0 else NULL,
        &X_indices[0] if X_indices.shape[0] > 0 else NULL,
        &X_indptr[0],
        &rows[0],
        rows.shape[0],
        batch_size,
        &centers[0, 0],
        centers.shape[0],
        centers.shape[1],
        &weight_ema[0],
        &step_sizes[0],
        step_sizes.shape[0],
        NULL if epoch_sums is None else &epoch_sums[0, 0],
        NULL if epoch_counts is None else &epoch_counts[0],
        first_step,
        revival_period,
        revive,
        n_threads,
    )


def _labels_inertia_sparse(
    X,                               # IN
    const floating[:, ::1] centers,  # IN
    int[::1] labels,                 # OUT
    int n_threads,
):
    """Assign each row of the CSR matrix `X` to its nearest center.

    Same as `_labels_inertia`, with `X` a CSR matrix with int32 indices.
    """
    cdef:
        const floating[::1] X_data = X.data
        const int[::1] X_indices = X.indices
        const int[::1] X_indptr = X.indptr
        intp_t n_samples = X.shape[0]
        intp_t n_clusters = centers.shape[0]
        intp_t n_features = centers.shape[1]
        intp_t n_chunks = (n_samples + CHUNK_SIZE - 1) // CHUNK_SIZE
        intp_t chunk_idx, start, j
        intp_t n_ready = 0
        intp_t n_failed = 0
        double inertia = 0.0
        double *chunk_inertia
        floating *centers_T
        floating *centers_squared_norms
        double *centers_squared_norms_64
        floating *distances = NULL

    if n_samples == 0:
        return 0.0

    n_threads = min(n_threads, n_chunks)
    chunk_inertia = <double*> malloc(n_chunks * sizeof(double))
    centers_T = <floating*> malloc(n_features * n_clusters * sizeof(floating))
    centers_squared_norms = <floating*> malloc(n_clusters * sizeof(floating))
    centers_squared_norms_64 = <double*> malloc(n_clusters * sizeof(double))
    if (
        chunk_inertia == NULL
        or centers_T == NULL
        or centers_squared_norms == NULL
        or centers_squared_norms_64 == NULL
    ):
        free(chunk_inertia)
        free(centers_T)
        free(centers_squared_norms)
        free(centers_squared_norms_64)
        raise MemoryError()

    with nogil:
        for j in range(n_clusters):
            centers_squared_norms[j] = _squared_norm(&centers[j, 0], n_features)
            centers_squared_norms_64[j] = _squared_norm_64(&centers[j, 0],
                                                           n_features)

        _transpose(&centers[0, 0], n_clusters, n_features, centers_T, n_threads)

        with parallel(num_threads=n_threads):
            distances = <floating*> malloc(n_clusters * sizeof(floating))

            if emeans_buffers_ready(&n_ready, &n_failed, distances == NULL):
                for chunk_idx in prange(n_chunks, schedule='static'):
                    start = chunk_idx * CHUNK_SIZE
                    _assign_rows_sparse(
                        &X_data[0],
                        &X_indices[0],
                        &X_indptr[0],
                        NULL,
                        start,
                        min(CHUNK_SIZE, n_samples - start),
                        centers_T,
                        centers_squared_norms,
                        centers_squared_norms_64,
                        n_clusters,
                        &labels[start],
                        distances,
                        &chunk_inertia[chunk_idx],
                    )

            free(distances)

        if not n_failed:
            # Sum the inertias of the chunks in a fixed order.
            for chunk_idx in range(n_chunks):
                inertia += chunk_inertia[chunk_idx]

    free(chunk_inertia)
    free(centers_T)
    free(centers_squared_norms)
    free(centers_squared_norms_64)
    if n_failed:
        raise MemoryError()
    return inertia


def _update_min_distances(
    const floating[:, ::1] X,        # IN
    const floating[:, ::1] centers,  # IN
    floating[::1] min_distances,     # INOUT
    int[::1] labels,                 # INOUT, or None
    int label_offset,
    int n_threads,
):
    """Update the squared distances of the samples to their nearest center.

    For each sample x_i, with c_j the given centers:
        d²_i <- min(d²_i, min_j ||x_i - c_j||²)
    If `labels` is not None, the samples whose distance decreases get the
    label `label_offset + argmin_j ||x_i - c_j||²`. The distances use the GEMM
    trick, clipped at 0. The result doesn't depend on the number of threads.
    """
    cdef:
        intp_t n_samples = X.shape[0]
        intp_t n_clusters = centers.shape[0]
        intp_t n_features = X.shape[1]
        intp_t n_chunks = (n_samples + CHUNK_SIZE - 1) // CHUNK_SIZE
        intp_t chunk_idx, start, n_rows, r, i, j
        intp_t n_ready = 0
        intp_t n_failed = 0
        int *labels_ptr = NULL
        int *chunk_labels = NULL
        floating *centers_squared_norms
        floating *distances = NULL
        floating dist

    if n_samples == 0 or n_clusters == 0:
        return

    if labels is not None:
        labels_ptr = &labels[0]

    n_threads = min(n_threads, n_chunks)
    centers_squared_norms = <floating*> malloc(n_clusters * sizeof(floating))
    if centers_squared_norms == NULL:
        raise MemoryError()

    with nogil:
        for j in range(n_clusters):
            centers_squared_norms[j] = _squared_norm(&centers[j, 0], n_features)

        with parallel(num_threads=n_threads):
            distances = <floating*> malloc(CHUNK_SIZE * n_clusters * sizeof(floating))
            chunk_labels = <int*> malloc(CHUNK_SIZE * sizeof(int))

            if emeans_buffers_ready(
                &n_ready, &n_failed, distances == NULL or chunk_labels == NULL
            ):
                for chunk_idx in prange(n_chunks, schedule='static'):
                    start = chunk_idx * CHUNK_SIZE
                    n_rows = min(CHUNK_SIZE, n_samples - start)
                    # distances = ||c||² - 2 x.c, and its argmin
                    _assign_rows(
                        &X[start, 0],
                        n_rows,
                        &centers[0, 0],
                        centers_squared_norms,
                        n_clusters,
                        n_features,
                        chunk_labels,
                        distances,
                        NULL,
                    )
                    for r in range(n_rows):
                        i = start + r
                        # d² = max(||x||² + ||c||² - 2 x.c, 0)
                        dist = (_squared_norm(&X[i, 0], n_features)
                                + distances[r * n_clusters + chunk_labels[r]])
                        if dist < 0:
                            dist = 0
                        if dist < min_distances[i]:
                            min_distances[i] = dist
                            if labels_ptr != NULL:
                                labels_ptr[i] = label_offset + chunk_labels[r]

            free(distances)
            free(chunk_labels)

    free(centers_squared_norms)
    if n_failed:
        raise MemoryError()


def _update_min_distances_sparse(
    X,                               # IN
    const floating[:, ::1] centers,  # IN
    floating[::1] min_distances,     # INOUT
    int[::1] labels,                 # INOUT, or None
    int label_offset,
    int n_threads,
):
    """Same as `_update_min_distances`, with `X` a CSR matrix with int32 indices.

    The distances are computed over the nonzeros of each row against the
    transposed centers, as in `_labels_inertia_sparse`. The distance to the
    nearest center is exact (no cancellation of ||x||²), clipped at 0.
    """
    cdef:
        const floating[::1] X_data = X.data
        const int[::1] X_indices = X.indices
        const int[::1] X_indptr = X.indptr
        intp_t n_samples = X.shape[0]
        intp_t n_clusters = centers.shape[0]
        intp_t n_features = centers.shape[1]
        intp_t i, j
        intp_t n_ready = 0
        intp_t n_failed = 0
        int label
        double dist
        int *labels_ptr = NULL
        floating *centers_T
        floating *centers_squared_norms
        double *centers_squared_norms_64
        floating *distances = NULL

    if n_samples == 0 or n_clusters == 0:
        return

    if labels is not None:
        labels_ptr = &labels[0]

    n_threads = min(n_threads, (n_samples + CHUNK_SIZE - 1) // CHUNK_SIZE)
    centers_T = <floating*> malloc(n_features * n_clusters * sizeof(floating))
    centers_squared_norms = <floating*> malloc(n_clusters * sizeof(floating))
    centers_squared_norms_64 = <double*> malloc(n_clusters * sizeof(double))
    if (
        centers_T == NULL
        or centers_squared_norms == NULL
        or centers_squared_norms_64 == NULL
    ):
        free(centers_T)
        free(centers_squared_norms)
        free(centers_squared_norms_64)
        raise MemoryError()

    with nogil:
        for j in range(n_clusters):
            centers_squared_norms[j] = _squared_norm(&centers[j, 0], n_features)
            centers_squared_norms_64[j] = _squared_norm_64(&centers[j, 0],
                                                           n_features)

        _transpose(&centers[0, 0], n_clusters, n_features, centers_T, n_threads)

        with parallel(num_threads=n_threads):
            distances = <floating*> malloc(n_clusters * sizeof(floating))

            if emeans_buffers_ready(&n_ready, &n_failed, distances == NULL):
                for i in prange(
                    n_samples, schedule='static', chunksize=CHUNK_SIZE
                ):
                    # Assigned here so that they are private to each thread.
                    label = 0
                    dist = 0.0
                    # label = argmin_j ||x_i - c_j||², dist = ||x_i - c_label||²
                    _assign_rows_sparse(
                        &X_data[0],
                        &X_indices[0],
                        &X_indptr[0],
                        NULL,
                        i,
                        1,
                        centers_T,
                        centers_squared_norms,
                        centers_squared_norms_64,
                        n_clusters,
                        &label,
                        distances,
                        &dist,
                    )
                    if dist < min_distances[i]:
                        min_distances[i] = <floating> dist
                        if labels_ptr != NULL:
                            labels_ptr[i] = label_offset + label

            free(distances)

    free(centers_T)
    free(centers_squared_norms)
    free(centers_squared_norms_64)
    if n_failed:
        raise MemoryError()


cdef int _run_steps(
    const floating *X,           # IN (n_samples, n_features), or NULL
    const floating *X_data,      # IN (nnz,), if X is NULL
    const int *X_indices,        # IN (nnz,), if X is NULL
    const int *X_indptr,         # IN (n_samples + 1,), if X is NULL
    const intp_t *rows,          # IN (n_rows,)
    intp_t n_rows,
    intp_t batch_size,
    floating *centers,           # INOUT (n_clusters, n_features)
    intp_t n_clusters,
    intp_t n_features,
    double *weight_ema,          # INOUT (n_clusters,)
    const double *step_sizes,    # IN (n_steps,)
    intp_t n_steps,
    double *epoch_sums,          # INOUT (n_clusters, n_features), or NULL
    double *epoch_counts,        # INOUT (n_clusters,), or NULL
    intp_t first_step,
    intp_t revival_period,
    object revive,
    int n_threads,
) except -1:
    """Run consecutive steps in a single parallel region.

    One parallel region per phase of each step would be simpler, but with
    thousands of short regions per second, the fit times are unstable when the
    idle OpenMP threads wait actively, which is the default of libgomp. The
    threads don't sleep between regions, so the scheduler never places them
    again and two of them can share a core for a while (seen on CPUs with
    performance and efficiency cores). All the threads must reach the barrier
    that ends each region, and the one that shares its core only gets there
    when the other, which spins at that barrier, is preempted: a few ms per
    region instead of a few microseconds. See
    https://github.com/cakedev0/sklearn-contrib-emeans/issues/8.

    Here the threads enter a single region and each of them runs
    `_steps_worker`, which goes through the phases of all the steps. In each
    phase, the threads claim pieces of work with an atomic counter until none
    is left, then wait until all the pieces are done before the next phase. So
    a phase waits for its work, not for its threads: a thread that doesn't get
    a CPU delays nothing unless it holds a piece of work, and in that case the
    threads that wait yield their CPU to it.
    """
    cdef:
        bint is_sparse = X == NULL
        intp_t max_batch_size = min(batch_size, n_rows)
        intp_t distances_size = n_clusters * (1 if is_sparse else CHUNK_SIZE)
        # Small enough to balance the threads when the centers have different
        # numbers of rows, as with a guided schedule.
        intp_t block_size = max(1, n_clusters // (16 * n_threads))
        intp_t failed = 0
        intp_t n_ready = 0
        intp_t n_failed = 0
        list errors = []
        int *labels = <int*> malloc(max_batch_size * sizeof(int))
        intp_t *cluster_starts = <intp_t*> malloc((n_clusters + 1) * sizeof(intp_t))
        intp_t *batch_order = <intp_t*> malloc(max_batch_size * sizeof(intp_t))
        intp_t *counters = <intp_t*> calloc(N_COUNTERS * n_steps, sizeof(intp_t))
        floating *centers_squared_norms = <floating*> malloc(
            n_clusters * sizeof(floating)
        )
        floating *centers_T = NULL
        floating *X_chunk = NULL
        floating *distances = NULL
        double *sums = NULL

    if is_sparse:
        centers_T = <floating*> malloc(n_features * n_clusters * sizeof(floating))

    if (
        labels == NULL
        or cluster_starts == NULL
        or batch_order == NULL
        or counters == NULL
        or centers_squared_norms == NULL
        or (is_sparse and centers_T == NULL)
    ):
        free(labels)
        free(cluster_starts)
        free(batch_order)
        free(counters)
        free(centers_squared_norms)
        free(centers_T)
        raise MemoryError()

    with nogil, parallel(num_threads=n_threads):
        if not is_sparse:
            X_chunk = <floating*> malloc(CHUNK_SIZE * n_features * sizeof(floating))
        distances = <floating*> malloc(distances_size * sizeof(floating))
        sums = <double*> malloc(n_features * sizeof(double))

        if emeans_buffers_ready(
            &n_ready,
            &n_failed,
            distances == NULL or sums == NULL or (not is_sparse and X_chunk == NULL),
        ):
            _steps_worker(
                X, X_data, X_indices, X_indptr, rows, n_rows, batch_size,
                centers, n_clusters, n_features, weight_ema, step_sizes,
                n_steps, epoch_sums, epoch_counts, first_step, revival_period,
                <PyObject*> revive, <PyObject*> errors, &failed, counters,
                block_size, labels, cluster_starts, batch_order,
                centers_squared_norms, centers_T, X_chunk, distances, sums,
            )

        free(X_chunk)
        free(distances)
        free(sums)

    free(labels)
    free(cluster_starts)
    free(batch_order)
    free(counters)
    free(centers_squared_norms)
    free(centers_T)

    if errors:
        raise errors[0]
    if n_failed:
        raise MemoryError()
    return 0


cdef void _steps_worker(
    const floating *X,
    const floating *X_data,
    const int *X_indices,
    const int *X_indptr,
    const intp_t *rows,
    intp_t n_rows,
    intp_t batch_size,
    floating *centers,
    intp_t n_clusters,
    intp_t n_features,
    double *weight_ema,
    const double *step_sizes,
    intp_t n_steps,
    double *epoch_sums,
    double *epoch_counts,
    intp_t first_step,
    intp_t revival_period,
    PyObject *revive,
    PyObject *errors,
    intp_t *failed,                   # shared: != 0 to stop at the next step
    intp_t *counters,                 # shared (n_steps, N_COUNTERS)
    intp_t block_size,
    int *labels,                      # shared (batch_size,)
    intp_t *cluster_starts,           # shared (n_clusters + 1,)
    intp_t *batch_order,              # shared (batch_size,)
    floating *centers_squared_norms,  # shared (n_clusters,)
    floating *centers_T,              # shared (n_features, n_clusters), or NULL
    floating *X_chunk,                # thread-local
    floating *distances,              # thread-local
    double *sums,                     # thread-local
) noexcept nogil:
    """What each thread of `_run_steps` runs: the phases of all the steps.

    A thread can be late, or even go through steps that the others have
    already completed: it then finds nothing to claim and nothing to wait for.
    """
    cdef:
        bint is_sparse = X == NULL
        bint is_main_thread = emeans_omp_thread_num() == 0
        intp_t n_transpose_blocks = (
            (n_features + TRANSPOSE_BLOCK_SIZE - 1) // TRANSPOSE_BLOCK_SIZE
        )
        intp_t s, n_batch, n_chunks, chunk_idx, start, n_chunk_rows, r, j
        intp_t block_idx, first, last
        const intp_t *batch
        intp_t *counter

    for s in range(n_steps):
        if emeans_load(failed):
            return
        if is_main_thread:
            # Only the main thread of the interpreter handles the signals, as
            # KeyboardInterrupt.
            with gil:
                try:
                    PyErr_CheckSignals()
                except BaseException as exc:
                    (<list> errors).append(exc)
                    emeans_fetch_add(failed, 1)
                    return

        batch = rows + s * batch_size
        n_batch = min(batch_size, n_rows - s * batch_size)
        n_chunks = (n_batch + CHUNK_SIZE - 1) // CHUNK_SIZE
        counter = counters + s * N_COUNTERS

        # Squared norms of the centers, by blocks of centers.
        while True:
            first = emeans_fetch_add(&counter[NORMS], block_size)
            if first >= n_clusters:
                break
            last = min(first + block_size, n_clusters)
            for j in range(first, last):
                centers_squared_norms[j] = _squared_norm(
                    centers + j * n_features, n_features
                )
            emeans_fetch_add(&counter[NORMS + 1], last - first)
        emeans_wait(&counter[NORMS + 1], n_clusters)

        if is_sparse:
            # Transposed centers, by blocks of features.
            while True:
                block_idx = emeans_fetch_add(&counter[TRANSPOSE], 1)
                if block_idx >= n_transpose_blocks:
                    break
                _transpose_block(centers, n_clusters, n_features, centers_T,
                                 block_idx)
                emeans_fetch_add(&counter[TRANSPOSE + 1], 1)
            emeans_wait(&counter[TRANSPOSE + 1], n_transpose_blocks)

        # Assignment, by chunks of batch rows.
        while True:
            chunk_idx = emeans_fetch_add(&counter[ASSIGN], 1)
            if chunk_idx >= n_chunks:
                break
            start = chunk_idx * CHUNK_SIZE
            n_chunk_rows = min(CHUNK_SIZE, n_batch - start)
            if is_sparse:
                _assign_rows_sparse(
                    X_data, X_indices, X_indptr, batch + start, 0, n_chunk_rows,
                    centers_T, centers_squared_norms, NULL, n_clusters,
                    labels + start, distances, NULL,
                )
            else:
                for r in range(n_chunk_rows):
                    memcpy(&X_chunk[r * n_features],
                           &X[batch[start + r] * n_features],
                           n_features * sizeof(floating))
                _assign_rows(
                    X_chunk, n_chunk_rows, centers, centers_squared_norms,
                    n_clusters, n_features, labels + start, distances, NULL,
                )
            emeans_fetch_add(&counter[ASSIGN + 1], 1)
        emeans_wait(&counter[ASSIGN + 1], n_chunks)

        # Sort of the batch rows by label, by the first thread to get here.
        if emeans_fetch_add(&counter[SORT], 1) == 0:
            _sort_by_label(batch, labels, n_batch, n_clusters, cluster_starts,
                           batch_order)
            emeans_fetch_add(&counter[SORT + 1], 1)
        emeans_wait(&counter[SORT + 1], 1)

        # EMA update, by blocks of centers.
        while True:
            first = emeans_fetch_add(&counter[UPDATE], block_size)
            if first >= n_clusters:
                break
            last = min(first + block_size, n_clusters)
            for j in range(first, last):
                if is_sparse:
                    _update_center_sparse(
                        X_data, X_indices, X_indptr,
                        batch_order + cluster_starts[j],
                        cluster_starts[j + 1] - cluster_starts[j], n_features,
                        centers + j * n_features, &weight_ema[j], step_sizes[s],
                        NULL if epoch_sums == NULL
                        else epoch_sums + j * n_features,
                        NULL if epoch_counts == NULL else epoch_counts + j,
                        sums,
                    )
                else:
                    _update_center(
                        X, batch_order + cluster_starts[j],
                        cluster_starts[j + 1] - cluster_starts[j], n_features,
                        centers + j * n_features, &weight_ema[j], step_sizes[s],
                        NULL if epoch_sums == NULL
                        else epoch_sums + j * n_features,
                        NULL if epoch_counts == NULL else epoch_counts + j,
                        sums,
                    )
            emeans_fetch_add(&counter[UPDATE + 1], last - first)
        emeans_wait(&counter[UPDATE + 1], n_clusters)

        if revival_period > 0 and (first_step + s + 1) % revival_period == 0:
            # Revival of the dead clusters, by the first thread to get here.
            if emeans_fetch_add(&counter[REVIVE], 1) == 0:
                with gil:
                    try:
                        (<object> revive)()
                    except BaseException as exc:
                        (<list> errors).append(exc)
                        emeans_fetch_add(failed, 1)
                emeans_fetch_add(&counter[REVIVE + 1], 1)
            emeans_wait(&counter[REVIVE + 1], 1)


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


cdef inline double _squared_norm_64(
    const floating *x,
    intp_t n_features,
) noexcept nogil:
    cdef:
        double result = 0.0
        intp_t k

    for k in range(n_features):
        result += <double> x[k] * x[k]
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


cdef void _transpose(
    const floating *A,  # IN (n_rows, n_cols)
    intp_t n_rows,
    intp_t n_cols,
    floating *A_T,      # OUT (n_cols, n_rows)
    int n_threads,
) noexcept nogil:
    """A_T = A^T, parallel over blocks of columns of A."""
    cdef:
        intp_t n_blocks = (
            (n_cols + TRANSPOSE_BLOCK_SIZE - 1) // TRANSPOSE_BLOCK_SIZE
        )
        intp_t block_idx

    for block_idx in prange(n_blocks, num_threads=n_threads, schedule='static'):
        _transpose_block(A, n_rows, n_cols, A_T, block_idx)


cdef inline void _transpose_block(
    const floating *A,  # IN (n_rows, n_cols)
    intp_t n_rows,
    intp_t n_cols,
    floating *A_T,      # OUT (n_cols, n_rows)
    intp_t block_idx,
) noexcept nogil:
    """Transpose the block `block_idx` of TRANSPOSE_BLOCK_SIZE columns of A."""
    cdef intp_t i, k

    for i in range(n_rows):
        for k in range(block_idx * TRANSPOSE_BLOCK_SIZE,
                       min((block_idx + 1) * TRANSPOSE_BLOCK_SIZE, n_cols)):
            A_T[k * n_rows + i] = A[i * n_cols + k]


cdef void _assign_rows_sparse(
    const floating *X_data,                   # IN (nnz,)
    const int *X_indices,                     # IN (nnz,)
    const int *X_indptr,                      # IN (n_samples + 1,)
    const intp_t *rows,                       # IN (n_rows,), or NULL
    intp_t first_row,
    intp_t n_rows,
    const floating *centers_T,                # IN (n_features, n_clusters)
    const floating *centers_squared_norms,    # IN (n_clusters,)
    const double *centers_squared_norms_64,   # IN (n_clusters,), or NULL
    intp_t n_clusters,
    int *labels,                              # OUT (n_rows,)
    floating *distances,                      # TMP (n_clusters,)
    double *inertia,                          # OUT, or NULL
) noexcept nogil:
    """Assign the CSR rows `X[rows]` to their nearest center.

    If `rows` is NULL, the rows are `first_row, ..., first_row + n_rows - 1`.
    The inertia needs `centers_squared_norms_64`, the squared norms of the
    centers in float64.
    """
    cdef:
        intp_t r, i, p, j, label
        floating minus_2x, dist, min_dist
        const floating *c_k
        double c, diff, squared_distance, total = 0.0

    for r in range(n_rows):
        i = first_row + r if rows == NULL else rows[r]

        # distances = ||c||² - 2 x.c, with x.c = sum_k x_k c_k over the
        # nonzeros of x
        for j in range(n_clusters):
            distances[j] = centers_squared_norms[j]
        for p in range(X_indptr[i], X_indptr[i + 1]):
            minus_2x = -2 * X_data[p]
            c_k = &centers_T[X_indices[p] * n_clusters]
            for j in range(n_clusters):
                distances[j] += minus_2x * c_k[j]

        min_dist = distances[0]
        label = 0
        for j in range(1, n_clusters):
            dist = distances[j]
            if dist < min_dist:
                min_dist = dist
                label = j
        labels[r] = <int> label

        if inertia != NULL:
            # ||x - c||² = ||c||² + sum_k ((x_k - c_k)² - c_k²) over the
            # nonzeros of x, without the cancellation of ||x||².
            squared_distance = centers_squared_norms_64[label]
            for p in range(X_indptr[i], X_indptr[i + 1]):
                c = centers_T[X_indices[p] * n_clusters + label]
                diff = X_data[p] - c
                squared_distance += diff * diff - c * c
            total += max(squared_distance, 0.0)

    if inertia != NULL:
        inertia[0] = total


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
    """EMA update of one center from its batch rows `X[members]`."""
    cdef:
        intp_t m, k
        const floating *x

    if n_members > 0:
        # s_j = sum of the rows of center j
        memset(sums, 0, n_features * sizeof(double))
        for m in range(n_members):
            x = &X[members[m] * n_features]
            for k in range(n_features):
                sums[k] += x[k]

    _ema_update(center, weight_ema, sums, n_members, step_size, epoch_sum,
                epoch_count, n_features)


cdef void _update_center_sparse(
    const floating *X_data,  # IN (nnz,)
    const int *X_indices,    # IN (nnz,)
    const int *X_indptr,     # IN (n_samples + 1,)
    const intp_t *members,   # IN (n_members,)
    intp_t n_members,
    intp_t n_features,
    floating *center,        # INOUT (n_features,)
    double *weight_ema,      # INOUT
    double step_size,
    double *epoch_sum,       # INOUT (n_features,), or NULL
    double *epoch_count,     # INOUT, or NULL
    double *sums,            # TMP (n_features,)
) noexcept nogil:
    """EMA update of one center from its CSR batch rows `X[members]`."""
    cdef intp_t m, i, p

    if n_members > 0:
        # s_j = sum of the rows of center j
        memset(sums, 0, n_features * sizeof(double))
        for m in range(n_members):
            i = members[m]
            for p in range(X_indptr[i], X_indptr[i + 1]):
                sums[X_indices[p]] += X_data[p]

    _ema_update(center, weight_ema, sums, n_members, step_size, epoch_sum,
                epoch_count, n_features)


cdef inline void _ema_update(
    floating *center,     # INOUT (n_features,)
    double *weight_ema,   # INOUT
    const double *sums,   # IN (n_features,), unused if n_members == 0
    intp_t n_members,
    double step_size,
    double *epoch_sum,    # INOUT (n_features,), or NULL
    double *epoch_count,  # INOUT, or NULL
    intp_t n_features,
) noexcept nogil:
    """EMA update of one center from the sum s_j of its n_j batch rows.

    The center c_j = S_j / N_j is the ratio of two EMAs with step alpha, of the
    sum s_j and of the number n_j of its batch rows:
        S_j <- (1 - alpha) S_j + alpha s_j
        N_j <- (1 - alpha) N_j + alpha n_j
    so with N'_j the new N_j:
        c_j <- ((1 - alpha) N_j c_j + alpha s_j) / N'_j
    """
    cdef:
        intp_t k
        double retained = (1.0 - step_size) * weight_ema[0]
        double new_weight, center_coef, sums_coef

    if n_members == 0:
        # Only N_j decays, c_j stays.
        weight_ema[0] = retained
        return

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
