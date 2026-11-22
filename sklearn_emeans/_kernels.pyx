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

For the cost-based revival, the steps also keep three statistics of each
cluster: its removal cost, its axis sums and its axis, see `_removal_update`.

The `_sparse` variants take a CSR matrix. Without a sparse BLAS, the assignment
computes the dot products row by row against the transposed centers, so that
each nonzero x_k adds -2 x_k c[:, k] to the distances of its row with one
contiguous loop over the centers.

`_update_min_distances` and `_update_min_distances_sparse` use the same
assignment to update the distances of the samples to their nearest center, for
the k-means|| initialization.

`_greedy_stats` and `_greedy_stats_sparse` compute the statistics of the
clusters for the greedy revival, with the same assignment and the same
parallelism over centers. `_trial_splits` and `_trial_splits_sparse` then try
to split some clusters in two, in parallel over these clusters.

`_emeans_steps_two_level` and `_labels_inertia_two_level` are the two-level
assignment, for dense X: see the section at the end of the file.

`_all_finite` and `_shuffle` do what `fit` would otherwise spend sequential
time on with many samples: the check of the input and the permutation of each
epoch.
"""

from cython cimport floating
from cython.parallel cimport parallel, prange
from cpython.exc cimport PyErr_CheckSignals
from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_IsValid
from cpython.ref cimport PyObject
from libc.float cimport DBL_MAX, FLT_MAX
from libc.math cimport INFINITY, isfinite, sqrt
from libc.stdint cimport uint32_t, uint64_t
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

    /* The interface of the bit generators of NumPy, as in
       numpy/random/bitgen.h. It is copied here so that the build doesn't
       need the headers of NumPy. */
    #include <stdint.h>
    typedef struct {
        void *state;
        uint64_t (*next_uint64)(void *state);
        uint32_t (*next_uint32)(void *state);
        double (*next_double)(void *state);
        uint64_t (*next_raw)(void *state);
    } emeans_bitgen_t;

    /* Prefetch for a write. */
    #if defined(__GNUC__) || defined(__clang__)
    #define emeans_prefetch(p) __builtin_prefetch((p), 1)
    #elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    #include <xmmintrin.h>
    #define emeans_prefetch(p) _mm_prefetch((const char *) (p), _MM_HINT_T0)
    #else
    #define emeans_prefetch(p) ((void) 0)
    #endif
    """
    ctypedef struct emeans_bitgen_t:
        void *state
        uint64_t (*next_uint64)(void *state) noexcept nogil
        uint32_t (*next_uint32)(void *state) noexcept nogil
    void emeans_prefetch(void *p) noexcept nogil
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


# Number of values checked by one iteration of the loop of `_all_finite`.
cdef intp_t FINITE_BLOCK_SIZE = 1 << 18

# Number of swaps of `_shuffle` whose indices are drawn ahead.
cdef enum:
    SHUFFLE_BATCH_SIZE = 1024


def _all_finite(const floating[:, ::1] X, int n_threads):
    """Whether X has no NaN and no infinity.

    scikit-learn checks it with a sum of X on one thread. Here the blocks of
    values are checked in parallel.
    """
    cdef:
        intp_t n_values = X.shape[0] * X.shape[1]
        intp_t n_blocks = (n_values + FINITE_BLOCK_SIZE - 1) // FINITE_BLOCK_SIZE
        const floating *values
        intp_t b
        intp_t n_not_finite = 0

    if n_values == 0:
        return True
    values = &X[0, 0]
    n_threads = max(1, min(n_threads, n_blocks))
    for b in prange(n_blocks, nogil=True, schedule="static", num_threads=n_threads):
        n_not_finite += _has_not_finite(
            values + b * FINITE_BLOCK_SIZE,
            min(FINITE_BLOCK_SIZE, n_values - b * FINITE_BLOCK_SIZE),
        )
    return n_not_finite == 0


cdef inline bint _has_not_finite(const floating *x, intp_t n) noexcept nogil:
    cdef:
        # 8 partial sums, since the compiler must add floats in order.
        floating[8] sums
        floating total = 0.0
        intp_t n_blocks = n // 8
        intp_t i, j

    for j in range(8):
        sums[j] = 0.0
    # x * 0 is 0 if x is finite and NaN otherwise: unlike the sum of x, the
    # sum can't overflow.
    for i in range(n_blocks):
        for j in range(8):
            sums[j] += x[8 * i + j] * 0
    for i in range(8 * n_blocks, n):
        total += x[i] * 0
    for j in range(8):
        total += sums[j]
    return total != total


def _shuffle(capsule, intp_t[::1] values):
    """Shuffle `values` inplace as `RandomState.shuffle` does.

    `capsule` is the capsule of the bit generator of the RandomState, whose
    lock is held by the caller. The draws and the swaps are the ones of NumPy,
    so the result and the state of the generator after the call are the same.
    NumPy draws an index and swaps right away, and each swap is a cache miss
    when `values` is large. Here the indices of the next swaps are drawn and
    prefetched before these swaps are done, which is about twice as fast on
    100M values or more.

    Return False, without drawing anything, if `capsule` is not a bit
    generator.
    """
    cdef:
        emeans_bitgen_t *bitgen
        intp_t *a
        uint64_t[SHUFFLE_BATCH_SIZE] indices
        uint64_t i, n_batch, b, last, mask, index
        intp_t value

    if not PyCapsule_IsValid(capsule, "BitGenerator"):
        return False
    if values.shape[0] < 2:
        return True
    bitgen = <emeans_bitgen_t *> PyCapsule_GetPointer(capsule, "BitGenerator")
    a = &values[0]
    with nogil:
        i = values.shape[0] - 1
        while i >= 1:
            n_batch = min(<uint64_t> SHUFFLE_BATCH_SIZE, i)
            for b in range(n_batch):
                # `random_interval(bitgen, last)` of NumPy: a value in
                # [0, last], by rejection among the values that have as many
                # bits as `last`.
                last = i - b
                mask = last
                mask |= mask >> 1
                mask |= mask >> 2
                mask |= mask >> 4
                mask |= mask >> 8
                mask |= mask >> 16
                mask |= mask >> 32
                if last <= 0xffffffffUL:
                    index = bitgen.next_uint32(bitgen.state) & mask
                    while index > last:
                        index = bitgen.next_uint32(bitgen.state) & mask
                else:
                    index = bitgen.next_uint64(bitgen.state) & mask
                    while index > last:
                        index = bitgen.next_uint64(bitgen.state) & mask
                indices[b] = index
                emeans_prefetch(&a[index])
            for b in range(n_batch):
                value = a[i - b]
                a[i - b] = a[indices[b]]
                a[indices[b]] = value
            i -= n_batch
    return True


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
    double[::1] removal_cost=None,         # INOUT
    double[:, ::1] axis_sums=None,         # INOUT
    floating[:, ::1] axes=None,            # INOUT
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

    removal_cost : ndarray of shape (n_clusters,), dtype=float64, default=None
        If not None, the statistics of the cost-based revival are updated
        inplace by each step: `removal_cost`, `axis_sums` and `axes`, see
        `_removal_update`. It needs at least 2 centers. EMA of the removal
        cost of each cluster.

    axis_sums : ndarray of shape (n_clusters, n_features), dtype=float64, \
            default=None
        EMA of the axis sums of each cluster.

    axes : ndarray of shape (n_clusters, n_features), dtype=floating, \
            default=None
        Unit axis of each cluster.
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
        NULL if removal_cost is None else &removal_cost[0],
        NULL if removal_cost is None else &axis_sums[0, 0],
        NULL if removal_cost is None else &axes[0, 0],
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
    double[::1] removal_cost=None,         # INOUT
    double[:, ::1] axis_sums=None,         # INOUT
    floating[:, ::1] axes=None,            # INOUT
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
        NULL if removal_cost is None else &removal_cost[0],
        NULL if removal_cost is None else &axis_sums[0, 0],
        NULL if removal_cost is None else &axes[0, 0],
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
    double *removal_cost,        # INOUT (n_clusters,), or NULL
    double *axis_sums,           # INOUT (n_clusters, n_features), or NULL
    floating *axes,              # INOUT (n_clusters, n_features), or NULL
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
        bint with_removal = removal_cost != NULL
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
        # Gap of each batch row, in batch order then sorted by label.
        double *gaps = NULL
        double *sorted_gaps = NULL
        floating *X_chunk = NULL
        floating *distances = NULL
        double *sums = NULL

    if is_sparse:
        centers_T = <floating*> malloc(n_features * n_clusters * sizeof(floating))
    if with_removal:
        gaps = <double*> malloc(max_batch_size * sizeof(double))
        sorted_gaps = <double*> malloc(max_batch_size * sizeof(double))

    if (
        labels == NULL
        or cluster_starts == NULL
        or batch_order == NULL
        or counters == NULL
        or centers_squared_norms == NULL
        or (is_sparse and centers_T == NULL)
        or (with_removal and (gaps == NULL or sorted_gaps == NULL))
    ):
        free(labels)
        free(cluster_starts)
        free(batch_order)
        free(counters)
        free(centers_squared_norms)
        free(centers_T)
        free(gaps)
        free(sorted_gaps)
        raise MemoryError()

    with nogil, parallel(num_threads=n_threads):
        if not is_sparse:
            X_chunk = <floating*> malloc(CHUNK_SIZE * n_features * sizeof(floating))
        distances = <floating*> malloc(distances_size * sizeof(floating))
        # With the removal statistics, the second half is for the axis sums of
        # the batch.
        sums = <double*> malloc(
            (2 if with_removal else 1) * n_features * sizeof(double)
        )

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
                removal_cost, axis_sums, axes, gaps, sorted_gaps,
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
    free(gaps)
    free(sorted_gaps)

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
    double *removal_cost,             # shared (n_clusters,), or NULL
    double *axis_sums,                # shared (n_clusters, n_features), or NULL
    floating *axes,                   # shared (n_clusters, n_features), or NULL
    double *gaps,                     # shared (batch_size,), or NULL
    double *sorted_gaps,              # shared (batch_size,), or NULL
) noexcept nogil:
    """What each thread of `_run_steps` runs: the phases of all the steps.

    A thread can be late, or even go through steps that the others have
    already completed: it then finds nothing to claim and nothing to wait for.
    """
    cdef:
        bint is_sparse = X == NULL
        bint with_removal = removal_cost != NULL
        bint is_main_thread = emeans_omp_thread_num() == 0
        intp_t n_transpose_blocks = (
            (n_features + TRANSPOSE_BLOCK_SIZE - 1) // TRANSPOSE_BLOCK_SIZE
        )
        intp_t s, n_batch, n_chunks, chunk_idx, start, n_chunk_rows, r, j
        intp_t block_idx, first, last, n_members
        double *axis_batch = sums + n_features
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
            if is_sparse and with_removal:
                for r in range(start, start + n_chunk_rows):
                    _sparse_row_distances(
                        X_data, X_indices, X_indptr, batch[r], centers_T,
                        centers_squared_norms, n_clusters, distances,
                    )
                    _two_nearest(distances, n_clusters, &labels[r], &gaps[r])
            elif is_sparse:
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
                if with_removal:
                    _two_nearest_rows(
                        X_chunk, n_chunk_rows, centers, centers_squared_norms,
                        n_clusters, n_features, labels + start, distances,
                        gaps + start,
                    )
                else:
                    _assign_rows(
                        X_chunk, n_chunk_rows, centers, centers_squared_norms,
                        n_clusters, n_features, labels + start, distances, NULL,
                    )
            emeans_fetch_add(&counter[ASSIGN + 1], 1)
        emeans_wait(&counter[ASSIGN + 1], n_chunks)

        # Sort of the batch rows by label, by the first thread to get here.
        if emeans_fetch_add(&counter[SORT], 1) == 0:
            _sort_by_label(batch, labels, n_batch, n_clusters, cluster_starts,
                           batch_order, gaps, sorted_gaps)
            emeans_fetch_add(&counter[SORT + 1], 1)
        emeans_wait(&counter[SORT + 1], 1)

        # EMA update, by blocks of centers.
        while True:
            first = emeans_fetch_add(&counter[UPDATE], block_size)
            if first >= n_clusters:
                break
            last = min(first + block_size, n_clusters)
            for j in range(first, last):
                if with_removal:
                    # Before the update of the center: the statistics are
                    # relative to the center that the rows were assigned to.
                    n_members = cluster_starts[j + 1] - cluster_starts[j]
                    if n_members > 0 and is_sparse:
                        _axis_batch_sparse(
                            X_data, X_indices, X_indptr,
                            batch_order + cluster_starts[j], n_members,
                            n_features, centers + j * n_features,
                            axes + j * n_features, axis_batch,
                        )
                    elif n_members > 0:
                        _axis_batch(
                            X, batch_order + cluster_starts[j], n_members,
                            n_features, centers + j * n_features,
                            axes + j * n_features, axis_batch,
                        )
                    _removal_update(
                        &removal_cost[j], axis_sums + j * n_features,
                        axes + j * n_features, axis_batch,
                        sorted_gaps + cluster_starts[j], n_members,
                        step_sizes[s], n_features,
                    )
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
    const double *values,         # IN (n_batch,), or NULL
    double *sorted_values,        # OUT (n_batch,), or NULL
) noexcept nogil:
    """Counting sort of the batch rows by label.

    The rows assigned to center j are
    `batch_order[cluster_starts[j]:cluster_starts[j + 1]]`, in batch order.
    If `values` is not NULL, `sorted_values` gets the value of each batch row
    in the same order.
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
        if values != NULL:
            sorted_values[cluster_starts[labels[p]]] = values[p]
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


cdef void _axis_batch(
    const floating *X,       # IN (n_samples, n_features)
    const intp_t *members,   # IN (n_members,)
    intp_t n_members,
    intp_t n_features,
    const floating *center,  # IN (n_features,)
    const floating *axis,    # IN (n_features,)
    double *axis_batch,      # OUT (n_features,)
) noexcept nogil:
    """Axis sums of one center over its batch rows `X[members]`.

    The sum of p y over the rows x, with y = x - center and p = <y, axis>,
    computed as sum(p x) - sum(p) center.
    """
    cdef:
        intp_t m, k
        const floating *x
        double p, center_p = 0.0, total_p = 0.0

    memset(axis_batch, 0, n_features * sizeof(double))
    for k in range(n_features):
        center_p += <double> center[k] * axis[k]
    for m in range(n_members):
        x = &X[members[m] * n_features]
        p = -center_p
        for k in range(n_features):
            p += <double> x[k] * axis[k]
        total_p += p
        for k in range(n_features):
            axis_batch[k] += p * x[k]
    for k in range(n_features):
        axis_batch[k] -= total_p * center[k]


cdef void _axis_batch_sparse(
    const floating *X_data,  # IN (nnz,)
    const int *X_indices,    # IN (nnz,)
    const int *X_indptr,     # IN (n_samples + 1,)
    const intp_t *members,   # IN (n_members,)
    intp_t n_members,
    intp_t n_features,
    const floating *center,  # IN (n_features,)
    const floating *axis,    # IN (n_features,)
    double *axis_batch,      # OUT (n_features,)
) noexcept nogil:
    """`_axis_batch` of the CSR rows `X[members]`."""
    cdef:
        intp_t m, i, k, q
        double p, center_p = 0.0, total_p = 0.0

    memset(axis_batch, 0, n_features * sizeof(double))
    for k in range(n_features):
        center_p += <double> center[k] * axis[k]
    for m in range(n_members):
        i = members[m]
        p = -center_p
        for q in range(X_indptr[i], X_indptr[i + 1]):
            p += <double> X_data[q] * axis[X_indices[q]]
        total_p += p
        for q in range(X_indptr[i], X_indptr[i + 1]):
            axis_batch[X_indices[q]] += p * X_data[q]
    for k in range(n_features):
        axis_batch[k] -= total_p * center[k]


cdef inline void _removal_update(
    double *removal_cost,      # INOUT
    double *axis_sum,          # INOUT (n_features,)
    floating *axis,            # INOUT (n_features,)
    const double *axis_batch,  # IN (n_features,), unused if n_members == 0
    const double *gaps,        # IN (n_members,)
    intp_t n_members,
    double step_size,
    intp_t n_features,
) noexcept nogil:
    """Update of the statistics of one center for the cost-based revival.

    They are EMAs with step alpha, as the weight N_j of the center:
    - the removal cost R_j, of the sum over its batch rows of their gap: how
      much further their second nearest center is, in squared distance. It
      estimates how much the inertia of a batch would grow without this
      center.
    - the axis sums U_j, of the sum of <y, u_j> y over its batch rows x, with
      y = x - c_j. The axis u_j = U_j / ||U_j|| then follows the direction in
      which the cluster is the longest (Oja's rule), where ||U_j|| / N_j
      estimates its variance. The axis only changes when the center gets rows.
    """
    cdef:
        intp_t m, k
        double decay = 1.0 - step_size
        double gap_sum = 0.0, squared_norm = 0.0, norm

    for m in range(n_members):
        if isfinite(gaps[m]):
            gap_sum += gaps[m]
    removal_cost[0] = decay * removal_cost[0] + step_size * gap_sum

    if n_members == 0:
        for k in range(n_features):
            axis_sum[k] *= decay
        return

    for k in range(n_features):
        axis_sum[k] = decay * axis_sum[k] + step_size * axis_batch[k]
        squared_norm += axis_sum[k] * axis_sum[k]
    norm = sqrt(squared_norm)
    if norm > 0:
        for k in range(n_features):
            axis[k] = <floating> (axis_sum[k] / norm)


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


def _greedy_stats(
    const floating[:, ::1] X,        # IN
    const floating[:, ::1] centers,  # IN
    int[::1] labels,                 # OUT (n_samples,)
    intp_t[::1] order,               # OUT (n_samples,)
    intp_t[::1] cluster_starts,      # OUT (n_clusters + 1,)
    double[::1] cost,                # OUT (n_clusters,)
    double[::1] max_scatter,         # OUT (n_clusters,)
    intp_t[::1] feature,             # OUT (n_clusters,)
    int n_threads,
):
    """Statistics of the clusters for the greedy revival.

    Assigns each sample to its nearest center, then computes for each cluster:

    - `cost`: the increase of the inertia when its samples go to their second
      nearest center;
    - `max_scatter` and `feature`: the largest sum of squared deviations to
      the mean of its samples along one feature, and this feature.

    The samples of the cluster j are
    `order[cluster_starts[j]:cluster_starts[j + 1]]`, in increasing order.
    The results don't depend on the number of threads.
    """
    cdef:
        intp_t n_samples = X.shape[0]
        intp_t n_clusters = centers.shape[0]
        intp_t n_features = centers.shape[1]
        intp_t n_chunks = (n_samples + CHUNK_SIZE - 1) // CHUNK_SIZE
        intp_t chunk_idx, start, j
        intp_t n_ready = 0
        intp_t n_ready_scatter = 0
        intp_t n_failed = 0
        floating *centers_squared_norms
        floating *distances = NULL
        double *gaps
        double *sums = NULL

    gaps = <double*> malloc(max(n_samples, 1) * sizeof(double))
    centers_squared_norms = <floating*> malloc(n_clusters * sizeof(floating))
    if gaps == NULL or centers_squared_norms == NULL:
        free(gaps)
        free(centers_squared_norms)
        raise MemoryError()

    with nogil:
        for j in range(n_clusters):
            centers_squared_norms[j] = _squared_norm(&centers[j, 0], n_features)

        with parallel(num_threads=max(1, min(n_threads, n_chunks))):
            distances = <floating*> malloc(CHUNK_SIZE * n_clusters * sizeof(floating))

            if emeans_buffers_ready(&n_ready, &n_failed, distances == NULL):
                for chunk_idx in prange(n_chunks, schedule='static'):
                    start = chunk_idx * CHUNK_SIZE
                    _two_nearest_rows(
                        &X[start, 0],
                        min(CHUNK_SIZE, n_samples - start),
                        &centers[0, 0],
                        centers_squared_norms,
                        n_clusters,
                        n_features,
                        &labels[start],
                        distances,
                        &gaps[start],
                    )

            free(distances)

        if not n_failed:
            _sort_rows_by_label(&labels[0], n_samples, n_clusters,
                                &cluster_starts[0], &order[0])

            with parallel(num_threads=n_threads):
                sums = <double*> malloc(2 * n_features * sizeof(double))

                if emeans_buffers_ready(
                    &n_ready_scatter, &n_failed, sums == NULL
                ):
                    for j in prange(n_clusters, schedule='dynamic'):
                        _cluster_scatter(
                            &X[0, 0] if n_samples > 0 else NULL,
                            n_features,
                            &order[0] + cluster_starts[j],
                            cluster_starts[j + 1] - cluster_starts[j],
                            &centers[j, 0],
                            gaps,
                            sums,
                            &cost[j],
                            &max_scatter[j],
                            &feature[j],
                        )

                free(sums)

    free(gaps)
    free(centers_squared_norms)
    if n_failed:
        raise MemoryError()


def _greedy_stats_sparse(
    X,                               # IN
    const floating[:, ::1] centers,  # IN
    int[::1] labels,                 # OUT (n_samples,)
    intp_t[::1] order,               # OUT (n_samples,)
    intp_t[::1] cluster_starts,      # OUT (n_clusters + 1,)
    double[::1] cost,                # OUT (n_clusters,)
    double[::1] max_scatter,         # OUT (n_clusters,)
    intp_t[::1] feature,             # OUT (n_clusters,)
    int n_threads,
):
    """Same as `_greedy_stats`, with `X` a CSR matrix with int32 indices."""
    cdef:
        const floating[::1] X_data = X.data
        const int[::1] X_indices = X.indices
        const int[::1] X_indptr = X.indptr
        intp_t n_samples = X.shape[0]
        intp_t n_clusters = centers.shape[0]
        intp_t n_features = centers.shape[1]
        intp_t n_chunks = (n_samples + CHUNK_SIZE - 1) // CHUNK_SIZE
        intp_t chunk_idx, start, i, j
        intp_t n_ready = 0
        intp_t n_ready_scatter = 0
        intp_t n_failed = 0
        floating *centers_T
        floating *centers_squared_norms
        floating *distances = NULL
        double *gaps
        double *sums = NULL

    gaps = <double*> malloc(max(n_samples, 1) * sizeof(double))
    centers_T = <floating*> malloc(n_features * n_clusters * sizeof(floating))
    centers_squared_norms = <floating*> malloc(n_clusters * sizeof(floating))
    if gaps == NULL or centers_T == NULL or centers_squared_norms == NULL:
        free(gaps)
        free(centers_T)
        free(centers_squared_norms)
        raise MemoryError()

    with nogil:
        for j in range(n_clusters):
            centers_squared_norms[j] = _squared_norm(&centers[j, 0], n_features)

        _transpose(&centers[0, 0], n_clusters, n_features, centers_T, n_threads)

        with parallel(num_threads=max(1, min(n_threads, n_chunks))):
            distances = <floating*> malloc(n_clusters * sizeof(floating))

            if emeans_buffers_ready(&n_ready, &n_failed, distances == NULL):
                for chunk_idx in prange(n_chunks, schedule='static'):
                    start = chunk_idx * CHUNK_SIZE
                    for i in range(start, min(start + CHUNK_SIZE, n_samples)):
                        _sparse_row_distances(
                            &X_data[0], &X_indices[0], &X_indptr[0], i, centers_T,
                            centers_squared_norms, n_clusters, distances,
                        )
                        _two_nearest(distances, n_clusters, &labels[i], &gaps[i])

            free(distances)

        if not n_failed:
            _sort_rows_by_label(&labels[0], n_samples, n_clusters,
                                &cluster_starts[0], &order[0])

            with parallel(num_threads=n_threads):
                sums = <double*> malloc(2 * n_features * sizeof(double))

                if emeans_buffers_ready(
                    &n_ready_scatter, &n_failed, sums == NULL
                ):
                    for j in prange(n_clusters, schedule='dynamic'):
                        _cluster_scatter_sparse(
                            &X_data[0] if X_data.shape[0] > 0 else NULL,
                            &X_indices[0] if X_indices.shape[0] > 0 else NULL,
                            &X_indptr[0],
                            n_features,
                            &order[0] + cluster_starts[j],
                            cluster_starts[j + 1] - cluster_starts[j],
                            gaps,
                            sums,
                            &cost[j],
                            &max_scatter[j],
                            &feature[j],
                        )

                free(sums)

    free(gaps)
    free(centers_T)
    free(centers_squared_norms)
    if n_failed:
        raise MemoryError()


cdef inline void _two_nearest(
    const floating *distances,  # IN (n_clusters,)
    intp_t n_clusters,
    int *label,                 # OUT
    double *gap,                # OUT
) noexcept nogil:
    """Nearest center, and how much further the second nearest one is.

    The gap between the two squared distances is 0 for a single center.
    """
    cdef:
        intp_t j, nearest = 0
        floating dist
        floating min_dist = distances[0]
        floating second_dist = <floating> INFINITY

    for j in range(1, n_clusters):
        dist = distances[j]
        if dist < min_dist:
            second_dist = min_dist
            min_dist = dist
            nearest = j
        elif dist < second_dist:
            second_dist = dist
    label[0] = <int> nearest
    gap[0] = <double> second_dist - min_dist if n_clusters > 1 else 0.0


cdef void _two_nearest_rows(
    const floating *X_rows,                 # IN (n_rows, n_features)
    intp_t n_rows,
    const floating *centers,                # IN (n_clusters, n_features)
    const floating *centers_squared_norms,  # IN (n_clusters,)
    intp_t n_clusters,
    intp_t n_features,
    int *labels,                            # OUT (n_rows,)
    floating *distances,                    # TMP (n_rows, n_clusters)
    double *gaps,                           # OUT (n_rows,)
) noexcept nogil:
    """`_two_nearest` of the contiguous rows `X_rows`, with one GEMM."""
    cdef intp_t r, j

    for r in range(n_rows):
        for j in range(n_clusters):
            distances[r * n_clusters + j] = centers_squared_norms[j]

    # distances = ||c||² - 2 x.c, as in `_assign_rows`.
    _gemm_nt(n_rows, n_clusters, n_features, -2.0, X_rows, centers, 1.0,
             distances)

    for r in range(n_rows):
        _two_nearest(&distances[r * n_clusters], n_clusters, &labels[r], &gaps[r])


cdef inline void _sparse_row_distances(
    const floating *X_data,                 # IN (nnz,)
    const int *X_indices,                   # IN (nnz,)
    const int *X_indptr,                    # IN (n_samples + 1,)
    intp_t i,
    const floating *centers_T,              # IN (n_features, n_clusters)
    const floating *centers_squared_norms,  # IN (n_clusters,)
    intp_t n_clusters,
    floating *distances,                    # OUT (n_clusters,)
) noexcept nogil:
    """distances = ||c||² - 2 x.c for the CSR row i, as in `_assign_rows_sparse`."""
    cdef:
        intp_t p, j
        floating minus_2x
        const floating *c_k

    for j in range(n_clusters):
        distances[j] = centers_squared_norms[j]
    for p in range(X_indptr[i], X_indptr[i + 1]):
        minus_2x = -2 * X_data[p]
        c_k = &centers_T[X_indices[p] * n_clusters]
        for j in range(n_clusters):
            distances[j] += minus_2x * c_k[j]


cdef void _sort_rows_by_label(
    const int *labels,       # IN (n_samples,)
    intp_t n_samples,
    intp_t n_clusters,
    intp_t *cluster_starts,  # OUT (n_clusters + 1,)
    intp_t *order,           # OUT (n_samples,)
) noexcept nogil:
    """Counting sort of all the rows by label, as in `_sort_by_label`."""
    cdef intp_t i, j

    memset(cluster_starts, 0, (n_clusters + 1) * sizeof(intp_t))
    for i in range(n_samples):
        cluster_starts[labels[i] + 1] += 1
    for j in range(n_clusters):
        cluster_starts[j + 1] += cluster_starts[j]
    for i in range(n_samples):
        order[cluster_starts[labels[i]]] = i
        cluster_starts[labels[i]] += 1
    for j in range(n_clusters, 0, -1):
        cluster_starts[j] = cluster_starts[j - 1]
    cluster_starts[0] = 0


cdef inline void _max_scatter(
    const double *sums,   # IN (2 * n_features,): sums, then sums of squares
    intp_t n_features,
    intp_t n_rows,
    double *max_scatter,  # OUT
    intp_t *feature,      # OUT
) noexcept nogil:
    """Largest sum of squared deviations to the mean along one feature."""
    cdef:
        intp_t k
        double scatter

    max_scatter[0] = 0.0
    feature[0] = 0
    if n_rows == 0:
        return
    for k in range(n_features):
        scatter = sums[n_features + k] - sums[k] * sums[k] / n_rows
        if scatter > max_scatter[0]:
            max_scatter[0] = scatter
            feature[0] = k


cdef void _cluster_scatter(
    const floating *X,       # IN (n_samples, n_features)
    intp_t n_features,
    const intp_t *rows,      # IN (n_rows,)
    intp_t n_rows,
    const floating *center,  # IN (n_features,)
    const double *gaps,      # IN (n_samples,)
    double *sums,            # TMP (2 * n_features,)
    double *cost,            # OUT
    double *max_scatter,     # OUT
    intp_t *feature,         # OUT
) noexcept nogil:
    """Removal cost and largest scatter along one feature of a cluster.

    The sums are those of the deviations to the center, which is near the
    mean, for their accuracy.
    """
    cdef:
        intp_t r, k
        const floating *x
        double diff, total = 0.0

    memset(sums, 0, 2 * n_features * sizeof(double))
    for r in range(n_rows):
        total += gaps[rows[r]]
        x = &X[rows[r] * n_features]
        for k in range(n_features):
            diff = <double> x[k] - center[k]
            sums[k] += diff
            sums[n_features + k] += diff * diff
    cost[0] = total
    _max_scatter(sums, n_features, n_rows, max_scatter, feature)


cdef void _cluster_scatter_sparse(
    const floating *X_data,  # IN (nnz,)
    const int *X_indices,    # IN (nnz,)
    const int *X_indptr,     # IN (n_samples + 1,)
    intp_t n_features,
    const intp_t *rows,      # IN (n_rows,)
    intp_t n_rows,
    const double *gaps,      # IN (n_samples,)
    double *sums,            # TMP (2 * n_features,)
    double *cost,            # OUT
    double *max_scatter,     # OUT
    intp_t *feature,         # OUT
) noexcept nogil:
    """Same as `_cluster_scatter` for CSR rows, with sums of the values."""
    cdef:
        intp_t r, p
        double value, total = 0.0

    memset(sums, 0, 2 * n_features * sizeof(double))
    for r in range(n_rows):
        total += gaps[rows[r]]
        for p in range(X_indptr[rows[r]], X_indptr[rows[r] + 1]):
            value = X_data[p]
            sums[X_indices[p]] += value
            sums[n_features + X_indices[p]] += value * value
    cost[0] = total
    _max_scatter(sums, n_features, n_rows, max_scatter, feature)


def _trial_splits(
    const floating[:, ::1] X,              # IN
    const intp_t[::1] order,               # IN (n_samples,)
    const intp_t[::1] cluster_starts,      # IN (n_clusters + 1,)
    const intp_t[::1] clusters,            # IN (n_trials,)
    const intp_t[::1] feature,             # IN (n_clusters,)
    const double[::1] std,                 # IN (n_clusters,)
    int n_iterations,
    double[::1] gains,                     # OUT (n_trials,)
    double[:, :, ::1] split_centers,       # OUT (n_trials, 2, n_features)
    int n_threads,
):
    """2-means of the samples of some clusters, on a half of them.

    The samples of the cluster j are
    `order[cluster_starts[j]:cluster_starts[j + 1]]`. Its 2-means is fitted on
    the samples at the even positions, with `n_iterations` iterations from
    their mean moved by `std[j]` in both directions along `feature[j]`.
    `gains` is the decrease of the inertia of the cluster, estimated on the
    samples at the odd positions. Each cluster of `clusters` must have at
    least 2 samples.
    """
    cdef:
        intp_t n_trials = clusters.shape[0]
        intp_t n_features = X.shape[1]
        intp_t t, j
        intp_t n_ready = 0
        intp_t n_failed = 0
        double *buffer = NULL

    with nogil, parallel(num_threads=n_threads):
        buffer = <double*> malloc(3 * n_features * sizeof(double))

        if emeans_buffers_ready(&n_ready, &n_failed, buffer == NULL):
            for t in prange(n_trials, schedule='dynamic'):
                j = clusters[t]
                _trial_split(
                    &X[0, 0],
                    n_features,
                    &order[0] + cluster_starts[j],
                    cluster_starts[j + 1] - cluster_starts[j],
                    feature[j],
                    std[j],
                    n_iterations,
                    buffer,
                    &split_centers[t, 0, 0],
                    &gains[t],
                )

        free(buffer)

    if n_failed:
        raise MemoryError()


def _trial_splits_sparse(
    const floating[::1] X_data,            # IN (nnz,)
    const int[::1] X_indices,              # IN (nnz,)
    const int[::1] X_indptr,               # IN (n_samples + 1,)
    intp_t n_features,
    const intp_t[::1] order,               # IN (n_samples,)
    const intp_t[::1] cluster_starts,      # IN (n_clusters + 1,)
    const intp_t[::1] clusters,            # IN (n_trials,)
    const intp_t[::1] feature,             # IN (n_clusters,)
    const double[::1] std,                 # IN (n_clusters,)
    int n_iterations,
    double[::1] gains,                     # OUT (n_trials,)
    double[:, :, ::1] split_centers,       # OUT (n_trials, 2, n_features)
    int n_threads,
):
    """Same as `_trial_splits`, for a CSR matrix with int32 indices."""
    cdef:
        intp_t n_trials = clusters.shape[0]
        intp_t t, j
        intp_t n_ready = 0
        intp_t n_failed = 0
        double *buffer = NULL

    with nogil, parallel(num_threads=n_threads):
        buffer = <double*> malloc(3 * n_features * sizeof(double))

        if emeans_buffers_ready(&n_ready, &n_failed, buffer == NULL):
            for t in prange(n_trials, schedule='dynamic'):
                j = clusters[t]
                _trial_split_sparse(
                    &X_data[0] if X_data.shape[0] > 0 else NULL,
                    &X_indices[0] if X_indices.shape[0] > 0 else NULL,
                    &X_indptr[0],
                    n_features,
                    &order[0] + cluster_starts[j],
                    cluster_starts[j + 1] - cluster_starts[j],
                    feature[j],
                    std[j],
                    n_iterations,
                    buffer,
                    &split_centers[t, 0, 0],
                    &gains[t],
                )

        free(buffer)

    if n_failed:
        raise MemoryError()


cdef inline double _squared_distance_64(
    const floating *x,
    const double *y,
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


cdef inline double _sparse_squared_distance_64(
    const floating *X_data,  # IN (nnz,)
    const int *X_indices,    # IN (nnz,)
    const int *X_indptr,     # IN (n_samples + 1,)
    intp_t i,
    const double *y,         # IN (n_features,)
    double y_squared_norm,
) noexcept nogil:
    """||x - y||² for the CSR row i: ||y||² + sum_k ((x_k - y_k)² - y_k²)."""
    cdef:
        double result = y_squared_norm
        double diff
        intp_t p

    for p in range(X_indptr[i], X_indptr[i + 1]):
        diff = X_data[p] - y[X_indices[p]]
        result += diff * diff - y[X_indices[p]] * y[X_indices[p]]
    return result


cdef inline double _squared_norm_of_double(
    const double *x,
    intp_t n_features,
) noexcept nogil:
    cdef:
        double result = 0.0
        intp_t k

    for k in range(n_features):
        result += x[k] * x[k]
    return result


cdef inline bint _update_split_centers(
    const double *sums,  # IN (2, n_features)
    intp_t n_features,
    intp_t n_first,
    intp_t n_second,
    double *centers,     # INOUT (2, n_features)
) noexcept nogil:
    """Means of the two sides, unless one is empty. Returns whether it is."""
    cdef intp_t k

    if n_first == 0 or n_second == 0:
        return True
    for k in range(n_features):
        centers[k] = sums[k] / n_first
        centers[n_features + k] = sums[n_features + k] / n_second
    return False


cdef void _trial_split(
    const floating *X,   # IN (n_samples, n_features)
    intp_t n_features,
    const intp_t *rows,  # IN (n_rows,)
    intp_t n_rows,
    intp_t feature,
    double std,
    int n_iterations,
    double *buffer,      # TMP (3, n_features)
    double *centers,     # OUT (2, n_features)
    double *gain,        # OUT
) noexcept nogil:
    """2-means of the rows at the even positions, gain on the other ones."""
    cdef:
        double *mean = buffer
        double *sums = buffer + n_features
        intp_t n_fitted = (n_rows + 1) // 2
        intp_t r, k, side, n_second
        const floating *x
        double whole = 0.0, split = 0.0

    memset(mean, 0, n_features * sizeof(double))
    for r in range(0, n_rows, 2):
        x = &X[rows[r] * n_features]
        for k in range(n_features):
            mean[k] += x[k]
    for k in range(n_features):
        mean[k] /= n_fitted
        centers[k] = mean[k]
        centers[n_features + k] = mean[k]
    centers[feature] += std
    centers[n_features + feature] -= std

    for _ in range(n_iterations):
        memset(sums, 0, 2 * n_features * sizeof(double))
        n_second = 0
        for r in range(0, n_rows, 2):
            x = &X[rows[r] * n_features]
            side = (_squared_distance_64(x, centers, n_features)
                    > _squared_distance_64(x, centers + n_features, n_features))
            n_second += side
            for k in range(n_features):
                sums[side * n_features + k] += x[k]
        if _update_split_centers(sums, n_features, n_fitted - n_second, n_second,
                                 centers):
            break

    for r in range(1, n_rows, 2):
        x = &X[rows[r] * n_features]
        whole += _squared_distance_64(x, mean, n_features)
        split += min(_squared_distance_64(x, centers, n_features),
                     _squared_distance_64(x, centers + n_features, n_features))
    gain[0] = (whole - split) * n_rows / (n_rows // 2)


cdef void _trial_split_sparse(
    const floating *X_data,  # IN (nnz,)
    const int *X_indices,    # IN (nnz,)
    const int *X_indptr,     # IN (n_samples + 1,)
    intp_t n_features,
    const intp_t *rows,      # IN (n_rows,)
    intp_t n_rows,
    intp_t feature,
    double std,
    int n_iterations,
    double *buffer,          # TMP (3, n_features)
    double *centers,         # OUT (2, n_features)
    double *gain,            # OUT
) noexcept nogil:
    """Same as `_trial_split` for CSR rows."""
    cdef:
        double *mean = buffer
        double *sums = buffer + n_features
        intp_t n_fitted = (n_rows + 1) // 2
        intp_t r, k, p, i, side, n_second
        double whole = 0.0, split = 0.0
        double mean_squared_norm, first_squared_norm, second_squared_norm

    memset(mean, 0, n_features * sizeof(double))
    for r in range(0, n_rows, 2):
        i = rows[r]
        for p in range(X_indptr[i], X_indptr[i + 1]):
            mean[X_indices[p]] += X_data[p]
    for k in range(n_features):
        mean[k] /= n_fitted
        centers[k] = mean[k]
        centers[n_features + k] = mean[k]
    centers[feature] += std
    centers[n_features + feature] -= std

    for _ in range(n_iterations):
        first_squared_norm = _squared_norm_of_double(centers, n_features)
        second_squared_norm = _squared_norm_of_double(centers + n_features,
                                                      n_features)
        memset(sums, 0, 2 * n_features * sizeof(double))
        n_second = 0
        for r in range(0, n_rows, 2):
            i = rows[r]
            side = (
                _sparse_squared_distance_64(X_data, X_indices, X_indptr, i,
                                            centers, first_squared_norm)
                > _sparse_squared_distance_64(X_data, X_indices, X_indptr, i,
                                              centers + n_features,
                                              second_squared_norm)
            )
            n_second += side
            for p in range(X_indptr[i], X_indptr[i + 1]):
                sums[side * n_features + X_indices[p]] += X_data[p]
        if _update_split_centers(sums, n_features, n_fitted - n_second, n_second,
                                 centers):
            break

    mean_squared_norm = _squared_norm_of_double(mean, n_features)
    first_squared_norm = _squared_norm_of_double(centers, n_features)
    second_squared_norm = _squared_norm_of_double(centers + n_features, n_features)
    for r in range(1, n_rows, 2):
        i = rows[r]
        whole += _sparse_squared_distance_64(X_data, X_indices, X_indptr, i, mean,
                                             mean_squared_norm)
        split += min(
            _sparse_squared_distance_64(X_data, X_indices, X_indptr, i, centers,
                                        first_squared_norm),
            _sparse_squared_distance_64(X_data, X_indices, X_indptr, i,
                                        centers + n_features, second_squared_norm),
        )
    gain[0] = (whole - split) * n_rows / (n_rows // 2)


# Two-level assignment
# --------------------
#
# The centers are grouped in cells: a cell is a coarse center and the centers
# that are nearer to it than to the other coarse centers. A row is compared to
# the coarse centers, then only to the centers of the cells it probes:
#
# - its nearest cell;
# - the cells whose squared distance to the row is at most `ratio`² times the
#   one of the nearest cell, limited to the `max_probes` nearest cells (ties go
#   to the lowest cell index).
#
# The row gets the nearest of these centers, the lowest index on ties. A cell
# without center is never probed.
#
# The cells follow the centers: before each assignment (`_refresh_cells`),
# each coarse center moves to the mean of the centers of its cell, weighted by
# their EMA weights, then each center goes to the cell of its nearest coarse
# center.
#
# The (row, cell) pairs are sorted by cell, so that the centers of a cell are
# compared to the rows that probe it with one GEMM per chunk of pairs
# (`_search`). The chunks don't depend on the number of threads, and neither
# does the result.

# Number of rows compared to the coarse centers by one GEMM, and number of
# (row, cell) pairs of a cell compared to its centers by one GEMM. These GEMMs
# have about sqrt(n_clusters) columns: with the 256 rows of CHUNK_SIZE, they
# are small enough for OpenBLAS to be 2 to 3 times slower per product (64
# threads, AVX-512).
cdef intp_t COARSE_CHUNK_SIZE = 1024
cdef intp_t PAIR_CHUNK_SIZE = 1024

# The centers of a cell are compared by blocks of this number of centers, which
# bounds the size of the buffer of distances of each thread.
cdef intp_t CELL_BLOCK_SIZE = 1024

# Number of rows assigned together by `_labels_inertia_two_level`, a multiple
# of CHUNK_SIZE. It bounds the size of the buffers of the pairs.
cdef intp_t LABELS_BLOCK_SIZE = 1024 * CHUNK_SIZE

# Lower bound of the EMA weight of a center in the mean of its cell.
cdef double MIN_CELL_WEIGHT = 1e-12

# Atomic counters of a two-level step, as the ones of `_run_steps`: for each
# phase, the work claimed so far, then the work done so far. N_TASKS is the
# number of GEMM tasks of the step, set by the TASKS phase.
cdef enum:
    TL_GATHER = 0
    TL_CELL_LISTS = 2
    TL_CELL_MEANS = 4
    TL_CELL_ASSIGN = 6
    TL_CELL_SORT = 8
    TL_CELL_COPY = 10
    TL_COARSE = 12
    TL_COUNT = 14
    TL_TASKS = 16
    TL_FILL = 18
    TL_FINE = 20
    TL_BEST = 22
    TL_SORT_COUNT = 24
    TL_SORT_STARTS = 26
    TL_SORT_FILL = 28
    TL_SORT_CELLS = 30
    TL_UPDATE = 32
    TL_REVIVE = 34
    TL_N_TASKS = 36
    TL_N_COUNTERS = 38


# The sizes and the integer arrays shared by the threads of a two-level
# assignment. The floating arrays are passed next to it: a struct can't have
# fields of a fused type.
cdef struct TwoLevel:
    intp_t n_clusters
    intp_t n_cells
    intp_t n_features
    intp_t max_probes
    # Squared `ratio`.
    double ratio2
    # Number of blocks of rows of the counting sort of the pairs.
    intp_t n_blocks
    # Number of centers claimed together.
    intp_t block_size
    # (n_clusters,) cell of each center
    int *cell_of_center
    # (n_clusters,) centers sorted by cell, then by index
    intp_t *order
    # (n_cells + 1,) the centers of the cell c are
    # order[cell_starts[c]:cell_starts[c + 1]]
    intp_t *cell_starts
    # (max_rows, max_probes) cells probed by each row, ended by -1
    int *probes
    # (n_blocks, n_cells) pairs of each cell in each block of rows, then where
    # the block writes its next pair of each cell
    intp_t *block_counts
    # (n_cells + 1,) the pairs of the cell c are
    # pair_slots[pair_starts[c]:pair_starts[c + 1]]
    intp_t *pair_starts
    # (max_rows * max_probes,) pairs sorted by cell, as indices in `probes`
    intp_t *pair_slots
    # (max_rows, max_probes) nearest center of each pair
    int *pair_labels
    # (max_tasks,) cell and first pair of each GEMM task
    intp_t *task_cells
    intp_t *task_starts
    # (n_clusters,) position of each center in `order`
    intp_t *position


cdef bint _two_level_alloc(TwoLevel *tl, intp_t max_rows) noexcept nogil:
    """Allocate the work arrays of `tl`. Returns whether it succeeded."""
    cdef:
        intp_t max_pairs = max_rows * tl.max_probes
        intp_t max_tasks = max_pairs // PAIR_CHUNK_SIZE + tl.n_cells + 1

    tl.order = <intp_t*> malloc(tl.n_clusters * sizeof(intp_t))
    tl.cell_starts = <intp_t*> malloc((tl.n_cells + 1) * sizeof(intp_t))
    tl.probes = <int*> malloc(max_pairs * sizeof(int))
    tl.block_counts = <intp_t*> malloc(tl.n_blocks * tl.n_cells * sizeof(intp_t))
    tl.pair_starts = <intp_t*> malloc((tl.n_cells + 1) * sizeof(intp_t))
    tl.pair_slots = <intp_t*> malloc(max_pairs * sizeof(intp_t))
    tl.pair_labels = <int*> malloc(max_pairs * sizeof(int))
    tl.task_cells = <intp_t*> malloc(max_tasks * sizeof(intp_t))
    tl.task_starts = <intp_t*> malloc(max_tasks * sizeof(intp_t))
    tl.position = <intp_t*> malloc(tl.n_clusters * sizeof(intp_t))
    return (
        tl.order != NULL
        and tl.cell_starts != NULL
        and tl.probes != NULL
        and tl.block_counts != NULL
        and tl.pair_starts != NULL
        and tl.pair_slots != NULL
        and tl.pair_labels != NULL
        and tl.task_cells != NULL
        and tl.task_starts != NULL
        and tl.position != NULL
    )


cdef void _two_level_free(TwoLevel *tl) noexcept nogil:
    free(tl.order)
    free(tl.cell_starts)
    free(tl.probes)
    free(tl.block_counts)
    free(tl.pair_starts)
    free(tl.pair_slots)
    free(tl.pair_labels)
    free(tl.task_cells)
    free(tl.task_starts)
    free(tl.position)


cdef void _two_level_init(
    TwoLevel *tl,
    intp_t n_clusters,
    intp_t n_cells,
    intp_t n_features,
    intp_t max_probes,
    double ratio,
    int *cell_of_center,
    int n_threads,
) noexcept nogil:
    memset(tl, 0, sizeof(TwoLevel))
    tl.n_clusters = n_clusters
    tl.n_cells = n_cells
    tl.n_features = n_features
    tl.max_probes = max(1, min(max_probes, n_cells))
    tl.ratio2 = ratio * ratio
    tl.n_blocks = n_threads
    tl.block_size = max(1, n_clusters // (16 * n_threads))
    tl.cell_of_center = cell_of_center


cdef inline floating _squared_norm_unrolled(
    const floating *x,
    intp_t n_features,
) noexcept nogil:
    """Squared norm with 8 partial sums.

    `_squared_norm` adds the terms one after the other, which the compiler
    can't vectorize: it takes as long as comparing the row to tens of cells.
    """
    cdef:
        floating[8] partial
        floating result = 0.0
        intp_t k, u
        intp_t n_unrolled = n_features - n_features % 8

    for u in range(8):
        partial[u] = 0.0
    for k in range(0, n_unrolled, 8):
        for u in range(8):
            partial[u] += x[k + u] * x[k + u]
    for k in range(n_unrolled, n_features):
        result += x[k] * x[k]
    for u in range(8):
        result += partial[u]
    return result


cdef inline intp_t _two_level_distances_size(TwoLevel *tl) noexcept nogil:
    """Size of the buffer of distances of a thread."""
    return max(
        COARSE_CHUNK_SIZE * tl.n_cells,
        PAIR_CHUNK_SIZE * min(CELL_BLOCK_SIZE, tl.n_clusters),
    )


def _emeans_steps_two_level(
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
    floating[:, ::1] cells,                # INOUT
    int[::1] cell_of_center,               # INOUT
    intp_t max_probes,
    double ratio,
):
    """Consecutive e-means steps on the rows `X[rows]`, two-level assignment.

    Same as `_emeans_steps`, but each row is only compared to the centers of
    the cells it probes. The result doesn't depend on the number of threads.

    Parameters
    ----------
    cells : ndarray of shape (n_cells, n_features), dtype=floating
        Coarse centers, updated inplace.

    cell_of_center : ndarray of shape (n_clusters,), dtype=int32
        Cell of each center, updated inplace. `revive` can change it, e.g. to
        put a center that it moves in the cell of the center it now is next
        to.

    max_probes : int
        Maximum number of cells probed by a row.

    ratio : float
        A cell is probed by a row when its squared distance to the row is at
        most `ratio**2` times the one of the nearest cell. `inf` to always
        probe the `max_probes` nearest cells.
    """
    cdef:
        TwoLevel tl
        intp_t n_rows = rows.shape[0]
        intp_t n_clusters = centers.shape[0]
        intp_t n_features = centers.shape[1]
        intp_t n_cells = cells.shape[0]
        intp_t n_steps = step_sizes.shape[0]
        intp_t max_batch_size = min(batch_size, n_rows)
        intp_t distances_size
        intp_t failed = 0
        intp_t n_ready = 0
        intp_t n_failed = 0
        bint allocated
        list errors = []
        int *labels
        intp_t *cluster_starts
        intp_t *batch_order
        intp_t *counters
        floating *X_batch
        floating *cell_squared_norms
        floating *sorted_centers
        floating *sorted_squared_norms
        floating *pair_distances
        floating *X_chunk = NULL
        floating *distances = NULL
        int *chunk_labels = NULL
        double *sums = NULL

    if n_rows == 0:
        return

    _two_level_init(&tl, n_clusters, n_cells, n_features, max_probes, ratio,
                    &cell_of_center[0], n_threads)
    distances_size = _two_level_distances_size(&tl)

    labels = <int*> malloc(max_batch_size * sizeof(int))
    cluster_starts = <intp_t*> malloc((n_clusters + 1) * sizeof(intp_t))
    batch_order = <intp_t*> malloc(max_batch_size * sizeof(intp_t))
    counters = <intp_t*> calloc(TL_N_COUNTERS * n_steps, sizeof(intp_t))
    X_batch = <floating*> malloc(max_batch_size * n_features * sizeof(floating))
    cell_squared_norms = <floating*> malloc(n_cells * sizeof(floating))
    sorted_centers = <floating*> malloc(n_clusters * n_features * sizeof(floating))
    sorted_squared_norms = <floating*> malloc(n_clusters * sizeof(floating))
    pair_distances = <floating*> malloc(
        max_batch_size * tl.max_probes * sizeof(floating)
    )
    allocated = _two_level_alloc(&tl, max_batch_size)

    if (
        not allocated
        or labels == NULL
        or cluster_starts == NULL
        or batch_order == NULL
        or counters == NULL
        or X_batch == NULL
        or cell_squared_norms == NULL
        or sorted_centers == NULL
        or sorted_squared_norms == NULL
        or pair_distances == NULL
    ):
        n_failed = 1
    else:
        with nogil, parallel(num_threads=n_threads):
            X_chunk = <floating*> malloc(
                PAIR_CHUNK_SIZE * n_features * sizeof(floating)
            )
            distances = <floating*> malloc(distances_size * sizeof(floating))
            chunk_labels = <int*> malloc(
                max(COARSE_CHUNK_SIZE, PAIR_CHUNK_SIZE) * sizeof(int)
            )
            sums = <double*> malloc(n_features * sizeof(double))

            if emeans_buffers_ready(
                &n_ready,
                &n_failed,
                X_chunk == NULL
                or distances == NULL
                or chunk_labels == NULL
                or sums == NULL,
            ):
                _two_level_steps_worker(
                    &tl, &X[0, 0], &rows[0], n_rows, batch_size, &centers[0, 0],
                    &weight_ema[0], &step_sizes[0], n_steps,
                    NULL if epoch_sums is None else &epoch_sums[0, 0],
                    NULL if epoch_counts is None else &epoch_counts[0],
                    first_step, revival_period, <PyObject*> revive,
                    <PyObject*> errors, &failed, counters, labels,
                    cluster_starts, batch_order, X_batch, &cells[0, 0],
                    cell_squared_norms, sorted_centers, sorted_squared_norms,
                    pair_distances, X_chunk, distances, chunk_labels, sums,
                )

            free(X_chunk)
            free(distances)
            free(chunk_labels)
            free(sums)

    free(labels)
    free(cluster_starts)
    free(batch_order)
    free(counters)
    free(X_batch)
    free(cell_squared_norms)
    free(sorted_centers)
    free(sorted_squared_norms)
    free(pair_distances)
    _two_level_free(&tl)

    if errors:
        raise errors[0]
    if n_failed:
        raise MemoryError()


def _labels_inertia_two_level(
    const floating[:, ::1] X,        # IN
    const floating[:, ::1] centers,  # IN
    const double[::1] weight_ema,    # IN
    floating[:, ::1] cells,          # INOUT
    int[::1] cell_of_center,         # INOUT
    intp_t max_probes,
    double ratio,
    int[::1] labels,                 # OUT
    int n_threads,
):
    """Assign each sample to the nearest center of the cells it probes.

    The cells are refreshed first, as before a step of
    `_emeans_steps_two_level`. Fills `labels` and returns the sum of the
    squared distances of the samples to the center of their label. The result
    doesn't depend on the number of threads.
    """
    cdef:
        TwoLevel tl
        intp_t n_samples = X.shape[0]
        intp_t n_clusters = centers.shape[0]
        intp_t n_features = centers.shape[1]
        intp_t n_cells = cells.shape[0]
        intp_t n_chunks = (n_samples + CHUNK_SIZE - 1) // CHUNK_SIZE
        intp_t n_blocks = (n_samples + LABELS_BLOCK_SIZE - 1) // LABELS_BLOCK_SIZE
        intp_t max_rows = min(n_samples, LABELS_BLOCK_SIZE)
        intp_t distances_size, chunk_idx
        intp_t failed = 0
        intp_t n_ready = 0
        intp_t n_failed = 0
        bint allocated
        list errors = []
        double inertia = 0.0
        double *chunk_inertia
        intp_t *counters
        floating *cell_squared_norms
        floating *sorted_centers
        floating *sorted_squared_norms
        floating *pair_distances
        floating *X_chunk = NULL
        floating *distances = NULL
        int *chunk_labels = NULL
        double *sums = NULL

    if n_samples == 0:
        return 0.0

    n_threads = min(n_threads, n_chunks)
    _two_level_init(&tl, n_clusters, n_cells, n_features, max_probes, ratio,
                    &cell_of_center[0], n_threads)
    distances_size = _two_level_distances_size(&tl)

    chunk_inertia = <double*> malloc(n_chunks * sizeof(double))
    counters = <intp_t*> calloc(TL_N_COUNTERS * (n_blocks + 1), sizeof(intp_t))
    cell_squared_norms = <floating*> malloc(n_cells * sizeof(floating))
    sorted_centers = <floating*> malloc(n_clusters * n_features * sizeof(floating))
    sorted_squared_norms = <floating*> malloc(n_clusters * sizeof(floating))
    pair_distances = <floating*> malloc(max_rows * tl.max_probes * sizeof(floating))
    allocated = _two_level_alloc(&tl, max_rows)

    if (
        not allocated
        or chunk_inertia == NULL
        or counters == NULL
        or cell_squared_norms == NULL
        or sorted_centers == NULL
        or sorted_squared_norms == NULL
        or pair_distances == NULL
    ):
        n_failed = 1
    else:
        with nogil:
            with parallel(num_threads=n_threads):
                X_chunk = <floating*> malloc(
                    PAIR_CHUNK_SIZE * n_features * sizeof(floating)
                )
                distances = <floating*> malloc(distances_size * sizeof(floating))
                chunk_labels = <int*> malloc(
                    max(COARSE_CHUNK_SIZE, PAIR_CHUNK_SIZE) * sizeof(int)
                )
                sums = <double*> malloc(n_features * sizeof(double))

                if emeans_buffers_ready(
                    &n_ready,
                    &n_failed,
                    X_chunk == NULL
                    or distances == NULL
                    or chunk_labels == NULL
                    or sums == NULL,
                ):
                    _two_level_labels_worker(
                        &tl, &X[0, 0], n_samples, &centers[0, 0],
                        &weight_ema[0], <PyObject*> errors, &failed, counters,
                        &labels[0], chunk_inertia, &cells[0, 0],
                        cell_squared_norms, sorted_centers,
                        sorted_squared_norms, pair_distances, X_chunk,
                        distances, chunk_labels, sums,
                    )

                free(X_chunk)
                free(distances)
                free(chunk_labels)
                free(sums)

            if not n_failed and not failed:
                # Sum the inertias of the chunks in a fixed order.
                for chunk_idx in range(n_chunks):
                    inertia += chunk_inertia[chunk_idx]

    free(chunk_inertia)
    free(counters)
    free(cell_squared_norms)
    free(sorted_centers)
    free(sorted_squared_norms)
    free(pair_distances)
    _two_level_free(&tl)

    if errors:
        raise errors[0]
    if n_failed:
        raise MemoryError()
    return inertia


cdef void _two_level_steps_worker(
    TwoLevel *tl,                     # shared
    const floating *X,
    const intp_t *rows,
    intp_t n_rows,
    intp_t batch_size,
    floating *centers,
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
    intp_t *counters,                 # shared (n_steps, TL_N_COUNTERS)
    int *labels,                      # shared (batch_size,)
    intp_t *cluster_starts,           # shared (n_clusters + 1,)
    intp_t *batch_order,              # shared (batch_size,)
    floating *X_batch,                # shared (batch_size, n_features)
    floating *cells,                  # shared (n_cells, n_features)
    floating *cell_squared_norms,     # shared (n_cells,)
    floating *sorted_centers,         # shared (n_clusters, n_features)
    floating *sorted_squared_norms,   # shared (n_clusters,)
    floating *pair_distances,         # shared (batch_size, max_probes)
    floating *X_chunk,                # thread-local
    floating *distances,              # thread-local
    int *chunk_labels,                # thread-local
    double *sums,                     # thread-local
) noexcept nogil:
    """What each thread of `_emeans_steps_two_level` runs.

    Same phases as `_steps_worker`, with the two-level assignment on a
    contiguous copy of the batch: its rows are read once per probed cell.
    """
    cdef:
        bint is_main_thread = emeans_omp_thread_num() == 0
        intp_t n_clusters = tl.n_clusters
        intp_t n_features = tl.n_features
        intp_t block_size = tl.block_size
        intp_t n_cells = tl.n_cells
        intp_t n_blocks = tl.n_blocks
        intp_t s, n_batch, n_chunks, chunk_idx, start, n_chunk_rows, r, j, m, c
        intp_t first, last, block_idx, count
        const intp_t *batch
        intp_t *counter
        intp_t *counts

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
        counter = counters + s * TL_N_COUNTERS

        # Copy of the batch rows, by chunks of rows.
        while True:
            chunk_idx = emeans_fetch_add(&counter[TL_GATHER], 1)
            if chunk_idx >= n_chunks:
                break
            start = chunk_idx * CHUNK_SIZE
            n_chunk_rows = min(CHUNK_SIZE, n_batch - start)
            for r in range(start, start + n_chunk_rows):
                memcpy(&X_batch[r * n_features], &X[batch[r] * n_features],
                       n_features * sizeof(floating))
            emeans_fetch_add(&counter[TL_GATHER + 1], 1)
        emeans_wait(&counter[TL_GATHER + 1], n_chunks)

        _refresh_cells(
            tl, counter, centers, weight_ema, cells, cell_squared_norms,
            sorted_centers, sorted_squared_norms, distances, sums,
        )
        _search(
            tl, counter, X_batch, n_batch, cells, cell_squared_norms,
            sorted_centers, sorted_squared_norms, pair_distances, labels,
            <const floating*> NULL, NULL, X_chunk, distances, chunk_labels,
        )

        # Sort of the batch rows by center, in the order of `tl.order`, then
        # by row: a counting sort by the cell of their center as for the
        # pairs, with the same work arrays, then one by center in each cell.
        # 1. Number of rows of each cell, by blocks of rows.
        while True:
            block_idx = emeans_fetch_add(&counter[TL_SORT_COUNT], 1)
            if block_idx >= n_blocks:
                break
            counts = tl.block_counts + block_idx * n_cells
            memset(counts, 0, n_cells * sizeof(intp_t))
            for r in range(block_idx * n_batch // n_blocks,
                           (block_idx + 1) * n_batch // n_blocks):
                counts[tl.cell_of_center[labels[r]]] += 1
            emeans_fetch_add(&counter[TL_SORT_COUNT + 1], 1)
        emeans_wait(&counter[TL_SORT_COUNT + 1], n_blocks)

        # 2. First row of each cell and of each block in it, by the first
        # thread to get here.
        if emeans_fetch_add(&counter[TL_SORT_STARTS], 1) == 0:
            start = 0
            for c in range(n_cells):
                tl.pair_starts[c] = start
                for block_idx in range(n_blocks):
                    count = tl.block_counts[block_idx * n_cells + c]
                    tl.block_counts[block_idx * n_cells + c] = start
                    start += count
            tl.pair_starts[n_cells] = start
            cluster_starts[0] = 0
            emeans_fetch_add(&counter[TL_SORT_STARTS + 1], 1)
        emeans_wait(&counter[TL_SORT_STARTS + 1], 1)

        # 3. Rows of each cell, by blocks of rows.
        while True:
            block_idx = emeans_fetch_add(&counter[TL_SORT_FILL], 1)
            if block_idx >= n_blocks:
                break
            counts = tl.block_counts + block_idx * n_cells
            for r in range(block_idx * n_batch // n_blocks,
                           (block_idx + 1) * n_batch // n_blocks):
                c = tl.cell_of_center[labels[r]]
                tl.pair_slots[counts[c]] = r
                counts[c] += 1
            emeans_fetch_add(&counter[TL_SORT_FILL + 1], 1)
        emeans_wait(&counter[TL_SORT_FILL + 1], n_blocks)

        # 4. Rows of each center, cell by cell. The rows of the center
        # tl.order[m] are batch_order[cluster_starts[m]:cluster_starts[m + 1]].
        while True:
            c = emeans_fetch_add(&counter[TL_SORT_CELLS], 1)
            if c >= n_cells:
                break
            first = tl.cell_starts[c]
            last = tl.cell_starts[c + 1]
            # cluster_starts[m + 1] is the number of rows of the center, then
            # where its next row goes: it ends at the first row of the next
            # center.
            memset(cluster_starts + first + 1, 0, (last - first) * sizeof(intp_t))
            for r in range(tl.pair_starts[c], tl.pair_starts[c + 1]):
                cluster_starts[tl.position[labels[tl.pair_slots[r]]] + 1] += 1
            start = tl.pair_starts[c]
            for m in range(first, last):
                count = cluster_starts[m + 1]
                cluster_starts[m + 1] = start
                start += count
            for r in range(tl.pair_starts[c], tl.pair_starts[c + 1]):
                m = tl.position[labels[tl.pair_slots[r]]]
                batch_order[cluster_starts[m + 1]] = tl.pair_slots[r]
                cluster_starts[m + 1] += 1
            emeans_fetch_add(&counter[TL_SORT_CELLS + 1], 1)
        emeans_wait(&counter[TL_SORT_CELLS + 1], n_cells)

        # EMA update, by blocks of centers in the order of `tl.order`.
        while True:
            first = emeans_fetch_add(&counter[TL_UPDATE], block_size)
            if first >= n_clusters:
                break
            last = min(first + block_size, n_clusters)
            for m in range(first, last):
                j = tl.order[m]
                _update_center(
                    X_batch, batch_order + cluster_starts[m],
                    cluster_starts[m + 1] - cluster_starts[m], n_features,
                    centers + j * n_features, &weight_ema[j], step_sizes[s],
                    NULL if epoch_sums == NULL
                    else epoch_sums + j * n_features,
                    NULL if epoch_counts == NULL else epoch_counts + j,
                    sums,
                )
            emeans_fetch_add(&counter[TL_UPDATE + 1], last - first)
        emeans_wait(&counter[TL_UPDATE + 1], n_clusters)

        if revival_period > 0 and (first_step + s + 1) % revival_period == 0:
            # Revival of the dead clusters, by the first thread to get here.
            if emeans_fetch_add(&counter[TL_REVIVE], 1) == 0:
                with gil:
                    try:
                        (<object> revive)()
                    except BaseException as exc:
                        (<list> errors).append(exc)
                        emeans_fetch_add(failed, 1)
                emeans_fetch_add(&counter[TL_REVIVE + 1], 1)
            emeans_wait(&counter[TL_REVIVE + 1], 1)


cdef void _two_level_labels_worker(
    TwoLevel *tl,                     # shared
    const floating *X,
    intp_t n_samples,
    const floating *centers,
    const double *weight_ema,
    PyObject *errors,
    intp_t *failed,                   # shared: != 0 to stop at the next block
    intp_t *counters,                 # shared (n_blocks + 1, TL_N_COUNTERS)
    int *labels,                      # shared (n_samples,)
    double *chunk_inertia,            # shared (n_chunks,)
    floating *cells,                  # shared (n_cells, n_features)
    floating *cell_squared_norms,     # shared (n_cells,)
    floating *sorted_centers,         # shared (n_clusters, n_features)
    floating *sorted_squared_norms,   # shared (n_clusters,)
    floating *pair_distances,         # shared (max_rows, max_probes)
    floating *X_chunk,                # thread-local
    floating *distances,              # thread-local
    int *chunk_labels,                # thread-local
    double *sums,                     # thread-local
) noexcept nogil:
    """What each thread of `_labels_inertia_two_level` runs."""
    cdef:
        bint is_main_thread = emeans_omp_thread_num() == 0
        intp_t n_blocks = (n_samples + LABELS_BLOCK_SIZE - 1) // LABELS_BLOCK_SIZE
        intp_t block_idx, start

    _refresh_cells(
        tl, counters, centers, weight_ema, cells, cell_squared_norms,
        sorted_centers, sorted_squared_norms, distances, sums,
    )

    for block_idx in range(n_blocks):
        if emeans_load(failed):
            return
        if is_main_thread:
            with gil:
                try:
                    PyErr_CheckSignals()
                except BaseException as exc:
                    (<list> errors).append(exc)
                    emeans_fetch_add(failed, 1)
                    return

        start = block_idx * LABELS_BLOCK_SIZE
        _search(
            tl, counters + (block_idx + 1) * TL_N_COUNTERS,
            X + start * tl.n_features, min(LABELS_BLOCK_SIZE, n_samples - start),
            cells, cell_squared_norms, sorted_centers, sorted_squared_norms,
            pair_distances, labels + start, centers,
            chunk_inertia + start // CHUNK_SIZE, X_chunk, distances,
            chunk_labels,
        )


cdef void _refresh_cells(
    TwoLevel *tl,                     # shared
    intp_t *counter,                  # shared (TL_N_COUNTERS,)
    const floating *centers,          # IN (n_clusters, n_features)
    const double *weight_ema,         # IN (n_clusters,)
    floating *cells,                  # shared (n_cells, n_features)
    floating *cell_squared_norms,     # shared (n_cells,)
    floating *sorted_centers,         # shared (n_clusters, n_features)
    floating *sorted_squared_norms,   # shared (n_clusters,)
    floating *distances,              # thread-local
    double *sums,                     # thread-local
) noexcept nogil:
    """Make the cells follow the centers. Run by each thread of the team.

    1. Each coarse center moves to the mean of the centers of its cell,
       weighted by their EMA weights. The one of an empty cell stays.
    2. Each center goes to the cell of its nearest coarse center.

    Then the centers are copied sorted by cell in `sorted_centers`, so that
    the ones of a cell are contiguous for the GEMM. The squared norm of an
    empty cell is set to infinity: no row has it as a nearest cell.
    """
    cdef:
        intp_t n_clusters = tl.n_clusters
        intp_t n_cells = tl.n_cells
        intp_t n_features = tl.n_features
        intp_t block_size = tl.block_size
        intp_t n_chunks = (
            (n_clusters + COARSE_CHUNK_SIZE - 1) // COARSE_CHUNK_SIZE
        )
        intp_t c, m, j, k, chunk_idx, start, first, last
        double weight, total
        const floating *center

    # Centers of each cell, by the first thread to get here. The cells of the
    # centers can have been changed by the revival since the last sort.
    if emeans_fetch_add(&counter[TL_CELL_LISTS], 1) == 0:
        _sort_rows_by_label(tl.cell_of_center, n_clusters, n_cells,
                            tl.cell_starts, tl.order)
        emeans_fetch_add(&counter[TL_CELL_LISTS + 1], 1)
    emeans_wait(&counter[TL_CELL_LISTS + 1], 1)

    # Coarse centers, cell by cell.
    while True:
        c = emeans_fetch_add(&counter[TL_CELL_MEANS], 1)
        if c >= n_cells:
            break
        if tl.cell_starts[c + 1] > tl.cell_starts[c]:
            memset(sums, 0, n_features * sizeof(double))
            total = 0.0
            for m in range(tl.cell_starts[c], tl.cell_starts[c + 1]):
                j = tl.order[m]
                weight = max(weight_ema[j], MIN_CELL_WEIGHT)
                total += weight
                center = centers + j * n_features
                for k in range(n_features):
                    sums[k] += weight * center[k]
            for k in range(n_features):
                cells[c * n_features + k] = <floating> (sums[k] / total)
        cell_squared_norms[c] = _squared_norm(cells + c * n_features, n_features)
        emeans_fetch_add(&counter[TL_CELL_MEANS + 1], 1)
    emeans_wait(&counter[TL_CELL_MEANS + 1], n_cells)

    # Nearest coarse center of each center, by chunks of centers.
    while True:
        chunk_idx = emeans_fetch_add(&counter[TL_CELL_ASSIGN], 1)
        if chunk_idx >= n_chunks:
            break
        start = chunk_idx * COARSE_CHUNK_SIZE
        _assign_rows(
            centers + start * n_features,
            min(COARSE_CHUNK_SIZE, n_clusters - start),
            cells, cell_squared_norms, n_cells, n_features,
            tl.cell_of_center + start, distances, NULL,
        )
        emeans_fetch_add(&counter[TL_CELL_ASSIGN + 1], 1)
    emeans_wait(&counter[TL_CELL_ASSIGN + 1], n_chunks)

    # Centers of each cell again, by the first thread to get here.
    if emeans_fetch_add(&counter[TL_CELL_SORT], 1) == 0:
        _sort_rows_by_label(tl.cell_of_center, n_clusters, n_cells,
                            tl.cell_starts, tl.order)
        for c in range(n_cells):
            if tl.cell_starts[c + 1] == tl.cell_starts[c]:
                cell_squared_norms[c] = <floating> INFINITY
        emeans_fetch_add(&counter[TL_CELL_SORT + 1], 1)
    emeans_wait(&counter[TL_CELL_SORT + 1], 1)

    # Copy of the centers sorted by cell, by blocks of centers.
    while True:
        first = emeans_fetch_add(&counter[TL_CELL_COPY], block_size)
        if first >= n_clusters:
            break
        last = min(first + block_size, n_clusters)
        for m in range(first, last):
            tl.position[tl.order[m]] = m
            memcpy(sorted_centers + m * n_features,
                   centers + tl.order[m] * n_features,
                   n_features * sizeof(floating))
            sorted_squared_norms[m] = _squared_norm(
                sorted_centers + m * n_features, n_features
            )
        emeans_fetch_add(&counter[TL_CELL_COPY + 1], last - first)
    emeans_wait(&counter[TL_CELL_COPY + 1], n_clusters)


cdef void _search(
    TwoLevel *tl,                               # shared
    intp_t *counter,                            # shared (TL_N_COUNTERS,)
    const floating *X_rows,                     # IN (n_rows, n_features)
    intp_t n_rows,
    const floating *cells,                      # IN (n_cells, n_features)
    const floating *cell_squared_norms,         # IN (n_cells,)
    const floating *sorted_centers,             # IN (n_clusters, n_features)
    const floating *sorted_squared_norms,       # IN (n_clusters,)
    floating *pair_distances,                   # shared (n_rows, max_probes)
    int *labels,                                # OUT (n_rows,)
    const floating *centers,                    # IN, if chunk_inertia != NULL
    double *chunk_inertia,                      # OUT (n_chunks,), or NULL
    floating *X_chunk,                          # thread-local
    floating *distances,                        # thread-local
    int *chunk_labels,                          # thread-local
) noexcept nogil:
    """Two-level assignment of contiguous rows. Run by each thread of the team.

    `chunk_inertia` gets the sum of the squared distances of the rows of each
    chunk of CHUNK_SIZE rows to the center of their label.
    """
    cdef:
        intp_t n_cells = tl.n_cells
        intp_t n_features = tl.n_features
        intp_t max_probes = tl.max_probes
        intp_t n_blocks = tl.n_blocks
        intp_t n_chunks = (n_rows + CHUNK_SIZE - 1) // CHUNK_SIZE
        intp_t n_coarse_chunks = (
            (n_rows + COARSE_CHUNK_SIZE - 1) // COARSE_CHUNK_SIZE
        )
        intp_t chunk_idx, start, n_chunk_rows, r, i, c, q
        intp_t n_probes, block_idx, n_pairs, n_tasks, task_idx, count
        intp_t first, n_members, block_start, n_block, slot
        int nearest, label
        int *probes
        intp_t *counts
        const floating *row
        floating squared_norm, nearest_distance, threshold, dist, min_dist
        floating max_finite
        double threshold_64, total

    if floating is float:
        max_finite = FLT_MAX
    else:
        max_finite = DBL_MAX

    # Cells probed by each row, by chunks of rows.
    while True:
        chunk_idx = emeans_fetch_add(&counter[TL_COARSE], 1)
        if chunk_idx >= n_coarse_chunks:
            break
        start = chunk_idx * COARSE_CHUNK_SIZE
        n_chunk_rows = min(COARSE_CHUNK_SIZE, n_rows - start)
        # distances = ||g||² - 2 x.g, and the nearest cell
        _assign_rows(
            X_rows + start * n_features, n_chunk_rows, cells,
            cell_squared_norms, n_cells, n_features, chunk_labels, distances,
            NULL,
        )
        for r in range(n_chunk_rows):
            i = start + r
            row = distances + r * n_cells
            nearest = chunk_labels[r]
            probes = tl.probes + i * max_probes

            # ||x - g||² <= ratio² ||x - g_nearest||², without the ||x||² of
            # the left-hand side. The threshold is finite: the distance of an
            # empty cell is infinite.
            squared_norm = _squared_norm_unrolled(
                X_rows + i * n_features, n_features
            )
            nearest_distance = squared_norm + row[nearest]
            if nearest_distance < 0:
                nearest_distance = 0
            threshold_64 = tl.ratio2 * nearest_distance - squared_norm
            if threshold_64 < max_finite:
                threshold = <floating> threshold_64
            else:
                threshold = max_finite

            # The probed cells, kept sorted by distance then by index while
            # going through the cells: the last one leaves when a nearer one
            # comes.
            n_probes = 0
            for c in range(n_cells):
                dist = row[c]
                if dist <= threshold or c == nearest:
                    if n_probes < max_probes:
                        q = n_probes
                        n_probes += 1
                    elif dist < row[probes[max_probes - 1]]:
                        q = max_probes - 1
                    else:
                        continue
                    while q > 0 and row[probes[q - 1]] > dist:
                        probes[q] = probes[q - 1]
                        q -= 1
                    probes[q] = <int> c
            if n_probes < max_probes:
                probes[n_probes] = -1
        emeans_fetch_add(&counter[TL_COARSE + 1], 1)
    emeans_wait(&counter[TL_COARSE + 1], n_coarse_chunks)

    # Counting sort of the (row, cell) pairs by cell, then by row. Its three
    # phases split the rows in the same blocks.
    # 1. Number of pairs of each cell, by blocks of rows.
    while True:
        block_idx = emeans_fetch_add(&counter[TL_COUNT], 1)
        if block_idx >= n_blocks:
            break
        counts = tl.block_counts + block_idx * n_cells
        memset(counts, 0, n_cells * sizeof(intp_t))
        for i in range(block_idx * n_rows // n_blocks,
                       (block_idx + 1) * n_rows // n_blocks):
            probes = tl.probes + i * max_probes
            for q in range(max_probes):
                if probes[q] < 0:
                    break
                counts[probes[q]] += 1
        emeans_fetch_add(&counter[TL_COUNT + 1], 1)
    emeans_wait(&counter[TL_COUNT + 1], n_blocks)

    # 2. First pair of each cell and of each block in it, and the GEMM tasks:
    # chunks of the pairs of a cell. By the first thread to get here.
    if emeans_fetch_add(&counter[TL_TASKS], 1) == 0:
        n_pairs = 0
        n_tasks = 0
        for c in range(n_cells):
            tl.pair_starts[c] = n_pairs
            for block_idx in range(n_blocks):
                count = tl.block_counts[block_idx * n_cells + c]
                tl.block_counts[block_idx * n_cells + c] = n_pairs
                n_pairs += count
            start = tl.pair_starts[c]
            while start < n_pairs:
                tl.task_cells[n_tasks] = c
                tl.task_starts[n_tasks] = start
                n_tasks += 1
                start += PAIR_CHUNK_SIZE
        tl.pair_starts[n_cells] = n_pairs
        # The number of tasks is in the counters of the step: a thread that
        # is late must not read the one of a later step.
        emeans_fetch_add(&counter[TL_N_TASKS], n_tasks)
        emeans_fetch_add(&counter[TL_TASKS + 1], 1)
    emeans_wait(&counter[TL_TASKS + 1], 1)
    n_tasks = emeans_load(&counter[TL_N_TASKS])

    # 3. Pairs of each cell, by blocks of rows.
    while True:
        block_idx = emeans_fetch_add(&counter[TL_FILL], 1)
        if block_idx >= n_blocks:
            break
        counts = tl.block_counts + block_idx * n_cells
        for i in range(block_idx * n_rows // n_blocks,
                       (block_idx + 1) * n_rows // n_blocks):
            probes = tl.probes + i * max_probes
            for q in range(max_probes):
                if probes[q] < 0:
                    break
                tl.pair_slots[counts[probes[q]]] = i * max_probes + q
                counts[probes[q]] += 1
        emeans_fetch_add(&counter[TL_FILL + 1], 1)
    emeans_wait(&counter[TL_FILL + 1], n_blocks)

    # Nearest center of each pair in its cell, by chunks of pairs of a cell.
    while True:
        task_idx = emeans_fetch_add(&counter[TL_FINE], 1)
        if task_idx >= n_tasks:
            break
        c = tl.task_cells[task_idx]
        start = tl.task_starts[task_idx]
        n_chunk_rows = min(PAIR_CHUNK_SIZE, tl.pair_starts[c + 1] - start)
        for r in range(n_chunk_rows):
            memcpy(X_chunk + r * n_features,
                   X_rows + (tl.pair_slots[start + r] // max_probes) * n_features,
                   n_features * sizeof(floating))
        first = tl.cell_starts[c]
        n_members = tl.cell_starts[c + 1] - first
        block_start = 0
        while block_start < n_members:
            n_block = min(CELL_BLOCK_SIZE, n_members - block_start)
            _assign_rows(
                X_chunk, n_chunk_rows,
                sorted_centers + (first + block_start) * n_features,
                sorted_squared_norms + first + block_start, n_block,
                n_features, chunk_labels, distances, NULL,
            )
            for r in range(n_chunk_rows):
                slot = tl.pair_slots[start + r]
                dist = distances[r * n_block + chunk_labels[r]]
                # The centers of a cell are sorted by index: on ties, the
                # first block has the lowest index.
                if block_start == 0 or dist < pair_distances[slot]:
                    pair_distances[slot] = dist
                    tl.pair_labels[slot] = <int> tl.order[
                        first + block_start + chunk_labels[r]
                    ]
            block_start += CELL_BLOCK_SIZE
        emeans_fetch_add(&counter[TL_FINE + 1], 1)
    emeans_wait(&counter[TL_FINE + 1], n_tasks)

    # Nearest center of each row among its pairs, by chunks of rows.
    while True:
        chunk_idx = emeans_fetch_add(&counter[TL_BEST], 1)
        if chunk_idx >= n_chunks:
            break
        start = chunk_idx * CHUNK_SIZE
        total = 0.0
        for i in range(start, min(start + CHUNK_SIZE, n_rows)):
            slot = i * max_probes
            min_dist = pair_distances[slot]
            label = tl.pair_labels[slot]
            for q in range(1, max_probes):
                if tl.probes[slot + q] < 0:
                    break
                dist = pair_distances[slot + q]
                if dist < min_dist or (
                    dist == min_dist and tl.pair_labels[slot + q] < label
                ):
                    min_dist = dist
                    label = tl.pair_labels[slot + q]
            labels[i] = label
            if chunk_inertia != NULL:
                total += _squared_distance(X_rows + i * n_features,
                                           centers + label * n_features,
                                           n_features)
        if chunk_inertia != NULL:
            chunk_inertia[chunk_idx] = total
        emeans_fetch_add(&counter[TL_BEST + 1], 1)
    emeans_wait(&counter[TL_BEST + 1], n_chunks)
