"""E-means on array API inputs (PyTorch, CuPy...), e.g. to fit on a GPU.

The Cython kernels of `_kernels.pyx` only run on NumPy arrays and SciPy sparse
matrices. The functions here do the same computations with array operations
of the namespace of the input, on its device. The random numbers are still
drawn on the host, in the same order as for NumPy inputs, so both code paths
give the same result up to rounding.

Most of the code only uses the array API standard. The few operations that
the standard lacks, or that are much faster with a library-specific function,
are in the helpers at the top of this module. On Apple GPUs, the steps on
low-dimensional data run with the Metal kernels of `_mps_kernels.py`.
"""

# SPDX-License-Identifier: BSD-3-Clause

from functools import cache

import numpy as np
import scipy.sparse as sp
from sklearn import get_config
from sklearn.externals import array_api_compat

# Upper bound on the number of elements of the (n_rows, n_clusters) scores of
# a chunk of rows: 64 MB in float32. Larger chunks are not faster on the GPUs
# we tried, and up to 1.7x slower with PyTorch on Apple GPUs.
_MAX_SCORES_SIZE = 2**24


def get_namespace(X):
    """Array namespace of X, or None if X goes through the NumPy code path.

    As in scikit-learn, array API inputs are only dispatched when
    `sklearn.set_config(array_api_dispatch=True)` is set. Otherwise they are
    converted to NumPy arrays, when that is possible.
    """
    if not get_config().get("array_api_dispatch", False):
        return None
    if isinstance(X, (np.ndarray, np.generic, list, tuple)) or sp.issparse(X):
        return None
    if not array_api_compat.is_array_api_obj(X) or array_api_compat.is_numpy_array(X):
        return None
    return array_api_compat.array_namespace(X)


def device(X):
    return array_api_compat.device(X)


def _is_torch(xp):
    return array_api_compat.is_torch_namespace(xp)


def _is_cupy(xp):
    return array_api_compat.is_cupy_namespace(xp)


def _use_mps_kernels(xp, X):
    """Whether to run the steps with the Metal kernels of `_mps_kernels.py`."""
    if not _is_torch(xp) or X.device.type != "mps":
        return False
    from sklearn_emeans import _mps_kernels

    return _mps_kernels.supports(X)


def to_numpy(xp, array):
    """Copy an array to the host."""
    if _is_torch(xp):
        return array.detach().cpu().numpy()
    if _is_cupy(xp):
        return array.get()
    try:
        return np.from_dlpack(array)
    except (AttributeError, TypeError, BufferError):
        return np.asarray(array)


def from_numpy(xp, array, dtype, device):
    """Copy a NumPy array to a device, with the dtype of the given name."""
    # Cast on the host: the device may not support the dtype of `array`. Always
    # copy: PyTorch rejects negative strides and warns on read-only arrays.
    array = np.array(array, dtype=dtype, order="C", copy=True)
    return xp.asarray(array, device=device)


def index_array(xp, indices, device):
    """Copy a NumPy array of indices to a device."""
    return from_numpy(xp, indices, "int64", device)


def accumulator_dtype(xp, device):
    """Name of the widest float dtype of the device.

    Some devices have no float64 (Apple GPUs, some Intel GPUs).
    """
    try:
        probe = xp.zeros(2, dtype=xp.float64, device=device)
        float(xp.sum(probe + 1.0))
    except Exception:
        return "float32"
    return "float64"


def dtype_name(xp, array):
    for name in ("float32", "float64"):
        if array.dtype == getattr(xp, name):
            return name
    return None


def _scores(xp, X, centers, centers_half_squared_norms):
    """`x·c - ||c||² / 2` for each row x and center c.

    The nearest center of a row has the largest score, and the squared
    distance is `||x||² - 2 * score`.
    """
    if _is_torch(xp):
        import torch

        # A single kernel on GPUs, without temporary array. `addmm` does the
        # same but is 3x slower on Apple GPUs (PyTorch 2.14).
        return torch.nn.functional.linear(X, centers, -centers_half_squared_norms)
    scores = X @ xp.matrix_transpose(centers)
    if _is_cupy(xp):
        # Inplace, to save an allocation.
        scores -= centers_half_squared_norms[None, :]
        return scores
    return scores - centers_half_squared_norms[None, :]


def _max_argmax(xp, scores):
    """Largest score of each row and its index."""
    if _is_torch(xp):
        import torch

        # A single pass over the scores.
        return tuple(torch.max(scores, dim=1))
    return xp.max(scores, axis=1), xp.argmax(scores, axis=1)


def _half_squared_norms(xp, centers):
    return 0.5 * xp.sum(centers * centers, axis=1)


def _cluster_sums(xp, X, labels, n_clusters):
    """Sum and number of the rows of each cluster, as arrays of dtype X.dtype.

    The standard has no scatter-add: use the one of the library if we know it,
    and a product with the one-hot encoding of the labels otherwise.
    """
    if _is_torch(xp):
        import torch

        sums = torch.zeros(
            (n_clusters, X.shape[1]), dtype=X.dtype, device=X.device
        ).index_add_(0, labels, X)
        counts = torch.zeros(n_clusters, dtype=X.dtype, device=X.device).index_add_(
            0, labels, torch.ones(1, dtype=X.dtype, device=X.device).expand(len(labels))
        )
        return sums, counts
    if _is_cupy(xp):
        import cupy

        sums = cupy.zeros((n_clusters, X.shape[1]), dtype=X.dtype)
        cupy.add.at(sums, labels, X)
        # Faster than `bincount` for the small batches of low-dimensional data.
        counts = cupy.zeros(n_clusters, dtype=X.dtype)
        cupy.add.at(counts, labels, X.dtype.type(1))
        return sums, counts
    one_hot = xp.astype(
        labels[:, None] == xp.arange(n_clusters, device=device(X))[None, :], X.dtype
    )
    return xp.matrix_transpose(one_hot) @ X, xp.sum(one_hot, axis=0)


def _ema_update(xp, centers, weight_ema, sums, counts, step_size):
    """EMA update of the centers from the sums and numbers of their batch rows.

        c_j <- ((1 - alpha) N_j c_j + alpha s_j) / N'_j

    with N_j the EMA weight and N'_j the updated one, see `_ema_update` in
    `_kernels.pyx`. The centers without rows keep their position.
    """
    if _is_cupy(xp) and weight_ema.dtype == xp.float64:
        # One kernel in place of 12: the launch of the kernels dominates the
        # steps on small batches.
        return _cupy_ema_kernel()(
            centers, sums, counts[:, None], weight_ema[:, None], step_size
        )
    assigned = counts > 0
    retained = (1.0 - step_size) * weight_ema
    weight_ema = retained + step_size * xp.astype(counts, weight_ema.dtype)
    safe_weight = xp.where(assigned, weight_ema, xp.ones_like(weight_ema))
    centers_coef = xp.astype(retained / safe_weight, centers.dtype)
    sums_coef = xp.astype(step_size / safe_weight, centers.dtype)
    centers = xp.where(
        assigned[:, None],
        centers_coef[:, None] * centers + sums_coef[:, None] * sums,
        centers,
    )
    return centers, weight_ema


@cache
def _cupy_ema_kernel():
    import cupy

    kernel = cupy.ElementwiseKernel(
        "T center, T sum, T count, float64 weight, float64 step_size",
        "T new_center, float64 new_weight",
        """
        double retained = (1.0 - step_size) * weight;
        new_weight = retained + step_size * count;
        if (count > 0) {
            new_center = (T) ((retained / new_weight) * center
                              + (step_size / new_weight) * sum);
        } else {
            new_center = center;
        }
        """,
        "sklearn_emeans_ema_update",
    )

    def ema_update(centers, sums, counts, weight_ema, step_size):
        centers, weight_ema = kernel(centers, sums, counts, weight_ema, step_size)
        return centers, weight_ema[:, 0]

    return ema_update


def _set_rows(xp, array, indices, rows):
    """`array[indices] = rows` for distinct indices, not always inplace."""
    if _is_torch(xp) or _is_cupy(xp):
        array[indices] = rows
        return array
    # The standard has no assignment with integer arrays.
    one_hot = (
        xp.arange(array.shape[0], device=device(array))[:, None] == indices[None, :]
    )
    selected = xp.any(one_hot, axis=1)
    if array.ndim == 1:
        return xp.where(selected, xp.astype(one_hot, array.dtype) @ rows, array)
    return xp.where(selected[:, None], xp.astype(one_hot, array.dtype) @ rows, array)


def _chunks(n_rows, n_centers):
    """Slices of rows whose scores have at most `_MAX_SCORES_SIZE` elements."""
    chunk_size = max(1, _MAX_SCORES_SIZE // n_centers)
    # The standard doesn't allow slices that stop out of bounds.
    return [
        slice(start, min(start + chunk_size, n_rows))
        for start in range(0, n_rows, chunk_size)
    ]


def _assign(xp, X, centers):
    """Label of the nearest center of each row."""
    half_squared_norms = _half_squared_norms(xp, centers)
    labels = [
        xp.argmax(_scores(xp, X[chunk, :], centers, half_squared_norms), axis=1)
        for chunk in _chunks(X.shape[0], centers.shape[0])
    ]
    return labels[0] if len(labels) == 1 else xp.concat(labels)


def emeans_steps(
    xp,
    X,
    rows,
    batch_size,
    centers,
    weight_ema,
    step_sizes,
    epoch_sums,
    epoch_counts,
    first_step,
    revival_period,
    revive,
):
    """Consecutive e-means steps on the batches of `rows`.

    Same computations as `_kernels._emeans_steps`, but the sums of the batch
    rows and the EMA updates are in the dtype of X and not in float64.

    Parameters
    ----------
    xp : array namespace of X

    X : array of shape (n_samples, n_features), dtype float32 or float64

    rows : integer array of shape (n_rows,), on the device of X
        Indices of the rows of X to go through, `batch_size` rows per step.

    centers : array of shape (n_clusters, n_features), dtype of X

    weight_ema : array of shape (n_clusters,), accumulator dtype

    step_sizes : sequence of n_steps floats

    epoch_sums : array of shape (n_clusters, n_features), accumulator dtype, \
            or None
        If not None, the sums of the rows assigned to each center are added
        to it, and their numbers to `epoch_counts`.

    epoch_counts : array of shape (n_clusters,), accumulator dtype, or None

    first_step : int
        Index of the first step, to schedule the revivals.

    revival_period : int
        `revive` is called every `revival_period` steps, never if 0.

    revive : callable
        Takes `(centers, weight_ema)` and returns them with the dead clusters
        revived.

    Returns
    -------
    centers, weight_ema, epoch_sums, epoch_counts : the updated arrays
        The inputs may have been modified inplace or not.
    """
    if _use_mps_kernels(xp, X):
        from sklearn_emeans import _mps_kernels

        return _mps_kernels.emeans_steps(
            X,
            rows,
            batch_size,
            centers,
            weight_ema,
            step_sizes,
            epoch_sums,
            epoch_counts,
            first_step,
            revival_period,
            revive,
        )
    n_clusters = centers.shape[0]
    n_rows = rows.shape[0]
    acc_dtype = weight_ema.dtype
    for s, step_size in enumerate(step_sizes):
        # The standard doesn't allow slices that stop out of bounds.
        batch = rows[s * batch_size : min((s + 1) * batch_size, n_rows)]
        X_batch = xp.take(X, batch, axis=0)
        labels = _assign(xp, X_batch, centers)
        sums, counts = _cluster_sums(xp, X_batch, labels, n_clusters)
        if epoch_sums is not None:
            epoch_sums = epoch_sums + xp.astype(sums, acc_dtype)
            epoch_counts = epoch_counts + xp.astype(counts, acc_dtype)
        centers, weight_ema = _ema_update(
            xp, centers, weight_ema, sums, counts, step_size
        )

        if revival_period > 0 and (first_step + s + 1) % revival_period == 0:
            centers, weight_ema = revive(centers, weight_ema)
    return centers, weight_ema, epoch_sums, epoch_counts


def revive_dead_clusters(xp, centers, weight_ema, random_state):
    """Split heavy clusters to replace the dead ones.

    Same as `_emeans._revive_dead_clusters`, which this function runs on the
    host: only the EMA weights and the centers to split are copied there.
    """
    from sklearn_emeans._emeans import _revival_indices, _split_centers

    indices = _revival_indices(
        to_numpy(xp, weight_ema).astype(np.float64), random_state
    )
    if indices is None:
        return centers, weight_ema
    dead_indices, donor_indices = (
        index_array(xp, indices_, device(centers)) for indices_ in indices
    )
    donor_centers = to_numpy(xp, xp.take(centers, donor_indices, axis=0))
    dtype = dtype_name(xp, centers)
    for indices_, rows in zip(
        (dead_indices, donor_indices), _split_centers(donor_centers)
    ):
        centers = _set_rows(
            xp, centers, indices_, from_numpy(xp, rows, dtype, device(centers))
        )
    donor_weights = xp.take(weight_ema, donor_indices, axis=0) / 2
    weight_ema = _set_rows(xp, weight_ema, donor_indices, donor_weights)
    weight_ema = _set_rows(xp, weight_ema, dead_indices, donor_weights)
    return centers, weight_ema


def labels_inertia(xp, X, centers):
    """Labels (int32) and inertia (float) of X."""
    half_squared_norms = _half_squared_norms(xp, centers)
    acc_dtype = getattr(xp, accumulator_dtype(xp, device(X)))
    labels = []
    inertia = 0.0
    for chunk in _chunks(X.shape[0], centers.shape[0]):
        X_chunk = X[chunk, :]
        chunk_labels = xp.argmax(
            _scores(xp, X_chunk, centers, half_squared_norms), axis=1
        )
        # Not from the scores: in float32, they lose digits by cancellation.
        # The sums of the rows are added in the widest dtype of the device.
        difference = X_chunk - xp.take(centers, chunk_labels, axis=0)
        inertia = inertia + xp.sum(
            xp.astype(xp.sum(difference * difference, axis=1), acc_dtype)
        )
        labels.append(xp.astype(chunk_labels, xp.int32))
    return xp.concat(labels), float(inertia)


def euclidean_distances(xp, X, centers):
    """Distances between the rows of X and the centers."""
    squared_distances = xp.sum(X * X, axis=1)[:, None] - 2.0 * _scores(
        xp, X, centers, _half_squared_norms(xp, centers)
    )
    zero = xp.zeros_like(squared_distances)
    return xp.sqrt(xp.where(squared_distances > 0, squared_distances, zero))


def kmeans_parallel(xp, X, n_clusters, random_state, n_threads):
    """Initial centers given by k-means||, as a NumPy array.

    Same as `_kmeans_parallel._kmeans_parallel`. The distances of the sampling
    rounds are computed on the device of X, and the candidates are sampled on
    the host, from the same random numbers as for NumPy inputs. The final
    weighted k-means++ on the candidates runs on the host.
    """
    from sklearn.utils.extmath import row_norms

    from sklearn_emeans._kmeans_parallel import (
        _KMEANS_PARALLEL_N_ROUNDS,
        _complete_candidates,
        _sample_candidates,
    )
    from sklearn_emeans._kmeans_plusplus import _kmeans_plusplus

    n_samples = X.shape[0]
    dev = device(X)
    dtype = dtype_name(xp, X)
    X_squared_norms = xp.sum(X * X, axis=1)
    min_distances = xp.full(n_samples, xp.inf, dtype=X.dtype, device=dev)
    labels = xp.zeros(n_samples, dtype=xp.int64, device=dev)
    candidates = []
    n_candidates = 0

    def add_candidates(new):
        nonlocal n_candidates, min_distances, labels
        new_xp = index_array(xp, new, dev)
        new_centers = xp.take(X, new_xp, axis=0)
        half_squared_norms = _half_squared_norms(xp, new_centers)
        max_scores, new_labels = [], []
        for chunk in _chunks(n_samples, len(new)):
            chunk_max_scores, chunk_labels = _max_argmax(
                xp, _scores(xp, X[chunk, :], new_centers, half_squared_norms)
            )
            max_scores.append(chunk_max_scores)
            new_labels.append(chunk_labels)
        new_distances = X_squared_norms - 2.0 * xp.concat(max_scores)
        new_distances = xp.where(
            new_distances > 0, new_distances, xp.zeros_like(new_distances)
        )
        closer = new_distances < min_distances
        min_distances = xp.where(closer, new_distances, min_distances)
        labels = xp.where(closer, xp.concat(new_labels) + n_candidates, labels)
        # A candidate is its own nearest candidate, even if it has duplicates.
        min_distances = _set_rows(
            xp, min_distances, new_xp, xp.zeros(len(new), dtype=X.dtype, device=dev)
        )
        labels = _set_rows(
            xp,
            labels,
            new_xp,
            xp.arange(n_candidates, n_candidates + len(new), device=dev),
        )
        candidates.append(new)
        n_candidates += len(new)

    add_candidates(np.array([random_state.randint(n_samples)], dtype=np.intp))
    for _ in range(_KMEANS_PARALLEL_N_ROUNDS):
        new = _sample_candidates(to_numpy(xp, min_distances), n_clusters, random_state)
        if new is None:
            break
        if len(new):
            add_candidates(new)
    if n_candidates < n_clusters:
        add_candidates(
            _complete_candidates(n_samples, candidates, n_clusters, random_state)
        )

    indices = np.concatenate(candidates)
    weights = np.bincount(to_numpy(xp, labels), minlength=n_candidates)
    candidates = to_numpy(xp, xp.take(X, index_array(xp, indices, dev), axis=0))
    centers, _ = _kmeans_plusplus(
        np.ascontiguousarray(candidates),
        n_clusters,
        row_norms(candidates, squared=True),
        weights.astype(dtype),
        random_state,
        n_threads=n_threads,
    )
    return centers
