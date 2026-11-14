"""E-means clustering."""

# SPDX-License-Identifier: BSD-3-Clause

import math
import os
from functools import partial
from numbers import Integral, Real

import numpy as np
import scipy.sparse as sp
from joblib import cpu_count
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    ClusterMixin,
    TransformerMixin,
)
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import row_norms
from sklearn.utils.validation import check_is_fitted, validate_data
from threadpoolctl import ThreadpoolController

from sklearn_emeans import _array_api
from sklearn_emeans._greedy_revival import GreedyRevival, greedy_sample_indices
from sklearn_emeans._kernels import (
    _chunk_size,
    _emeans_steps,
    _emeans_steps_sparse,
    _labels_inertia,
    _labels_inertia_sparse,
    _omp_max_threads,
)
from sklearn_emeans._kmeans_parallel import _kmeans_parallel
from sklearn_emeans._kmeans_plusplus import _kmeans_plusplus

# The constants below are the ones of the reference implementation
# (faiss/contrib/e_means/base.py), which differ slightly from the paper.

# Ratio between the last and the first step size of the cosine schedule.
_MIN_STEP_SIZE_FACTOR = 0.1

# Dead clusters are revived every `_REVIVAL_PERIOD` steps.
_REVIVAL_PERIOD = 5

# Default of `revival`: a cluster is dead when its EMA weight is below this
# fraction of the mean EMA weight.
_DEAD_CLUSTER_THRESHOLD = 0.1

# Relative perturbation of the two halves of a split donor center.
_SPLIT_EPSILON = 1e-4

# Initial EMA weight of each center. Tiny, so that the first batch moves the
# centers to the mean of their rows.
_INIT_WEIGHT = 2.0**-16

# The reference implementation turns the revival of the dead clusters off for
# n_features <= 2, where it was found to hurt the held-out inertia.
_NO_REVIVAL_MAX_FEATURES = 2

# The "quality" batch law (8x smaller batches) is used for n_features <= 16,
# where more steps help more, and the "throughput" law otherwise.
_QUALITY_BATCH_MAX_FEATURES = 16

# Batch law: batch_size = c * n_clusters**a_k * n_samples**a_n * n_epochs**a_t,
# stored as (c, a_k, a_n, a_t).
_BATCH_LAWS = {
    "throughput": (0.035134, 0.134749, 0.876473, 0.813344),
    "quality": (0.00439175, 0.134749, 0.876473, 0.813344),
}

# Step law: step_size = c * n_clusters**a_k * batch_size**a_b * n_samples**a_n
# * n_epochs**a_t, stored as (c, a_k, a_b, a_n, a_t).
_STEP_LAW = (0.8244, -0.4617, 0.9003, -0.4498, -0.1057)

_CHUNK_SIZE = _chunk_size()

_THREADPOOL_CONTROLLER = None
_N_CPUS = None


def _threadpool_controller():
    # Creating a ThreadpoolController inspects the loaded libraries, which is
    # slow: do it once.
    global _THREADPOOL_CONTROLLER
    if _THREADPOOL_CONTROLLER is None:
        _THREADPOOL_CONTROLLER = ThreadpoolController()
    return _THREADPOOL_CONTROLLER


def _openmp_effective_n_threads():
    """Number of OpenMP threads, as in scikit-learn.

    `omp_get_max_threads()` if `OMP_NUM_THREADS` is set, otherwise bounded by
    the number of physical cores (taking cgroups quotas into account).
    """
    global _N_CPUS
    if os.getenv("OMP_NUM_THREADS"):
        return _omp_max_threads()
    if _N_CPUS is None:
        _N_CPUS = cpu_count(only_physical_cores=True)
    return max(1, min(_omp_max_threads(), _N_CPUS))


def _labels_inertia_threadpool_limit(X, centers, n_threads):
    """Labels and inertia of X, with sequential BLAS inside the OpenMP loop."""
    labels = np.empty(X.shape[0], dtype=np.int32)
    labels_inertia = _labels_inertia_sparse if sp.issparse(X) else _labels_inertia
    with _threadpool_controller().limit(limits=1, user_api="blas"):
        inertia = labels_inertia(X, centers, labels, n_threads)
    return labels, inertia


def _round_power_of_two(value):
    """Round a positive value to the nearest power of two."""
    if value <= 1:
        return 1
    low = 2 ** math.floor(math.log2(value))
    return low if value / low < 2 * low / value else 2 * low


def _resolve_batch_size(n_samples, n_clusters, n_features, n_epochs):
    """Batch size given by the fitted batch law."""
    if n_features <= _QUALITY_BATCH_MAX_FEATURES:
        law = "quality"
        max_batch_size = n_samples // 2
    else:
        law = "throughput"
        max_batch_size = _round_power_of_two(n_samples / 2)
    c, a_k, a_n, a_t = _BATCH_LAWS[law]
    batch_size = _round_power_of_two(
        c * n_clusters**a_k * n_samples**a_n * n_epochs**a_t
    )
    return max(1, min(batch_size, max_batch_size, n_samples))


def _greedy_revival_period(n_samples, n_clusters, n_epochs, batch_size):
    """Number of steps between two greedy revivals.

    `_REVIVAL_PERIOD` batches of the "throughput" law: the smaller batches of
    the "quality" law don't get more revivals for the same number of samples,
    since the time of a greedy revival doesn't depend on the batch size.
    """
    throughput_batch_size = _resolve_batch_size(
        n_samples, n_clusters, _QUALITY_BATCH_MAX_FEATURES + 1, n_epochs
    )
    return _REVIVAL_PERIOD * max(1, round(throughput_batch_size / batch_size))


def _resolve_initial_step_size(n_samples, n_clusters, n_epochs, batch_size):
    """Initial EMA step size given by the fitted step law."""
    c, a_k, a_b, a_n, a_t = _STEP_LAW
    step_size = c * n_clusters**a_k * batch_size**a_b * n_samples**a_n * n_epochs**a_t
    return min(max(step_size, 1e-4), 1.0)


def _epochs_n_rows(n_samples, n_epochs):
    """Number of rows seen in each epoch.

    A fractional epoch only sees a prefix of its permutation. It comes before
    the last epoch, which is always complete since the final centers are the
    means of its assignments.
    """
    n_full = math.floor(n_epochs)
    remainder = round((n_epochs - n_full) * n_samples)
    if n_full == 0:
        return [max(remainder, 1)]
    return [n_samples] * (n_full - 1) + ([remainder] if remainder else []) + [n_samples]


def _is_greedy_step(greedy, step):
    """Whether the greedy revival runs after `step` steps of the fit."""
    return greedy is not None and greedy.is_revival_step(step)


def _greedy_segments(greedy, first_step, n_epoch_steps):
    """Split the steps of an epoch at the steps of the greedy revival.

    Returns the (start, stop) of each segment, as indices of steps in the
    epoch. `first_step` is the number of steps done before the epoch.
    """
    stops = [
        stop
        for stop in range(1, n_epoch_steps)
        if _is_greedy_step(greedy, first_step + stop)
    ]
    return zip([0] + stops, stops + [n_epoch_steps])


def _scheduled_step_size(initial_step_size, step, n_steps):
    """Cosine annealing from `initial_step_size` to a tenth of it."""
    cosine = 0.5 * (1.0 + math.cos(math.pi * step / n_steps))
    return initial_step_size * (
        _MIN_STEP_SIZE_FACTOR + (1.0 - _MIN_STEP_SIZE_FACTOR) * cosine
    )


def _revive_dead_clusters(
    centers, weight_ema, random_state, threshold=_DEAD_CLUSTER_THRESHOLD
):
    """Split heavy clusters to replace the dead ones, inplace.

    A cluster is dead when its EMA weight is below `threshold` times the mean
    EMA weight. Each dead cluster takes half of a donor cluster, drawn without
    replacement with a probability proportional to its EMA weight. The donor
    center is split into two centers moved by a tiny relative offset in
    opposite directions, and its EMA weight is shared between them.

    Returns the number of revived clusters.
    """
    indices = _revival_indices(weight_ema, random_state, threshold)
    if indices is None:
        return 0
    dead_indices, donor_indices = indices
    centers[dead_indices], centers[donor_indices] = _split_centers(
        centers[donor_indices]
    )
    weight_ema[donor_indices] /= 2
    weight_ema[dead_indices] = weight_ema[donor_indices]
    return len(dead_indices)


def _revival_indices(weight_ema, random_state, threshold=_DEAD_CLUSTER_THRESHOLD):
    """Indices of the dead clusters to revive and of their donors, or None."""
    dead = weight_ema < threshold * weight_ema.mean()
    donors = ~dead & (weight_ema > 0)
    n_revived = min(np.count_nonzero(dead), np.count_nonzero(donors))
    if n_revived == 0:
        return None

    # Gumbel top-k trick: the n_revived largest log(w) + Gumbel noise are a
    # sample without replacement with probabilities proportional to w.
    uniform = np.maximum(random_state.uniform(size=len(weight_ema)), 1e-300)
    keys = np.full(len(weight_ema), -np.inf)
    keys[donors] = np.log(weight_ema[donors]) - np.log(-np.log(uniform[donors]))
    donor_indices = np.argsort(keys)[::-1][:n_revived]
    dead_indices = np.flatnonzero(dead)[:n_revived]
    return dead_indices, donor_indices


def _split_centers(donor_centers):
    """The two halves of each donor center: for the dead cluster and the donor."""
    offset = np.full(donor_centers.shape[1], _SPLIT_EPSILON, dtype=donor_centers.dtype)
    offset[1::2] *= -1
    dead_centers = donor_centers * (1 + offset)
    donor_centers = donor_centers * (1 - offset)

    # A null center would give two identical halves: separate them along the
    # first feature.
    same = (dead_centers == donor_centers).all(axis=1)
    if same.any():
        dead_centers[same, 0] = np.nextafter(
            dead_centers[same, 0], donor_centers.dtype.type(np.inf)
        )
    return dead_centers, donor_centers


class EMeans(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, ClusterMixin, BaseEstimator
):
    """E-means clustering.

    E-means is a mini-batch variant of K-means where each center is an
    exponential moving average (EMA) of the samples assigned to it. The EMA
    step size decreases along a cosine schedule, and clusters that receive
    too few samples are revived by splitting heavier ones (see `revival`). It
    typically gets close to the inertia of :class:`~sklearn.cluster.KMeans` in
    a few passes over the data (epochs) instead of tens of iterations.

    The batch size and the initial step size are given by scaling laws fitted
    by the authors, as functions of the number of samples, of clusters and of
    epochs, so that the number of epochs is the only parameter to choose.

    The input can be dense or a sparse CSR matrix, in float32 or float64.

    With `sklearn.set_config(array_api_dispatch=True)`, dense array API inputs
    (PyTorch tensors, CuPy arrays...) are fitted on their device, e.g. a GPU,
    and the fitted attributes are arrays of the same library on that device.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

    init : {'random', 'k-means++', 'k-means||'}, callable or array-like of \
            shape (n_clusters, n_features), default='random'
        Method for initialization:

        'random': choose `n_clusters` observations (rows) at random from data
        for the initial centroids.

        'k-means++': greedy k-means++, as in
        :func:`sklearn.cluster.kmeans_plusplus`. Mostly for experimenting:
        'k-means||' gives a similar inertia much faster.

        'k-means||': scalable k-means++ [2]_. A few rounds sample candidates
        with probabilities proportional to their squared distance to the
        candidates so far, then a weighted k-means++ on the candidates gives
        the initial centroids. It costs about as much as 2 to 3 epochs.

        If an array is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

        If a callable is passed, it should take arguments X, n_clusters and a
        random state and return an initialization.

    n_epochs : float, default=5
        Number of passes over the data. Fractional values are supported: when
        `n_epochs > 1`, the fractional pass comes before the last epoch, which
        is always a full pass.

    revival : float in [0, 0.5] or 'greedy', default=0.1
        How clusters are revived during the fit:

        If a float, every 5 steps, a cluster is dead when its weight (an
        exponential moving average of the number of samples it gets per
        batch) is below `revival` times the mean weight of the clusters. Each
        dead cluster is replaced by half of a cluster drawn with a probability
        proportional to its weight. 0 turns the revival off. This revival is
        always off when `n_features <= 2`.

        The default 0.1 is the value of the reference implementation. It is
        a good choice to get clusters of homogeneous sizes, but it keeps
        removing the clusters that have less than about a tenth of the mean
        cluster size. When the data really has small clusters (e.g. groups of
        outliers in heavy-tailed data), the fit can then end far from the
        best clustering, whatever the number of epochs: use a lower value, 0
        or 'greedy'.

        'greedy': a cluster is replaced by half of another one only when this
        is estimated to lower the inertia, whatever their sizes. The cost of
        removing a cluster and the gain of splitting one are estimated on a
        sample of the data, during the first half of the fit. It keeps the
        small clusters that are worth keeping and usually gives a lower
        inertia, for a longer fit. This option is experimental: its rule may
        be tuned in future versions, which would change its results.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization, the
        order of the samples in each epoch and the revival of clusters.
        Use an int to make the randomness deterministic.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.

    labels_ : ndarray of shape (n_samples,)
        Labels of each point.

    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.

    batch_size_ : int
        Size of the mini-batches, given by the scaling law of the authors. It
        grows almost linearly with the number of samples and is smaller when
        `n_features <= 16`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    See Also
    --------
    sklearn.cluster.KMeans : The classic implementation of the clustering
        method based on the Lloyd's algorithm.
    sklearn.cluster.MiniBatchKMeans : Mini-batch K-means with per-center
        learning rates.

    Notes
    -----
    The algorithm is described in [1]_. For each mini-batch of samples, each
    sample is assigned to its nearest center, then each center :math:`c_j` is
    updated from the sum :math:`s_j` and the number :math:`n_j` of the samples
    assigned to it:

    .. math::

        N_j \\leftarrow (1 - \\alpha) N_j + \\alpha n_j, \\qquad
        c_j \\leftarrow \\frac{(1 - \\alpha) N_j c_j + \\alpha s_j}{N_j'}

    where :math:`N_j'` is the updated :math:`N_j` and :math:`\\alpha` the step
    size. The final centers are the means of the samples assigned to each
    center during the last epoch. `labels_` and `inertia_` are then computed
    with an extra pass over the data, so that they match the final centers.

    For array API inputs, the Cython kernels are replaced by array operations
    of the library of the input. The random numbers are drawn on the host in
    the same order, so the results are the same up to rounding. On devices
    without float64, the EMA weights and the sums of the last epoch are in
    float32. With `init="k-means||"`, the final k-means++ on the candidates
    runs on the host, and `init="k-means++"` copies the data to the host.
    `revival="greedy"` copies its sample of the data to the host, where it
    estimates the costs and the gains.

    References
    ----------
    .. [1] S. Bessa, P.-E. Mazaré, G. Szilvasy, H. Touvron, D. Picard,
       M. Douze and H. Jégou, "e-means: outperforming k-means (in one-tenth
       the epochs)", 2026. `arXiv:2611.09501 <https://arxiv.org/abs/2611.09501>`_

    .. [2] B. Bahmani, B. Moseley, A. Vattani, R. Kumar and S. Vassilvitskii,
       "Scalable K-Means++", Proceedings of the VLDB Endowment, 2012.
       `arXiv:1203.6402 <https://arxiv.org/abs/1203.6402>`_

    Examples
    --------
    >>> from sklearn_emeans import EMeans
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> emeans = EMeans(n_clusters=2, random_state=0).fit(X)
    >>> emeans.labels_
    array([0, 0, 0, 1, 1, 1], dtype=int32)
    >>> emeans.predict([[0, 0], [12, 3]])
    array([0, 1], dtype=int32)
    >>> emeans.cluster_centers_
    array([[ 1.,  2.],
           [10.,  2.]])
    """

    def __init__(
        self,
        n_clusters=8,
        *,
        init="random",
        n_epochs=5,
        revival=_DEAD_CLUSTER_THRESHOLD,
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.n_epochs = n_epochs
        self.revival = revival
        self.random_state = random_state

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        tags.transformer_tags.preserves_dtype = ["float64", "float32"]
        return tags

    def _check_params(self):
        if (
            not isinstance(self.n_clusters, Integral)
            or isinstance(self.n_clusters, bool)
            or self.n_clusters < 1
        ):
            raise ValueError(
                f"n_clusters must be an int >= 1, got {self.n_clusters!r}."
            )
        if (
            not isinstance(self.n_epochs, Real)
            or isinstance(self.n_epochs, bool)
            or not self.n_epochs > 0
        ):
            raise ValueError(f"n_epochs must be a float > 0, got {self.n_epochs!r}.")
        if isinstance(self.revival, str):
            valid_revival = self.revival == "greedy"
        else:
            valid_revival = (
                isinstance(self.revival, Real)
                and not isinstance(self.revival, bool)
                and 0 <= self.revival <= 0.5
            )
        if not valid_revival:
            raise ValueError(
                "revival must be a float in [0, 0.5] or 'greedy', got "
                f"{self.revival!r}."
            )
        if isinstance(self.init, str):
            if self.init not in ("random", "k-means++", "k-means||"):
                raise ValueError(
                    "init must be 'random', 'k-means++', 'k-means||', a callable "
                    f"or an array-like, got {self.init!r}."
                )
        elif not callable(self.init) and not (
            hasattr(self.init, "__len__") or hasattr(self.init, "shape")
        ):
            raise ValueError(
                "init must be 'random', 'k-means++', 'k-means||', a callable "
                f"or an array-like, got {self.init!r}."
            )

    def _check_centers(self, centers, X, dtype=None):
        """Validate initial centers given as an array or by a callable."""
        centers = check_array(
            centers, dtype=X.dtype if dtype is None else dtype, copy=True, order="C"
        )
        if centers.shape[0] != self.n_clusters:
            raise ValueError(
                f"The shape of the initial centers {centers.shape} does not "
                f"match the number of clusters {self.n_clusters}."
            )
        if centers.shape[1] != X.shape[1]:
            raise ValueError(
                f"The shape of the initial centers {centers.shape} does not "
                f"match the number of features of the data {X.shape[1]}."
            )
        return centers

    def _init_centers(self, X, random_state):
        if isinstance(self.init, str) and self.init == "k-means++":
            sample_weight = np.ones(X.shape[0], dtype=X.dtype)
            x_squared_norms = row_norms(X, squared=True)
            with _threadpool_controller().limit(limits=1, user_api="blas"):
                centers, _ = _kmeans_plusplus(
                    X,
                    self.n_clusters,
                    x_squared_norms,
                    sample_weight,
                    random_state,
                    n_threads=_openmp_effective_n_threads(),
                )
            return centers
        if isinstance(self.init, str) and self.init == "k-means||":
            with _threadpool_controller().limit(limits=1, user_api="blas"):
                return _kmeans_parallel(
                    X, self.n_clusters, random_state, _openmp_effective_n_threads()
                )
        if isinstance(self.init, str):
            # Same draw as scikit-learn's KMeans with init="random" and unit
            # sample weights.
            n_samples = X.shape[0]
            weights = np.ones(n_samples, dtype=X.dtype)
            seeds = random_state.choice(
                n_samples,
                size=self.n_clusters,
                replace=False,
                p=weights / weights.sum(),
            )
            return X[seeds].toarray() if sp.issparse(X) else X[seeds]
        if callable(self.init):
            return self._check_centers(
                self.init(X, self.n_clusters, random_state=random_state), X
            )
        return self._check_centers(self.init, X)

    def _revival_period(self, n_features):
        """Number of steps between two revivals of the dead clusters, or 0."""
        if (
            isinstance(self.revival, str)
            or self.revival == 0
            or n_features <= _NO_REVIVAL_MAX_FEATURES
        ):
            return 0
        return _REVIVAL_PERIOD

    def _greedy_revival(self, X, to_numpy, n_steps, batch_size, random_state):
        """The greedy revival on a sample of X, or None for the other revivals.

        `to_numpy(X, indices)` gives the rows of X of the given indices, all
        of them if None, as a NumPy array or a SciPy CSR matrix.
        """
        if self.revival != "greedy" or self.n_clusters < 2:
            return None
        indices = greedy_sample_indices(X.shape[0], self.n_clusters, random_state)
        period = _greedy_revival_period(
            X.shape[0], self.n_clusters, self.n_epochs, batch_size
        )
        return GreedyRevival(
            to_numpy(X, indices), n_steps, period, _openmp_effective_n_threads()
        )

    def fit(self, X, y=None):
        """Compute e-means clustering.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory copy
            if the given data is not C-contiguous. If a sparse matrix is
            passed, a copy will be made if it's not in CSR format.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self._check_params()
        xp = _array_api.get_namespace(X)
        if xp is not None:
            return self._fit_array_api(xp, X)
        X = validate_data(
            self,
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
        )
        n_samples, n_features = X.shape
        n_clusters = self.n_clusters
        if n_samples < n_clusters:
            raise ValueError(
                f"n_samples={n_samples} should be >= n_clusters={n_clusters}."
            )
        random_state = check_random_state(self.random_state)
        max_n_threads = _openmp_effective_n_threads()

        centers = np.array(
            self._init_centers(X, random_state), dtype=X.dtype, order="C", copy=True
        )

        batch_size = _resolve_batch_size(
            n_samples, n_clusters, n_features, self.n_epochs
        )
        initial_step_size = _resolve_initial_step_size(
            n_samples, n_clusters, self.n_epochs, batch_size
        )
        # No more threads than chunks of a batch: small batches are faster on
        # fewer threads.
        n_threads = min(max_n_threads, math.ceil(batch_size / _CHUNK_SIZE))

        epochs_n_rows = _epochs_n_rows(n_samples, self.n_epochs)
        n_steps = sum(math.ceil(n_rows / batch_size) for n_rows in epochs_n_rows)
        revival_period = self._revival_period(n_features)

        weight_ema = np.full(n_clusters, _INIT_WEIGHT)
        epoch_sums = np.zeros((n_clusters, n_features), dtype=np.float64)
        epoch_counts = np.zeros(n_clusters, dtype=np.float64)

        emeans_steps = _emeans_steps_sparse if sp.issparse(X) else _emeans_steps
        revive = partial(
            _revive_dead_clusters, centers, weight_ema, random_state, self.revival
        )
        greedy = self._greedy_revival(
            X,
            lambda X, indices: X if indices is None else X[indices],
            n_steps,
            batch_size,
            random_state,
        )

        step = 0
        with _threadpool_controller().limit(limits=1, user_api="blas"):
            for epoch, n_rows in enumerate(epochs_n_rows):
                is_last_epoch = epoch == len(epochs_n_rows) - 1
                # intp for the kernel: RandomState gives C longs, which are
                # 32-bit on Windows.
                permutation = random_state.permutation(n_samples).astype(
                    np.intp, copy=False
                )
                n_epoch_steps = math.ceil(n_rows / batch_size)
                step_sizes = np.array(
                    [
                        _scheduled_step_size(initial_step_size, step + i, n_steps)
                        for i in range(n_epoch_steps)
                    ]
                )
                # All the steps of the epoch in one call, which revives the
                # dead clusters every `revival_period` steps: see `_run_steps`
                # in `_kernels.pyx`. The greedy revival runs between two
                # calls instead, to use all the threads.
                for start, stop in _greedy_segments(greedy, step, n_epoch_steps):
                    emeans_steps(
                        X,
                        permutation[
                            start * batch_size : min(stop * batch_size, n_rows)
                        ],
                        batch_size,
                        centers,
                        weight_ema,
                        step_sizes[start:stop],
                        epoch_sums if is_last_epoch else None,
                        epoch_counts if is_last_epoch else None,
                        step + start,
                        revival_period,
                        revive,
                        n_threads,
                    )
                    if _is_greedy_step(greedy, step + stop):
                        greedy(centers, weight_ema)
                step += n_epoch_steps

        # The final centers are the means of the last epoch's assignments.
        # Centers that got no sample in the last epoch keep their EMA position.
        assigned = epoch_counts > 0
        centers[assigned] = epoch_sums[assigned] / epoch_counts[assigned, None]

        self.cluster_centers_ = centers
        self._n_features_out = n_clusters
        self.labels_, self.inertia_ = _labels_inertia_threadpool_limit(
            X, centers, max_n_threads
        )
        self.batch_size_ = batch_size
        return self

    def _validate_array_api(self, xp, X, reset):
        """Validate an array API input, kept on its device."""
        if _array_api.dtype_name(xp, X) is None:
            # Not all the devices have float64.
            X = xp.astype(
                X, getattr(xp, _array_api.accumulator_dtype(xp, _array_api.device(X)))
            )
        return validate_data(
            self, X, dtype=[xp.float64, xp.float32], order="C", reset=reset
        )

    def _init_centers_array_api(self, xp, X, random_state):
        """Initial centers of an array API input, on its device."""
        dtype = _array_api.dtype_name(xp, X)
        device = _array_api.device(X)
        if isinstance(self.init, str) and self.init == "random":
            # Same draw as for NumPy inputs.
            n_samples = X.shape[0]
            weights = np.ones(n_samples, dtype=dtype)
            seeds = random_state.choice(
                n_samples,
                size=self.n_clusters,
                replace=False,
                p=weights / weights.sum(),
            )
            return xp.take(X, _array_api.index_array(xp, seeds, device), axis=0)
        if isinstance(self.init, str) and self.init == "k-means||":
            with _threadpool_controller().limit(limits=1, user_api="blas"):
                centers = _array_api.kmeans_parallel(
                    xp, X, self.n_clusters, random_state, _openmp_effective_n_threads()
                )
        elif isinstance(self.init, str):
            # The greedy k-means++ only runs on the host.
            centers = self._init_centers(_array_api.to_numpy(xp, X), random_state)
        else:
            centers = self.init
            if callable(centers):
                centers = centers(X, self.n_clusters, random_state=random_state)
            if _array_api.get_namespace(centers) is not None:
                centers = _array_api.to_numpy(
                    _array_api.get_namespace(centers), centers
                )
            centers = self._check_centers(centers, X, dtype=dtype)
        return _array_api.from_numpy(xp, centers, dtype, device)

    def _fit_array_api(self, xp, X):
        """Fit on an array API input, on its device.

        Same algorithm and same random numbers as in `fit`, with the array
        operations of `_array_api.py` in place of the Cython kernels.
        """
        X = self._validate_array_api(xp, X, reset=True)
        n_samples, n_features = X.shape
        n_clusters = self.n_clusters
        if n_samples < n_clusters:
            raise ValueError(
                f"n_samples={n_samples} should be >= n_clusters={n_clusters}."
            )
        random_state = check_random_state(self.random_state)
        device = _array_api.device(X)
        acc_dtype = getattr(xp, _array_api.accumulator_dtype(xp, device))

        centers = self._init_centers_array_api(xp, X, random_state)

        batch_size = _resolve_batch_size(
            n_samples, n_clusters, n_features, self.n_epochs
        )
        initial_step_size = _resolve_initial_step_size(
            n_samples, n_clusters, self.n_epochs, batch_size
        )
        epochs_n_rows = _epochs_n_rows(n_samples, self.n_epochs)
        n_steps = sum(math.ceil(n_rows / batch_size) for n_rows in epochs_n_rows)
        revival_period = self._revival_period(n_features)

        weight_ema = xp.full(n_clusters, _INIT_WEIGHT, dtype=acc_dtype, device=device)
        epoch_sums = epoch_counts = None
        revive = partial(
            _array_api.revive_dead_clusters,
            xp,
            random_state=random_state,
            threshold=self.revival,
        )
        greedy = self._greedy_revival(
            X,
            lambda X, indices: _array_api.to_numpy(
                xp,
                X
                if indices is None
                else xp.take(X, _array_api.index_array(xp, indices, device), axis=0),
            ),
            n_steps,
            batch_size,
            random_state,
        )
        if greedy is not None:
            # Called every `greedy.period` steps, with a sequential BLAS for
            # its OpenMP threads.
            revival_period = greedy.period
            n_calls = 0

            def revive(centers, weight_ema):
                nonlocal n_calls
                n_calls += 1
                if not greedy.is_revival_step(n_calls * greedy.period):
                    return centers, weight_ema
                with _threadpool_controller().limit(limits=1, user_api="blas"):
                    return _array_api.revive_greedy(xp, greedy, centers, weight_ema)

        step = 0
        for epoch, n_rows in enumerate(epochs_n_rows):
            if epoch == len(epochs_n_rows) - 1:
                epoch_sums = xp.zeros(
                    (n_clusters, n_features), dtype=acc_dtype, device=device
                )
                epoch_counts = xp.zeros(n_clusters, dtype=acc_dtype, device=device)
            permutation = random_state.permutation(n_samples)[:n_rows]
            n_epoch_steps = math.ceil(n_rows / batch_size)
            step_sizes = [
                _scheduled_step_size(initial_step_size, step + i, n_steps)
                for i in range(n_epoch_steps)
            ]
            centers, weight_ema, epoch_sums, epoch_counts = _array_api.emeans_steps(
                xp,
                X,
                _array_api.index_array(xp, permutation, device),
                batch_size,
                centers,
                weight_ema,
                step_sizes,
                epoch_sums,
                epoch_counts,
                step,
                revival_period,
                revive,
            )
            step += n_epoch_steps

        # The final centers are the means of the last epoch's assignments.
        # Centers that got no sample in the last epoch keep their EMA position.
        assigned = epoch_counts > 0
        safe_counts = xp.where(assigned, epoch_counts, xp.ones_like(epoch_counts))
        centers = xp.where(
            assigned[:, None],
            xp.astype(epoch_sums / safe_counts[:, None], X.dtype),
            centers,
        )

        self.cluster_centers_ = centers
        self._n_features_out = n_clusters
        self.labels_, self.inertia_ = _array_api.labels_inertia(xp, X, centers)
        self.batch_size_ = batch_size
        return self

    def _check_test_data(self, X):
        """Validate X, and return its array namespace and the centers.

        The centers are converted to the dtype of X, and moved to its
        namespace and device if needed. The namespace is None for the inputs
        that go through the Cython kernels.
        """
        centers = self.cluster_centers_
        centers_xp = _array_api.get_namespace(centers)
        xp = _array_api.get_namespace(X)
        if xp is not None:
            X = self._validate_array_api(xp, X, reset=False)
            if centers_xp is not None and centers_xp is not xp:
                centers = _array_api.to_numpy(centers_xp, centers)
            if centers_xp is not xp:
                centers = np.asarray(centers)
            return (
                X,
                xp,
                xp.asarray(centers, dtype=X.dtype, device=_array_api.device(X)),
            )
        X = validate_data(
            self,
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
            reset=False,
        )
        if not isinstance(centers, np.ndarray):
            # Fitted on an array API input.
            centers = _array_api.to_numpy(
                _array_api.array_api_compat.array_namespace(centers), centers
            )
        return X, None, np.ascontiguousarray(centers, dtype=X.dtype)

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self)
        X, xp, centers = self._check_test_data(X)
        if xp is not None:
            return _array_api.labels_inertia(xp, X, centers)[0]
        labels, _ = _labels_inertia_threadpool_limit(
            X, centers, _openmp_effective_n_threads()
        )
        return labels

    def transform(self, X):
        """Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster
        centers.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_clusters)
            X transformed in the new space.
        """
        check_is_fitted(self)
        X, xp, centers = self._check_test_data(X)
        if xp is not None:
            return _array_api.euclidean_distances(xp, X, centers)
        return euclidean_distances(X, centers)

    def score(self, X, y=None):
        """Opposite of the value of X on the K-means objective.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        score : float
            Opposite of the value of X on the K-means objective.
        """
        check_is_fitted(self)
        X, xp, centers = self._check_test_data(X)
        if xp is not None:
            return -_array_api.labels_inertia(xp, X, centers)[1]
        _, inertia = _labels_inertia_threadpool_limit(
            X, centers, _openmp_effective_n_threads()
        )
        return -inertia
