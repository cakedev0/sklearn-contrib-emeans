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
from sklearn.utils.validation import check_is_fitted, validate_data
from threadpoolctl import ThreadpoolController

from sklearn_emeans._kernels import (
    _chunk_size,
    _emeans_step,
    _emeans_step_sparse,
    _labels_inertia,
    _labels_inertia_sparse,
    _omp_max_threads,
)

# The constants below are the ones of the reference implementation
# (faiss/contrib/e_means/base.py), which differ slightly from the paper.

# Ratio between the last and the first step size of the cosine schedule.
_MIN_STEP_SIZE_FACTOR = 0.1

# Dead clusters are revived every `_REVIVAL_PERIOD` steps.
_REVIVAL_PERIOD = 5

# A cluster is dead when its EMA weight is below this fraction of the mean EMA
# weight.
_DEAD_CLUSTER_THRESHOLD = 0.1

# Relative perturbation of the two halves of a split donor center.
_SPLIT_EPSILON = 1e-4

# Initial EMA weight of each center. Tiny, so that the first batch moves the
# centers to the mean of their rows.
_INIT_WEIGHT = 2.0**-16

# The reference implementation turns revival off for n_features <= 2, where it
# was found to hurt the held-out inertia.
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


def _scheduled_step_size(initial_step_size, step, n_steps):
    """Cosine annealing from `initial_step_size` to a tenth of it."""
    cosine = 0.5 * (1.0 + math.cos(math.pi * step / n_steps))
    return initial_step_size * (
        _MIN_STEP_SIZE_FACTOR + (1.0 - _MIN_STEP_SIZE_FACTOR) * cosine
    )


def _revive_dead_clusters(centers, weight_ema, random_state):
    """Split heavy clusters to replace the dead ones, inplace.

    A cluster is dead when its EMA weight is below a fraction of the mean EMA
    weight. Each dead cluster takes half of a donor cluster, drawn without
    replacement with a probability proportional to its EMA weight. The donor
    center is split into two centers moved by a tiny relative offset in
    opposite directions, and its EMA weight is shared between them.

    Returns the number of revived clusters.
    """
    dead = weight_ema < _DEAD_CLUSTER_THRESHOLD * weight_ema.mean()
    donors = ~dead & (weight_ema > 0)
    n_revived = min(np.count_nonzero(dead), np.count_nonzero(donors))
    if n_revived == 0:
        return 0

    # Gumbel top-k trick: the n_revived largest log(w) + Gumbel noise are a
    # sample without replacement with probabilities proportional to w.
    uniform = np.maximum(random_state.uniform(size=len(weight_ema)), 1e-300)
    keys = np.full(len(weight_ema), -np.inf)
    keys[donors] = np.log(weight_ema[donors]) - np.log(-np.log(uniform[donors]))
    donor_indices = np.argsort(keys)[::-1][:n_revived]
    dead_indices = np.flatnonzero(dead)[:n_revived]

    offset = np.full(centers.shape[1], _SPLIT_EPSILON, dtype=centers.dtype)
    offset[1::2] *= -1
    donor_centers = centers[donor_indices]
    centers[dead_indices] = donor_centers * (1 + offset)
    centers[donor_indices] = donor_centers * (1 - offset)

    # A null center would give two identical halves: separate them along the
    # first feature.
    same = (centers[dead_indices] == centers[donor_indices]).all(axis=1)
    if same.any():
        same_indices = dead_indices[same]
        centers[same_indices, 0] = np.nextafter(
            centers[same_indices, 0], centers.dtype.type(np.inf)
        )

    weight_ema[donor_indices] /= 2
    weight_ema[dead_indices] = weight_ema[donor_indices]
    return n_revived


class EMeans(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, ClusterMixin, BaseEstimator
):
    """E-means clustering.

    E-means is a mini-batch variant of K-means where each center is an
    exponential moving average (EMA) of the samples assigned to it. The EMA
    step size decreases along a cosine schedule, and clusters that receive
    too few samples are revived by splitting heavier ones. It typically
    reaches the inertia of :class:`~sklearn.cluster.KMeans` in a few passes
    over the data (epochs) instead of tens of iterations.

    The batch size and the initial step size are given by scaling laws fitted
    by the authors, as functions of the number of samples, of clusters and of
    epochs, so that the number of epochs is the only parameter to choose.

    The input can be dense or a sparse CSR matrix, in float32 or float64.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

    init : 'random', callable or array-like of shape (n_clusters, n_features), \
            default='random'
        Method for initialization:

        'random': choose `n_clusters` observations (rows) at random from data
        for the initial centroids.

        If an array is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

        If a callable is passed, it should take arguments X, n_clusters and a
        random state and return an initialization.

    n_epochs : float, default=5
        Number of passes over the data. Fractional values are supported: when
        `n_epochs > 1`, the fractional pass comes before the last epoch, which
        is always a full pass.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization, the
        order of the samples in each epoch and the revival of dead clusters.
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

    References
    ----------
    .. [1] S. Bessa, P.-E. Mazaré, G. Szilvasy, H. Touvron, D. Picard,
       M. Douze and H. Jégou, "e-means: outperforming k-means (in one-tenth
       the epochs)", 2026. `arXiv:2611.09501 <https://arxiv.org/abs/2611.09501>`_

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

    def __init__(self, n_clusters=8, *, init="random", n_epochs=5, random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.n_epochs = n_epochs
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
        if isinstance(self.init, str):
            if self.init != "random":
                raise ValueError(
                    "init must be 'random', a callable or an array-like, got "
                    f"{self.init!r}."
                )
        elif not callable(self.init) and not hasattr(self.init, "__len__"):
            raise ValueError(
                "init must be 'random', a callable or an array-like, got "
                f"{self.init!r}."
            )

    def _check_centers(self, centers, X):
        """Validate initial centers given as an array or by a callable."""
        centers = check_array(centers, dtype=X.dtype, copy=True, order="C")
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
        revival = n_features > _NO_REVIVAL_MAX_FEATURES

        weight_ema = np.full(n_clusters, _INIT_WEIGHT)
        epoch_sums = np.zeros((n_clusters, n_features), dtype=np.float64)
        epoch_counts = np.zeros(n_clusters, dtype=np.float64)

        if sp.issparse(X):
            emeans_step = partial(
                _emeans_step_sparse,
                centers_T=np.empty((n_features, n_clusters), dtype=X.dtype),
            )
        else:
            emeans_step = _emeans_step

        step = 0
        with _threadpool_controller().limit(limits=1, user_api="blas"):
            for epoch, n_rows in enumerate(epochs_n_rows):
                is_last_epoch = epoch == len(epochs_n_rows) - 1
                # intp for the kernel: RandomState gives C longs, which are
                # 32-bit on Windows.
                permutation = random_state.permutation(n_samples).astype(
                    np.intp, copy=False
                )
                for start in range(0, n_rows, batch_size):
                    emeans_step(
                        X,
                        permutation[start : min(start + batch_size, n_rows)],
                        centers,
                        weight_ema,
                        _scheduled_step_size(initial_step_size, step, n_steps),
                        epoch_sums if is_last_epoch else None,
                        epoch_counts if is_last_epoch else None,
                        n_threads,
                    )
                    step += 1
                    if revival and step % _REVIVAL_PERIOD == 0:
                        _revive_dead_clusters(centers, weight_ema, random_state)

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

    def _check_test_data(self, X):
        X = validate_data(
            self,
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
            reset=False,
        )
        return X, self.cluster_centers_.astype(X.dtype, copy=False)

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
        X, centers = self._check_test_data(X)
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
        X, centers = self._check_test_data(X)
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
        X, centers = self._check_test_data(X)
        _, inertia = _labels_inertia_threadpool_limit(
            X, centers, _openmp_effective_n_threads()
        )
        return -inertia
