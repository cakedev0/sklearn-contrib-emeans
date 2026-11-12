from unittest.mock import Mock

import numpy as np
import pytest
import sklearn
from numpy.testing import assert_array_equal
from sklearn.cluster import kmeans_plusplus
from sklearn.datasets import make_blobs

from sklearn_emeans import EMeans
from sklearn_emeans._kmeans_plusplus import _kmeans_plusplus
from sklearn_emeans._kmeans_plusplus_kernels import (
    CHUNK_SIZE,
    _kmeans_plusplus_sample_candidates,
)

# The vendored k-means++ is the one of scikit-learn 1.10. Older versions draw the
# same random numbers, but sample with a cumulative sum over all the samples,
# whose rounding errors can pick other samples.
SKLEARN_HAS_SAME_KMEANS_PLUSPLUS = tuple(
    int(part) for part in sklearn.__version__.split(".")[:2]
) >= (1, 10)


def _kmeans_plusplus_indices(X, n_clusters, sample_weight=None, random_state=0):
    if sample_weight is None:
        sample_weight = np.ones(X.shape[0], dtype=X.dtype)
    _, indices = _kmeans_plusplus(
        X,
        n_clusters,
        np.einsum("ij,ij->i", X, X),
        sample_weight.astype(X.dtype),
        np.random.RandomState(random_state),
        n_threads=4,
    )
    return indices


@pytest.mark.skipif(
    not SKLEARN_HAS_SAME_KMEANS_PLUSPLUS, reason="needs scikit-learn >= 1.10"
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("weighted", [False, True])
def test_kmeans_plusplus_matches_sklearn(dtype, weighted):
    X, _ = make_blobs(n_samples=5000, n_features=10, centers=30, random_state=0)
    X = X.astype(dtype)
    rng = np.random.RandomState(0)
    sample_weight = rng.uniform(size=X.shape[0]) if weighted else None

    indices = _kmeans_plusplus_indices(X, 30, sample_weight)
    _, expected = kmeans_plusplus(X, 30, sample_weight=sample_weight, random_state=0)
    assert_array_equal(indices, expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_kmeans_plusplus_output(dtype):
    X, _ = make_blobs(n_samples=1000, n_features=5, centers=10, random_state=0)
    X = X.astype(dtype)
    centers, indices = _kmeans_plusplus(
        X,
        10,
        np.einsum("ij,ij->i", X, X),
        np.ones(1000, dtype=dtype),
        np.random.RandomState(0),
    )
    assert centers.dtype == dtype
    assert_array_equal(centers, X[indices])
    assert len(np.unique(indices)) == 10


def test_kmeans_plusplus_sample_weight():
    # Samples with a zero weight are never picked.
    X, _ = make_blobs(n_samples=2000, n_features=5, centers=20, random_state=0)
    sample_weight = np.ones(2000)
    sample_weight[::2] = 0
    for random_state in range(5):
        indices = _kmeans_plusplus_indices(X, 20, sample_weight, random_state)
        assert np.all(indices % 2 == 1)

    # Unit weights give the same centers as no weights.
    assert_array_equal(
        _kmeans_plusplus_indices(X, 20, np.ones(2000)),
        _kmeans_plusplus_indices(X, 20),
    )


def test_kmeans_plusplus_heavy_cluster():
    # A heavy sample far from the others is the first center, and the other
    # centers are spread over the light samples.
    rng = np.random.RandomState(0)
    X = np.vstack([rng.normal(size=(500, 3)), [[100.0, 100.0, 100.0]]])
    sample_weight = np.ones(501)
    sample_weight[-1] = 1e6
    indices = _kmeans_plusplus_indices(X, 5, sample_weight)
    assert indices[0] == 500
    assert len(np.unique(indices)) == 5


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_kmeans_plusplus_n_clusters_equal_n_samples(dtype):
    X = np.random.RandomState(0).normal(size=(300, 4)).astype(dtype)
    indices = _kmeans_plusplus_indices(X, 300)
    assert_array_equal(np.sort(indices), np.arange(300))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_kmeans_plusplus_duplicates_far_from_origin(dtype):
    # A sample identical to a chosen center is never chosen again, even far from
    # the origin, where ||x||² - 2 x.c + ||c||² is dominated by rounding errors
    # in float32.
    rng = np.random.RandomState(0)
    X = np.repeat(rng.uniform(size=(10, 2)) + 1e3, 50, axis=0).astype(dtype)
    indices = _kmeans_plusplus_indices(X, 10)
    assert np.unique(X[indices], axis=0).shape[0] == 10


@pytest.mark.parametrize("shape", [(30_000, 10), (600, 2_000)])
def test_kmeans_plusplus_does_not_depend_on_n_threads(shape):
    X, _ = make_blobs(
        n_samples=shape[0], n_features=shape[1], centers=10, random_state=0
    )
    args = (
        X,
        10,
        np.einsum("ij,ij->i", X, X),
        np.ones(X.shape[0]),
    )
    _, indices_sequential = _kmeans_plusplus(
        *args, np.random.RandomState(0), n_threads=1
    )
    _, indices_parallel = _kmeans_plusplus(*args, np.random.RandomState(0), n_threads=4)
    assert_array_equal(indices_sequential, indices_parallel)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_kmeans_plusplus_sample_candidates(dtype):
    # The sampling per chunk of samples picks the same samples as a search in
    # the cumulative sum of the potentials of all the samples.
    rng = np.random.RandomState(0)
    n_samples, n_candidates = 1000, 50
    closest_dist_sq = rng.uniform(size=n_samples).astype(dtype)
    closest_dist_sq[rng.uniform(size=n_samples) < 0.3] = 0
    sample_weight = rng.uniform(size=n_samples).astype(dtype)
    sample_potentials = (sample_weight * closest_dist_sq).astype(np.float64)
    chunk_potentials = np.add.reduceat(
        sample_potentials, np.arange(0, n_samples, CHUNK_SIZE)
    )

    candidate_indices = np.empty(n_candidates, dtype=np.intp)
    _kmeans_plusplus_sample_candidates(
        np.random.RandomState(0),
        chunk_potentials,
        closest_dist_sq,
        sample_weight,
        candidate_indices,
    )

    random_values = np.random.RandomState(0).uniform(size=n_candidates)
    potential_cumsum = np.cumsum(sample_potentials)
    expected = np.searchsorted(potential_cumsum, random_values * potential_cumsum[-1])
    assert_array_equal(candidate_indices, expected)
    assert np.all(closest_dist_sq[candidate_indices] > 0)


def test_kmeans_plusplus_sample_candidates_rounding():
    # A sample with a zero potential is never picked, when the random value is 0
    # or larger than the cumulative potential of the chunk because of rounding
    # errors.
    closest_dist_sq = np.ones(CHUNK_SIZE)
    closest_dist_sq[:3] = 0
    closest_dist_sq[-3:] = 0
    sample_weight = np.ones(CHUNK_SIZE)
    chunk_potentials = np.array([closest_dist_sq.sum() * (1 + 1e-12)])
    random_state = Mock()
    random_state.uniform.return_value = np.array([0, 1 - 1e-15])

    candidate_indices = np.empty(2, dtype=np.intp)
    _kmeans_plusplus_sample_candidates(
        random_state,
        chunk_potentials,
        closest_dist_sq,
        sample_weight,
        candidate_indices,
    )

    assert_array_equal(candidate_indices, [3, CHUNK_SIZE - 4])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_emeans_kmeans_plusplus_init(dtype):
    # init="k-means++" gives the same result as the vendored k-means++ passed as
    # a callable, so the centers come from it and the RNG is shared with the fit.
    X, _ = make_blobs(n_samples=3000, n_features=8, centers=20, random_state=0)
    X = X.astype(dtype)

    def init(X, n_clusters, random_state):
        centers, _ = _kmeans_plusplus(
            X,
            n_clusters,
            np.einsum("ij,ij->i", X, X),
            np.ones(X.shape[0], dtype=X.dtype),
            random_state,
        )
        return centers

    emeans = EMeans(n_clusters=20, init="k-means++", random_state=0).fit(X)
    expected = EMeans(n_clusters=20, init=init, random_state=0).fit(X)
    assert_array_equal(emeans.cluster_centers_, expected.cluster_centers_)


@pytest.mark.skipif(
    not SKLEARN_HAS_SAME_KMEANS_PLUSPLUS, reason="needs scikit-learn >= 1.10"
)
def test_emeans_kmeans_plusplus_init_matches_sklearn():
    X, _ = make_blobs(n_samples=3000, n_features=8, centers=20, random_state=0)

    def init(X, n_clusters, random_state):
        return kmeans_plusplus(X, n_clusters, random_state=random_state)[0]

    emeans = EMeans(n_clusters=20, init="k-means++", random_state=0).fit(X)
    expected = EMeans(n_clusters=20, init=init, random_state=0).fit(X)
    assert_array_equal(emeans.cluster_centers_, expected.cluster_centers_)
