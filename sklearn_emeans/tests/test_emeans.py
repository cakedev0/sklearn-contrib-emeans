from functools import partial

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from sklearn_emeans import EMeans
from sklearn_emeans._emeans import (
    _epochs_n_rows,
    _resolve_batch_size,
    _resolve_initial_step_size,
    _revive_dead_clusters,
)
from sklearn_emeans._kernels import (
    _emeans_step,
    _emeans_step_sparse,
    _labels_inertia,
    _labels_inertia_sparse,
    _update_min_distances,
    _update_min_distances_sparse,
)
from sklearn_emeans._kmeans_parallel import (
    _kmeans_parallel,
    _kmeans_parallel_candidates,
)

CSR_CONTAINERS = [sp.csr_matrix, sp.csr_array]


def _reference_step(X, batch_indices, centers, weight_ema, step_size, epoch_stats):
    """NumPy implementation of one e-means step, updating its inputs inplace."""
    X_batch = X[batch_indices]
    if sp.issparse(X_batch):
        X_batch = X_batch.toarray()
    distances = ((X_batch[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = distances.argmin(axis=1)

    for j in range(centers.shape[0]):
        mask = labels == j
        count = mask.sum()
        retained = (1 - step_size) * weight_ema[j]
        if count == 0:
            weight_ema[j] = retained
            continue
        sums = X_batch[mask].sum(axis=0)
        if epoch_stats is not None:
            epoch_stats[0][j] += sums
            epoch_stats[1][j] += count
        new_weight = retained + step_size * count
        centers[j] = (retained * centers[j] + step_size * sums) / new_weight
        weight_ema[j] = new_weight


@pytest.mark.parametrize("csr_container", [None] + CSR_CONTAINERS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("accumulate", [False, True])
def test_emeans_step_matches_reference(csr_container, dtype, accumulate):
    rng = np.random.RandomState(0)
    n_samples, n_features, n_clusters, batch_size = 1000, 7, 13, 600
    X = rng.normal(size=(n_samples, n_features)).astype(dtype)
    if csr_container is not None:
        # Sparse rows, some of them empty.
        X[X < 0.5] = 0
        X[:50] = 0
        X = csr_container(X)
    centers = rng.normal(size=(n_clusters, n_features)).astype(dtype)
    # A center far from the data gets no row.
    centers[0] = 100
    weight_ema = rng.uniform(size=n_clusters)
    batch_indices = rng.permutation(n_samples)[:batch_size].astype(np.intp)

    def epoch_stats():
        if not accumulate:
            return None
        return [np.ones((n_clusters, n_features)), np.ones(n_clusters)]

    centers_ref, weight_ema_ref, stats_ref = (
        centers.copy(),
        weight_ema.copy(),
        epoch_stats(),
    )
    _reference_step(X, batch_indices, centers_ref, weight_ema_ref, 0.3, stats_ref)

    stats = epoch_stats() or [None, None]
    if csr_container is None:
        step = _emeans_step
    else:
        centers_T = np.empty((n_features, n_clusters), dtype=dtype)
        step = partial(_emeans_step_sparse, centers_T=centers_T)
    step(
        X,
        batch_indices,
        centers,
        weight_ema,
        0.3,
        *stats,
        2,
    )

    rtol = 1e-5 if dtype == np.float32 else 1e-10
    assert_allclose(centers, centers_ref, rtol=rtol, atol=rtol)
    assert_allclose(weight_ema, weight_ema_ref, rtol=rtol)
    if accumulate:
        assert_allclose(stats[0], stats_ref[0], rtol=rtol)
        assert_allclose(stats[1], stats_ref[1], rtol=rtol)


def test_emeans_step_full_step_size_gives_batch_means():
    # With a step size of 1, each center becomes the mean of its batch rows.
    rng = np.random.RandomState(0)
    X = rng.normal(size=(200, 3))
    centers = X[:4].copy()
    batch_indices = np.arange(200, dtype=np.intp)
    labels = ((X[:, None] - centers) ** 2).sum(axis=2).argmin(axis=1)
    _emeans_step(
        X,
        batch_indices,
        centers,
        np.full(4, 1e-5),
        1.0,
        None,
        None,
        1,
    )
    expected = np.array([X[labels == j].mean(axis=0) for j in range(4)])
    assert_allclose(centers, expected)


@pytest.mark.parametrize("csr_container", [None] + CSR_CONTAINERS)
def test_emeans_does_not_depend_on_n_threads(csr_container):
    X, _ = make_blobs(n_samples=5000, n_features=20, centers=30, random_state=0)
    if csr_container is not None:
        X[X < 0] = 0
        X = csr_container(X)
    emeans = EMeans(n_clusters=30, random_state=0)
    with threadpool_limits(limits=1, user_api="openmp"):
        centers_sequential = emeans.fit(X).cluster_centers_
    centers_parallel = emeans.fit(X).cluster_centers_
    assert_array_equal(centers_sequential, centers_parallel)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_emeans_dense_sparse(csr_container, dtype):
    X, _ = make_blobs(n_samples=2000, n_features=10, centers=10, random_state=0)
    X[X < 0] = 0
    X = X.astype(dtype)
    emeans_dense = EMeans(n_clusters=10, random_state=0).fit(X)
    emeans_sparse = EMeans(n_clusters=10, random_state=0).fit(csr_container(X))
    rtol = 1e-5 if dtype == np.float32 else 1e-10
    assert emeans_sparse.cluster_centers_.dtype == dtype
    assert_allclose(
        emeans_dense.cluster_centers_, emeans_sparse.cluster_centers_, rtol=rtol
    )
    assert_array_equal(emeans_dense.labels_, emeans_sparse.labels_)
    assert emeans_dense.inertia_ == pytest.approx(emeans_sparse.inertia_, rel=rtol)


def test_emeans_float32_float64():
    X, _ = make_blobs(n_samples=2000, n_features=10, centers=10, random_state=0)
    emeans_64 = EMeans(n_clusters=10, random_state=0).fit(X)
    emeans_32 = EMeans(n_clusters=10, random_state=0).fit(X.astype(np.float32))
    assert emeans_32.cluster_centers_.dtype == np.float32
    assert_allclose(emeans_64.cluster_centers_, emeans_32.cluster_centers_, rtol=1e-4)


@pytest.mark.parametrize("n_features", [8, 32])
def test_emeans_inertia_close_to_kmeans(n_features):
    # Fewer epochs than KMeans iterations, for a similar inertia (within 2% in
    # practice).
    X, _ = make_blobs(
        n_samples=20_000,
        n_features=n_features,
        centers=200,
        cluster_std=3,
        random_state=0,
    )
    kmeans = KMeans(n_clusters=100, n_init=1, random_state=0).fit(X)
    emeans = EMeans(n_clusters=100, random_state=0).fit(X)
    assert emeans.n_epochs < kmeans.n_iter_
    assert emeans.inertia_ < 1.03 * kmeans.inertia_


@pytest.mark.parametrize("init", ["random", "k-means++", "k-means||"])
@pytest.mark.parametrize("csr_container", [None] + CSR_CONTAINERS)
def test_emeans_recovers_blobs(init, csr_container):
    X, y = make_blobs(n_samples=3000, n_features=5, centers=5, random_state=42)
    if csr_container is not None:
        X = csr_container(X)
    emeans = EMeans(n_clusters=5, init=init, random_state=0).fit(X)
    assert adjusted_rand_score(y, emeans.labels_) == pytest.approx(1)


def test_emeans_labels_inertia_match_final_centers():
    X, _ = make_blobs(n_samples=1000, n_features=5, centers=10, random_state=0)
    emeans = EMeans(n_clusters=10, random_state=0).fit(X)
    assert_array_equal(emeans.labels_, emeans.predict(X))
    assert emeans.inertia_ == pytest.approx(-emeans.score(X))


def test_emeans_revives_dead_clusters():
    # All initial centers are the same point: only the first one gets
    # samples, and the others are revived by splitting it.
    X, _ = make_blobs(n_samples=5000, n_features=5, centers=10, random_state=0)
    init = np.repeat(X[:1], 10, axis=0)
    emeans = EMeans(n_clusters=10, init=init, random_state=0).fit(X)
    assert len(np.unique(emeans.cluster_centers_, axis=0)) == 10
    assert np.all(np.bincount(emeans.labels_, minlength=10) > 0)


def test_revive_dead_clusters():
    rng = np.random.RandomState(0)
    centers = rng.uniform(1, 2, size=(4, 3))
    original_centers = centers.copy()
    original_weight_ema = np.array([10.0, 0.01, 20.0, 30.0])
    weight_ema = original_weight_ema.copy()

    assert _revive_dead_clusters(centers, weight_ema, rng) == 1

    # The dead cluster 1 takes half of a donor, which keeps the other half.
    (donor,) = [j for j in (0, 2, 3) if weight_ema[j] != original_weight_ema[j]]
    assert weight_ema[1] == weight_ema[donor] == original_weight_ema[donor] / 2
    assert not np.array_equal(centers[1], centers[donor])
    assert_allclose(centers[1], original_centers[donor], rtol=1e-3)
    assert_allclose((centers[1] + centers[donor]) / 2, original_centers[donor])
    others = [j for j in range(4) if j not in (1, donor)]
    assert_array_equal(centers[others], original_centers[others])


def test_revive_dead_clusters_null_center():
    centers = np.zeros((2, 3))
    weight_ema = np.array([1.0, 0.0])
    _revive_dead_clusters(centers, weight_ema, np.random.RandomState(0))
    assert not np.array_equal(centers[0], centers[1])
    assert_array_equal(weight_ema, [0.5, 0.5])


def test_scaling_laws():
    # Values of the reference implementation (faiss/contrib/e_means).
    assert _resolve_batch_size(1_000_000, 1024, 64, 5) == 65536
    assert _resolve_batch_size(1_000_000, 1024, 16, 5) == 8192
    assert _resolve_batch_size(100, 3, 2, 5) == 1
    assert _resolve_batch_size(10, 3, 64, 5) == 1
    assert _resolve_batch_size(12345, 100, 128, 2.5) == 512
    assert _resolve_initial_step_size(1_000_000, 1024, 5, 2048) == pytest.approx(
        0.8244 * 1024**-0.4617 * 2048**0.9003 * 1e6**-0.4498 * 5**-0.1057
    )
    assert _resolve_initial_step_size(1_000_000, 1024, 5, 65536) == 1


def test_emeans_batch_size():
    X = np.random.RandomState(0).normal(size=(1000, 3))
    emeans = EMeans(n_clusters=5, random_state=0).fit(X)
    assert emeans.batch_size_ == _resolve_batch_size(1000, 5, 3, 5)


@pytest.mark.parametrize(
    "n_epochs, expected",
    [(1, [100]), (3, [100, 100, 100]), (2.5, [100, 50, 100]), (0.25, [25])],
)
def test_epochs_n_rows(n_epochs, expected):
    assert _epochs_n_rows(100, n_epochs) == expected


def test_emeans_n_samples_lower_than_n_clusters():
    with pytest.raises(ValueError, match="n_samples=3 should be >= n_clusters=4"):
        EMeans(n_clusters=4).fit(np.zeros((3, 2)))


@pytest.mark.parametrize("csr_container", [None] + CSR_CONTAINERS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n_samples", [1, 255, 1000])
@pytest.mark.parametrize("offset", [0, 100])
def test_labels_inertia_matches_reference(csr_container, dtype, n_samples, offset):
    rng = np.random.RandomState(0)
    X = (rng.normal(size=(n_samples, 7)) + offset).astype(dtype)
    centers = (rng.normal(size=(13, 7)) + offset).astype(dtype)

    labels = np.empty(n_samples, dtype=np.int32)
    if csr_container is None:
        inertia = _labels_inertia(X, centers, labels, 2)
    else:
        X[rng.uniform(size=X.shape) < 0.5] = 0
        inertia = _labels_inertia_sparse(csr_container(X), centers, labels, 2)

    X64, centers64 = X.astype(np.float64), centers.astype(np.float64)
    distances = ((X64[:, None, :] - centers64[None, :, :]) ** 2).sum(axis=2)
    # Far from the origin, the argmin of the GEMM trick can pick a near tie in
    # float32, but the inertia of the chosen centers is computed exactly.
    if offset == 0:
        assert_array_equal(labels, distances.argmin(axis=1))
    chosen_distances = distances[np.arange(n_samples), labels]
    assert inertia == pytest.approx(chosen_distances.sum(), rel=1e-6)
    assert inertia == pytest.approx(distances.min(axis=1).sum(), rel=1e-5)


def test_emeans_predict_transform_score():
    X, _ = make_blobs(n_samples=1000, n_features=5, centers=10, random_state=0)
    emeans = EMeans(n_clusters=10, random_state=0).fit(X)
    X_new = np.random.RandomState(1).normal(size=(50, 5))

    distances = ((X_new[:, None, :] - emeans.cluster_centers_) ** 2).sum(axis=2)
    assert_array_equal(emeans.predict(X_new), distances.argmin(axis=1))
    assert_allclose(emeans.transform(X_new), np.sqrt(distances))
    assert emeans.score(X_new) == pytest.approx(-distances.min(axis=1).sum())
    assert_array_equal(emeans.fit_predict(X), emeans.labels_)
    assert emeans.fit_transform(X).shape == (1000, 10)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_emeans_predict_transform_score_sparse(csr_container):
    X, _ = make_blobs(n_samples=1000, n_features=5, centers=10, random_state=0)
    emeans = EMeans(n_clusters=10, random_state=0).fit(X)
    X_new = np.random.RandomState(1).normal(size=(50, 5))
    X_new[X_new < 0] = 0
    X_new_sparse = csr_container(X_new)

    assert_array_equal(emeans.predict(X_new_sparse), emeans.predict(X_new))
    assert_allclose(emeans.transform(X_new_sparse), emeans.transform(X_new))
    assert emeans.score(X_new_sparse) == pytest.approx(emeans.score(X_new))


def test_emeans_predict_other_dtype():
    X, _ = make_blobs(n_samples=500, n_features=4, centers=5, random_state=0)
    emeans = EMeans(n_clusters=5, random_state=0).fit(X.astype(np.float32))
    assert_array_equal(emeans.predict(X), emeans.labels_)


def test_emeans_init_array_and_callable():
    X, _ = make_blobs(n_samples=1000, n_features=5, centers=5, random_state=0)
    init = X[:5].copy()
    from_array = EMeans(n_clusters=5, init=init, random_state=0).fit(X)

    def init_callable(X, n_clusters, random_state):
        return X[:n_clusters]

    from_callable = EMeans(n_clusters=5, init=init_callable, random_state=0).fit(X)
    assert_array_equal(from_array.cluster_centers_, from_callable.cluster_centers_)
    # The init array is not modified.
    assert_array_equal(init, X[:5])


@pytest.mark.parametrize(
    "params, match",
    [
        ({"n_clusters": 0}, "n_clusters must be an int >= 1"),
        ({"n_clusters": 2.5}, "n_clusters must be an int >= 1"),
        ({"n_epochs": 0}, "n_epochs must be a float > 0"),
        ({"init": "unknown"}, "init must be 'random', 'k-means\\+\\+'"),
        ({"init": np.zeros((3, 2))}, "does not match the number of clusters"),
        ({"init": np.zeros((2, 3))}, "does not match the number of features"),
    ],
)
def test_emeans_invalid_params(params, match):
    X = np.random.RandomState(0).normal(size=(20, 2))
    with pytest.raises(ValueError, match=match):
        EMeans(**{"n_clusters": 2, **params}).fit(X)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("with_labels", [False, True])
@pytest.mark.parametrize("csr_container", [None] + CSR_CONTAINERS)
def test_update_min_distances_matches_reference(dtype, with_labels, csr_container):
    rng = np.random.RandomState(0)
    X = rng.normal(size=(1000, 7)).astype(dtype)
    if csr_container is not None:
        X[rng.uniform(size=X.shape) < 0.5] = 0
    centers = rng.normal(size=(13, 7)).astype(dtype)
    # Some samples are already closer to a previous center than to the new ones.
    min_distances = rng.uniform(0, 15, size=1000).astype(dtype)
    labels = rng.randint(5, size=1000).astype(np.int32)

    distances = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    closer = distances.min(axis=1) < min_distances
    assert 0 < closer.sum() < 1000
    expected_distances = np.minimum(min_distances, distances.min(axis=1))
    expected_labels = np.where(closer, 5 + distances.argmin(axis=1), labels)

    if csr_container is None:
        update_min_distances = _update_min_distances
    else:
        update_min_distances = _update_min_distances_sparse
        X = csr_container(X)
    update_min_distances(
        X, centers, min_distances, labels if with_labels else None, 5, 2
    )
    rtol = 1e-4 if dtype == np.float32 else 1e-10
    assert_allclose(min_distances, expected_distances, rtol=rtol, atol=rtol)
    if with_labels:
        assert_array_equal(labels, expected_labels)


def test_update_min_distances_clipped_at_zero():
    # Far from the origin, the GEMM trick can give negative distances.
    X = np.full((300, 4), 1e4, dtype=np.float32) + np.arange(4, dtype=np.float32)
    min_distances = np.full(300, np.inf, dtype=np.float32)
    _update_min_distances(X, X[:1].copy(), min_distances, None, 0, 2)
    assert np.all(min_distances >= 0)


@pytest.mark.parametrize("n_clusters", [1, 5, 50])
def test_kmeans_parallel_candidates(n_clusters):
    X, _ = make_blobs(n_samples=2000, n_features=5, centers=10, random_state=0)
    candidates, weights = _kmeans_parallel_candidates(
        X, n_clusters, np.random.RandomState(0), 2
    )
    assert len(candidates) >= n_clusters
    assert len(np.unique(candidates)) == len(candidates)
    assert weights.sum() == X.shape[0]
    assert np.all(weights >= 1)
    # The weights count the samples whose nearest candidate each one is.
    distances = ((X[:, None, :] - X[candidates][None, :, :]) ** 2).sum(axis=2)
    expected = np.bincount(distances.argmin(axis=1), minlength=len(candidates))
    assert_array_equal(weights, expected)


@pytest.mark.parametrize(
    "X",
    [
        # As many samples as clusters.
        np.arange(10, dtype=np.float64).reshape(5, 2),
        # Only 3 distinct samples.
        np.repeat(np.eye(3), 100, axis=0),
        # All samples identical.
        np.ones((50, 3)),
    ],
    ids=["n_samples_eq_n_clusters", "duplicates", "constant"],
)
@pytest.mark.parametrize("csr_container", [None] + CSR_CONTAINERS)
def test_kmeans_parallel_few_distinct_samples(X, csr_container):
    n_clusters = 5
    n_distinct = min(n_clusters, len(np.unique(X, axis=0)))
    if csr_container is not None:
        X = csr_container(X)
    candidates, weights = _kmeans_parallel_candidates(
        X, n_clusters, np.random.RandomState(0), 2
    )
    assert len(candidates) >= n_clusters
    assert len(np.unique(candidates)) == len(candidates)
    assert weights.sum() == X.shape[0]

    centers = _kmeans_parallel(X, n_clusters, np.random.RandomState(0), 2)
    assert centers.shape == (n_clusters, X.shape[1])
    # All the distinct samples are centers.
    assert len(np.unique(centers, axis=0)) == n_distinct
    EMeans(n_clusters=n_clusters, init="k-means||", random_state=0).fit(X)


@pytest.mark.parametrize("csr_container", [None] + CSR_CONTAINERS)
def test_kmeans_parallel_does_not_depend_on_n_threads(csr_container):
    X, _ = make_blobs(n_samples=5000, n_features=20, centers=30, random_state=0)
    X = X.astype(np.float32)
    if csr_container is not None:
        X[np.abs(X) < 3] = 0
        X = csr_container(X)
    sequential = _kmeans_parallel(X, 30, np.random.RandomState(0), 1)
    parallel = _kmeans_parallel(X, 30, np.random.RandomState(0), 4)
    assert_array_equal(sequential, parallel)

    emeans = EMeans(n_clusters=30, init="k-means||", random_state=0)
    with threadpool_limits(limits=1, user_api="openmp"):
        centers_sequential = emeans.fit(X).cluster_centers_
    assert_array_equal(centers_sequential, emeans.fit(X).cluster_centers_)


def test_kmeans_parallel_one_center_per_blob():
    # With well-separated blobs, k-means|| puts one initial center in each
    # blob, where random init leaves some blobs without a center.
    X, y = make_blobs(
        n_samples=20_000, n_features=10, centers=100, cluster_std=0.1, random_state=0
    )
    centers = _kmeans_parallel(X, 100, np.random.RandomState(0), 2)
    distances = ((centers[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    blobs = y[distances.argmin(axis=1)]
    assert len(np.unique(blobs)) == 100

    emeans = EMeans(n_clusters=100, init="k-means||", random_state=0).fit(X)
    assert adjusted_rand_score(y, emeans.labels_) == pytest.approx(1)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_kmeans_parallel_sparse_matches_dense(dtype, csr_container):
    X, _ = make_blobs(n_samples=3000, n_features=20, centers=30, random_state=0)
    X[np.abs(X) < 3] = 0
    X = X.astype(dtype)
    X_csr = csr_container(X)

    candidates, weights = _kmeans_parallel_candidates(
        X, 30, np.random.RandomState(0), 2
    )
    candidates_csr, weights_csr = _kmeans_parallel_candidates(
        X_csr, 30, np.random.RandomState(0), 2
    )
    centers = _kmeans_parallel(X, 30, np.random.RandomState(0), 2)
    centers_csr = _kmeans_parallel(X_csr, 30, np.random.RandomState(0), 2)
    # The sparse distances round differently than the GEMM: a near tie can
    # change a label or a sampled candidate.
    if dtype == np.float64:
        assert_array_equal(candidates, candidates_csr)
        assert_array_equal(weights, weights_csr)
    else:
        assert len(np.intersect1d(candidates, candidates_csr)) > 0.9 * len(candidates)
    # The k-means++ on the candidates can break an exact tie differently, in
    # float64 too: here, two candidates of weight 50 are closer to each other
    # than to their closest center, so adding either one gives the same
    # potential. The rounding decides which one is picked, and it differs
    # between the GEMM and the sparse kernel on macOS arm64. The next steps
    # then sample other candidates.
    inertia = EMeans(n_clusters=30, init=centers).fit(X).inertia_
    inertia_csr = EMeans(n_clusters=30, init=centers_csr).fit(X).inertia_
    assert inertia_csr == pytest.approx(inertia, rel=0.01)
