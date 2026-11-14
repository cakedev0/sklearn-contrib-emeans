import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from sklearn_emeans import EMeans, _emeans
from sklearn_emeans._emeans import (
    _epochs_n_rows,
    _greedy_revival_period,
    _greedy_segments,
    _resolve_batch_size,
    _resolve_initial_step_size,
    _revive_dead_clusters,
)
from sklearn_emeans._greedy_revival import GreedyRevival, _pair
from sklearn_emeans._kernels import (
    _emeans_steps,
    _emeans_steps_sparse,
    _greedy_stats,
    _greedy_stats_sparse,
    _labels_inertia,
    _labels_inertia_sparse,
    _trial_splits,
    _trial_splits_sparse,
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
    steps = _emeans_steps if csr_container is None else _emeans_steps_sparse
    steps(
        X,
        batch_indices,
        batch_size,
        centers,
        weight_ema,
        np.array([0.3]),
        *stats,
        0,
        0,
        None,
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
    _emeans_steps(
        X,
        batch_indices,
        200,
        centers,
        np.full(4, 1e-5),
        np.array([1.0]),
        None,
        None,
        0,
        0,
        None,
        1,
    )
    expected = np.array([X[labels == j].mean(axis=0) for j in range(4)])
    assert_allclose(centers, expected)


@pytest.mark.parametrize("csr_container", [None] + CSR_CONTAINERS)
@pytest.mark.parametrize("n_threads", [1, 4])
def test_emeans_steps_same_as_one_call_per_step(csr_container, n_threads):
    # Several steps in one call, with revivals in between, give the same
    # centers as one call per step. The last batch is smaller.
    rng = np.random.RandomState(0)
    n_samples, n_features, n_clusters, batch_size = 3000, 5, 11, 400
    X = rng.normal(size=(n_samples, n_features))
    if csr_container is not None:
        X[X < 0] = 0
        X = csr_container(X)
    steps = _emeans_steps if csr_container is None else _emeans_steps_sparse
    rows = rng.permutation(n_samples).astype(np.intp)
    n_steps = 8
    step_sizes = np.linspace(0.5, 0.1, n_steps)
    first_step, revival_period = 4, 3

    def fit(one_call):
        centers = rng_centers.copy()
        weight_ema = np.full(n_clusters, 0.5)
        revived_at = []

        def revive():
            # Stands for the revival: changes the centers and the weights.
            revived_at.append(centers[0, 0])
            centers[len(revived_at) % n_clusters] += 0.25
            weight_ema[0] *= 2

        if one_call:
            steps(
                X,
                rows,
                batch_size,
                centers,
                weight_ema,
                step_sizes,
                None,
                None,
                first_step,
                revival_period,
                revive,
                n_threads,
            )
        else:
            for s in range(n_steps):
                batch = rows[s * batch_size : (s + 1) * batch_size]
                steps(
                    X,
                    batch,
                    batch_size,
                    centers,
                    weight_ema,
                    step_sizes[s : s + 1],
                    None,
                    None,
                    0,
                    0,
                    None,
                    n_threads,
                )
                if (first_step + s + 1) % revival_period == 0:
                    revive()
        return centers, weight_ema, revived_at

    rng_centers = rng.normal(size=(n_clusters, n_features))
    centers, weight_ema, revived_at = fit(one_call=True)
    centers_ref, weight_ema_ref, revived_at_ref = fit(one_call=False)

    # Revivals after the steps 6, 9 and 12 of the fit.
    assert len(revived_at) == 3
    assert_array_equal(revived_at, revived_at_ref)
    assert_array_equal(centers, centers_ref)
    assert_array_equal(weight_ema, weight_ema_ref)


def test_emeans_steps_propagates_revive_error():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(500, 3))
    centers = X[:4].copy()
    n_calls = []

    def revive():
        n_calls.append(1)
        raise ValueError("revival failed")

    with pytest.raises(ValueError, match="revival failed"):
        _emeans_steps(
            X,
            np.arange(500, dtype=np.intp),
            50,
            centers,
            np.ones(4),
            np.full(10, 0.1),
            None,
            None,
            0,
            2,
            revive,
            4,
        )
    # The steps stop at the first error.
    assert len(n_calls) == 1


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


def test_revive_dead_clusters_threshold():
    # The cluster 1 has 5% of the mean weight.
    centers = np.random.RandomState(0).uniform(1, 2, size=(4, 3))
    weight_ema = np.array([10.0, 0.5, 14.5, 15.0])
    rng = np.random.RandomState(0)
    assert _revive_dead_clusters(centers, weight_ema.copy(), rng, 0.01) == 0
    assert _revive_dead_clusters(centers, weight_ema.copy(), rng, 0.1) == 1


def _big_and_tiny_blobs(dtype=np.float32):
    """20 blobs of 1000 samples and 10 far blobs of 15 samples."""
    rng = np.random.RandomState(0)
    X, _ = make_blobs(n_samples=20_000, n_features=8, centers=20, random_state=0)
    tiny_centers = rng.uniform(-60, 60, size=(10, 8))
    X_tiny = np.concatenate([c + rng.normal(size=(15, 8)) for c in tiny_centers])
    X = np.concatenate([X, X_tiny]).astype(dtype)
    return X[rng.permutation(X.shape[0])]


@pytest.mark.parametrize("revival", [0, "greedy"])
def test_emeans_revival_keeps_small_clusters(revival):
    # The default revival removes the clusters with less than a tenth of the
    # mean cluster size, here the tiny blobs, which costs a lot of inertia.
    X = _big_and_tiny_blobs()
    params = dict(n_clusters=30, init="k-means||", random_state=0)
    default = EMeans(**params).fit(X)
    emeans = EMeans(revival=revival, **params).fit(X)
    assert np.bincount(default.labels_).min() > 100
    assert np.bincount(emeans.labels_).min() == 15
    assert emeans.inertia_ < 0.2 * default.inertia_


def test_emeans_revival_zero_never_revives(monkeypatch):
    def revive(*args):
        raise AssertionError("revival called")

    monkeypatch.setattr(_emeans, "_revive_dead_clusters", revive)
    X, _ = make_blobs(n_samples=5000, n_features=5, centers=10, random_state=0)
    EMeans(n_clusters=10, revival=0, random_state=0).fit(X)
    with pytest.raises(AssertionError, match="revival called"):
        EMeans(n_clusters=10, random_state=0).fit(X)


@pytest.mark.parametrize("init", ["random", "k-means||"])
def test_emeans_greedy_revival_from_random_init(init):
    # With a random init, some tiny blobs have no center and some big ones
    # have several: the greedy revival moves these centers, where a revival
    # based on the weights of the clusters can't.
    X = _big_and_tiny_blobs()
    params = dict(n_clusters=30, init=init, random_state=0)
    greedy = EMeans(revival="greedy", **params).fit(X)
    assert np.bincount(greedy.labels_).min() == 15
    for revival in (0.1, 0.01):
        assert greedy.inertia_ < 0.5 * EMeans(revival=revival, **params).fit(X).inertia_


def test_emeans_greedy_revival_revives_dead_clusters():
    X, _ = make_blobs(n_samples=5000, n_features=5, centers=10, random_state=0)
    init = np.repeat(X[:1], 10, axis=0)
    emeans = EMeans(n_clusters=10, init=init, revival="greedy", random_state=0).fit(X)
    assert len(np.unique(emeans.cluster_centers_, axis=0)) == 10
    assert np.bincount(emeans.labels_, minlength=10).min() > 100


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_emeans_greedy_revival_sparse_same_as_dense(csr_container, dtype):
    X = _big_and_tiny_blobs(dtype)
    params = dict(n_clusters=30, revival="greedy", random_state=0)
    dense = EMeans(**params).fit(X)
    sparse = EMeans(**params).fit(csr_container(X))
    assert_allclose(
        sparse.cluster_centers_, dense.cluster_centers_, rtol=1e-4, atol=1e-4
    )
    assert_array_equal(sparse.labels_, dense.labels_)


def test_emeans_greedy_revival_independent_of_n_threads(monkeypatch):
    X = _big_and_tiny_blobs()
    params = dict(n_clusters=30, revival="greedy", random_state=0)
    expected = EMeans(**params).fit(X).cluster_centers_
    monkeypatch.setattr(_emeans, "_openmp_effective_n_threads", lambda: 1)
    assert_array_equal(EMeans(**params).fit(X).cluster_centers_, expected)


@pytest.mark.parametrize("n_clusters", [1, 2])
def test_emeans_greedy_revival_few_clusters(n_clusters):
    X, _ = make_blobs(n_samples=500, n_features=3, centers=3, random_state=0)
    emeans = EMeans(n_clusters=n_clusters, revival="greedy", random_state=0).fit(X)
    assert emeans.cluster_centers_.shape == (n_clusters, 3)
    assert np.isfinite(emeans.cluster_centers_).all()


def test_greedy_revival_period():
    # Throughput law: every 5 steps, as the revival of the dead clusters.
    batch_size = _resolve_batch_size(1_000_000, 1024, 64, 5)
    assert _greedy_revival_period(1_000_000, 1024, 5, batch_size) == 5
    # Quality law: 8x smaller batches, so 8x more steps between two revivals.
    batch_size = _resolve_batch_size(1_000_000, 1024, 16, 5)
    assert _greedy_revival_period(1_000_000, 1024, 5, batch_size) == 40
    assert _greedy_revival_period(10, 3, 5, 1) == 5


def test_greedy_segments():
    X = np.zeros((10, 2))
    # 40 steps: revivals after the steps 5, 10, 15 and 20, in the first half.
    greedy = GreedyRevival(X, n_steps=40, period=5, n_threads=1)
    assert list(_greedy_segments(greedy, 0, 12)) == [(0, 5), (5, 10), (10, 12)]
    assert list(_greedy_segments(greedy, 12, 12)) == [(0, 3), (3, 8), (8, 12)]
    assert list(_greedy_segments(greedy, 24, 16)) == [(0, 16)]
    assert list(_greedy_segments(None, 0, 12)) == [(0, 12)]


def test_greedy_revival_replaces_useless_cluster():
    # Two centers in the first blob, one in the middle of the two others: the
    # cheapest cluster of the first blob is replaced by a half of the last
    # cluster, split between its two blobs.
    rng = np.random.RandomState(0)
    blobs = np.array([[0.0, 0.0], [20.0, 0.0], [30.0, 0.0]])
    X = np.concatenate([blob + rng.normal(size=(200, 2)) for blob in blobs])
    centers = np.array([[-0.5, 0.0], [0.5, 0.0], [25.0, 0.0]])
    weight_ema = np.array([1.0, 2.0, 4.0])

    assert GreedyRevival(X, n_steps=10, period=5, n_threads=2)(centers, weight_ema) == 1

    order = np.argsort(centers[:, 0])
    assert_allclose(centers[order], blobs, atol=0.5)
    assert_array_equal(np.sort(weight_ema), [1.0, 2.0, 2.0])
    assert weight_ema[order[0]] != 2.0

    # Nothing left to gain.
    expected = centers.copy()
    assert GreedyRevival(X, n_steps=10, period=5, n_threads=2)(centers, weight_ema) == 0
    assert_array_equal(centers, expected)


@pytest.mark.parametrize("csr_container", [None] + CSR_CONTAINERS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n_threads", [1, 3])
def test_greedy_stats(csr_container, dtype, n_threads):
    rng = np.random.RandomState(0)
    X = (rng.normal(size=(700, 6)) * [1, 3, 1, 1, 2, 1] + 5).astype(dtype)
    # The last center gets no row.
    centers = np.concatenate([X[:9], np.full((1, 6), 100.0)]).astype(dtype)
    n_clusters = centers.shape[0]

    distances = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    expected_labels = distances.argmin(axis=1)
    sorted_distances = np.sort(distances.astype(np.float64), axis=1)
    expected_cost = np.bincount(
        expected_labels,
        weights=sorted_distances[:, 1] - sorted_distances[:, 0],
        minlength=n_clusters,
    )
    expected_scatter = np.zeros((n_clusters, 6))
    for j in range(n_clusters - 1):
        rows = X[expected_labels == j].astype(np.float64)
        expected_scatter[j] = ((rows - rows.mean(axis=0)) ** 2).sum(axis=0)

    labels = np.empty(700, dtype=np.int32)
    order = np.empty(700, dtype=np.intp)
    cluster_starts = np.empty(n_clusters + 1, dtype=np.intp)
    cost, max_scatter = np.empty(n_clusters), np.empty(n_clusters)
    feature = np.empty(n_clusters, dtype=np.intp)
    greedy_stats = _greedy_stats if csr_container is None else _greedy_stats_sparse
    greedy_stats(
        X if csr_container is None else csr_container(X),
        centers,
        labels,
        order,
        cluster_starts,
        cost,
        max_scatter,
        feature,
        n_threads,
    )

    assert_array_equal(labels, expected_labels)
    assert_array_equal(np.diff(cluster_starts), np.bincount(labels, minlength=10))
    for j in range(n_clusters):
        assert_array_equal(
            order[cluster_starts[j] : cluster_starts[j + 1]],
            np.flatnonzero(labels == j),
        )
    rtol = 1e-3 if dtype == np.float32 else 1e-9
    assert_allclose(cost, expected_cost, rtol=rtol)
    assert_allclose(max_scatter, expected_scatter.max(axis=1), rtol=rtol)
    assert_array_equal(feature[:-1], expected_scatter.argmax(axis=1)[:-1])
    assert cost[-1] == max_scatter[-1] == 0


@pytest.mark.parametrize("csr_container", [None] + CSR_CONTAINERS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_trial_splits(csr_container, dtype):
    # Two clusters: the second one is made of two groups, 10 apart along the
    # feature 1.
    rng = np.random.RandomState(0)
    X = (rng.normal(size=(600, 3)) + 1).astype(dtype)
    X[200 + np.flatnonzero(rng.uniform(size=400) < 0.5), 1] += 10
    order = rng.permutation(600).astype(np.intp)
    order = np.concatenate([np.sort(order[order < 200]), np.sort(order[order >= 200])])
    cluster_starts = np.array([0, 200, 600], dtype=np.intp)

    rows = X[200:].astype(np.float64)
    groups = rows[:, 1] > 6
    inertia = ((rows - rows.mean(axis=0)) ** 2).sum()
    split_inertia = sum(
        ((rows[mask] - rows[mask].mean(axis=0)) ** 2).sum()
        for mask in (groups, ~groups)
    )

    gains = np.empty(2)
    split_centers = np.empty((2, 2, 3))
    args = (
        order,
        cluster_starts,
        np.array([1, 0], dtype=np.intp),
        np.array([0, 1], dtype=np.intp),
        np.array([1.0, rows[:, 1].std()]),
        3,
        gains,
        split_centers,
        2,
    )
    if csr_container is None:
        _trial_splits(X, *args)
    else:
        X_csr = csr_container(X)
        _trial_splits_sparse(X_csr.data, X_csr.indices, X_csr.indptr, 3, *args)

    # The split of the second cluster separates its two groups.
    assert gains[0] == pytest.approx(inertia - split_inertia, rel=0.05)
    assert_allclose(split_centers[0, :, 1], [11, 1], atol=0.3)
    assert_allclose(split_centers[0][:, [0, 2]], 1, atol=0.3)
    # Not much to gain from the split of a Gaussian blob in 3 dimensions.
    assert 0 < gains[1] < 0.5 * 200 * 3


def test_pair():
    # The clusters 3 and 0 are the cheapest to remove, 2 and 1 the best to
    # split. The second pair is not worth it.
    removed, donors = _pair(
        cost=np.array([2.0, 9.0, 8.0, 1.0]), gain=np.array([0.0, 1.5, 5.0, 0.0])
    )
    assert_array_equal(removed, [3])
    assert_array_equal(donors, [2])

    # A cluster is not its own donor, and is in one pair at most.
    removed, donors = _pair(cost=np.array([0.0, 1.0]), gain=np.array([5.0, 4.0]))
    assert_array_equal(removed, [0])
    assert_array_equal(donors, [1])

    removed, donors = _pair(cost=np.zeros(3), gain=np.zeros(3))
    assert len(removed) == len(donors) == 0


@pytest.mark.parametrize("revival", [-0.1, 0.6, "dead", None, True])
def test_emeans_invalid_revival(revival):
    X = np.zeros((10, 2))
    with pytest.raises(ValueError, match="revival must be"):
        EMeans(n_clusters=2, revival=revival).fit(X)


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
