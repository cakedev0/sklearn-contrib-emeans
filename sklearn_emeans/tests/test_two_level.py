"""Tests of the two-level assignment of EMeans."""

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.datasets import make_blobs
from threadpoolctl import threadpool_limits

from sklearn_emeans import EMeans, _emeans
from sklearn_emeans._emeans import _init_cells, _revive_dead_clusters
from sklearn_emeans._kernels import (
    _emeans_steps,
    _emeans_steps_two_level,
    _labels_inertia,
    _labels_inertia_two_level,
)
from sklearn_emeans.tests.test_emeans import _reference_step


def _squared_distances(X, Y):
    return ((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2)


def _reference_refresh(centers, weight_ema, cells, cell_of_center):
    """NumPy implementation of the refresh of the cells, inplace."""
    for c in range(len(cells)):
        members = cell_of_center == c
        if members.any():
            weights = np.maximum(weight_ema[members], 1e-12)
            cells[c] = (weights[:, None] * centers[members]).sum(axis=0) / weights.sum()
    cell_of_center[:] = _squared_distances(centers, cells).argmin(axis=1)


def _reference_labels(X, centers, cells, cell_of_center, max_probes, ratio):
    """NumPy implementation of the two-level assignment, once refreshed."""
    distances = _squared_distances(X, cells).astype(np.float64)
    distances[:, np.bincount(cell_of_center, minlength=len(cells)) == 0] = np.inf
    labels = np.empty(len(X), dtype=np.int32)
    for i in range(len(X)):
        nearest_cells = np.argsort(distances[i], kind="stable")[:max_probes]
        probes = [nearest_cells[0]]
        for c in nearest_cells[1:]:
            if np.isfinite(distances[i, c]) and (
                ratio == np.inf
                or distances[i, c] <= ratio**2 * distances[i, nearest_cells[0]]
            ):
                probes.append(c)
        candidates = np.flatnonzero(np.isin(cell_of_center, probes))
        labels[i] = candidates[
            _squared_distances(X[i : i + 1], centers[candidates])[0].argmin()
        ]
    return labels


def _data(dtype, n_samples=1500, n_features=6, n_clusters=40, n_cells=6, seed=0):
    """Blobs, centers among the samples, and cells from far-apart centers."""
    rng = np.random.RandomState(seed)
    blobs = 4 * rng.normal(size=(12, n_features))
    X = (
        blobs[rng.randint(12, size=n_samples)]
        + rng.normal(size=(n_samples, n_features))
    ).astype(dtype)
    centers = X[rng.choice(n_samples, size=n_clusters, replace=False)].copy()
    weight_ema = rng.uniform(0.1, 1, size=n_clusters)
    cells = centers[rng.choice(n_clusters, size=n_cells, replace=False)].copy()
    cell_of_center = _squared_distances(centers, cells).argmin(axis=1).astype(np.int32)
    return X, centers, weight_ema, cells, cell_of_center


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("max_probes, ratio", [(1, 1.15), (3, 1.3), (3, np.inf)])
@pytest.mark.parametrize("accumulate", [False, True])
def test_two_level_steps_match_reference(dtype, max_probes, ratio, accumulate):
    X, centers, weight_ema, cells, cell_of_center = _data(dtype)
    n_samples, n_features = X.shape
    n_clusters = len(centers)
    # An empty cell far from the data, and a center far from the data.
    cells[0] = 100
    cell_of_center[cell_of_center == 0] = 1
    centers[3] = -100
    batch_size, n_steps = 400, 3
    rng = np.random.RandomState(1)
    rows = rng.permutation(n_samples)[:1000].astype(np.intp)
    step_sizes = np.array([0.5, 0.3, 0.2])

    def epoch_stats():
        if not accumulate:
            return None
        return [np.ones((n_clusters, n_features)), np.ones(n_clusters)]

    ref = [a.copy() for a in (centers, weight_ema, cells, cell_of_center)]
    stats_ref = epoch_stats()
    for s in range(n_steps):
        batch = rows[s * batch_size : (s + 1) * batch_size]
        _reference_refresh(*ref)
        labels = _reference_labels(X[batch], ref[0], ref[2], ref[3], max_probes, ratio)
        _reference_step(
            X, batch, ref[0], ref[1], step_sizes[s], stats_ref, labels=labels
        )

    stats = epoch_stats() or [None, None]
    _emeans_steps_two_level(
        X,
        rows,
        batch_size,
        centers,
        weight_ema,
        step_sizes,
        *stats,
        0,
        0,
        None,
        2,
        cells,
        cell_of_center,
        max_probes,
        ratio,
    )

    rtol = 1e-4 if dtype == np.float32 else 1e-10
    assert_allclose(centers, ref[0], rtol=rtol, atol=rtol)
    assert_allclose(weight_ema, ref[1], rtol=rtol)
    # The cells are the ones of the refresh of the last step.
    assert_allclose(cells, ref[2], rtol=rtol, atol=rtol)
    assert_array_equal(cell_of_center, ref[3])
    assert np.bincount(cell_of_center, minlength=len(cells))[0] == 0
    if accumulate:
        assert_allclose(stats[0], stats_ref[0], rtol=rtol)
        assert_allclose(stats[1], stats_ref[1], rtol=rtol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("max_probes, ratio", [(1, 1.15), (4, 1.2), (6, np.inf)])
@pytest.mark.parametrize("n_samples", [1, 255, 1500])
def test_two_level_labels_inertia_match_reference(dtype, max_probes, ratio, n_samples):
    X, centers, weight_ema, cells, cell_of_center = _data(dtype)
    X = X[:n_samples]
    cells[0] = 100
    cell_of_center[cell_of_center == 0] = 1

    ref = [a.copy() for a in (centers, weight_ema, cells, cell_of_center)]
    _reference_refresh(*ref)
    labels_ref = _reference_labels(X, centers, ref[2], ref[3], max_probes, ratio)
    inertia_ref = ((X - centers[labels_ref]) ** 2).sum()

    labels = np.empty(n_samples, dtype=np.int32)
    inertia = _labels_inertia_two_level(
        X, centers, weight_ema, cells, cell_of_center, max_probes, ratio, labels, 2
    )
    assert_array_equal(labels, labels_ref)
    assert inertia == pytest.approx(inertia_ref, rel=1e-5)
    assert_array_equal(cell_of_center, ref[3])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_two_level_all_cells_probed_same_as_exact(dtype):
    # When each sample probes all the cells, it is compared to all the
    # centers: same result as the exact assignment, up to rounding.
    X, centers, weight_ema, cells, cell_of_center = _data(dtype)
    centers_exact, weight_ema_exact = centers.copy(), weight_ema.copy()
    rows = np.random.RandomState(1).permutation(len(X)).astype(np.intp)
    batch_size = 256
    n_steps = -(-len(X) // batch_size)
    step_sizes = np.linspace(0.5, 0.1, n_steps)
    stats = [np.zeros_like(centers, dtype=np.float64), np.zeros(len(centers))]
    stats_exact = [np.zeros_like(stats[0]), np.zeros_like(stats[1])]

    _emeans_steps(
        X,
        rows,
        batch_size,
        centers_exact,
        weight_ema_exact,
        step_sizes,
        *stats_exact,
        0,
        0,
        None,
        2,
    )
    _emeans_steps_two_level(
        X,
        rows,
        batch_size,
        centers,
        weight_ema,
        step_sizes,
        *stats,
        0,
        0,
        None,
        2,
        cells,
        cell_of_center,
        len(cells),
        np.inf,
    )
    rtol = 1e-4 if dtype == np.float32 else 1e-10
    assert_allclose(centers, centers_exact, rtol=rtol, atol=rtol)
    assert_allclose(weight_ema, weight_ema_exact, rtol=rtol)
    assert_allclose(stats[0], stats_exact[0], rtol=rtol, atol=rtol)
    assert_array_equal(stats[1], stats_exact[1])

    labels = np.empty(len(X), dtype=np.int32)
    labels_exact = np.empty(len(X), dtype=np.int32)
    inertia_exact = _labels_inertia(X, centers, labels_exact, 2)
    inertia = _labels_inertia_two_level(
        X, centers, weight_ema, cells, cell_of_center, len(cells), np.inf, labels, 2
    )
    assert_array_equal(labels, labels_exact)
    assert inertia == pytest.approx(inertia_exact, rel=1e-12)


def test_two_level_ties_go_to_lowest_index():
    # Duplicated centers: as with the exact assignment, a sample goes to the
    # one of lowest index.
    rng = np.random.RandomState(0)
    X = rng.normal(size=(600, 4))
    centers = np.concatenate([X[:5], X[:5]])
    cells = X[5:8].copy()
    cell_of_center = np.zeros(10, dtype=np.int32)
    labels = np.empty(len(X), dtype=np.int32)
    labels_exact = np.empty(len(X), dtype=np.int32)

    _labels_inertia_two_level(
        X, centers, np.ones(10), cells, cell_of_center, 3, np.inf, labels, 2
    )
    _labels_inertia(X, centers, labels_exact, 2)
    assert_array_equal(labels, labels_exact)
    assert labels.max() < 5


@pytest.mark.parametrize("n_threads", [3, 8])
def test_two_level_does_not_depend_on_n_threads(n_threads):
    X, centers, weight_ema, cells, cell_of_center = _data(
        np.float32, n_samples=6000, n_clusters=150, n_cells=12
    )
    rows = np.random.RandomState(1).permutation(len(X)).astype(np.intp)
    step_sizes = np.linspace(0.5, 0.1, 4)

    def fit(n_threads):
        state = [a.copy() for a in (centers, weight_ema, cells, cell_of_center)]
        labels = np.empty(len(X), dtype=np.int32)
        with threadpool_limits(limits=1, user_api="blas"):
            _emeans_steps_two_level(
                X,
                rows,
                1500,
                state[0],
                state[1],
                step_sizes,
                None,
                None,
                0,
                0,
                None,
                n_threads,
                state[2],
                state[3],
                4,
                1.15,
            )
            inertia = _labels_inertia_two_level(
                X, state[0], state[1], state[2], state[3], 4, 1.15, labels, n_threads
            )
        return state + [labels, inertia]

    for result, expected in zip(fit(n_threads), fit(1)):
        assert_array_equal(result, expected)


def _fit_two_level(X, monkeypatch, **params):
    """Fit with the two-level assignment whatever the number of clusters."""
    monkeypatch.setattr(_emeans, "_TWO_LEVEL_MIN_CLUSTERS", 1)
    return EMeans(assignment="two-level", **params).fit(X)


def test_emeans_two_level_does_not_depend_on_n_threads(monkeypatch):
    X, _ = make_blobs(n_samples=5000, n_features=20, centers=30, random_state=0)
    with threadpool_limits(limits=1, user_api="openmp"):
        sequential = _fit_two_level(X, monkeypatch, n_clusters=100, random_state=0)
    parallel = _fit_two_level(X, monkeypatch, n_clusters=100, random_state=0)
    assert_array_equal(sequential.cluster_centers_, parallel.cluster_centers_)
    assert_array_equal(sequential.labels_, parallel.labels_)
    assert sequential.inertia_ == parallel.inertia_


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("init", ["random", "k-means++", "k-means||"])
def test_emeans_two_level_close_to_exact(dtype, init, monkeypatch):
    # Overlapping blobs: on separated blobs, the inertia depends too much on
    # the random numbers, which the two assignments don't draw in the same
    # order.
    rng = np.random.RandomState(0)
    X = (
        rng.normal(size=(20_000, 16))
        + 2 * rng.normal(size=(30, 16))[rng.randint(30, size=20_000)]
    )
    X = X.astype(dtype)
    exact = EMeans(n_clusters=300, init=init, random_state=0).fit(X)
    two_level = _fit_two_level(
        X, monkeypatch, n_clusters=300, init=init, random_state=0
    )
    assert two_level.cluster_centers_.dtype == dtype
    assert two_level.labels_.dtype == np.int32
    assert two_level.inertia_ < 1.01 * exact.inertia_

    # labels_ and inertia_ are the ones of the two-level assignment: almost all
    # the samples have their nearest center, and predict is exact.
    labels = two_level.predict(X)
    assert np.mean(two_level.labels_ == labels) > 0.95
    assert two_level.inertia_ >= -two_level.score(X) * (1 - 1e-6)
    assert two_level.inertia_ == pytest.approx(
        ((X - two_level.cluster_centers_[two_level.labels_]) ** 2).sum(), rel=1e-5
    )


def test_emeans_two_level_few_clusters_is_exact():
    # Below _TWO_LEVEL_MIN_CLUSTERS, the assignment is exact: same result and
    # same random numbers as assignment="exact".
    X, _ = make_blobs(n_samples=2000, n_features=5, centers=10, random_state=0)
    exact = EMeans(n_clusters=10, random_state=0).fit(X)
    two_level = EMeans(n_clusters=10, assignment="two-level", random_state=0).fit(X)
    assert_array_equal(two_level.cluster_centers_, exact.cluster_centers_)
    assert_array_equal(two_level.labels_, exact.labels_)
    assert two_level.inertia_ == exact.inertia_


@pytest.mark.parametrize("n_clusters", [1, 2, 3, 5])
def test_emeans_two_level_tiny_n_clusters(n_clusters, monkeypatch):
    X, _ = make_blobs(n_samples=500, n_features=5, centers=5, random_state=0)
    emeans = _fit_two_level(X, monkeypatch, n_clusters=n_clusters, random_state=0)
    exact = EMeans(n_clusters=n_clusters, random_state=0).fit(X)
    assert emeans.cluster_centers_.shape == (n_clusters, 5)
    assert emeans.inertia_ < 1.5 * exact.inertia_
    if n_clusters <= 2:
        # A single cell: each sample is compared to all the centers.
        assert_array_equal(emeans.labels_, emeans.predict(X))


@pytest.mark.parametrize("n_clusters", [150, 299, 300])
def test_emeans_two_level_n_clusters_close_to_n_samples(n_clusters, monkeypatch):
    X, _ = make_blobs(n_samples=300, n_features=5, centers=20, random_state=0)
    emeans = _fit_two_level(X, monkeypatch, n_clusters=n_clusters, random_state=0)
    assert np.isfinite(emeans.cluster_centers_).all()
    assert emeans.labels_.min() >= 0 and emeans.labels_.max() < n_clusters
    assert emeans.inertia_ == pytest.approx(
        ((X - emeans.cluster_centers_[emeans.labels_]) ** 2).sum()
    )


def test_emeans_two_level_few_distinct_samples(monkeypatch):
    # Duplicated centers, and cells that end up empty.
    rng = np.random.RandomState(0)
    X = rng.normal(size=(4, 5))[rng.randint(4, size=1000)]
    emeans = _fit_two_level(X, monkeypatch, n_clusters=50, revival=0, random_state=0)
    assert np.isfinite(emeans.cluster_centers_).all()
    assert emeans.inertia_ == pytest.approx(0, abs=1e-8)


def test_emeans_two_level_revives_dead_clusters(monkeypatch):
    # All the initial centers are the same point: only the first one gets
    # samples, and the others are revived by splitting it.
    X, _ = make_blobs(n_samples=5000, n_features=5, centers=10, random_state=0)
    init = np.repeat(X[:1], 30, axis=0)
    emeans = _fit_two_level(X, monkeypatch, n_clusters=30, init=init, random_state=0)
    assert len(np.unique(emeans.cluster_centers_, axis=0)) == 30
    assert np.all(np.bincount(emeans.labels_, minlength=30) > 0)


def test_revive_dead_clusters_moves_cells():
    # A revived center goes to the cell of its donor.
    centers = np.arange(12, dtype=np.float64).reshape(4, 3)
    weight_ema = np.array([1.0, 1e-9, 1.0, 5.0])
    cell_of_center = np.array([0, 1, 2, 3], dtype=np.int32)
    n_revived = _revive_dead_clusters(
        centers, weight_ema, np.random.RandomState(0), 0.1, cell_of_center
    )
    assert n_revived == 1
    donor = np.flatnonzero(weight_ema == weight_ema[1])
    donor = donor[donor != 1][0]
    assert cell_of_center[1] == cell_of_center[donor] == donor


def test_emeans_two_level_greedy_revival(monkeypatch):
    X, _ = make_blobs(n_samples=5000, n_features=5, centers=40, random_state=0)
    emeans = _fit_two_level(
        X, monkeypatch, n_clusters=40, revival="greedy", random_state=0
    )
    exact = EMeans(n_clusters=40, revival="greedy", random_state=0).fit(X)
    assert emeans.inertia_ < 1.1 * exact.inertia_


def test_two_level_steps_propagate_revive_error():
    X, centers, weight_ema, cells, cell_of_center = _data(np.float64)
    n_calls = []

    def revive():
        n_calls.append(1)
        raise ValueError("revival failed")

    with pytest.raises(ValueError, match="revival failed"):
        _emeans_steps_two_level(
            X,
            np.arange(len(X), dtype=np.intp),
            100,
            centers,
            weight_ema,
            np.full(15, 0.1),
            None,
            None,
            0,
            2,
            revive,
            4,
            cells,
            cell_of_center,
            4,
            1.15,
        )
    # The steps stop at the first error.
    assert len(n_calls) == 1


def test_init_cells():
    X, _ = make_blobs(n_samples=400, n_features=5, centers=20, random_state=0)
    X = X.astype(np.float32)
    cells, cell_of_center = _init_cells(X, np.random.RandomState(0))
    assert cells.shape == (20, 5)
    assert cells.dtype == np.float32
    assert cell_of_center.dtype == np.int32
    assert_array_equal(
        cell_of_center, _squared_distances(X, cells.astype(np.float64)).argmin(axis=1)
    )


def test_emeans_two_level_sparse_error():
    X = sp.csr_matrix(np.eye(10))
    with pytest.raises(ValueError, match="not supported for sparse X"):
        EMeans(n_clusters=2, assignment="two-level").fit(X)


@pytest.mark.parametrize("assignment", ["approximate", None, 2])
def test_emeans_invalid_assignment(assignment):
    X = np.eye(10)
    with pytest.raises(ValueError, match="assignment must be"):
        EMeans(n_clusters=2, assignment=assignment).fit(X)
