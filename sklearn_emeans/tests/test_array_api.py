"""Tests of EMeans on array API inputs.

They need `SCIPY_ARRAY_API=1` in the environment, and run
on the namespaces and devices that are available: array-api-strict, PyTorch
(CPU, CUDA, MPS, XPU) and CuPy.
"""

import numpy as np
import pytest
import sklearn
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.datasets import make_blobs

from sklearn_emeans import EMeans, _array_api, _emeans
from sklearn_emeans._emeans import _revive_dead_clusters
from sklearn_emeans.tests.test_emeans import _big_and_tiny_blobs, _reference_step
from sklearn_emeans.tests.test_two_level import (
    _data,
    _reference_labels,
    _reference_refresh,
)

try:
    with sklearn.config_context(array_api_dispatch=True):
        pass
except Exception as exc:  # pragma: no cover
    pytest.skip(f"array API dispatch can't be enabled: {exc}", allow_module_level=True)


def _namespaces_and_devices():
    """The (namespace name, device) pairs that are available."""
    params = []
    try:
        import array_api_strict  # noqa: F401

        params.append(("array_api_strict", None))
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        params.append(("torch", "cpu"))
        if torch.cuda.is_available():  # pragma: no cover
            params.append(("torch", "cuda"))
        if torch.backends.mps.is_available():  # pragma: no cover
            params.append(("torch", "mps"))
        if hasattr(torch, "xpu") and torch.xpu.is_available():  # pragma: no cover
            params.append(("torch", "xpu"))
    except ImportError:  # pragma: no cover
        pass
    try:  # pragma: no cover
        import cupy

        if cupy.cuda.is_available():
            params.append(("cupy", None))
    except ImportError:
        pass
    return [pytest.param(*param, id="-".join(filter(None, param))) for param in params]


pytestmark = pytest.mark.parametrize("xp_name, device", _namespaces_and_devices())


def _xp(xp_name):
    return _array_api.array_api_compat.array_namespace(
        pytest.importorskip(xp_name).asarray(np.zeros(1, dtype=np.float32))
    )


def _to_device(xp_name, device, array):
    return _xp(xp_name).asarray(np.ascontiguousarray(array), device=device)


def _to_numpy(xp_name, array):
    return _array_api.to_numpy(_xp(xp_name), array)


def _has_float64(xp_name, device):
    xp = _xp(xp_name)
    probe = xp.asarray(np.zeros(1, dtype=np.float32), device=device)
    return _array_api.accumulator_dtype(xp, _array_api.device(probe)) == "float64"


def _is_cuda(xp_name, device):
    return (xp_name, device) in (("torch", "cuda"), ("cupy", None))


def _blobs(dtype=np.float32, n_samples=3000, n_features=10, centers=20):
    X, _ = make_blobs(
        n_samples=n_samples, n_features=n_features, centers=centers, random_state=0
    )
    return X.astype(dtype)


@pytest.mark.parametrize("init", ["random", "k-means||", "k-means++"])
@pytest.mark.parametrize("n_epochs", [0.5, 2.5, 5])
def test_fit_float64_same_as_numpy(xp_name, device, init, n_epochs):
    # Same random numbers and same computations as with the Cython kernels:
    # in float64, the results only differ by the rounding errors.
    if not _has_float64(xp_name, device):
        pytest.skip("no float64 on this device")
    X = _blobs(np.float64)
    params = dict(n_clusters=20, init=init, n_epochs=n_epochs, random_state=0)
    expected = EMeans(**params).fit(X)
    with sklearn.config_context(array_api_dispatch=True):
        emeans = EMeans(**params).fit(_to_device(xp_name, device, X))

    assert_allclose(
        _to_numpy(xp_name, emeans.cluster_centers_),
        expected.cluster_centers_,
        rtol=1e-9,
        atol=1e-9,
    )
    assert_array_equal(_to_numpy(xp_name, emeans.labels_), expected.labels_)
    assert emeans.inertia_ == pytest.approx(expected.inertia_, rel=1e-9)
    assert emeans.batch_size_ == expected.batch_size_
    assert emeans.n_features_in_ == expected.n_features_in_


def test_fit_float64_revival_same_as_numpy(xp_name, device):
    # Duplicated initial centers: all but one of each group are dead after the
    # first steps and are revived.
    if not _has_float64(xp_name, device):
        pytest.skip("no float64 on this device")
    X = _blobs(np.float64)
    init = np.repeat(X[:5], 4, axis=0)
    params = dict(n_clusters=20, init=init, random_state=0)
    expected = EMeans(**params).fit(X)
    assert len(np.unique(expected.cluster_centers_, axis=0)) == 20
    with sklearn.config_context(array_api_dispatch=True):
        emeans = EMeans(**params).fit(_to_device(xp_name, device, X))
    assert_allclose(
        _to_numpy(xp_name, emeans.cluster_centers_),
        expected.cluster_centers_,
        rtol=1e-9,
        atol=1e-9,
    )


@pytest.mark.parametrize("revival", [0, 0.01, "greedy"])
def test_fit_float64_revival_param_same_as_numpy(xp_name, device, revival):
    if not _has_float64(xp_name, device):
        pytest.skip("no float64 on this device")
    X = _big_and_tiny_blobs(np.float64)
    params = dict(n_clusters=30, revival=revival, random_state=0)
    expected = EMeans(**params).fit(X)
    with sklearn.config_context(array_api_dispatch=True):
        emeans = EMeans(**params).fit(_to_device(xp_name, device, X))
    assert_allclose(
        _to_numpy(xp_name, emeans.cluster_centers_),
        expected.cluster_centers_,
        rtol=1e-9,
        atol=1e-9,
    )


def test_fit_float32_greedy_revival(xp_name, device):
    X = _big_and_tiny_blobs(np.float32)
    params = dict(n_clusters=30, init="k-means||", random_state=0)
    X_xp = _to_device(xp_name, device, X)
    with sklearn.config_context(array_api_dispatch=True):
        default = EMeans(**params).fit(X_xp)
        greedy = EMeans(revival="greedy", **params).fit(X_xp)
    assert greedy.cluster_centers_.dtype == X_xp.dtype
    assert _array_api.device(greedy.cluster_centers_) == _array_api.device(X_xp)
    assert greedy.inertia_ < 0.2 * default.inertia_


@pytest.mark.parametrize("init", ["random", "k-means||"])
def test_fit_float32(xp_name, device, init):
    xp = _xp(xp_name)
    X = _blobs(np.float32)
    X_xp = _to_device(xp_name, device, X)
    params = dict(n_clusters=20, init=init, random_state=0)
    expected = EMeans(**params).fit(X)
    with sklearn.config_context(array_api_dispatch=True):
        emeans = EMeans(**params).fit(X_xp)
        labels = emeans.predict(X_xp)
        score = emeans.score(X_xp)

    # The outputs are in the namespace and on the device of the input.
    for array in (emeans.cluster_centers_, emeans.labels_, labels):
        assert _array_api.array_api_compat.array_namespace(array) is xp
        assert _array_api.device(array) == _array_api.device(X_xp)
    assert emeans.cluster_centers_.dtype == xp.float32
    assert emeans.cluster_centers_.shape == (20, 10)
    assert emeans.labels_.dtype == xp.int32
    assert isinstance(emeans.inertia_, float)

    # In float32, a row at about the same distance of two centers can go to
    # either of them: the centers are not exactly the same.
    assert emeans.inertia_ == pytest.approx(expected.inertia_, rel=0.02)
    centers = _to_numpy(xp_name, emeans.cluster_centers_)
    labels_ = _to_numpy(xp_name, emeans.labels_)
    assert_array_equal(labels_, _to_numpy(xp_name, labels))
    assert score == pytest.approx(-emeans.inertia_, rel=1e-5)
    distances = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    assert np.mean(labels_ == distances.argmin(axis=1)) > 0.999
    assert emeans.inertia_ == pytest.approx(distances.min(axis=1).sum(), rel=1e-4)


def test_steps_match_reference(xp_name, device):
    xp = _xp(xp_name)
    rng = np.random.RandomState(0)
    n_samples, n_features, n_clusters, batch_size = 1000, 7, 13, 300
    X = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    centers = rng.normal(size=(n_clusters, n_features)).astype(np.float32)
    # A center far from the data gets no row.
    centers[0] = 100
    weight_ema = rng.uniform(size=n_clusters)
    rows = rng.permutation(n_samples)[:700]
    step_sizes = [0.5, 0.3, 0.2]

    centers_ref, weight_ema_ref = centers.astype(np.float64), weight_ema.copy()
    stats_ref = [np.zeros((n_clusters, n_features)), np.zeros(n_clusters)]
    for s, step_size in enumerate(step_sizes):
        _reference_step(
            X.astype(np.float64),
            rows[s * batch_size : (s + 1) * batch_size],
            centers_ref,
            weight_ema_ref,
            step_size,
            stats_ref,
        )

    X_xp = _to_device(xp_name, device, X)
    dev = _array_api.device(X_xp)
    acc_dtype = _array_api.accumulator_dtype(xp, dev)
    revived = []

    def revive(centers, weight_ema):
        revived.append(None)
        return centers, weight_ema

    new_centers, new_weight_ema, epoch_sums, epoch_counts = _array_api.emeans_steps(
        xp,
        X_xp,
        _array_api.index_array(xp, rows, dev),
        batch_size,
        _array_api.from_numpy(xp, centers, "float32", dev),
        _array_api.from_numpy(xp, weight_ema, acc_dtype, dev),
        step_sizes,
        _array_api.from_numpy(xp, stats_ref[0] * 0, acc_dtype, dev),
        _array_api.from_numpy(xp, stats_ref[1] * 0, acc_dtype, dev),
        first_step=1,
        revival_period=2,
        revive=revive,
    )
    assert_allclose(_to_numpy(xp_name, new_centers), centers_ref, rtol=1e-4, atol=1e-5)
    assert_allclose(_to_numpy(xp_name, new_weight_ema), weight_ema_ref, rtol=1e-5)
    assert_allclose(_to_numpy(xp_name, epoch_sums), stats_ref[0], rtol=1e-4, atol=1e-4)
    assert_array_equal(_to_numpy(xp_name, epoch_counts), stats_ref[1])
    # One revival every 2 steps: after the steps 1 and 3, the first one being
    # the step 1.
    assert len(revived) == 2


def test_revive_dead_clusters_same_as_numpy(xp_name, device):
    xp = _xp(xp_name)
    rng = np.random.RandomState(0)
    centers = rng.normal(size=(12, 5)).astype(np.float32)
    # A null center among the donors.
    centers[7] = 0
    weight_ema = rng.uniform(1, 2, size=12)
    weight_ema[[1, 4, 9]] = 1e-6
    weight_ema[7] = 1e3

    centers_ref, weight_ema_ref = centers.copy(), weight_ema.copy()
    assert _revive_dead_clusters(centers_ref, weight_ema_ref, np.random.RandomState(3))

    probe = _to_device(xp_name, device, centers)
    dev = _array_api.device(probe)
    acc_dtype = _array_api.accumulator_dtype(xp, dev)
    new_centers, new_weight_ema = _array_api.revive_dead_clusters(
        xp,
        probe,
        _array_api.from_numpy(xp, weight_ema, acc_dtype, dev),
        np.random.RandomState(3),
        0.1,
    )
    assert_array_equal(_to_numpy(xp_name, new_centers), centers_ref)
    assert_allclose(_to_numpy(xp_name, new_weight_ema), weight_ema_ref, rtol=1e-6)


def test_no_dead_cluster_draws_no_random_number(xp_name, device):
    xp = _xp(xp_name)
    centers = _to_device(xp_name, device, np.ones((4, 3), dtype=np.float32))
    dev = _array_api.device(centers)
    weight_ema = _array_api.from_numpy(
        xp, np.ones(4), _array_api.accumulator_dtype(xp, dev), dev
    )
    random_state = np.random.RandomState(0)
    _array_api.revive_dead_clusters(xp, centers, weight_ema, random_state, 0.1)
    assert random_state.uniform() == np.random.RandomState(0).uniform()


def test_small_chunks(xp_name, device, monkeypatch):
    # Several chunks of rows in each assignment.
    X = _blobs(np.float32, n_samples=1000, centers=8)
    X_xp = _to_device(xp_name, device, X)
    params = dict(n_clusters=8, init="k-means||", n_epochs=2, random_state=0)
    with sklearn.config_context(array_api_dispatch=True):
        expected = EMeans(**params).fit(X_xp)
        monkeypatch.setattr(_array_api, "_MAX_SCORES_SIZE", 8 * 37)
        emeans = EMeans(**params).fit(X_xp)
        assert_array_equal(
            _to_numpy(xp_name, emeans.predict(X_xp)),
            _to_numpy(xp_name, expected.labels_),
        )
    assert_allclose(
        _to_numpy(xp_name, emeans.cluster_centers_),
        _to_numpy(xp_name, expected.cluster_centers_),
        rtol=1e-4,
        atol=1e-4,
    )
    assert emeans.inertia_ == pytest.approx(expected.inertia_, rel=1e-5)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_predict_transform_score(xp_name, device, dtype):
    if dtype == np.float64 and not _has_float64(xp_name, device):
        pytest.skip("no float64 on this device")
    xp = _xp(xp_name)
    X = _blobs(dtype)
    X_test = _blobs(dtype, n_samples=200)[::-1]
    reference = EMeans(n_clusters=20, random_state=0).fit(X)
    with sklearn.config_context(array_api_dispatch=True):
        X_test_xp = _to_device(xp_name, device, X_test)
        emeans = EMeans(
            n_clusters=20,
            init=reference.cluster_centers_,
            n_epochs=0.1,
            random_state=0,
        ).fit(_to_device(xp_name, device, X))
        # Compare with the NumPy code path on the same centers.
        reference.cluster_centers_ = _to_numpy(xp_name, emeans.cluster_centers_)
        labels = emeans.predict(X_test_xp)
        distances = emeans.transform(X_test_xp)
        score = emeans.score(X_test_xp)
        assert distances.dtype == getattr(xp, dtype.__name__)
        assert _array_api.device(distances) == _array_api.device(X_test_xp)
        assert_array_equal(_to_numpy(xp_name, labels), reference.predict(X_test))
        assert_allclose(
            _to_numpy(xp_name, distances),
            reference.transform(X_test),
            rtol=1e-3,
            atol=1e-3,
        )
        assert score == pytest.approx(reference.score(X_test), rel=1e-5)

        # NumPy input on an estimator fitted on an array API input, and the
        # other way around.
        assert_array_equal(emeans.predict(X_test), reference.predict(X_test))
        assert_array_equal(
            _to_numpy(xp_name, reference.predict(X_test_xp)),
            reference.predict(X_test),
        )
    # Without the dispatch, the centers are still converted for NumPy inputs.
    assert_array_equal(emeans.predict(X_test), reference.predict(X_test))


@pytest.mark.parametrize("as_callable", [False, True])
def test_init_array_and_callable(xp_name, device, as_callable):
    X = _blobs(np.float32)
    X_xp = _to_device(xp_name, device, X)
    init = _to_device(xp_name, device, X[:20].astype(np.float32))
    calls = []

    def init_callable(X, n_clusters, random_state):
        calls.append(X)
        return init

    params = dict(n_clusters=20, n_epochs=1, random_state=0)
    expected = EMeans(init=X[:20], **params).fit(X)
    with sklearn.config_context(array_api_dispatch=True):
        emeans = EMeans(init=init_callable if as_callable else init, **params).fit(X_xp)
        # Initial centers given as a NumPy array.
        emeans_numpy_init = EMeans(init=X[:20], **params).fit(X_xp)
    if as_callable:
        # The callable gets the input on its device.
        assert type(calls[0]) is type(X_xp)
        assert _array_api.device(calls[0]) == _array_api.device(X_xp)
    assert emeans.inertia_ == pytest.approx(expected.inertia_, rel=0.02)
    # Not exactly equal: the order of the sums is not deterministic on GPUs.
    assert_allclose(
        _to_numpy(xp_name, emeans.cluster_centers_),
        _to_numpy(xp_name, emeans_numpy_init.cluster_centers_),
        rtol=1e-4,
        atol=1e-4,
    )

    with sklearn.config_context(array_api_dispatch=True):
        with pytest.raises(ValueError, match="does not match the number of clusters"):
            EMeans(init=init, n_clusters=5).fit(X_xp)


def test_integer_input(xp_name, device):
    # Converted to the widest float dtype of the device.
    xp = _xp(xp_name)
    X = np.round(_blobs(np.float64, n_samples=500, centers=5)).astype(np.int64)
    X_xp = _to_device(xp_name, device, X)
    with sklearn.config_context(array_api_dispatch=True):
        emeans = EMeans(n_clusters=5, random_state=0).fit(X_xp)
        assert emeans.predict(X_xp).shape == (500,)
    dtype = "float64" if _has_float64(xp_name, device) else "float32"
    assert emeans.cluster_centers_.dtype == getattr(xp, dtype)
    expected = EMeans(n_clusters=5, random_state=0).fit(X.astype(dtype))
    assert emeans.inertia_ == pytest.approx(expected.inertia_, rel=0.02)


def test_errors(xp_name, device):
    X_xp = _to_device(xp_name, device, _blobs(np.float32, n_samples=10))
    with sklearn.config_context(array_api_dispatch=True):
        with pytest.raises(ValueError, match="should be >= n_clusters"):
            EMeans(n_clusters=11).fit(X_xp)
        emeans = EMeans(n_clusters=2).fit(X_xp)
        with pytest.raises(ValueError, match="features"):
            emeans.predict(X_xp[:, :3])
        if not _is_cuda(xp_name, device):
            with pytest.raises(ValueError, match="not supported for array API"):
                EMeans(n_clusters=2, assignment="two-level").fit(X_xp)


def test_no_dispatch_without_config(xp_name, device):
    # As for the scikit-learn estimators, array API inputs are converted to
    # NumPy arrays (when their library allows it) without
    # `array_api_dispatch=True`.
    if (xp_name, device) != ("torch", "cpu"):
        pytest.skip("only for an input that converts to a NumPy array")
    X = _blobs(np.float32)
    emeans = EMeans(n_clusters=20, random_state=0).fit(_to_device(xp_name, device, X))
    assert isinstance(emeans.cluster_centers_, np.ndarray)
    assert isinstance(emeans.labels_, np.ndarray)


# Two-level assignment: Triton kernels, on CUDA devices only.


def _two_level_triton(xp_name, device, monkeypatch, precision="ieee"):
    """The module of the kernels, with exact float32 products by default.

    The default float16 inputs are rounded to 11 bits: the tests that compare
    labels need the float32 products.
    """
    if not _is_cuda(xp_name, device):
        pytest.skip("the two-level assignment needs a CUDA device")
    pytest.importorskip("triton")
    from sklearn_emeans import _two_level_triton

    monkeypatch.setattr(_two_level_triton, "PRECISION", precision)
    return _two_level_triton


def _torch_cuda(array, dtype=None):
    import torch

    return torch.asarray(np.ascontiguousarray(array), dtype=dtype, device="cuda")


def _assigner_labels(module, X, centers, weight_ema, cells, cell_of_center, **params):
    """Labels of the kernels after a refresh, and the refreshed cells."""
    import torch

    params = {
        "max_probes": 3,
        "ratio": 1.3,
        "max_rows": len(X),
        "max_abs": float(np.abs(X).max()),
        **params,
    }
    assigner = module.TwoLevelAssigner(
        _torch_cuda(cells), _torch_cuda(cell_of_center, torch.int64), **params
    )
    assigner.refresh(_torch_cuda(centers), _torch_cuda(weight_ema))
    labels = assigner.assign(_torch_cuda(X))
    return (
        labels.cpu().numpy(),
        assigner.cells.cpu().numpy(),
        assigner.cell_of_center.cpu().numpy(),
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "max_probes, ratio", [(1, 1.15), (3, 1.3), (4, 1.2), (3, np.inf), (6, np.inf)]
)
@pytest.mark.parametrize("n_features", [6, 32, 70])
def test_two_level_assigner_matches_reference(
    xp_name, device, monkeypatch, dtype, max_probes, ratio, n_features
):
    module = _two_level_triton(xp_name, device, monkeypatch)
    X, centers, weight_ema, cells, cell_of_center = _data(dtype, n_features=n_features)
    labels, new_cells, new_cell_of_center = _assigner_labels(
        module,
        X,
        centers,
        weight_ema,
        cells,
        cell_of_center,
        max_probes=max_probes,
        ratio=ratio,
    )

    _reference_refresh(centers, weight_ema, cells, cell_of_center)
    assert_allclose(new_cells, cells, rtol=1e-5 if dtype == np.float32 else 1e-12)
    assert_array_equal(new_cell_of_center, cell_of_center)
    expected = _reference_labels(X, centers, cells, cell_of_center, max_probes, ratio)
    # A row at about the same distance of two centers, or of the threshold of
    # a cell, can get another label in float32.
    assert np.mean(labels == expected) > 0.995
    assert labels.dtype == np.int64


@pytest.mark.parametrize("max_rows", [1, 100, 255, 4096])
def test_two_level_assigner_blocks_of_rows(xp_name, device, monkeypatch, max_rows):
    # The rows are assigned by blocks of max_rows rows.
    module = _two_level_triton(xp_name, device, monkeypatch)
    X, centers, weight_ema, cells, cell_of_center = _data(np.float32, n_samples=300)
    labels, _, _ = _assigner_labels(
        module, X, centers, weight_ema, cells, cell_of_center, max_rows=max_rows
    )
    expected, _, _ = _assigner_labels(
        module, X, centers, weight_ema, cells, cell_of_center
    )
    assert_array_equal(labels, expected)


def test_two_level_assigner_empty_cells(xp_name, device, monkeypatch):
    # Cells without center are never probed, and their coarse center stays.
    module = _two_level_triton(xp_name, device, monkeypatch)
    X, centers, weight_ema, cells, cell_of_center = _data(np.float32)
    far = np.full((3, cells.shape[1]), 1e3, dtype=np.float32)
    far[1] *= -1
    far[2, ::2] = 0
    cells = np.concatenate([far[:1], cells, far[1:]])
    cell_of_center = cell_of_center + 1
    labels, new_cells, new_cell_of_center = _assigner_labels(
        module,
        X,
        centers,
        weight_ema,
        cells,
        cell_of_center,
        max_probes=len(cells),
        ratio=np.inf,
    )
    assert_array_equal(new_cells[[0, -2, -1]], far)
    assert not np.isin(new_cell_of_center, [0, len(cells) - 2, len(cells) - 1]).any()
    # All the non-empty cells are probed: the assignment is exact.
    distances = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    assert np.mean(labels == distances.argmin(axis=1)) > 0.995


def test_two_level_assigner_large_cell(xp_name, device, monkeypatch):
    # A cell with many more centers than the others, and more than a block of
    # the kernel: almost all the rows probe it.
    module = _two_level_triton(xp_name, device, monkeypatch)
    rng = np.random.RandomState(0)
    n_features = 20
    centers = np.concatenate(
        [
            rng.normal(size=(700, n_features)),
            10 + 0.1 * rng.normal(size=(5, n_features)),
            -10 + 0.1 * rng.normal(size=(3, n_features)),
        ]
    ).astype(np.float32)
    cells = np.stack(
        [np.zeros(n_features), np.full(n_features, 10), np.full(n_features, -10)]
    ).astype(np.float32)
    cell_of_center = np.repeat([0, 1, 2], [700, 5, 3])
    weight_ema = np.ones(len(centers))
    X = np.concatenate(
        [
            rng.normal(size=(5000, n_features)),
            10 + 0.1 * rng.normal(size=(100, n_features)),
            -10 + 0.1 * rng.normal(size=(1, n_features)),
        ]
    ).astype(np.float32)
    labels, _, new_cell_of_center = _assigner_labels(
        module, X, centers, weight_ema, cells, cell_of_center, max_probes=1
    )
    assert_array_equal(new_cell_of_center, cell_of_center)
    distances = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    assert np.mean(labels == distances.argmin(axis=1)) > 0.999
    assert np.bincount(new_cell_of_center[labels]).tolist() == [5000, 100, 1]


def test_two_level_assigner_ties_go_to_lowest_index(xp_name, device, monkeypatch):
    # Duplicated centers, in the same cell or not: the lowest index wins.
    module = _two_level_triton(xp_name, device, monkeypatch)
    rng = np.random.RandomState(0)
    base = rng.normal(size=(10, 8)).astype(np.float32)
    centers = np.concatenate([base, base, base])
    cells = base[:3].copy()
    cell_of_center = rng.randint(3, size=30)
    X = (base[rng.randint(10, size=500)] + 0.01 * rng.normal(size=(500, 8))).astype(
        np.float32
    )
    labels, _, _ = _assigner_labels(
        module,
        X,
        centers,
        np.ones(30),
        cells,
        cell_of_center,
        max_probes=3,
        ratio=np.inf,
    )
    assert labels.max() < 10


def test_two_level_assigner_fallback_without_coarse_kernel(
    xp_name, device, monkeypatch
):
    # When the coarse kernel doesn't fit in the shared memory of the GPU, the
    # probes are selected with PyTorch operations: same labels.
    module = _two_level_triton(xp_name, device, monkeypatch)
    X, centers, weight_ema, cells, cell_of_center = _data(np.float32)
    for max_probes, ratio in [(1, 1.15), (3, 1.3), (6, np.inf)]:
        params = dict(max_probes=max_probes, ratio=ratio)
        expected, _, _ = _assigner_labels(
            module, X, centers, weight_ema, cells, cell_of_center, **params
        )
        with monkeypatch.context() as context:
            context.setattr(module, "_COARSE_BLOCKS", ())
            labels, _, _ = _assigner_labels(
                module, X, centers, weight_ema, cells, cell_of_center, **params
            )
        assert np.mean(labels == expected) > 0.995


def _fit_two_level(xp_name, device, X, monkeypatch, **params):
    """Fit on the device with the two-level assignment, for any n_clusters."""
    monkeypatch.setattr(_emeans, "_TWO_LEVEL_MIN_CLUSTERS", 1)
    with sklearn.config_context(array_api_dispatch=True):
        return EMeans(assignment="two-level", **params).fit(
            _to_device(xp_name, device, X)
        )


def _overlapping_blobs(dtype, n_samples=20_000):
    rng = np.random.RandomState(0)
    X = (
        rng.normal(size=(n_samples, 16))
        + 2 * rng.normal(size=(30, 16))[rng.randint(30, size=n_samples)]
    )
    return X.astype(dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_two_level_steps_all_cells_probed_same_as_exact(
    xp_name, device, monkeypatch, dtype
):
    # When each sample probes all the cells, it is compared to all the
    # centers: same steps as with the exact assignment, up to the rounding of
    # the assignment, which is in float32.
    module = _two_level_triton(xp_name, device, monkeypatch)
    import torch

    xp = _array_api.array_api_compat.array_namespace(torch.zeros(1))
    X, centers, weight_ema, cells, cell_of_center = _data(dtype)
    rows = np.random.RandomState(1).permutation(len(X))
    batch_size = 256
    step_sizes = np.linspace(0.5, 0.1, -(-len(X) // batch_size)).tolist()
    assigner = module.TwoLevelAssigner(
        _torch_cuda(cells),
        _torch_cuda(cell_of_center, torch.int64),
        len(cells),
        np.inf,
        batch_size,
        float(np.abs(X).max()),
    )

    def steps(assigner):
        return _array_api.emeans_steps(
            xp,
            _torch_cuda(X),
            _torch_cuda(rows),
            batch_size,
            _torch_cuda(centers),
            _torch_cuda(weight_ema),
            step_sizes,
            torch.zeros(centers.shape, dtype=torch.float64, device="cuda"),
            torch.zeros(len(centers), dtype=torch.float64, device="cuda"),
            first_step=0,
            revival_period=0,
            revive=None,
            assigner=assigner,
        )

    results = [array.cpu().numpy() for array in steps(assigner)]
    expected = [array.cpu().numpy() for array in steps(None)]
    for array, expected_array in zip(results, expected):
        assert_allclose(array, expected_array, rtol=1e-4, atol=1e-4)
    assert_array_equal(results[3], expected[3])

    new_centers = _torch_cuda(results[0])
    assigner.refresh(new_centers, _torch_cuda(results[1]))
    labels, inertia = _array_api.labels_inertia(
        xp, _torch_cuda(X), new_centers, assigner
    )
    labels_exact, inertia_exact = _array_api.labels_inertia(
        xp, _torch_cuda(X), new_centers
    )
    assert_array_equal(labels.cpu().numpy(), labels_exact.cpu().numpy())
    assert inertia == pytest.approx(inertia_exact, rel=1e-6)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("init", ["random", "k-means||"])
def test_fit_two_level_same_as_numpy(xp_name, device, monkeypatch, dtype, init):
    # Same cells, same refreshes and same random numbers as the Cython
    # kernels: the results differ by the rounding of the assignment. Without
    # revival: the two halves of a split center are so close that the rounding
    # decides which one many samples get, also between float32 and float64
    # with the Cython kernels.
    _two_level_triton(xp_name, device, monkeypatch)
    X = _overlapping_blobs(dtype)
    params = dict(n_clusters=300, init=init, revival=0, random_state=0)
    two_level = _fit_two_level(xp_name, device, X, monkeypatch, **params)
    expected = EMeans(assignment="two-level", **params).fit(X)

    labels = _to_numpy(xp_name, two_level.labels_)
    centers = _to_numpy(xp_name, two_level.cluster_centers_)
    assert np.mean(labels == expected.labels_) > 0.95
    assert two_level.inertia_ == pytest.approx(expected.inertia_, rel=2e-3)
    assert np.mean(np.abs(centers - expected.cluster_centers_) < 0.01) > 0.95
    assert two_level.inertia_ == pytest.approx(
        ((X - centers[labels]) ** 2).sum(), rel=1e-5
    )
    assert two_level.batch_size_ == expected.batch_size_


@pytest.mark.parametrize("precision", ["float16", "tf32"])
@pytest.mark.parametrize("scale", [1e-8, 1, 1e8])
def test_two_level_assigner_rounded_inputs(
    xp_name, device, monkeypatch, precision, scale
):
    # The inputs of the products are rounded to 11 bits, whatever the scale
    # of the data: almost all the rows keep their label.
    module = _two_level_triton(xp_name, device, monkeypatch, precision=precision)
    X, centers, weight_ema, cells, cell_of_center = _data(np.float32, n_features=40)
    scale = np.float32(scale)
    labels, _, _ = _assigner_labels(
        module, scale * X, scale * centers, weight_ema, scale * cells, cell_of_center
    )
    _reference_refresh(centers, weight_ema, cells, cell_of_center)
    expected = _reference_labels(X, centers, cells, cell_of_center, 3, 1.3)
    assert np.mean(labels == expected) > 0.98


@pytest.mark.parametrize("precision", ["float16", "tf32"])
def test_fit_two_level_rounded_inputs_close_to_exact(
    xp_name, device, monkeypatch, precision
):
    # The default precision of the kernels and the other one on the tensor
    # cores, with more clusters than _TWO_LEVEL_MIN_CLUSTERS.
    _two_level_triton(xp_name, device, monkeypatch, precision=precision)
    rng = np.random.RandomState(0)
    X = (
        rng.normal(size=(60_000, 24))
        + 2 * rng.normal(size=(200, 24))[rng.randint(200, size=60_000)]
    ).astype(np.float32)
    X_xp = _to_device(xp_name, device, X)
    with sklearn.config_context(array_api_dispatch=True):
        two_level = EMeans(n_clusters=1500, assignment="two-level", random_state=0).fit(
            X_xp
        )
        exact = EMeans(n_clusters=1500, random_state=0).fit(X_xp)
        labels = _to_numpy(xp_name, two_level.labels_)
        assert two_level.inertia_ < 1.01 * exact.inertia_
        # labels_ and inertia_ are the ones of the two-level assignment, and
        # predict is exact.
        assert np.mean(labels == _to_numpy(xp_name, two_level.predict(X_xp))) > 0.9
        assert two_level.inertia_ >= -two_level.score(X_xp) * (1 - 1e-6)


def test_fit_two_level_few_clusters_is_exact(xp_name, device, monkeypatch):
    _two_level_triton(xp_name, device, monkeypatch)
    X_xp = _to_device(xp_name, device, _blobs(np.float32))
    with sklearn.config_context(array_api_dispatch=True):
        exact = EMeans(n_clusters=20, random_state=0).fit(X_xp)
        two_level = EMeans(n_clusters=20, assignment="two-level", random_state=0).fit(
            X_xp
        )
    # Not bitwise: the sums of the rows of a cluster are added in an order
    # that changes from one fit to the next on a GPU.
    assert type(two_level.cluster_centers_) is type(exact.cluster_centers_)
    assert_allclose(
        _to_numpy(xp_name, two_level.cluster_centers_),
        _to_numpy(xp_name, exact.cluster_centers_),
        rtol=1e-5,
        atol=1e-5,
    )
    assert_array_equal(
        _to_numpy(xp_name, two_level.labels_), _to_numpy(xp_name, exact.labels_)
    )


@pytest.mark.parametrize("n_clusters", [1, 2, 5, 299, 300])
def test_fit_two_level_n_clusters_edge_cases(xp_name, device, monkeypatch, n_clusters):
    _two_level_triton(xp_name, device, monkeypatch)
    X = _blobs(np.float32, n_samples=300)
    emeans = _fit_two_level(
        xp_name, device, X, monkeypatch, n_clusters=n_clusters, random_state=0
    )
    centers = _to_numpy(xp_name, emeans.cluster_centers_)
    labels = _to_numpy(xp_name, emeans.labels_)
    assert centers.shape == (n_clusters, X.shape[1])
    assert np.isfinite(centers).all()
    assert labels.min() >= 0 and labels.max() < n_clusters
    assert emeans.inertia_ == pytest.approx(
        ((X - centers[labels]) ** 2).sum(), rel=1e-4, abs=1e-3
    )


def test_fit_two_level_revives_dead_clusters(xp_name, device, monkeypatch):
    # All the initial centers are the same point: only the first one gets
    # samples, and the others are revived by splitting it. They go to the
    # cell of their donor.
    _two_level_triton(xp_name, device, monkeypatch)
    X, _ = make_blobs(n_samples=5000, n_features=5, centers=10, random_state=0)
    X = X.astype(np.float32)
    init = np.repeat(X[:1], 30, axis=0)
    emeans = _fit_two_level(
        xp_name, device, X, monkeypatch, n_clusters=30, init=init, random_state=0
    )
    centers = _to_numpy(xp_name, emeans.cluster_centers_)
    assert len(np.unique(centers, axis=0)) == 30
    assert np.all(np.bincount(_to_numpy(xp_name, emeans.labels_), minlength=30) > 0)
    expected = EMeans(
        n_clusters=30, init=init, assignment="two-level", random_state=0
    ).fit(X)
    assert emeans.inertia_ == pytest.approx(expected.inertia_, rel=0.05)


def test_revive_dead_clusters_moves_cells(xp_name, device, monkeypatch):
    module = _two_level_triton(xp_name, device, monkeypatch)
    import torch

    xp = _array_api.array_api_compat.array_namespace(torch.zeros(1))
    centers = _torch_cuda(np.arange(12, dtype=np.float32).reshape(4, 3))
    weight_ema = _torch_cuda(np.array([1.0, 1e-9, 1.0, 5.0]))
    assigner = module.TwoLevelAssigner(
        centers.clone(), _torch_cuda(np.arange(4), torch.int64), 2, 1.15, 16, 11.0
    )
    _, weight_ema = _array_api.revive_dead_clusters(
        xp, centers, weight_ema, np.random.RandomState(0), 0.1, assigner
    )
    weight_ema = weight_ema.cpu().numpy()
    donor = np.flatnonzero(weight_ema == weight_ema[1])
    donor = donor[donor != 1][0]
    cell_of_center = assigner.cell_of_center.cpu().numpy()
    assert cell_of_center[1] == cell_of_center[donor] == donor


def test_fit_two_level_greedy_revival(xp_name, device, monkeypatch):
    _two_level_triton(xp_name, device, monkeypatch)
    X, _ = make_blobs(n_samples=5000, n_features=5, centers=40, random_state=0)
    X = X.astype(np.float32)
    emeans = _fit_two_level(
        xp_name,
        device,
        X,
        monkeypatch,
        n_clusters=40,
        revival="greedy",
        random_state=0,
    )
    expected = EMeans(
        n_clusters=40, revival="greedy", assignment="two-level", random_state=0
    )
    expected.fit(X)
    assert emeans.inertia_ == pytest.approx(expected.inertia_, rel=0.05)


def test_fit_two_level_cells_on_device(xp_name, device, monkeypatch):
    # With many centers, the e-means of the initial centers runs on the
    # device: same cells up to rounding, and a close result.
    _two_level_triton(xp_name, device, monkeypatch)
    X = _overlapping_blobs(np.float32)
    params = dict(n_clusters=300, random_state=0)
    on_host = _fit_two_level(xp_name, device, X, monkeypatch, **params)
    monkeypatch.setattr(_emeans, "_TWO_LEVEL_CELLS_HOST_MAX_CLUSTERS", 0)
    on_device = _fit_two_level(xp_name, device, X, monkeypatch, **params)
    assert on_device.inertia_ == pytest.approx(on_host.inertia_, rel=5e-3)


def test_fit_two_level_needs_triton(xp_name, device, monkeypatch):
    if not _is_cuda(xp_name, device):
        pytest.skip("the two-level assignment needs a CUDA device")
    import sys

    monkeypatch.setitem(sys.modules, "triton", None)
    X_xp = _to_device(xp_name, device, _blobs(np.float32))
    with sklearn.config_context(array_api_dispatch=True):
        with pytest.raises(ImportError, match="needs PyTorch and Triton"):
            EMeans(n_clusters=20, assignment="two-level").fit(X_xp)
