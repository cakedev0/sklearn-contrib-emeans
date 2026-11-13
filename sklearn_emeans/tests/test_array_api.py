"""Tests of EMeans on array API inputs.

They need `SCIPY_ARRAY_API=1` in the environment, and run
on the namespaces and devices that are available: array-api-strict, PyTorch
(CPU, CUDA, MPS, XPU) and CuPy.
"""

import math

import numpy as np
import pytest
import sklearn
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.datasets import make_blobs

from sklearn_emeans import EMeans, _array_api
from sklearn_emeans._emeans import _revive_dead_clusters
from sklearn_emeans.tests.test_emeans import _reference_step

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
    _array_api.revive_dead_clusters(xp, centers, weight_ema, random_state)
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


@pytest.mark.parametrize(
    "n_samples, n_features, n_clusters",
    [(3000, 10, 20), (1000, 3, 300), (2500, 16, 130), (700, 32, 5), (700, 1, 5)],
)
@pytest.mark.parametrize("with_epoch", [True, False])
def test_mps_kernels_same_as_array_operations(
    xp_name, device, n_samples, n_features, n_clusters, with_epoch, monkeypatch
):
    # On Apple GPUs, the steps on low-dimensional data run with Metal kernels.
    xp = _xp(xp_name)
    X = _blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters)
    X_xp = _to_device(xp_name, device, X)
    if not _array_api._use_mps_kernels(xp, X_xp):
        pytest.skip("no Metal kernels for this input")
    dev = _array_api.device(X_xp)
    rng = np.random.RandomState(0)
    rows = rng.permutation(n_samples)[: n_samples - 100]
    centers = X[rng.choice(n_samples, n_clusters, replace=False)]
    # A center far from the data gets no row.
    centers[0] = 1000
    weight_ema = rng.uniform(size=n_clusters)
    batch_size = 500
    step_sizes = [0.5, 0.3, 0.2, 0.1, 0.1][: math.ceil(len(rows) / batch_size)]
    centers_xp = _array_api.from_numpy(xp, centers, "float32", dev)
    weight_ema_xp = _array_api.from_numpy(xp, weight_ema, "float32", dev)

    def steps():
        # No revival: in float32, it can amplify the rounding differences.
        epoch_sums = epoch_counts = None
        if with_epoch:
            epoch_sums = xp.zeros(
                (n_clusters, n_features), dtype=xp.float32, device=dev
            )
            epoch_counts = xp.zeros(n_clusters, dtype=xp.float32, device=dev)
        results = _array_api.emeans_steps(
            xp,
            X_xp,
            _array_api.index_array(xp, rows, dev),
            batch_size,
            centers_xp,
            weight_ema_xp,
            step_sizes,
            epoch_sums,
            epoch_counts,
            first_step=0,
            revival_period=0,
            revive=None,
        )
        assert results[0].dtype == xp.float32
        return [None if a is None else _to_numpy(xp_name, a) for a in results]

    results = steps()
    # The inputs are not modified.
    assert_array_equal(_to_numpy(xp_name, centers_xp), centers)
    assert_array_equal(_to_numpy(xp_name, weight_ema_xp), weight_ema.astype(np.float32))
    monkeypatch.setattr(_array_api, "_use_mps_kernels", lambda xp, X: False)
    expected = steps()

    assert_array_equal(results[0][0], centers[0])
    for array, expected_array in zip(results, expected):
        if expected_array is None:
            assert array is None
        else:
            assert_allclose(array, expected_array, rtol=1e-4, atol=1e-4)


def test_mps_kernels_only_for_low_dimensional_data(xp_name, device):
    xp = _xp(xp_name)
    for n_features, expected in [(32, device == "mps"), (33, False)]:
        X_xp = _to_device(xp_name, device, _blobs(n_features=n_features))
        assert _array_api._use_mps_kernels(xp, X_xp) == expected
