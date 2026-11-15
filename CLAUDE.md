# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`sklearn-contrib-emeans` provides `EMeans`, a scikit-learn compatible e-means
(mini-batch k-means with EMA centers, arXiv:2611.09501). The reference
implementation is `faiss/contrib/e_means`: from the same initial centers and
random numbers, `EMeans` must give the same centers as its `NumpyEmeans`, up to
float32 rounding.

## Commands

The package is built with meson-python (Cython + OpenMP). `.venv` holds an
editable install, which recompiles the changed `.pyx` files on the next import:

```
pip install --no-build-isolation -e ".[test]"   # needs meson-python, meson, ninja, Cython, scipy
.venv/bin/pytest                                 # all tests + doctests (testpaths and --doctest-modules are in pyproject.toml)
.venv/bin/pytest sklearn_emeans/tests/test_emeans.py -k batch_size
SCIPY_ARRAY_API=1 .venv/bin/pytest sklearn_emeans/tests/test_array_api.py
pre-commit run --all-files                       # ruff check + ruff format + cython-lint, as in CI
sphinx-build -W --keep-going -b html -d doc/_build/doctrees doc doc/_build/html   # needs the "docs" extra
```

- Without `SCIPY_ARRAY_API=1`, `test_array_api.py` is skipped as a whole. With
  it, the tests run on the namespaces and devices that are installed and
  available (array-api-strict, PyTorch CPU/CUDA/MPS/XPU, CuPy).
- CI (`.github/workflows/tests.yml`) installs the package non-editable and runs
  `pytest --pyargs sklearn_emeans` from another directory, so the tests must
  pass on the installed package.
- A new `.py` or test file must be added to `py.install_sources` in
  `sklearn_emeans/meson.build`, otherwise it is missing from the wheels.
- The version is in the top-level `meson.build`.
- The documentation (`doc/`, and the sphinx-gallery examples in `examples/`) is
  built from the installed package and deployed from main to the `gh-pages`
  branch by `.github/workflows/docs.yml`. Warnings are errors.

## Architecture

`EMeans.fit` (`_emeans.py`) owns everything that is not a hot loop: parameter
validation, the batch size and initial step size from the scaling laws of the
paper (constants at the top of the file, taken from the reference
implementation), the cosine schedule, the permutation of each epoch, and the
revival of the dead clusters. All the random numbers are drawn on the host
from `random_state`, in a fixed order. It then calls one of two backends,
which run all the steps of an epoch in one call and take the same arguments
(`rows`, `batch_size`, `centers`, `weight_ema`, `step_sizes`, `epoch_sums`,
`epoch_counts`, `first_step`, `revival_period`, `revive` callback):

1. `_kernels.pyx`, for NumPy arrays and CSR matrices (`_emeans_steps`,
   `_emeans_steps_sparse`, `_labels_inertia*`). A step assigns the batch by
   chunks of 256 rows with one GEMM each (BLAS is limited to one thread by the
   caller), then updates the centers in parallel over centers, so the result
   doesn't depend on the number of threads. The whole epoch is a single OpenMP
   parallel region where the threads synchronize with atomic counters
   (`_run_steps`). Sums and EMA weights are in float64.
2. `_array_api.py`, for array API inputs when `array_api_dispatch=True`
   (`get_namespace` returns `None` otherwise, and the input goes to the NumPy
   path). Same computations with array operations on the device of the input.
   The operations that the standard lacks, or that have a much faster
   library-specific version (PyTorch, CuPy), are in the helpers at the top of
   the module, with a standard-only fallback that array-api-strict tests.
   Devices without float64 accumulate in float32 (`accumulator_dtype`).

The two backends must stay equivalent: from the same `random_state` they give
the same result up to rounding, and the tests compare them with each other and
with the pure NumPy `_reference_step` of `tests/test_emeans.py`. A change to
the algorithm goes in both.

After the last epoch, the final centers are the means of the samples assigned
during that epoch (`epoch_sums / epoch_counts`), not the EMA centers, and
`labels_` / `inertia_` come from an extra assignment pass.

Initialization:

- `_kmeans_parallel.py` is k-means||: the sampling rounds use
  `_update_min_distances*` of `_kernels.pyx` (or the device, in
  `_array_api.kmeans_parallel`, which reuses its host-side sampling helpers),
  then a weighted k-means++ runs on the candidates on the host.
- `_kmeans_plusplus.py` and `_kmeans_plusplus_kernels.pyx` are copied from
  scikit-learn main (the commit is in the module docstring). Keep the changes
  to them minimal, so that they can be synced again.

## Constraints

- Supported range: Python >= 3.10, scikit-learn >= 1.7, and the minimum NumPy,
  SciPy, joblib and threadpoolctl of scikit-learn 1.10. CI has a job with
  exactly these minimum versions and one with the scikit-learn nightly.
- The Cython code must compile with MSVC (hence the hand-written atomics at the
  top of `_kernels.pyx`) and run without OpenMP (Apple Clang without libomp).
- BLAS comes from `scipy.linalg.cython_blas`, loaded from the installed SciPy at
  run time: no BLAS is linked at build time.
- `tests/test_common.py` runs the scikit-learn estimator checks: `EMeans` has to
  keep passing them on all the supported scikit-learn versions.
- CuPy, PyTorch and array-api-strict are optional: import them inside the
  functions, never at module level.
