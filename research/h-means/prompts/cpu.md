# Prompt: two-level assignment in `EMeans`, CPU backend

Implement the two-level ("h-means") assignment in the NumPy/Cython backend of
`EMeans`, in this repository, on a new branch `h-means-cpu` created from
`h-means`. Commit there. Don't push or open a PR unless the user asks.

## Why

At a fixed cluster size, e-means costs `O(n k d)`, which is quadratic in `n`.
A research prototype on the `h-means` branch brings the number of comparisons
to about `O(n sqrt(k) d)` for a loss of 0.2% to 0.3% of inertia. On one Xeon
socket: DEEP 10M at `k=10k` goes from 40.6 s to 10 s, DEEP 10M at `k=100k` from
788 s to 16 s. The prototype is not a package (PyTorch driver, gcc only, no
tests). Your job is the production version for dense NumPy input.

## Read first

- `CLAUDE.md` (architecture and constraints of the package, they all apply).
- `research/h-means/README.md`: methods, results, journal of what was tried.
- `research/h-means/hkm.py`: `IVFAssigner` (cells, refresh, `search`,
  `assign`), `emeans_fit`, `run`. This is the reference for the algorithm.
- `research/h-means/hkm_kernels.pyx`: `ivf_assign` (the fast kernel) and
  `accumulate`.

## The algorithm

One e-means on all the `k` centers, unchanged (laws, schedule, revival, final
centers from the sums of the last epoch, extra labelling pass). Only the
assignment changes:

- `g = round(sqrt(k))` cells. A cell is a coarse center and the fine centers
  that are closest to it. The cells are refreshed during the fit as the fine
  centers move (see `IVFAssigner.refresh` and `teleport` for revived centers).
- A sample is compared to the coarse centers, then to the fine centers of its
  nearest cells. Cell `q > 0` is probed only when its squared distance is at
  most `ratio^2` times that of the nearest cell, up to `p` cells. Defaults of
  the prototype: `p=16`, `ratio=1.15`, and the same for the final labelling
  pass (`p_final=16`).
- Empty cells must never be the only candidates of a sample.

## Already settled, don't redo

- Clustering each coarse cluster independently (frozen two-level) loses 2% to
  4%. Cells frozen after the init lose up to 5%. Not an option.
- The kNN-graph variant is less accurate and gets worse with `n`. Drop it.
- A row-wise kernel without BLAS was 3x to 4x slower than bucketing the
  (sample, cell) pairs by cell and running one GEMM per (cell, chunk).
- `sgemv` of OpenBLAS takes a global lock: don't call it from threads.
- Reuse the work buffers between steps (page faults), sort the pairs with a
  parallel counting sort, keep the same OpenMP team size in all the regions.

## Requirements

- Public API: propose it and keep it small. Suggested: a parameter such as
  `assignment="exact"` (default, unchanged behavior) or `"two-level"`, with
  the probing constants private at first. Say in your report what you chose
  and why.
- Exact assignment must stay bit-for-bit what it is today.
- All the package constraints: MSVC (no gcc-only atomics or pragmas), runs
  without OpenMP, BLAS only from `scipy.linalg.cython_blas`, no
  `-march=native`, float64 sums and EMA weights, random numbers drawn on the
  host from `random_state` in a fixed order.
- The result must not depend on the number of threads. If you can't get that
  at a reasonable cost, stop and report the trade-off.
- Move the whole step loop to Cython as for the exact path (the prototype
  only has the assignment and the accumulation there).
- `labels_` and `inertia_` come from the two-level labelling pass: document
  that they are approximate. `predict` stays exact.
- Out of scope, with a clear error or a documented fallback: CSR input, array
  API inputs (another agent does the GPU path; say in your report how your
  semantics should be mirrored there), `init="k-means||"` and
  `init="k-means++"` made two-level (they cost `O(n k)`: keep them working,
  document the cost).
- Small `k`: the prototype gains nothing at `k=1000` on 1M samples. Measure
  where two-level starts to win and decide what `"two-level"` does below that.

## Tests

- A pure NumPy reference of the two-level step in the tests, compared to the
  Cython kernels, like `_reference_step` in `tests/test_emeans.py`.
- When all the cells are probed (`p = g`, no ratio), the result equals the
  exact backend up to rounding.
- Same result with 1 and several threads.
- Empty cells, revived centers, `k` close to `n`, tiny `k`.
- The scikit-learn estimator checks keep passing. New files go in
  `py.install_sources` of `sklearn_emeans/meson.build`.

## Benchmarks

On `intel-xeon`, one socket, 64 threads, under `benchlock` (see the global
instructions). Data and a working environment of the prototype are in
`~/work/hkm` there (`data/`, `venv/`, `v4/` is the prototype as benchmarked).
Run `EMeans` exact, the prototype and your implementation in the same
`benchlock` call.

Done means, against exact `EMeans` with 5 epochs:

- loss on labels at most 0.3% on SIFT 1M, GloVe, CLIP 1M at `k=10k` and on
  DEEP 10M at `k=10k` and `k=100k` (the prototype numbers are in the README);
- at least as fast as the prototype on these cases (it reaches about a third
  of the arithmetic rate of the flat GEMM, so being faster is realistic);
- the scaling run at 1,000 samples per cluster on DEEP (1M, 3M, 10M, 30M),
  with the measured exponent.

Add the numbers, with the commands, to `research/h-means/README.md`.

## Report

What you implemented, the API you chose, the benchmark table, anything that
does not meet the requirements above, and the open questions for the user.
