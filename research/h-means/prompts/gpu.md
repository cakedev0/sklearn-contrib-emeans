# Prompt: two-level assignment in `EMeans`, GPU backend with custom kernels

Design and implement a GPU path for the two-level ("h-means") assignment of
`EMeans`, in this repository, on a new branch `h-means-gpu` created from
`h-means`. Commit there. Don't push or open a PR unless the user asks. Don't
post anything outside this repository (no issue, no comment on other
projects).

## Why

At a fixed cluster size, e-means costs `O(n k d)`, which is quadratic in `n`.
Comparing each sample only to the centers of a few cells out of `sqrt(k)`
brings this to about `O(n sqrt(k) d)` for a loss well under 1% of inertia.
`research/h-means/README.md` has the evidence. On GPU the idea is proven by
o-means (GPU only, Triton kernels): on an NVIDIA L4 with `sqrt(k)` cells, DEEP
10M at `k=10k` goes from 3.6 s (dense) to 0.74 s, and at `k=100k` from 30.8 s
to 1.6 s.

The prototype of this repository on GPU (`research/h-means/hkm.py`, plain
PyTorch operations) is bound by Python overhead and gathers: about 1.3 s of
GPU work in a 4 s fit. Cells have different sizes and each sample probes
different cells, so standard array operations fit badly. This is why custom
kernels are needed, and why the design is open.

## Read first

- `CLAUDE.md` (architecture and constraints of the package).
- `research/h-means/README.md`, in particular the GPU section and the journal.
- `research/h-means/hkm.py`: `IVFAssigner` is the reference for the
  algorithm (cells, refresh, adaptive probing, final pass).
- `sklearn_emeans/_array_api.py`: the current GPU backend and its
  library-specific helpers.
- o-means, for the kernel design: `contrib/o_means/` of the owlgebra-ai/faiss
  fork (`o_means.py`, `triton_two_level.py`). A clone is on `aws-l4-3` in
  `~/work/omeans/fork`, with an environment in `~/work/omeans/venv` (run with
  `PYTHONPATH=~/work/omeans/fork`). It is MIT licensed: if you reuse code,
  keep the copyright notice and say so in your report.
- If the branch `h-means-cpu` exists, read it: the semantics (cells, refresh
  schedule, random draws, parameter names) must be the same on both backends.
  If it does not exist yet, follow `hkm.py` and list what must be aligned
  later.

## The algorithm

One e-means on all the `k` centers, unchanged. Only the assignment changes:

- `g = round(sqrt(k))` cells: a coarse center and the fine centers closest to
  it. The cells are refreshed during the fit as the fine centers move.
- A sample is compared to the coarse centers, then to the fine centers of its
  nearest cells. On CPU the number of probed cells is adaptive: cell `q > 0`
  is probed only when its squared distance is at most `ratio^2` times that of
  the nearest cell, up to `p` cells (`p=16`, `ratio=1.15`).
- The final labelling pass uses more probes than the training steps. It
  removes most of the gap between the inertia of the centers and of the
  returned labels, for one pass.

## Step 1: design, with measurements

Before the full implementation, compare at least these options on the L4 with
a small benchmark of the assignment alone, and write the result (one page) in
`research/h-means/gpu_design.md`:

1. No custom kernel: cells padded to the same size, batched matmul. This is
   the baseline to beat.
2. Triton kernels, as o-means. PyTorch only.
3. CUDA C kernels through CuPy `RawKernel`, called on PyTorch tensors with
   DLPack. Works for CuPy and PyTorch inputs.

For each: speed, how it handles cells of different sizes, fixed against
adaptive number of probes (adaptive is irregular: masking may be enough),
float16 or float32 distances (o-means uses float16) and the effect on
inertia, new dependencies, and what happens when the dependency is missing.
Recommend one and continue with it unless the measurements say the gain is
not there. In that case stop and report.

## Known trap

o-means fails on the L4 at `k=100k` with its default cell count, and with 632
cells: `out of resource: shared memory, Required: 135168, Hardware limit:
101376` in the Triton kernel of the coarse assignment. Your kernels must run
on the L4 at `k=100k` and `k=1M`. Don't size blocks from constants that assume
a larger GPU.

## Requirements

- Same public API as the CPU path (suggested there:
  `assignment="two-level"`). With `array_api_dispatch=True` and a CUDA input,
  the fit stays on the device.
- Exact assignment unchanged.
- CuPy, PyTorch and Triton are optional: import them inside the functions.
  Without the dependency of the kernels, fall back or raise a clear error.
- Random numbers drawn on the host from `random_state`, in the same order as
  the CPU path. Devices without float64 accumulate in float32, as today.
- `labels_` and `inertia_` come from the two-level labelling pass.
- Other devices (MPS, XPU, array-api-strict): out of scope, with a clear
  error or a documented fallback to exact.

## Tests

In `tests/test_array_api.py` (run with `SCIPY_ARRAY_API=1`), on CUDA when
available:

- When all the cells are probed, the result equals the exact backend up to
  rounding.
- Equivalence with the CPU two-level path from the same `random_state`, up to
  rounding, once that path exists.
- Empty cells, revived centers, cells much larger than the average.

## Benchmarks

On `aws-l4-3` (the VM is powered on on demand), slot `all`, under `benchlock`
(see the global instructions). Datasets are in `~/real-emb-data`. The disk is
almost full and `/dev/shm` is wiped at logout: `research/h-means/exp.py`
downloads DEEP 10M into memory at start. `research/h-means/omeans_run.py`
runs the faiss dense e-means and o-means with the same metrics.

Compare, in the same `benchlock` call: exact `EMeans` on GPU, faiss dense
e-means, o-means with `sqrt(k)` cells (nprobe 4 and 8), and your path.

Done means:

- loss on labels at most 0.5% against exact with 5 epochs, on SIFT 1M, GloVe,
  CLIP 1M at `k=10k` and DEEP 10M at `k=10k` and `k=100k`;
- within 1.5x of the o-means times at equal loss on DEEP 10M (0.74 s at
  `k=10k`, 1.6 s at `k=100k`, with nprobe=4);
- the scaling run at 100 samples per cluster on DEEP (1M, 3M, 10M), with the
  measured exponent (o-means: 1.38 to 1.44, dense: 1.94).

Add the numbers, with the commands, to `research/h-means/README.md`.

## Report

The design you chose and why, the benchmark table, anything that does not
meet the requirements above, what must be aligned with the CPU path, and the
open questions for the user.
