# Two-level assignment on GPU: design

How to run the two-level assignment of `EMeans` on a CUDA device. Three
options were measured on one NVIDIA L4 (`aws-l4-3`), on the assignment alone.
The choice is Triton kernels, on float16 copies of the rows and centers.

## Benchmark

`gpu_design/bench_assign.py` assigns one batch of 2^20 rows of DEEP (96-d,
float32) to `k` centers through `round(sqrt(k))` cells. The centers are random
rows after one Lloyd iteration, and the cells a k-means of the centers. The
three probing rules are a fixed `p=4`, a fixed `p=8` and the adaptive rule of
the CPU path (ratio 1.15, at most 16 cells), which probes 6.4 cells per row at
`k=10k` and 7.0 at `k=100k` here. `gpu_design/run_design.sh`, `run_cuda.sh` and
`run_stages.sh` are the commands, run under `benchlock`, and the raw results
are in `results/l4/gpu_design.jsonl` and `gpu_design_stages.jsonl` (3 stages
at `k=10k`, then one stage and blocks of 16 rows at `k=100k`).

Time in ms for the 2^20 rows, adaptive rule unless a `p` is given:

| | k=10k | k=100k | k=1M |
|---|---|---|---|
| exact, one GEMM (float32) | 530 | 6074 | |
| exact, one GEMM (float16) | 134 | 1507 | |
| 1. padded cells, `baddbmm` (float32 / float16) | 340 / 225 | 1340 / 798 | |
| 1b. one GEMM per cell in a Python loop (float32 / float16) | 136 / 102 | 255 / 168 | |
| 2. Triton, float16 | 27 | 42 | 222 |
| 2. Triton, float32 on the tensor cores (TF32) | 34 | 67 | 364 |
| 2. Triton, float32 products (IEEE) | 65 | 169 | |
| 2. Triton, float16, `p=4` / `p=8` | 14 / 27 | 23 / 41 | 128 / 231 |
| o-means kernels (float16), `p=4` / `p=8` | 10.5 / 23 | 20 / 41 | fails |
| 3. CUDA C through CuPy `RawKernel` (float32) | 467 | 1318 | 2373 |

The `k=1M` column has 2M rows for 1M centers and timings that vary by 2x from
one run to the next: it shows that the kernels run, nothing more.

## The options

**1. No custom kernel.** The pairs (row, probed cell) are sorted by cell, the
cells are padded to the largest one and the pairs of each cell to the largest
number of pairs, then one `baddbmm` compares them. The padding costs 4x at
`k=10k` and 7x at `k=100k` (largest cell: 2 to 3 times the mean, and as much
for the number of pairs per cell), on top of a gather of the rows of all the
pairs. It is slower than a Python loop with one GEMM per cell (1b), which is
the prototype of `hkm.py`. The adaptive rule is a mask on the result of
`topk`. No dependency. Both are 5x to 30x slower than option 2.

**2. Triton kernels**, with the layout of o-means: a coarse kernel selects
the probes of a block of rows, the pairs are grouped by cell, a fine kernel
runs one program per tile of 128 pairs of a cell against the centers of the
cell, a last kernel reduces the pairs of each row. `tl.dot` runs on the tensor
cores, which CUDA C can't use. With float16 inputs and a fixed `p`, it is as
fast as the kernels of o-means (14 ms against 10.5 ms, 23 ms against 20 ms).
PyTorch only: Triton takes PyTorch tensors. A CuPy array is viewed as a
tensor with DLPack, without copy, when PyTorch and Triton are installed.

**3. CUDA C through CuPy `RawKernel`.** The coarse scores come from one GEMM
and a kernel selects the probes (53 ms at `k=10k`, against 4 ms for the Triton
kernel that keeps the scores in registers). The fine kernel is a tiled product
in shared memory with scalar FMAs, 15x to 30x slower than option 2. This
kernel is simple and could be tuned, but the float32 products of option 2
(IEEE line) are what a tuned scalar kernel can hope for, and they are 2.5x to
4x slower than float16 on the tensor cores. It would work for CuPy and PyTorch
inputs with CuPy as the only dependency.

## Choices in option 2

**Cells of different sizes.** The centers are sorted by cell, and a program of
the fine kernel loops over the centers of its cell by blocks of 128, with a
mask on the last block. Nothing is padded. A cell 100 times larger than the
others only makes its programs longer.

**Adaptive number of probes.** A mask is enough. The coarse kernel puts the
threshold of the ratio rule on the scores, then extracts the nearest cells
one per round, up to 16 rounds, and writes -1 once a row has no cell left.
The pairs are counted per cell with atomics in the same kernel, a prefix sum
gives the place of each cell, and a second kernel writes the pairs. o-means
writes them directly in buckets of `batch_size` pairs per cell, which needs
`n_cells * batch_size` integers (4 GB at `k=1M`) but one kernel less. The 16
rounds cost: the coarse kernel takes 13 ms at `k=100k` with the adaptive rule
against 6 ms with `p=4`. Stopping the rounds when no row of the block has a
cell left did not help (13.9 ms), since a block of 32 rows almost always has a
row that probes many cells. With the adaptive rule the loss of the assignment
alone is +0.19% at `k=10k` and +0.52% at `k=100k`, against +0.27% and +0.90%
for `p=8`, for fewer probes.

**float16 or float32.** On the batch alone, float16 inputs are 1.3x faster
than TF32 at `k=10k` and 1.6x at `k=100k`: the rows are read once per probed
cell, and at `k=10k` the fine kernel is bound by this memory traffic more than
by the products. TF32 rounds the inputs to 10 bits of mantissa and float16 to
11, so they find the nearest center as often: 97.66% and 97.70% of the rows
with the adaptive rule at `k=10k`, against 97.73% with float32 products, and
the same loss to 0.001%. float16 needs a scale: the copies are multiplied by
the power of two that brings the largest absolute value of the data near 1024.
In a full fit the gain is smaller, because the float16 copy of each batch is
one more pass over it: see the README for the fit times with the three
precisions.

**The known trap.** The coarse kernel of o-means fails on the L4 from 512
cells in float32 and 1024 cells in float16 (`out of resource: shared memory`).
The cause is the software pipelining of Triton (`num_stages=3` by default):
each extra stage keeps a copy of the operands of `tl.dot` in shared memory,
and all the cells are one operand (`n_cells * 32 * itemsize` bytes per
stage). With `num_stages=1` the same kernel runs with 2048 cells on the L4
(`gpu_design/shmem_probe.py`), and is not slower: 14.2 ms with one stage
against 13.4 ms with two at `k=100k`, 4.3 ms against 5.2 ms at `k=10k`. The
float32 products without tensor cores need more (`block * n_cells * 4` bytes):
the kernel is launched with blocks of 32 rows, then 16, and when Triton
refuses both the probes are selected with PyTorch operations (`topk`). Nothing
is sized from the memory of a larger GPU.

**Missing dependency.** PyTorch and Triton are imported when the fit starts.
Without them, `assignment="two-level"` on a CUDA input raises an
`ImportError` that says so. There is no silent fallback to the exact
assignment: at `k=100k` it would be 60x slower than what the user asked for.
Other devices (CPU tensors, MPS, XPU, array-api-strict) raise a `ValueError`.

## Recommendation

Option 2. It is the only one that beats the exact assignment by more than the
ratio of the dot products, and the only one that can use the tensor cores.
The cost is that the GPU path needs PyTorch and Triton, also for CuPy inputs.
