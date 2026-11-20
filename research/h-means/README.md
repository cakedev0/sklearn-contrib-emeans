# h-means: two-level e-means

Research prototype and experiment journal. The prototype (`hkm.py`,
`hkm_kernels.pyx`) is not part of the `sklearn_emeans` package. The coupled
two-level method with adaptive probing is now in the package for dense NumPy
input, as `EMeans(assignment="two-level")`: see
[CPU implementation in `EMeans`](#cpu-implementation-in-emeans).

At a fixed cluster size `s`, e-means costs `O(n k d) = O(n^2 d / s)`. The
question is whether a two-level scheme can bring this down to about
`O(n^1.5 d / sqrt(s))` while losing less than 1% of inertia on embedding
datasets, with 10 to 10k samples per cluster (main target: 1,000).

**Short answer: yes, when the two levels are coupled.** One e-means runs on all
the `k` centers, and a coarse level of `sqrt(k)` groups of centers is only used
to choose which centers each sample is compared to. The groups are refreshed
during the fit. With an adaptive number of probed groups, the loss is at most
+0.3% at 100 and 1,000 samples per cluster on four datasets, and the fit time
grows about like `n^1.0` to `n^1.3` between 1M and 100M samples, against
`n^1.9` for `EMeans`. Clustering each coarse cluster independently (the frozen
two-level scheme) loses 2% to 4%.

## Methods

All the methods use the e-means step of `EMeans` (same batch and step size
laws, cosine schedule, revival, final centers from the last epoch, extra
labelling pass), 5 epochs, random init. Only the assignment changes.

| name | assignment |
|---|---|
| `flat` | Exact, all the centers. Same algorithm as `EMeans`, in the prototype. |
| `h2` | Frozen two-level: e-means into `g = sqrt(k)` groups, then an independent e-means inside each group, with a number of sub-clusters proportional to the size of the group. |
| `ivf` | Coupled two-level: `g = sqrt(k)` coarse centers, each fine center belongs to its nearest coarse center. A sample is compared to the fine centers of its `p` nearest groups. The coarse level is refreshed during the fit (`coarse`: `lloyd`, `reassign`, or `frozen`). |
| `ivf`, adaptive | Same, but group `q > 0` is probed only when its distance is at most `ratio` times the distance of the nearest group, up to `p` groups. |
| `graph` | A sample is compared to the `m` nearest centers of its previous center. The kNN graph of the centers is rebuilt every `graph_period` steps. |

`p_final` is the number of probes of the final labelling pass.

"Work" is the number of sample-center dot products divided by `n k` (`flat`
with 5 epochs and the labelling pass is 6.0).

Competitors:

- `emeans`: `sklearn_emeans.EMeans` 0.4.0.
- `superemeans`: `SuperEMeans` of [cwida/SuperKMeans](https://github.com/cwida/SuperKMeans)
  (main at 50ee8ca), e-means with dimension pruning. It keeps the `O(n k)`
  dependence.
- `faiss-e` and `omeans`: the dense e-means and o-means of the
  owlgebra-ai/faiss fork (`contrib/e_means`, `contrib/o_means`), GPU only.
  o-means is the same design as `ivf`: `k / 128` cells and 12 to 24 probes by
  default, cells rebuilt during the fit.

## Setup

- Datasets: SIFT 1M (128-d), GloVe 1.18M (200-d), Wolt CLIP 1M (512-d),
  DEEP 10M and 100M (96-d).
- CPU: one socket of a 2-socket Xeon server, 64 threads (`benchlock` slot `s0`
  or `s1`).
- GPU: one NVIDIA L4, PyTorch, float32.
- Loss: inertia relative to the flat reference with 5 epochs on the same data.
  "centers" is the exact inertia of the returned centers, "labels" is the
  inertia of the returned partition (what a user gets from `labels_`). Inertia
  is computed on 1M random samples (200k when `k > 20000`).
- One seed. The seed-to-seed noise is about 0.05%.

## Results

### Scaling at 1,000 samples per cluster (DEEP, CPU)

Fit time in seconds. Loss is on labels, against `emeans` with 5 epochs.
Adaptive is `ivf` with `p=16, ratio=1.15, p_final=16`.

| n | k | `emeans` | `superemeans` | `h2` | `ivf` p=4 | adaptive | loss p=4 | loss adaptive |
|---|---|---|---|---|---|---|---|---|
| 1M | 1k | 0.68 | 0.55 | 1.57 | 1.13 | 1.19 | +0.19% | +0.20% |
| 3M | 3k | 3.48 | 2.72 | 3.37 | 2.73 | 3.00 | +0.22% | +0.19% |
| 10M | 10k | 40.6 | 22.9 | 9.73 | 8.19 | 10.0 | +0.27% | +0.21% |
| 30M | 30k | 414 | 193 | 27.1 | 26.2 | 32.6 | +0.33% | +0.24% |
| 100M | 100k | pending | pending | 95.4 | 113 | 149 | | |

- Exponent of the time against `n`, from 1M to 30M: `emeans` 1.89,
  `superemeans` 1.72, adaptive 0.97, `ivf` p=4 0.92.
- Exponent of the number of dot products: 1.50 for a fixed `p`, 1.54 for
  adaptive.
- The time of the two-level methods grows slower than `n^1.5` on this range
  because the per-sample overheads still matter at 1M, and the GEMMs get more
  efficient when the groups grow. The local exponent of adaptive rises with
  `n`: 0.84, 1.00, 1.08, 1.26. It should tend to 1.5.
- The loss of adaptive stays between 0.19% and 0.24% over the 30x range. With
  a fixed `p=4` it creeps up.
- `h2` loses 3.1% to 3.4% at all sizes.
- At 100M, `emeans` with one epoch takes 1,611 s. Adaptive with 5 epochs takes
  149 s for a 0.9% lower inertia, and adaptive with one epoch takes 51 s (32x)
  for a 0.11% higher inertia.

The 100M references with 5 epochs (`emeans`, `superemeans`) were still running
when this was written.

### Worst loss over the datasets (CPU, labels)

| samples per cluster | adaptive, ratio 1.15 | adaptive, ratio 1.2 | `ivf` p=4, p_final=8 | `h2` |
|---|---|---|---|---|
| 1,000 | +0.27% | +0.14% | +0.33% | +3.4% |
| 100 | +0.29% | +0.11% | +0.52% | +4.0% |
| 10 | +0.97% | +0.47% | +2.1% | +8.9% |

1,000 per cluster: SIFT, GloVe and CLIP at `k=1000`, DEEP from 1M to 30M.
100 per cluster: the 1M sets at `k=10k`, DEEP 10M at `k=100k`. 10 per cluster:
the 1M sets at `k=100k`.

### Fit time on CPU, adaptive against `emeans`

| data | k | `emeans` | `superemeans` | adaptive | speed-up | loss (labels) |
|---|---|---|---|---|---|---|
| SIFT 1M | 1k | 1.11 s | 1.52 s | 0.81 s | 1.4x | +0.27% |
| SIFT 1M | 10k | 7.26 s | 3.87 s | 1.23 s | 5.9x | +0.29% |
| SIFT 1M | 100k | 75.4 s | 13.6 s | 3.63 s | 21x | +0.97% |
| GloVe 1.18M | 10k | 6.97 s | 4.67 s | 2.60 s | 2.7x | +0.04% |
| GloVe 1.18M | 100k | 75.6 s | 31.2 s | 8.00 s | 9.5x | +0.12% |
| CLIP 1M | 10k | 11.8 s | 4.50 s | 2.16 s | 5.4x | +0.10% |
| CLIP 1M | 100k | 113 s | 28.9 s | 8.79 s | 13x | +0.63% |
| DEEP 10M | 10k | 67.1 s | 39.1 s | 11.0 s | 6.1x | +0.23% |
| DEEP 10M | 100k | 788 s | 243 s | 16.4 s | 48x | +0.28% |

- These runs (`v4_final.jsonl`) shared the machine with a run on the other
  socket, so they are slower than the scaling table (`emeans` on DEEP 10M:
  67 s here, 40.6 s there).
- At 1M samples and `k=1000`, flat takes about 1 s and there is nothing to
  gain.
- On GloVe the ratio rule prunes little (work 1.2 at `k=10k`). A fixed `p=4`
  takes 1.15 s there (6x) for +0.28%.
- `superemeans` is 1.5x to 5x faster than `emeans` at the same inertia, with
  the same growth in `n k`.

### GPU: o-means with `sqrt(k)` cells

Fit time in seconds on the L4, loss on labels against `faiss-e` with 5
epochs. DEEP, 100 samples per cluster:

| n | k | `faiss-e` | `omeans` nprobe=4 | loss | `omeans` nprobe=8 | loss |
|---|---|---|---|---|---|---|
| 1M | 10k | 0.343 | 0.065 | +0.79% | 0.090 | +0.23% |
| 3M | 30k | 2.77 | 0.257 | +0.84% | 0.401 | +0.29% |
| 10M | 100k | 29.6 | 1.57 | +0.94% | 2.47 | +0.39% |

- Exponents: `faiss-e` 1.94, `omeans` 1.38 (nprobe=4) and 1.44 (nprobe=8).
- With its default `k / 128` cells and 16 to 24 probes, `omeans` is 1.1x to
  2.7x faster than `faiss-e` with no measurable loss. At `k=100k` it fails on
  the L4, also with 632 cells (only 316 cells ran). The Triton kernel of the
  coarse assignment (`triton_two_level.py`) needs more CUDA shared memory than
  the L4 has: `out of resource: shared memory, Required: 135168, Hardware
  limit: 101376`. I did not try a GPU with a higher limit.
- On the 1M sets at `k=10k`, 100 cells and nprobe=4 give 3.8x to 4.8x for
  +0.34% to +0.70%. At `k=100k`, nprobe=8 gives 6x to 8.8x for +0.42% to
  +0.93%.
- The loss in the table comes from the settings chosen here (`sqrt(k)` cells,
  4 or 8 probes), not from o-means as shipped: with its defaults the loss is
  at most +0.05% in all the runs. The defaults just buy less speed.
- With these settings the loss grows with `n` (+0.79% to +0.94% with
  nprobe=4), as with a fixed `p` in the CPU prototype. o-means has neither the
  adaptive rule nor extra probes for the final labelling. They would probably
  help it as they help the prototype, but this was not tested in o-means.
- The PyTorch prototype on GPU is bound by Python overhead (about 1.3 s of GPU
  work in a 4 s fit on DEEP 10M), so its GPU timings (`l4/v1_main.jsonl`,
  `l4/v3_adaptive.jsonl`) are only good for the inertia.

## Journal

1. **Frozen two-level (`h2`).** +0.7% to +2.5% on centers and +1.6% to +4.0%
   on labels at `k=1000` and `k=10k`. Most of the loss on labels comes from
   samples close to a coarse boundary (recall of the exact nearest center:
   0.80 to 0.89).
2. **Coupled (`ivf`), `p=4`.** +0.1% to +0.4% on centers, +0.2% to +0.6% on
   labels, with 18x fewer dot products at `k=10k`.
3. **The coarse level must follow the fine centers.** With `coarse=frozen`
   (groups fixed after the init), the loss at `k=10k` is +2.1% on SIFT, +3.9%
   on GloVe and +4.9% on CLIP, against +0.4% to +0.6% when the groups are
   refreshed. `reassign` (only update the membership of the fine centers) is
   close to `lloyd` (also move the coarse centers).
4. **Init from `h2`** (`init=h2`): no clear gain over a random init for `ivf`.
5. **Graph.** The cheapest in dot products at large `k` (work 0.02 at
   `k=100k`), but less accurate: +0.7% to +1.7% on labels at 1,000 samples per
   cluster, and worse as `n` grows. Two hops (`hops`) did not help. Slow on GPU
   (gathers).
6. **More probes for the final labelling (`p_final`).** It removes most of the
   gap between centers and labels for one pass with more probes: DEEP 10M,
   `p=4`: +0.57% to +0.29% with `p_final=8`. Using more probes in the last
   epochs (`late_epochs`) was not worth its cost.
7. **Adaptive probing.** At equal work it halves the loss of a fixed `p` on
   SIFT, CLIP and DEEP. On GloVe the nearest groups are almost equidistant, so
   it probes many groups: better inertia, smaller speed-up.
8. **Loss against `k` for a fixed `p=4`** (DEEP 10M, centers): +0.21%, +0.31%
   and +0.55% at `k=1k`, `10k` and `100k`. At 10 samples per cluster a fixed
   `p=8` or the adaptive rule is needed to stay under 1%.
9. **CPU kernels** (`hkm_kernels.pyx`), what mattered:
   - Bucket the (sample, group) pairs by group and run one GEMM per (group,
     chunk of samples). A row-wise version without BLAS was 3x to 4x slower.
   - `sgemv` of OpenBLAS takes a global lock: 20 s to 78 s fits on 512-d data
     before it was removed.
   - Reuse the work buffers between the steps (page faults), sort the pairs
     with a parallel counting sort, update the centers in place.
   - Keep the same number of threads in all the parallel regions, otherwise
     libgomp recreates its threads (v3 regression on small inputs).
   - The prototype reaches about a third of the arithmetic rate of the flat
     GEMM, so a tuned kernel has room left.

## CPU implementation in `EMeans`

`EMeans(assignment="two-level")` is the `ivf` method with adaptive probing
(`p=16, ratio=1.2`), in Cython, for dense NumPy input. Differences with the
prototype:

- The whole step loop is in `_kernels.pyx`, in one parallel region, as for the
  exact assignment. The result does not depend on the number of threads.
- The ratio is 1.2 in place of 1.15, in the steps and in the labelling pass
  (see "The ratio" below).
- Below 1024 clusters, `"two-level"` is the exact assignment.
- Ties go to the lowest center index, as with the exact assignment. When all
  the cells are probed, the result is the one of the exact assignment up to
  rounding.

### Results

One socket of the Xeon 6787P server (`benchlock` slot `s1`), 64 threads, 5
epochs, random init, one seed. `EMeans` exact, the prototype (`v4`, adaptive,
with its ratio of 1.15 and with 1.2) and `EMeans` two-level ran in the same
`benchlock` call. Times are fit times in seconds, the median of 3 fits for
two-level. The loss is the inertia against `EMeans` exact, on 1M random
samples (200k at `k=100k`): "centers" is the exact inertia of the centers,
"labels" the one of `labels_`.

| data | k | exact | prototype 1.15 | prototype 1.2 | two-level | speed-up | loss (centers) | loss (labels) |
|---|---|---|---|---|---|---|---|---|
| SIFT 1M | 10k | 4.84 | 1.44 | 1.72 | 1.05 | 4.6x | +0.04% | +0.09% |
| GloVe 1.18M | 10k | 6.90 | 2.55 | 2.61 | 2.42 | 2.9x | +0.02% | +0.05% |
| CLIP 1M | 10k | 11.1 | 1.91 | 2.28 | 2.59 | 4.3x | +0.12% | +0.14% |
| DEEP 10M | 10k | 40.4 | 10.0 | 12.9 | 6.29 | 6.4x | +0.05% | +0.10% |
| DEEP 10M | 100k | 485 | 16.0 | 20.4 | 13.1 | 37x | +0.05% | +0.11% |

- The loss on labels is at most +0.14%. The loss of the prototype in these
  runs is +0.29%, +0.04%, +0.10%, +0.30% and +0.44% with its ratio of 1.15,
  and +0.09%, +0.06%, 0.00%, +0.17% and +0.28% with 1.2 (same seed, but its
  evaluation sample is another one, so compare to within about 0.05%).
- Two-level is faster than the prototype on four cases out of five: by 1.1x
  to 2.1x with the same ratio, and by 1.05x to 1.6x against the prototype
  with its ratio of 1.15. On CLIP it is slower (2.59 s against 2.28 s, and
  1.91 s with 1.15): see "What does not meet the target".

Scaling at 1,000 samples per cluster on DEEP (the first `n` samples of DEEP
100M):

| n | k | exact | prototype 1.15 | prototype 1.2 | two-level | loss (labels) |
|---|---|---|---|---|---|---|
| 1M | 1k | 0.68 | 1.24 | 1.53 | 0.62 | +0.01% |
| 3M | 3k | 3.44 | 3.10 | 3.69 | 1.86 | +0.05% |
| 10M | 10k | 40.3 | 10.3 | 12.6 | 6.27 | +0.08% |
| 30M | 30k | 405 | 34.7 | 42.9 | 24.6 | +0.08% |

- Exponent of the time against `n`, from 1M to 30M: 1.88 for exact, 0.98 for
  the prototype, 1.08 for two-level. The local exponent of two-level rises
  with `n`: 1.00, 1.01, 1.24. It should tend to 1.5: the number of
  comparisons grows like `n sqrt(k)`, and the costs that are linear in `n`
  (check of the input, permutations, copies of the batches) still take most
  of the time at 1M samples.
- At 1M samples and `k=1000`, `"two-level"` is the exact assignment in the
  package (below 1024 clusters). The two-level time in the table has this
  fallback turned off (`--min-clusters 1`).

Where two-level starts to win (same machine, `exact` / `two-level` in seconds,
median of 3 fits, 2 on DEEP 10M, with the fallback turned off):

| k | SIFT 1M | DEEP 1M | CLIP 1M | DEEP 10M |
|---|---|---|---|---|
| 512 | 0.55 / 0.54 | 0.48 / 0.65 | 1.02 / 1.27 | 3.78 / 4.33 |
| 1024 | 0.72 / 0.58 | 0.69 / 0.65 | 1.51 / 1.33 | 5.37 / 4.40 |
| 2048 | 1.02 / 0.64 | 0.98 / 0.69 | 2.40 / 1.59 | 8.12 / 4.75 |

The time of two-level hardly depends on `k` in this range (about 0.6 s per
million samples with 96 or 128 features), so it loses or ties at 512 clusters
and wins from 1024. Hence the fallback to the exact assignment below 1024
clusters. An earlier run from 128 to 4096 clusters, with a ratio of 1.15 for
the steps and 1.3 for the labels, gave the same crossover
(`results/xeon/cpu_small_k.jsonl`).

Commands (on the server, in a directory with a venv where the branch is
installed):

```
benchlock --tag "h-means cpu main" s1 -- ./scripts/xeon_cpu_main.sh
benchlock --tag "h-means cpu scaling" s1 -- ./scripts/xeon_cpu_scaling.sh
python show_cpu.py results/xeon/cpu_main.jsonl results/xeon/cpu_main_proto.jsonl
OMP_NUM_THREADS=64 python bench_cpu.py --data sift1m --k 512 1024 2048 \
    --min-clusters 1 --repeat 3 --out res/small_k.jsonl
```

### The ratio

A sample probes the cells that are at most `ratio` times as far as its
nearest cell. The prototype uses 1.15, in the steps and in the labelling
pass. With that, the loss on labels was +0.30% on SIFT 1M, right at the
target of 0.3%. Loss on labels and fit time in seconds, one fit each (2 for
the first setting), in one `benchlock` call (`results/xeon/cpu_ratio.jsonl`):

| data | k | 1.15 | 1.15, labels 1.3 | 1.2 |
|---|---|---|---|---|
| SIFT 1M | 10k | +0.30%, 0.97 | +0.18%, 1.06 | +0.09%, 1.05 |
| GloVe 1.18M | 10k | +0.06%, 2.52 | +0.06%, 2.61 | +0.05%, 2.70 |
| CLIP 1M | 10k | +0.21%, 2.41 | +0.16%, 2.36 | +0.14%, 2.66 |
| DEEP 10M | 10k | +0.21%, 5.62 | +0.11%, 6.22 | +0.10%, 6.84 |
| DEEP 10M | 100k | +0.23%, 11.3 | +0.16%, 12.3 | +0.11%, 13.8 |

- A wider ratio for the labelling pass only does not change the centers, but
  99.5% to 99.95% of the samples then get their closest center (98.4% to
  99.5% with 1.15).
- 1.2 everywhere gives better centers, for 8% to 22% more time than 1.15
  everywhere. 99.3% to 99.8% of the samples get their closest center.

The package uses 1.2 everywhere: one rule, and the lowest loss of the three.

### What mattered for the speed

Per-phase times are from a build with timers around each phase, not
committed. They were measured with a ratio of 1.15 for the steps and 1.3 for
the labelling pass.

- **Size of the GEMMs.** With 256 rows per GEMM as in the exact assignment
  and in the prototype, a product costs 2 to 3 times more than in the big GEMM
  of the exact assignment (4.8 ns against 2.1 ns for 128 features), for the
  coarse and for the fine search. With 1024 rows: DEEP 10M at `k=10k` goes
  from 9.0 s to 5.6 s. 512 to 2048 rows are within 10% of each other, on 96
  to 512 features. This is what "a third of the arithmetic rate" was in the
  prototype.
- **Sort of the batch by label**, for the update of the centers. Sequential,
  it took 0.36 s out of 6.4 s on DEEP 10M. It is now a counting sort by the
  cell of the label, in parallel over blocks of rows, then by center inside
  each cell, in parallel over cells: 0.02 s.
- **Probe selection.** Keeping the probed cells sorted while going through the
  cells, in place of 16 passes over the candidates. It matters on GloVe, where
  most samples have more than 16 candidates.
- The squared norm of each row, for the ratio rule, took as long as the rest
  of the selection: the compiler can't vectorize a sum of floats in order.
  It now has 8 partial sums.

On DEEP 10M at `k=10k`, in the build with timers, where the fit took 6.6 s
(5.8 s without them): the steps take 3.1 s (of which 0.45 s to copy the
batches), the check of the input by scikit-learn 1.3 s, the labelling pass
1.0 s, and the permutations, the inits and the rest of the Python code about
1.2 s. The fine search is about 60% of the steps and of the labelling pass,
the coarse search 15%.

### What does not meet the target

- **CLIP 1M is slower than the prototype**: 2.59 s against 2.28 s with the
  same ratio, and 1.91 s with the ratio of 1.15 of the prototype. Most of
  the difference is the check of the input by scikit-learn (0.55 s for
  1M x 512 in float32, sequential), which the prototype does not do. With
  `sklearn.config_context(assume_finite=True)`, the fit was about 0.4 s
  shorter on CLIP (2.0 s to 2.2 s against 2.4 s to 2.6 s) and 0.6 s shorter
  on DEEP 10M at `k=10k` (measured with the earlier ratios).
- **The ratio is not the one of the prototype**: 1.2 in place of 1.15, see
  "The ratio".
- Prefetching the rows in the copies of the batch, and running the short
  GEMMs last so that the threads wait less at the end of the fine search,
  changed nothing measurable (interleaved A/B on the 5 cases), so they are
  not in.

### Semantics, for another backend

1. `g = max(1, round(sqrt(n_clusters)))` cells. The initial cells are
   `EMeans(n_clusters=g, n_epochs=10, random_state=rs).fit(centers)` on the
   initial centers, with the exact assignment and the `RandomState` of the
   fit. It is the first draw after the init of the centers, before the sample
   of the greedy revival and the permutations.
2. Refresh, before each step and before the labelling pass: each cell center
   becomes the mean of the centers of its cell, with the weights
   `max(weight_ema, 1e-12)` (a cell without center keeps its cell center),
   then each center goes to the cell of its nearest cell center, among all
   the cells.
3. A sample probes its nearest cell among the cells that have centers, and
   the cells with `d <= ratio**2 * d_nearest`, `d` being the squared
   distance, limited to the 16 nearest cells. `ratio` is 1.2, in the steps
   and in the labelling pass.
4. The sample goes to the nearest center of the cells it probes. Ties go to
   the lowest index, for the cells and for the centers.
5. After a revival, a revived center takes the cell of its donor
   (`cell_of_center[dead] = cell_of_center[donor]`), for the mean of the
   cells of the next refresh. Same for the greedy revival.
6. The labelling pass refreshes the cells from the final centers and the
   last EMA weights. `inertia_` is the sum of the squared distances of the
   samples to the center of their label.
7. Below 1024 clusters, the assignment is exact and no random number is
   drawn for the cells.

The constants are `_TWO_LEVEL_*` in `sklearn_emeans/_emeans.py`, and
`tests/test_two_level.py` has a NumPy reference of the refresh and of the
assignment.

## Limits

- One seed per configuration, one machine per device type.
- All the datasets are dense embeddings with 96 to 512 dimensions.
- Random init only. k-means|| would cost `O(n k)` again unless it is also
  made two-level.
- `g = sqrt(k)` and the probing parameters were tuned by hand on these
  datasets.
- The 100M runs have no flat reference with 5 epochs yet.

## Files

- `hkm.py`: the prototype (PyTorch, CPU and GPU). `run(X, cfg, eval_rows)`
  runs one configuration.
- `hkm_kernels.pyx`, `setup_kernels.py`: Cython and OpenMP kernels for the CPU.
- `exp.py`: runs a grid of configurations on one dataset and appends JSON
  lines. Also runs `EMeans` and `SuperEMeans`.
- `omeans_run.py`: faiss dense e-means and o-means on GPU.
- `grids/`: the grids of configurations. `mkgrid.py` wrote the first ones.
- `scripts/`: the commands that were run on each machine, each under
  `benchlock`.
- `results/xeon/`, `results/l4/`: raw JSON lines, one per run. `v1` to `v4`
  are versions of the CPU kernels: the inertia is valid in all of them, the
  CPU timings only in `v4`. `cpu_*.jsonl` are the runs of `EMeans`
  (`bench_cpu.py`) and of the prototype in the same `benchlock` call:
  `cpu_main` and `cpu_scaling` are the tables above, `cpu_small_k` the
  search of the fallback, `cpu_ratio` the ratios (1.15 everywhere, 1.2
  everywhere, 1.3 for the labels, with an earlier version of the kernels) and
  `cpu_ab` the A/B of the prefetch and of the task order, with and without
  `assume_finite`.
- `results/all_runs.csv`, `results/scaling.csv`: written by `summarize.py`.
- `show.py`, `show_om.py`: print tables from JSON lines.
- `prof.py`, `prof_cpu.py`: profiling helpers.
- `bench_cpu.py`: times `EMeans` with the exact and the two-level
  assignment. `show_cpu.py` prints its results next to the ones of `exp.py`.

To run a grid on CPU:

```
python setup_kernels.py build_ext --inplace
python exp.py --data sift1m --device cpu --threads 64 \
    --configs grids/f_1m.json --out res.jsonl
python show.py res.jsonl
```

`exp.py` looks for `<name>.npy` (or `deep10M.fbin`, `deep100M.fbin`) in
`~/work/hkm/data` and `~/real-emb-data`.
