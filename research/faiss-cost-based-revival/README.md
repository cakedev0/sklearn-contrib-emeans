# Cost-based revival of faiss#5915 against `revival="greedy"`

[facebookresearch/faiss#5915](https://github.com/facebookresearch/faiss/pull/5915)
makes a cost-based revival the default of `NumpyEmeans` and `Emeans`. This
folder compares it with the revivals of `EMeans` on the datasets where the
revival matters: three of the k-means Pareto benchmark of scikit-learn-benchmarks
and the five tabular datasets of the PR thread.

Run on 2026-11-22, 3 seeds per case, sklearn-contrib-emeans 0.4.1 from PyPI:

- CPU: faiss built at the PR head `ba7781af` (`NumpyEmeans`), Core Ultra 9 285
  (24 threads).
- GPU: NVIDIA L4 (8 vCPUs), PyTorch 2.13, Triton 3.7.1, the `contrib/e_means`
  of `ba7781af` on top of the faiss 1.16.0 wheel (`Emeans`, Triton backend).
  Same datasets without SUSY.

## TLDR

- The two rules fix the same problem and mostly end within ~1% of each other.
  Both remove the gap of the mass-based revival: 16-18% on the road network,
  2% on YearPredictionMSD, 5% on Census, ~40% on household power, more than
  100% on GeoNames.
- The PR's rule is better from a random start on KDD Cup 99 at k <= 1024
  (held-out error 0.53 against 0.81 at k=1024, 5 epochs), without k-means||.
- Greedy with k-means|| is better on KDD at k=4096 (training inertia 0.097
  against 0.85) and on the training inertia of Shuttle (0.0098 against 0.017 at
  k=1024). The PR's training inertia on KDD at k=4096 changes a lot with the
  seed (1.18, 1.17, 0.19).
- On KDD and Shuttle, the PR's rule ends with many centers that are nearest to
  no training row: ~320 of 4096 on KDD and ~260 of 1024 on Shuttle at 5 epochs,
  against 0 to 1 for greedy.
- faiss's new top-2 search has a slow case: 75 ms against 0.8 ms for top-1 with
  65,536 queries, 1024 centers and 3 features (`knn.py`). It makes the
  cost-based fits on the road network at k=1024 and 10 epochs take 45 s,
  against 3.6 s with the mass-based revival.

- Cost: the cost-based rule adds 30 to 50% to the fit time of `NumpyEmeans`
  and 10 to 25% to the GPU `Emeans`, at any k. Greedy adds ~30% up to k=1024
  on CPU, ~2x at k=4096, and it is quadratic in k. On GPU greedy runs on the
  host and dominates the fit: 18x the fit time at k=4096 (median).
- On GPU the quality is the same as on CPU, and BF16 operands show no effect
  (the mass-based `Emeans` in BF16 and `QuantizedEmeans` in FP32 give the same
  inertia). The cost-based rule only exists in `Emeans` (BF16 on a CUDA FP32
  input): `QuantizedEmeans(quantizer=None)` keeps the mass-based default.

All the tables are in [results/tables.md](results/tables.md) (CPU) and
[results/tables_gpu.md](results/tables_gpu.md) (GPU), the raw runs in
`results/results.jsonl` and `results/results_gpu.jsonl`.
[faiss_pr_comment_draft.md](faiss_pr_comment_draft.md) is the draft of a comment
for the PR.

## What is compared

| Variant | Library | Settings |
| --- | --- | --- |
| PR cost | faiss `NumpyEmeans` | defaults of the PR (cost-based revival) |
| PR mass | faiss `NumpyEmeans` | `unused_threshold=0.1` |
| PR cost, km\|\| | faiss `NumpyEmeans` | defaults, initial centroids from our k-means\|\| |
| EM 0.1 | `EMeans` | defaults (`revival=0.1`, `init="random"`) |
| EM greedy | `EMeans` | `revival="greedy"` |
| EM km\|\| | `EMeans` | `init="k-means\|\|"` |
| EM greedy, km\|\| | `EMeans` | `revival="greedy"`, `init="k-means\|\|"` |

- Pareto benchmark datasets: `road_network_km_x10` (4.3M x 3),
  `year_prediction_msd_standardized` (515k x 90) and `susy` (5M x 18), at
  k = 64, 256, 1024 and 2, 5, 10 epochs.
- Datasets of the PR thread, prepared with the thread's own script (duplicates
  removed, 20,000 rows held out, z-score): `kdd_z` (1.05M x 122), `census_z`
  (1.98M x 68), `geonames` (1M x 2), `power_z` (1.89M x 7) and `shuttle_z`
  (38k x 9), at the k values of the thread and 2, 5 epochs. They also get the
  held-out error, on the test half of the held-out rows.

The inertia of every run is recomputed from the final centers in float64, with
an exact assignment.

On GPU (`bench_gpu.py`), the variants are the same, with `Emeans` for the PR
(BF16 operands), plus `QuantizedEmeans(quantizer=None)` (FP32, mass-based), and
`EMeans` on PyTorch CUDA tensors with `array_api_dispatch=True`.

## Caveats

- On GPU, the fit time of "PR cost, km||" includes our k-means|| called
  outside of `EMeans.fit`, without its BLAS limit: it is not meaningful (41 s
  at k=4096 on Census, against ~1.5 s inside `EMeans`).

- 3 seeds per case, and the two libraries draw different random starts.
- KDD and Shuttle are heavy-tailed: their means move a lot with one seed. The
  held-out error of Shuttle depends on one test row far from all the training
  rows.
- The fit times compare the NumPy reference path of faiss with the Cython
  kernels of `EMeans`: only the ratios within a library say something about
  the cost of a revival. The PR's cost-based fits take 1.2 to 1.6 times its
  mass-based ones. Greedy takes 1.05 to 1.3 times `revival=0.1` up to k=1024,
  and ~2 times at k=4096.

## Reproduce

The scripts expect to be in `~/work/faiss-pr5915`, with the data cache of
scikit-learn-benchmarks in `~/scikit-learn-benchmarks/data_cache`.

```
pixi install
pixi run bash build.sh        # faiss at the PR head, with its Python bindings
bash prep.sh > prep.log 2>&1  # datasets of the PR thread
benchlock --tag "revival comparison" all -- bash run_all.sh
python analyze.py > results/tables.md
```

The GPU run reuses an existing PyTorch environment (the root disk of the VM was
full), with `pip install --no-deps --target overlay faiss-cpu==1.16.0
sklearn-contrib-emeans==0.4.1`, the `contrib/` of the PR in `pr/contrib`, and
the datasets as `.npy` files in `npy/` and `data/`:

```
benchlock --tag "revival comparison, GPU" all -- bash run_gpu.sh
python analyze.py gpu > results/tables_gpu.md
```

`pixi.toml` asks for the pthreads build of OpenBLAS: with the OpenMP one, the
fits of the `EMeans` wheel are ~80x slower at k=1024
([#27](https://github.com/cakedev0/sklearn-contrib-emeans/issues/27)).
