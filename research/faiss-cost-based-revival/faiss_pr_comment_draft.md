Thanks for this PR! I had my agent benchmark the cost-based revival (`NumpyEmeans` at ba7781a) against the mass-based one, and against the `revival="greedy"` of [sklearn-contrib-emeans](https://github.com/cakedev0/sklearn-contrib-emeans), which is built on the same idea (removal cost against split gain) but estimates both on a sample, with trial 2-means splits. 8 datasets, 3 seeds, on CPU (`NumpyEmeans`) and on an L4 GPU (`Emeans`, Triton).

From these runs:

- The cost-based rule seems to remove most of the gap that the mass-based revival leaves (16-18% of inertia on a 3-D road network), and it ends within ~1% of greedy on most datasets. It looks better than greedy on KDD Cup 99 up to k=1024, and worse at k=4096.
- It also scales better: on GPU it adds 10 to 25% to the fit time at any k (30 to 50% in `NumpyEmeans`), while greedy is quadratic in k and runs on the host (~2x the fit time at k=4096 on CPU, much more on GPU). So I think the cost-based approach is the better one.
- A k-means|| init still seems useful on top of it: 0.5 to 4% lower inertia on the road network, Census and household power.
- One thing that may be worth a look: on KDD and Shuttle, many final centroids are nearest to no training row (~320 of 4096 on KDD, on CPU and GPU).

I didn't dig into it. Details below (with a slow case of `faiss.knn` with k=2 on CPU), and I can run more cases if that helps.

<details><summary>Details (from my agent)</summary>

Setup: sklearn-contrib-emeans 0.4.1, seeds 0 to 2, random init unless stated. CPU: faiss built at ba7781a, Core Ultra 9 285 (24 threads). GPU: NVIDIA L4, PyTorch 2.13, Triton 3.7.1, `contrib/e_means` of ba7781a on top of the faiss 1.16.0 wheel. The tables below are the CPU runs unless stated. The inertia is recomputed from the returned centroids in float64 with an exact assignment. Scripts, raw runs and all the tables are [here](https://github.com/cakedev0/sklearn-contrib-emeans/tree/h-means-bench/research/faiss-cost-based-revival).

Datasets: `road_network_km_x10` (4.3M x 3), YearPredictionMSD standardized (515k x 90) and SUSY (5M x 18) from a k-means benchmark of ours, and KDD Cup 99, Census, GeoNames, household power and Shuttle prepared with the script of [eyal-weiss/emeans-net-gain-revival](https://github.com/eyal-weiss/emeans-net-gain-revival) (duplicates removed, 20,000 rows held out, z-score).

-----

**Comparison.** Training inertia, % above the best variant of the row, mean of 3 seeds, longest budget (10 epochs for the first three datasets, 5 for the others). "km||" is the k-means|| init of sklearn-contrib-emeans, passed as `init_centroids` for the faiss columns.

| dataset, k | faiss mass 0.1 | faiss cost-based | faiss cost-based, km\|\| | greedy | greedy, km\|\| |
| --- | --- | --- | --- | --- | --- |
| road network, 64 | 11.3 | 1.3 | 0.5 | 1.2 | 0 |
| road network, 256 | 13.7 | 1.2 | 0 | 0.4 | 0 |
| road network, 1024 | 16.6 | 0.3 | 0 | 0.5 | 0.4 |
| MSD, 256 | 1.3 | 0.1 | 0 | 0.1 | 0 |
| MSD, 1024 | 2.2 | 0.1 | 0 | 0.4 | 0.4 |
| SUSY, 64 | 0.6 | 0.6 | 0.1 | 0.2 | 0.1 |
| SUSY, 1024 | 0.6 | 0.3 | 0 | 0 | 0 |
| Census, 1024 | 5.6 | 1.4 | 0.4 | 0.3 | 0 |
| Census, 4096 | 5.5 | 0.7 | 0.1 | 0.1 | 0 |
| GeoNames, 1024 | 104 | 0.8 | 0 | 0.4 | 0.1 |
| GeoNames, 4096 | 143 | 1.2 | 1.2 | 0.7 | 0 |
| power, 256 | 37.0 | 3.5 | 0 | 0.4 | 0 |
| power, 1024 | 43.0 | 4.2 | 0 | 0.7 | 0 |

KDD Cup 99 and Shuttle are heavy-tailed, so here are the raw values (5 epochs, mean of 3 seeds):

| | faiss mass 0.1 | faiss cost-based | faiss cost-based, km\|\| | greedy | greedy, km\|\| |
| --- | --- | --- | --- | --- | --- |
| KDD k=256, held-out error | 32.1 | 1.68 | 1.69 | 2.29 | 1.64 |
| KDD k=1024, held-out error | 11.0 | 0.53 | 0.51 | 0.81 | 0.63 |
| KDD k=4096, held-out error | 1.19 | 0.255 | 0.257 | 0.293 | 0.235 |
| KDD k=4096, training inertia | 13.3 | 0.85 | 0.82 | 0.30 | 0.097 |
| Shuttle k=1024, training inertia | 0.375 | 0.017 | 0.020 | 0.110 | 0.0098 |

The held-out error of Shuttle depends on one test row far from all the training rows, so I left it out. On KDD at k=4096, the training inertia of the cost-based rule changes a lot with the seed (1.18, 1.17 and 0.19).

**k-means|| on top of the cost-based rule.** In the first table, the k-means|| start lowers the inertia of the cost-based fits by 0.5 to 4% on the road network, Census, household power and SUSY at k=64, and changes little on MSD and KDD. So the cost-based rule recovers most of a bad random start, not all of it.

**Fit time of each rule.** Ratio of the fit time to the mass-based revival of the same library, median over the datasets and budgets (range in parentheses):

| k | CPU: faiss cost-based / faiss mass 0.1 | CPU: greedy / `revival=0.1` | GPU: faiss cost-based / faiss mass 0.1 | GPU: greedy / `revival=0.1` |
| --- | --- | --- | --- | --- |
| 64 | 1.27 (1.10 to 1.37) | 1.27 (1.07 to 3.35) | 1.11 (1.03 to 1.18) | 2.1 (1.1 to 3.6) |
| 256 | 1.32 (1.15 to 1.54) | 1.32 (1.10 to 1.76) | 1.19 (0.94 to 1.62) | 1.3 (1.0 to 4.5) |
| 1024 | 1.41 (0.97 to 1.54, and 12 in the slow case below) | 1.22 (1.03 to 2.10) | 1.24 (0.93 to 1.52) | 2.4 (1.1 to 13) |
| 4096 | 1.50 (0.89 to 1.58) | 2.22 (1.74 to 3.16) | 1.24 (1.19 to 1.33) | 18 (11 to 32) |

The GPU columns for faiss are `Emeans` with BF16 operands on both sides. The cost-based rule adds work to every step (top-2 instead of top-1, and statistics in O(batch x d)), so its ratio doesn't depend on the size. Greedy adds 5 to 15 revivals per fit, each an exact assignment of a sample of max(131072, 256 k) rows on the host: almost free when the data is much larger than the sample (+0.2 s on SUSY at k=1024 on CPU), about one more fit at k=4096 on CPU (+6 s on Census and KDD), and quadratic in k. On GPU it dominates the fit, since it runs on the 8 vCPUs of the VM (Census, k=4096, 5 epochs: 21 s against 1.9 s).

**On GPU, the quality is the same as on CPU.** `Emeans` with the cost-based rule (BF16 operands) is within 0 to 2% of greedy on the road network, MSD and Census, and removes the same gaps as on CPU. It is a bit further from greedy on GeoNames at k=4096 (4.6% against 0.6% above the best, 5 epochs) and household power at k=256 (8.2% against 0.4%). I see no effect of BF16: the mass-based `Emeans` (BF16) and `QuantizedEmeans(quantizer=None)` (FP32) give the same inertia, on the 3-D road network too. The centroids without training rows are there as well (265 of 4096 on KDD, 261 of 1024 on Shuttle, 5 epochs). For scale, the cost-based fit on Census (1.98M x 68) at k=4096 with 5 epochs takes 0.35 s on the L4.

-----

**On KDD Cup 99 and Shuttle, many final centroids are nearest to no training row.** The training rows are assigned to the returned centroids (exact search) and the centroids that get none are counted. Mean of 3 seeds, 5 epochs:

| | cost-based | `unused_threshold=0.1` |
| --- | --- | --- |
| KDD (1.05M x 122), k=1024 | 45 | 3 |
| KDD, k=4096 | 318 | 5 |
| Shuttle (38k x 9), k=256 | 34 | 0 |
| Shuttle, k=1024 | 258 | 1 |

On the other six datasets it is 0, or a handful at k=4096 on GeoNames. I don't know if these centroids are duplicates or centroids that the closing pass left at their running position.

-----

**On CPU, the top-2 search has a slow case in low dimension.** With 65,536 queries, 1024 centroids and d=3, `faiss.knn(x, c, 2)` takes 75 ms against 0.8 ms for `k=1`. With 32,768 queries there is no gap:

```python
import time, numpy as np, faiss

rng = np.random.RandomState(0)
for d in (3, 18):
    c = rng.rand(1024, d).astype("float32")
    for nq in (32768, 65536):
        x = rng.rand(nq, d).astype("float32")
        for k in (1, 2):
            faiss.knn(x, c, k)
            t = time.perf_counter()
            for _ in range(5):
                faiss.knn(x, c, k)
            print(
                f"d={d} nq={nq} top-{k}: {(time.perf_counter() - t) / 5 * 1e3:.1f} ms"
            )
```

```
d=3 nq=32768 top-1: 6.4 ms
d=3 nq=32768 top-2: 4.9 ms
d=3 nq=65536 top-1: 0.8 ms
d=3 nq=65536 top-2: 75.1 ms
d=18 nq=32768 top-1: 16.2 ms
d=18 nq=32768 top-2: 20.1 ms
d=18 nq=65536 top-1: 25.5 ms
d=18 nq=65536 top-2: 56.2 ms
```

On the road network at k=1024, the batch is 32,768 rows with 5 epochs and 65,536 with 10. The cost-based fit then takes 5.4 s with 5 epochs and 45 s with 10, against 3.6 s for `unused_threshold=0.1` with 10. I didn't look at which code path each case takes. This is specific to `faiss.knn` on CPU: on GPU (Triton), the same case (batch of 65,536, k=1024, d=3) takes 0.41 s with the cost-based rule against 0.33 s with the mass-based one, and no case is above 1.6x.

</details>
