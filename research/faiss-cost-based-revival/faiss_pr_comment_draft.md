Thanks for this PR! I had my agent benchmark the cost-based revival (`NumpyEmeans` at ba7781a) against the mass-based one, and against the `revival="greedy"` of [sklearn-contrib-emeans](https://github.com/cakedev0/sklearn-contrib-emeans), which is built on the same idea (removal cost against split gain) but estimates both on a sample, with trial 2-means splits. 8 datasets, 3 seeds, CPU only.

From these runs:

- The cost-based rule seems to remove most of the gap that the mass-based revival leaves (16-18% of inertia on a 3-D road network), and it ends within ~1% of greedy on most datasets. It looks better than greedy on KDD Cup 99 up to k=1024, and worse at k=4096.
- A k-means|| init still seems useful on top of it: 0.5 to 4% lower inertia on the road network, Census and household power.
- Two things that may be worth a look: the top-2 search is very slow in one low-dimension case (75 ms against 0.8 ms for top-1), and on KDD and Shuttle many final centroids are nearest to no training row (~320 of 4096 on KDD).

I didn't dig into these two. Details below, and I can run more cases if that helps.

<details><summary>Details (from my agent)</summary>

Setup: faiss built at ba7781a, sklearn-contrib-emeans 0.4.1, Core Ultra 9 285 (24 threads), seeds 0 to 2, random init unless stated. The inertia is recomputed from the returned centroids in float64 with an exact assignment. Scripts, raw runs and all the tables are [here](https://github.com/cakedev0/sklearn-contrib-emeans/tree/h-means-bench/research/faiss-cost-based-revival).

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

**Fit time of the rule.** The cost-based fits take 1.2 to 1.6 times the mass-based ones in `NumpyEmeans` (for example 5.8 s against 4.4 s on the road network at k=64 with 10 epochs), apart from the slow case below.

-----

**The top-2 search has a slow case in low dimension.** With 65,536 queries, 1024 centroids and d=3, `faiss.knn(x, c, 2)` takes 75 ms against 0.8 ms for `k=1`. With 32,768 queries there is no gap:

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
            print(f"d={d} nq={nq} top-{k}: {(time.perf_counter() - t) / 5 * 1e3:.1f} ms")
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

On the road network at k=1024, the batch is 32,768 rows with 5 epochs and 65,536 with 10. The cost-based fit then takes 5.4 s with 5 epochs and 45 s with 10, against 3.6 s for `unused_threshold=0.1` with 10. I didn't look at which code path each case takes.

-----

**On KDD Cup 99 and Shuttle, many final centroids are nearest to no training row.** The training rows are assigned to the returned centroids (exact search) and the centroids that get none are counted. Mean of 3 seeds, 5 epochs:

| | cost-based | `unused_threshold=0.1` |
| --- | --- | --- |
| KDD (1.05M x 122), k=1024 | 45 | 3 |
| KDD, k=4096 | 318 | 5 |
| Shuttle (38k x 9), k=256 | 34 | 0 |
| Shuttle, k=1024 | 258 | 1 |

On the other six datasets it is 0, or a handful at k=4096 on GeoNames. I don't know if these centroids are duplicates or centroids that the closing pass left at their running position.

</details>
