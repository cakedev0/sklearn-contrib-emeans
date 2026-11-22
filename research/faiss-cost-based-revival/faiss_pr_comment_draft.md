Thanks for this PR! I compared the cost-based revival (at ba7781a, `NumpyEmeans`) with the mass-based one and with the revivals of [sklearn-contrib-emeans](https://github.com/cakedev0/sklearn-contrib-emeans) on 8 datasets (3 seeds, 2 to 10 epochs, Core Ultra 9 285 with 24 threads). The benchmarks were run by my agent (Claude). Scripts, raw runs and tables are [here](https://github.com/cakedev0/sklearn-contrib-emeans/tree/h-means-bench/research/faiss-cost-based-revival).

**The cost-based rule removes the gap of the mass-based one** on every dataset where there is one: 16-18% of training inertia on a 3-D road network (4.3M rows), 2% on YearPredictionMSD, 5% on Census, ~40% on household power, more than 100% on GeoNames. On KDD Cup 99 at k=1024 (5 epochs), the held-out error goes from 11.0 to 0.53.

Two things I noticed on the way:

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

**On KDD Cup 99 and Shuttle, many final centroids are nearest to no training row.** I assign the training rows to the returned centroids (exact search) and count the centroids that get none, mean of 3 seeds, 5 epochs:

| | cost-based | `unused_threshold=0.1` |
| --- | --- | --- |
| KDD (deduplicated, z-scored, 1.05M x 122), k=1024 | 45 | 3 |
| KDD, k=4096 | 318 | 5 |
| Shuttle (38k x 9), k=256 | 34 | 0 |
| Shuttle, k=1024 | 258 | 1 |

On the other six datasets it is 0, or a handful at k=4096 on GeoNames. On KDD at k=4096 the training inertia also changes a lot with the seed (1.18, 1.17 and 0.19 with 5 epochs), while the held-out error is stable (0.255). I don't know if these centroids are duplicates or centroids that the closing pass left at their running position.

Let me know if more details on a case would help.
