import time

import faiss
import numpy as np

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
