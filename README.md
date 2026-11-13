# sklearn-contrib-emeans

`EMeans` is a scikit-learn compatible implementation of e-means, a mini-batch
k-means that typically gets close to the inertia of `KMeans` in a few passes
over the data instead of tens of iterations.

Each center is an exponential moving average (EMA) of the samples assigned to
it, with a cosine schedule on the EMA step. The batch size and the initial step
size come from scaling laws fitted by the authors of the paper, so the only
parameter to choose is the number of epochs.

```python
from sklearn_emeans import EMeans

emeans = EMeans(n_clusters=256, n_epochs=5).fit(X)
emeans.cluster_centers_, emeans.labels_, emeans.inertia_
emeans.predict(X_new)
```

## Installation

```
pip install sklearn-contrib-emeans
```

It needs Python >= 3.10 and scikit-learn >= 1.7. The other dependencies
(NumPy, SciPy, joblib, threadpoolctl) have the same minimum versions as in
scikit-learn 1.10. The wheels are built with OpenMP: `EMeans` uses all the
physical cores, or `OMP_NUM_THREADS` if it is set.

## API

`EMeans(n_clusters=8, *, init="random", n_epochs=5, random_state=None)`

- `n_epochs` can be fractional: the fractional pass comes before the last
  epoch, which is always a full pass.
- `init` is `"random"`, `"k-means||"`, `"k-means++"`, an array of initial
  centers or a callable. `"k-means||"` is scalable k-means++ (Bahmani et al.,
  [arXiv:1203.6402](https://arxiv.org/abs/1203.6402)): a few parallel
  sampling rounds over the data, then a weighted k-means++ on the sampled
  candidates. `"k-means++"` (greedy k-means++, copied from scikit-learn main)
  is mostly for experimenting: `"k-means||"` gives a similar inertia much
  faster.
- `fit`, `predict`, `fit_predict`, `transform`, `fit_transform` and `score`
  work as for `sklearn.cluster.KMeans`.
- Fitted attributes: `cluster_centers_`, `labels_`, `inertia_` and
  `batch_size_`.

`labels_` and `inertia_` come from an extra assignment pass on the final
centers, so that `fit(X).labels_` is the same as `predict(X)`.

The input can be dense or a sparse CSR matrix (other sparse formats are
converted to CSR), in float32 or float64 (float32 is faster). There is no
`sample_weight` yet.

## GPU and array API inputs

`EMeans` also fits on PyTorch tensors and CuPy arrays (and on the arrays of
other libraries that follow the [array API](https://data-apis.org/array-api/)),
on their device. As for the scikit-learn estimators, this must be
[enabled](https://scikit-learn.org/stable/modules/array_api.html) with
`array_api_dispatch=True`, and with `SCIPY_ARRAY_API=1` in the environment:

```python
import sklearn
import torch
from sklearn_emeans import EMeans

X = torch.asarray(X_numpy, device="mps")  # or "cuda", "xpu"
with sklearn.config_context(array_api_dispatch=True):
    emeans = EMeans(n_clusters=1024).fit(X)
    labels = emeans.predict(X)
emeans.cluster_centers_  # a tensor on the device of X
```

- The random numbers are drawn on the host, in the same order as for NumPy
  inputs: from the same `random_state`, the results are the same up to
  rounding. On devices without float64 (Apple GPUs), all the computations are
  in float32.
- With `init="k-means||"`, the sampling rounds run on the device and the final
  k-means++ on the candidates runs on the host. `init="k-means++"` copies the
  data to the host.
- Only dense inputs.

For example, with the default 5 epochs and `init="random"`, on float32 blobs,
against the same fit on the CPU cores of the same machine:

| | 1M × 128, 1024 clusters | 300k × 512, 1024 clusters | 2M × 16, 256 clusters |
| --- | ---: | ---: | ---: |
| MacBook Pro M2 (4 + 4 cores), PyTorch MPS | 1.20 s (6.1x) | 1.10 s (9.2x) | 0.99 s (1.7x) |
| Core Ultra 9 285 (24 cores), Arc B580, PyTorch XPU | 0.33 s (3.7x) | 0.25 s (5.4x) | 0.43 s (1.0x) |
| Core Ultra X7 358H (16 cores), integrated GPU, PyTorch XPU | 0.69 s (2.8x) | 0.53 s (6.1x) | 0.55 s (1.1x) |
| AWS VM (8 vCPUs), NVIDIA L4, PyTorch CUDA | 0.41 s (13.6x) | 0.31 s (17.9x) | 0.77 s (1.4x) |
| AWS VM (8 vCPUs), NVIDIA L4, CuPy | 0.59 s (9.1x) | 0.39 s (14.3x) | 0.73 s (1.5x) |

The GPU pays off when the matrix products dominate. On low-dimensional data,
the batches are small and each step is bound by the launch of the GPU kernels:
with 3 features, the fit is 2 to 3 times faster on the CPU.

## Algorithm

The algorithm is described in:

> S. Bessa, P.-E. Mazaré, G. Szilvasy, H. Touvron, D. Picard, M. Douze and
> H. Jégou, "e-means: outperforming k-means (in one-tenth the epochs)", 2026.
> [arXiv:2611.09501](https://arxiv.org/abs/2611.09501)

The reference implementation is in
[faiss/contrib/e_means](https://github.com/facebookresearch/faiss/tree/main/contrib/e_means).
From the same initial centers and random numbers, `EMeans` gives the same
centers as its `NumpyEmeans`, up to float32 rounding.

Each step assigns a mini-batch to the nearest centers with a chunked GEMM,
then updates the centers in parallel over centers, so the result doesn't
depend on the number of threads.
On sparse input, the GEMM is replaced by sparse dot products against the
transposed centers.

## Performance

On one socket of a Xeon 6787P with 43 threads, on float32 blobs, `EMeans`
with the default 5 epochs fits 6-17x faster than `KMeans(n_init=1)` for the
same or lower inertia. For example, 1M samples x 128 features with 1024
clusters take 0.70 s against 5.2 s. See
[cakedev0/scikit-learn#46](https://github.com/cakedev0/scikit-learn/pull/46)
for the full benchmark, which uses the same kernel.

On sparse input, with 22 threads on 22 cores of the same Xeon (one NUMA node,
with SMT siblings), `EMeans` fits 10-40x faster than `KMeans(n_init=1)`
(scikit-learn 1.10.dev0) on the TF-IDF of 20 newsgroups and on RCV1 (float32,
20 or 100 clusters), with an inertia within 0.2%. For example, RCV1 (804k
samples x 47k features, density 0.16%) with 100 clusters takes 1.2 s against
48 s. See [#3](https://github.com/cakedev0/sklearn-contrib-emeans/pull/3).

The best batch size depends on the dataset: on some datasets, batches 4x
smaller or larger than the scaling law's give a lower inertia for the same
fit time. A `batch_size` parameter may come later.

## Relation to scikit-learn

This is an independent package, not affiliated with scikit-learn or
[scikit-learn-contrib](https://github.com/scikit-learn-contrib). The plan is
to submit it to scikit-learn-contrib soon.

It is the place to collect feedback on `EMeans` and stabilize its API. Adding
it to scikit-learn itself is under discussion in
[scikit-learn#35383](https://github.com/scikit-learn/scikit-learn/issues/35383).

Feedback is welcome in the issues, especially on datasets where `EMeans` does
worse than `KMeans`.

## License

BSD 3-Clause.
