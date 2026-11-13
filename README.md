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

You can use the GPU in a laptop or desktop to speed up `EMeans`, including
Apple GPUs and Intel integrated or Arc GPUs. In tests on synthetic data with
128 or 512 features, these GPUs were about 3–9x faster than the CPU in the
same machine, using PyTorch with the default 5 epochs. With only a few
features, the CPU can be faster: on data with 3 features, CPU was 2–3x faster.

On a high-end Intel server, `EMeans` with the default 5 epochs was 6–17x
faster than scikit-learn's `KMeans(n_init=1)` on synthetic data, with equally
good or better clustering. On sparse text data, it was 10–40x faster with
nearly the same clustering quality. See the
[benchmarks](https://pr-187.sklbench-dashboard-preview.pages.dev/kmeans_pareto) for details.

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
