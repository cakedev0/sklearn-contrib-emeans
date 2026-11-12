# sklearn-contrib-emeans

`EMeans` is a scikit-learn compatible implementation of e-means, a mini-batch
k-means that reaches the inertia of `KMeans` in a few passes over the data
instead of tens of iterations.

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

It needs Python >= 3.10 and scikit-learn >= 1.6. The other dependencies
(NumPy, SciPy, joblib, threadpoolctl) have the same minimum versions as in
scikit-learn 1.10. The wheels are built with OpenMP: `EMeans` uses all the
physical cores, or `OMP_NUM_THREADS` if it is set.

## API

`EMeans(n_clusters=8, *, init="random", n_epochs=5, random_state=None)`

- `n_epochs` can be fractional: the fractional pass comes before the last
  epoch, which is always a full pass.
- `init` is `"random"`, `"k-means++"`, an array of initial centers or a
  callable. `"k-means++"` is the greedy k-means++ of scikit-learn, copied from
  its main branch, where it is 20-70x faster than in scikit-learn 1.9. It
  takes `n_clusters` passes over the data, often longer than the 5 epochs of
  e-means, but on data with a strong cluster structure it reaches a lower
  inertia than `"random"` with any number of epochs.
- `fit`, `predict`, `fit_predict`, `transform`, `fit_transform` and `score`
  work as for `sklearn.cluster.KMeans`.
- Fitted attributes: `cluster_centers_`, `labels_`, `inertia_` and
  `batch_size_`.

`labels_` and `inertia_` come from an extra assignment pass on the final
centers, so that `fit(X).labels_` is the same as `predict(X)`.

The input must be dense (float32 or float64, float32 is faster). There is no
`sample_weight` yet.

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

## Performance

On one socket of a Xeon 6787P with 43 threads, on float32 blobs, `EMeans`
with the default 5 epochs fits 6-17x faster than `KMeans(n_init=1)` for the
same or lower inertia. For example, 1M samples x 128 features with 1024
clusters take 0.70 s against 5.2 s. See
[cakedev0/scikit-learn#46](https://github.com/cakedev0/scikit-learn/pull/46)
for the full benchmark, which uses the same kernel.

The best batch size depends on the dataset: on some datasets, batches 4x
smaller or larger than the scaling law's give a lower inertia for the same
fit time. A `batch_size` parameter may come later.

## Relation to scikit-learn

This package is a step towards adding `EMeans` to scikit-learn itself, see
[scikit-learn#35383](https://github.com/scikit-learn/scikit-learn/issues/35383).
The plan is to collect feedback and stabilize the API here, then move it to
`sklearn.cluster` for scikit-learn 1.11.

Feedback is welcome in the issues, especially on datasets where `EMeans` does
worse than `KMeans`.

## License

BSD 3-Clause.
