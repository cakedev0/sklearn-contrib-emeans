.. _user_guide:

User guide
==========

:class:`~sklearn_emeans.EMeans` minimizes the inertia, the sum of the squared
distances of the samples to their closest center, like
:class:`~sklearn.cluster.KMeans` and :class:`~sklearn.cluster.MiniBatchKMeans`
(see :ref:`k_means`). It implements e-means [1]_, which reaches the inertia of
Lloyd's algorithm in a few passes over the data instead of tens of iterations.

What e-means does
-----------------

E-means is a mini-batch variant of k-means in which each center is the ratio of
two exponential moving averages, computed over the mini-batches: the running sum
:math:`S_j` and the running weight :math:`N_j` of the samples assigned to
cluster :math:`j`. With :math:`s_j` and :math:`n_j` the sum and the number of
the samples of a mini-batch assigned to the cluster, and :math:`\alpha` the step
size, each mini-batch updates them as:

.. math::

    S_j \leftarrow (1 - \alpha) S_j + \alpha s_j, \qquad
    N_j \leftarrow (1 - \alpha) N_j + \alpha n_j, \qquad
    c_j = S_j / N_j.

With a single mini-batch holding all the samples and :math:`\alpha = 1`, this is
an iteration of Lloyd's algorithm. Around this update:

- The step size is shared by all the centers and follows a cosine decay, from
  its initial value to a tenth of it. In
  :class:`~sklearn.cluster.MiniBatchKMeans`, each center has its own learning
  rate, the inverse of the number of samples it has seen.
- Every 5 mini-batches, the dead clusters, whose running weight is below a
  tenth of the mean one, are revived: each takes half of a heavier cluster,
  which is split in two. Clusters are not revived on data with 1 or 2 features.
- At the end of the last epoch, each center is set to the mean of the samples
  assigned to it during that epoch.
- `labels_` and `inertia_` are then computed with one more pass over the data,
  so that they match the final centers and `predict`.

The batch size and the initial step size are not parameters. They are given by
scaling laws that the authors of e-means fitted as functions of the number of
samples, of clusters and of epochs. The batch size grows almost linearly with
the number of samples, is rounded to a power of two, and is 8 times smaller for
data with up to 16 features. It is stored in `batch_size_` after a fit.

Choosing `n_epochs`
-------------------

`n_epochs`, the number of passes over the data, is the parameter that trades
fit time for inertia. The authors report that e-means reaches the inertia of
Lloyd's algorithm after 25 iterations in 1.4 to 2.6 passes over the data on a
wide range of embedding datasets.

.. figure:: auto_examples/images/sphx_glr_plot_emeans_comparison_001.png
   :target: auto_examples/plot_emeans_comparison.html
   :align: center
   :scale: 75

In :ref:`this example <sphx_glr_auto_examples_plot_emeans_comparison.py>`, 2 to
3 epochs reach the inertia that `KMeans` reaches after tens of iterations, and
the default of 5 epochs reaches its inertia after 100 iterations.

- Start from the default, `n_epochs=5`. Lower it to 2 or 3 when the fit time
  matters more than the last fraction of a percent of inertia, and raise it to
  compare: more epochs usually give a lower inertia, but not always.
- `n_epochs` can be fractional. With `n_epochs=2.5`, the half pass comes before
  the last epoch, which is always a full pass. With `n_epochs=0.5`, the fit sees
  half of the samples once.
- `n_epochs` is a fixed budget: there is no convergence criterion and no early
  stopping. The batch size and the schedule of the step size depend on it, so a
  fit with more epochs is not the continuation of a fit with fewer.

When to use `EMeans`
--------------------

**Rather than** `KMeans`, when an iteration of Lloyd's algorithm is expensive,
that is with many samples. An epoch of `EMeans` has the complexity of an
iteration of Lloyd's algorithm, and a few epochs do the work of tens of
iterations. Prefer `KMeans`:

- on small datasets, which it fits in a negligible time;
- with sample weights, which `EMeans` does not support yet;
- when the best clustering has many small clusters. Reviving the dead clusters
  keeps the running weight of every cluster above a tenth of the mean one, which
  pushes towards balanced clusters. On the Yolanda dataset (OpenML 42705,
  standardized: 400,000 samples, 100 features) with 256 clusters, `KMeans`
  leaves about 20 clusters under a tenth of the mean size and `EMeans` at most
  2, and from 2 to 50 epochs `EMeans` stays 1 to 2% above the inertia of
  `KMeans` after 100 iterations;
- when the centers must be a local minimum of the inertia, where an iteration of
  Lloyd's algorithm no longer moves them. `EMeans` stops after its budget, and
  its centers are not such a minimum. Iterations of Lloyd's algorithm from
  them, with `KMeans(n_clusters, init=emeans.cluster_centers_)`, can only lower
  the inertia.

**Rather than** `MiniBatchKMeans`, when the data fits in memory: for the same
number of passes, `EMeans` reaches a lower inertia in the example above.
`MiniBatchKMeans` remains the choice for data that arrives as a stream or does
not fit in memory, with its `partial_fit`. Each epoch of `EMeans` goes through
a permutation of the whole dataset.

Initialization
--------------

`init` can be:

- `"random"`, the default: `n_clusters` samples drawn at random;
- `"k-means||"`: scalable k-means++ [2]_. A few rounds over the data sample
  candidates far from the candidates so far, then a weighted k-means++ on the
  candidates gives the centers. It costs about as much as 2 to 3 epochs;
- `"k-means++"`: greedy k-means++, as in
  :func:`sklearn.cluster.kmeans_plusplus`. It draws the centers one after the
  other and is mostly there for experimenting: `"k-means||"` gives a similar
  inertia much faster;
- an array of centers, or a callable returning one.

The time of a better initialization can also be spent on more epochs: compare
both on your data.

Input data
----------

`X` is a dense array or a sparse CSR matrix. The other sparse formats are
converted to CSR. float32 and float64 data are used as they are, and float32
is faster. The other dtypes are converted to float64.

Complexity and parallelism
--------------------------

A fit makes `n_epochs` passes over the data, each with the complexity of an
iteration of Lloyd's algorithm, :math:`O(n_{samples} \, n_{clusters} \,
n_{features})` on dense data, and a last one to compute `labels_` and
`inertia_`. C-contiguous data is not copied, and the additional memory is
:math:`O(n_{clusters} \, n_{features} + n_{samples})`.

The samples of a mini-batch are assigned to their closest center in chunks of
256 samples, in parallel with OpenMP, then the centers are updated in parallel.
`EMeans` uses all the physical cores, or the number of threads set by the
`OMP_NUM_THREADS` environment variable or by
`threadpoolctl.threadpool_limits(limits=n_threads, user_api="openmp")`, and at most
one thread per chunk of a mini-batch: small mini-batches run on few threads.
The result does not depend on the number of threads.

References
----------

.. [1] S. Bessa, P.-E. Mazaré, G. Szilvasy, H. Touvron, D. Picard, M. Douze and
   H. Jégou, "e-means: outperforming k-means (in one-tenth the epochs)", 2026.
   `arXiv:2611.09501 <https://arxiv.org/abs/2611.09501>`_. The reference
   implementation is in `faiss/contrib/e_means
   <https://github.com/facebookresearch/faiss/tree/main/contrib/e_means>`_.

.. [2] B. Bahmani, B. Moseley, A. Vattani, R. Kumar and S. Vassilvitskii,
   "Scalable K-Means++", Proceedings of the VLDB Endowment, 2012.
   `arXiv:1203.6402 <https://arxiv.org/abs/1203.6402>`_
