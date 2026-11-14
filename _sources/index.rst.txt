sklearn-contrib-emeans
======================

:class:`~sklearn_emeans.EMeans` is a scikit-learn compatible implementation of
e-means, a mini-batch k-means that typically gets close to the inertia of
:class:`~sklearn.cluster.KMeans` in a few passes over the data instead of tens
of iterations.

.. code-block:: python

    from sklearn_emeans import EMeans

    emeans = EMeans(n_clusters=256, n_epochs=5).fit(X)
    emeans.cluster_centers_, emeans.labels_, emeans.inertia_
    emeans.predict(X_new)

It also fits on a GPU, on PyTorch tensors or CuPy arrays, for instance on the
GPU of a Mac:

.. code-block:: python

    import sklearn
    import torch

    X = torch.asarray(X, dtype=torch.float32, device="mps")  # or "cuda", "xpu"
    with sklearn.config_context(array_api_dispatch=True):
        emeans = EMeans(n_clusters=256, n_epochs=5).fit(X)
    emeans.cluster_centers_  # a tensor on the device of X

This needs `SCIPY_ARRAY_API=1` in the environment, see :ref:`array_api_inputs`.

Install it with `pip install sklearn-contrib-emeans`. It needs Python >= 3.10
and scikit-learn >= 1.7.

.. toctree::
   :maxdepth: 1

   user_guide
   api
   auto_examples/index

Relation to scikit-learn
------------------------

This is an independent package, not affiliated with scikit-learn or
`scikit-learn-contrib <https://github.com/scikit-learn-contrib>`_. Adding
`EMeans` to scikit-learn itself is under discussion in `scikit-learn#35383
<https://github.com/scikit-learn/scikit-learn/issues/35383>`_.

Feedback is welcome in the `issues
<https://github.com/cakedev0/sklearn-contrib-emeans/issues>`_, especially on
datasets where `EMeans` does worse than `KMeans`.
