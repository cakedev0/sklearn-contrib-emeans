sklearn-contrib-emeans
======================

:class:`~sklearn_emeans.EMeans` is a scikit-learn compatible implementation of
e-means, a mini-batch k-means that reaches the inertia of
:class:`~sklearn.cluster.KMeans` in a few passes over the data instead of tens
of iterations.

.. code-block:: python

    from sklearn_emeans import EMeans

    emeans = EMeans(n_clusters=256, n_epochs=5).fit(X)
    emeans.cluster_centers_, emeans.labels_, emeans.inertia_
    emeans.predict(X_new)

The algorithm is described in S. Bessa, P.-E. Mazaré, G. Szilvasy, H. Touvron,
D. Picard, M. Douze and H. Jégou, "e-means: outperforming k-means (in one-tenth
the epochs)", 2026, `arXiv:2611.09501 <https://arxiv.org/abs/2611.09501>`_. The
reference implementation is in `faiss/contrib/e_means
<https://github.com/facebookresearch/faiss/tree/main/contrib/e_means>`_.

Installation
------------

.. code-block:: bash

    pip install sklearn-contrib-emeans

It needs Python >= 3.10 and scikit-learn >= 1.6. The wheels are built with
OpenMP.

Contents
--------

.. toctree::
   :maxdepth: 1

   api
   auto_examples/index

Relation to scikit-learn
------------------------

This package is a step towards adding `EMeans` to scikit-learn itself, see
`scikit-learn#35383
<https://github.com/scikit-learn/scikit-learn/issues/35383>`_. Feedback is
welcome in the `issues
<https://github.com/cakedev0/sklearn-contrib-emeans/issues>`_, especially on
datasets where `EMeans` does worse than `KMeans`.
