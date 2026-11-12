"""k-means|| initialization (scalable k-means++)."""

# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import row_norms

from sklearn_emeans._kernels import (
    _update_min_distances,
    _update_min_distances_sparse,
)
from sklearn_emeans._kmeans_plusplus import _kmeans_plusplus

# Number of sampling rounds. The paper finds ~5 rounds enough in its
# experiments.
_KMEANS_PARALLEL_N_ROUNDS = 5

# Each round samples about `_KMEANS_PARALLEL_OVERSAMPLING * n_clusters`
# candidates, as in the experiments of the paper.
_KMEANS_PARALLEL_OVERSAMPLING = 2.0


def _kmeans_parallel(X, n_clusters, random_state, n_threads):
    """Initial centers given by k-means|| (Bahmani et al., 2012).

    The first candidate is a sample drawn uniformly. Then each round samples
    each sample x independently with probability `min(1, l * d²(x) / sum(d²))`,
    where `d²(x)` is the squared distance of x to its nearest candidate and
    `l = _KMEANS_PARALLEL_OVERSAMPLING * n_clusters`. Each candidate is weighted
    by the number of samples whose nearest candidate it is, and a weighted
    greedy k-means++ on the candidates gives the `n_clusters` centers.

    The caller must limit BLAS to one thread, since the BLAS calls of the
    kernels run inside OpenMP threads.

    Parameters
    ----------
    X : {ndarray, CSR matrix} of shape (n_samples, n_features), dtype=floating
        C-contiguous data, or a CSR matrix with int32 indices, with
        `n_samples >= n_clusters`.

    n_clusters : int
        Number of centers.

    random_state : RandomState instance
        Source of the random numbers. They are all drawn here, not in the
        threads, so the result doesn't depend on the number of threads.

    n_threads : int
        Number of OpenMP threads.

    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features), dtype=X.dtype
    """
    indices, weights = _kmeans_parallel_candidates(
        X, n_clusters, random_state, n_threads
    )
    # A CSR matrix if X is sparse: the k-means++ densifies only the centers.
    candidates = X[indices]
    centers, _ = _kmeans_plusplus(
        candidates,
        n_clusters,
        row_norms(candidates, squared=True),
        weights,
        random_state,
        n_threads=n_threads,
    )
    return centers


def _kmeans_parallel_candidates(X, n_clusters, random_state, n_threads):
    """Sampling rounds of k-means||.

    Returns the indices of the candidates, at least `n_clusters` of them, and
    their weights, the number of samples whose nearest candidate each one is
    (as an array of dtype X.dtype).
    """
    n_samples = X.shape[0]
    oversampling = _KMEANS_PARALLEL_OVERSAMPLING * n_clusters
    update_min_distances = (
        _update_min_distances_sparse if sp.issparse(X) else _update_min_distances
    )
    min_distances = np.full(n_samples, np.inf, dtype=X.dtype)
    labels = np.zeros(n_samples, dtype=np.int32)
    candidates = []
    n_candidates = 0

    def add_candidates(new):
        # Only the new candidates can decrease the distances: no need to
        # compute the distances to the previous ones again. With sparse X,
        # only the candidates of this round are densified.
        nonlocal n_candidates
        new_centers = X[new]
        if sp.issparse(new_centers):
            new_centers = new_centers.toarray()
        update_min_distances(
            X, new_centers, min_distances, labels, n_candidates, n_threads
        )
        # A candidate is its own nearest candidate, even if it has duplicates.
        # With d² = 0, it is never sampled again.
        min_distances[new] = 0
        labels[new] = np.arange(n_candidates, n_candidates + len(new))
        candidates.append(new)
        n_candidates += len(new)

    add_candidates(np.array([random_state.randint(n_samples)], dtype=np.intp))
    for _ in range(_KMEANS_PARALLEL_N_ROUNDS):
        total = min_distances.sum(dtype=np.float64)
        if total == 0:
            break
        # x is sampled with probability min(1, l * d²(x) / sum(d²))
        uniform = random_state.uniform(size=n_samples)
        new = np.flatnonzero(uniform * total < oversampling * min_distances)
        if len(new):
            add_candidates(new)

    if n_candidates < n_clusters:
        # Too few candidates, e.g. when many samples are duplicates or when
        # n_clusters is close to n_samples: complete with uniform samples.
        others = np.setdiff1d(np.arange(n_samples), np.concatenate(candidates))
        add_candidates(
            random_state.choice(others, size=n_clusters - n_candidates, replace=False)
        )

    weights = np.bincount(labels, minlength=n_candidates).astype(X.dtype)
    return np.concatenate(candidates), weights
