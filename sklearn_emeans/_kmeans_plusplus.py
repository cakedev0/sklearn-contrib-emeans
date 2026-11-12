"""Greedy k-means++ initialization.

Adapted from `sklearn/cluster/_kmeans.py` of scikit-learn (BSD-3-Clause), at
commit 6e37063f7ab7ef7a7e257b9d44bc3ec6ee1b1bcd, which has a faster k-means++
than the released versions up to 1.9. Only the private `_kmeans_plusplus` is
kept, and the caller limits BLAS to one thread instead of a decorator.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import scipy.sparse as sp

from sklearn_emeans._kmeans_plusplus_kernels import (
    CHUNK_SIZE,
    _kmeans_plusplus_sample_candidates,
    _kmeans_plusplus_trials_dense,
    _kmeans_plusplus_trials_sparse,
)


def _kmeans_plusplus(
    X,
    n_clusters,
    x_squared_norms,
    sample_weight,
    random_state,
    n_local_trials=None,
    n_threads=1,
):
    """Computational component for initialization of n_clusters by
    k-means++. Prior validation of data is assumed.

    The centers are picked among the samples as follows:

    1. The first center is drawn with a probability proportional to
       `sample_weight`.
    2. `n_local_trials` candidates for the next center are drawn with a
       probability proportional to their potential. The potential of a sample
       is its squared distance to the closest center already picked, weighted
       by `sample_weight`.
    3. Greedily pick the candidate that would produce the smallest inertia
       if added to the centers. The inertia is the sum of the potentials.

    Repeat 2-3 until `n_clusters` centers are picked. With `n_local_trials=1`,
    this is the original k-means++.

    The caller must limit BLAS to one thread, to avoid oversubscription in the
    BLAS calls made inside the OpenMP loops, e.g. with
    `_threadpool_controller().limit(limits=1, user_api="blas")`.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The data to pick seeds for.

    n_clusters : int
        The number of seeds to choose.

    sample_weight : ndarray of shape (n_samples,)
        The weights for each observation in `X`.

    x_squared_norms : ndarray of shape (n_samples,)
        Squared Euclidean norm of each data point.

    random_state : RandomState instance
        The generator used to initialize the centers.
        See :term:`Glossary <random_state>`.

    n_local_trials : int, default=None
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.

    n_threads : int, default=1
        The maximum number of OpenMP threads to use. It must be computed by the
        caller: with OpenBLAS's OpenMP threading layer, the BLAS limit of this
        function also sets the default number of OpenMP threads to 1.

    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        The initial centers for k-means.

    indices : ndarray of shape (n_clusters,)
        The index location of the chosen centers in the data array X. For a
        given index and center, X[index] = center.
    """
    n_samples = X.shape[0]
    n_chunks = int(np.ceil(n_samples / CHUNK_SIZE))
    # Each k-means++ step is a parallel region with little work for small
    # datasets, where starting and synchronizing many threads would dominate:
    # use at most one thread per 2**16 elements of X (or non-zero values).
    n_threads = min(n_threads, max(1, X.size // 2**16))

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # The potential of a sample is its squared distance to the closest center,
    # weighted by `sample_weight`. Candidates are sampled with a probability
    # proportional to the potential, and the candidate that gives the smallest
    # total potential (i.e. inertia) is added to the centers.
    #
    # trials_dist_sq[:, j, i] is the squared distance from sample i to its
    # closest center, with candidate j added to the centers.
    trials_dist_sq = np.empty((2, n_local_trials, n_samples), dtype=X.dtype)
    # trials_chunk_potentials[:, j, k] is the sum of the potentials of the
    # samples of the k-th chunk of CHUNK_SIZE samples, with candidate j added to
    # the centers.
    # Each chunk is summed by a single thread, without synchronization, and the
    # fixed chunk size makes the result independent of the number of threads
    # (see `_kmeans_plusplus_trials_dense`).
    trials_chunk_potentials = np.empty((2, n_local_trials, n_chunks), dtype=np.float64)
    # Both arrays are twice as large as needed to act as "ping-pong" buffers.
    # At each step, `closest_dist_sq` and `chunk_potentials` are read from views
    # inside the half written at the previous iteration.
    # This spares memory copies at the cost of a slightly larger memory usage.

    # The sparse kernel reads the columns of X: convert it once for all the steps
    X_csc = None
    if sp.issparse(X):
        X_csc = X.tocsc()
        X_csc.sort_indices()

    # Pick first center randomly and track index of point
    first_center_idx = random_state.choice(
        n_samples, p=sample_weight / sample_weight.sum()
    )
    center_indices = np.full(n_clusters, -1, dtype=int)
    center_indices[0] = first_center_idx

    # Initialize the closest distances and the potential per chunk, i.e.:
    #   trials_dist_sq[0, 0] = euclidean_distances(
    #       X[[first_center_idx]], X, squared=True
    #   )[0]
    #   trials_chunk_potentials[0, 0] = np.add.reduceat(
    #       sample_weight * trials_dist_sq[0, 0], np.arange(0, n_samples, CHUNK_SIZE)
    #   )

    # There is no center yet, hence closest distances of +inf. Half 1 of the
    # buffers is only written by the first step, so one of its rows can hold them.
    trials_dist_sq[1, 0] = np.inf
    _kmeans_plusplus_trials(
        X,
        X_csc,
        x_squared_norms,
        sample_weight,
        candidate_indices=np.array([first_center_idx], dtype=np.intp),
        closest_dist_sq=trials_dist_sq[1, 0],  # IN
        trials_dist_sq=trials_dist_sq[0, :1],  # OUT
        trials_chunk_potentials=trials_chunk_potentials[0, :1],  # OUT
        n_threads=n_threads,
    )
    # The first center is "trial" 0 in half 0 of the buffers
    best_trial_idx = 0

    candidate_indices = np.empty(n_local_trials, dtype=np.intp)

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Read the half written by the previous step, write in the other
        # half of the two "ping-pong" arrays.
        in_idx = (c - 1) % 2
        out_idx = c % 2
        closest_dist_sq = trials_dist_sq[in_idx, best_trial_idx]
        chunk_potentials = trials_chunk_potentials[in_idx, best_trial_idx]

        # Choose center candidates by sampling with probability proportional
        # to the potential, i.e. up to rounding errors:
        #   potential_cumsum = np.cumsum(sample_weight * closest_dist_sq)
        #   random_potentials = (
        #       random_state.uniform(size=n_local_trials) * potential_cumsum[-1]
        #   )
        #   candidate_indices[:] = np.searchsorted(potential_cumsum, random_potentials)
        _kmeans_plusplus_sample_candidates(
            random_state,
            chunk_potentials,
            closest_dist_sq,
            sample_weight,
            candidate_indices,
        )

        # Compute the closest distances and the potential per chunk for each
        # candidate, i.e.:
        #   dist_sq = euclidean_distances(X[candidate_indices], X, squared=True)
        #   trials_dist_sq[out_idx] = np.minimum(closest_dist_sq, dist_sq)
        #   trials_chunk_potentials[out_idx] = np.add.reduceat(
        #       sample_weight * trials_dist_sq[out_idx],
        #       np.arange(0, n_samples, CHUNK_SIZE),
        #       axis=1,
        #   )
        _kmeans_plusplus_trials(
            X,
            X_csc,
            x_squared_norms,
            sample_weight,
            candidate_indices,
            closest_dist_sq,
            trials_dist_sq[out_idx],
            trials_chunk_potentials[out_idx],
            n_threads,
        )

        # Pick the candidate with the smallest potential. The kernel only sums
        # within each chunk of samples, finish the sum over the chunks here.
        best_trial_idx = np.argmin(trials_chunk_potentials[out_idx].sum(axis=1))

        # Permanently add best center candidate found in local tries
        center_indices[c] = candidate_indices[best_trial_idx]

    centers = X[center_indices]
    if sp.issparse(X):
        centers = centers.toarray()

    return centers, center_indices


# fmt: off
def _kmeans_plusplus_trials(
    X,                          # IN (n_samples, n_features)
    X_csc,                      # IN (n_samples, n_features) or None
    x_squared_norms,            # IN (n_samples,)
    sample_weight,              # IN (n_samples,)
    candidate_indices,          # IN (n_trials,)
    closest_dist_sq,            # IN (n_samples,)
    trials_dist_sq,             # OUT (n_trials, n_samples)
    trials_chunk_potentials,    # OUT (n_trials, n_chunks)
    n_threads,
):
    # fmt: on
    """Evaluate k-means++ candidate centers, see `_kmeans_plusplus_trials_dense`.

    `X_csc` is X in CSC format if X is sparse, None otherwise.
    """
    candidates = X[candidate_indices]
    candidates_squared_norms = x_squared_norms[candidate_indices]
    if sp.issparse(X):
        _kmeans_plusplus_trials_sparse(
            X,
            X_csc,
            x_squared_norms,
            sample_weight,
            candidate_indices,
            candidates.toarray(),
            candidates_squared_norms,
            closest_dist_sq,
            trials_dist_sq,
            trials_chunk_potentials,
            n_threads,
        )
    else:
        _kmeans_plusplus_trials_dense(
            X,
            x_squared_norms,
            sample_weight,
            candidates,
            candidates_squared_norms,
            closest_dist_sq,
            trials_dist_sq,
            trials_chunk_potentials,
            n_threads,
        )
