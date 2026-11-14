"""Greedy revival of the clusters of e-means (`revival="greedy"`)."""

# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import scipy.sparse as sp

from sklearn_emeans._kernels import (
    _greedy_stats,
    _greedy_stats_sparse,
    _trial_splits,
    _trial_splits_sparse,
)

# The costs and the gains are estimated on a fixed sample of the data, of
# max(_MIN_SAMPLES, _SAMPLES_PER_CLUSTER * n_clusters) rows.
_MIN_SAMPLES = 2**17
_SAMPLES_PER_CLUSTER = 256

# A cluster is replaced when the gain of the split is above this many times
# the cost of removing it.
_MARGIN = 1.0

# No revival after this fraction of the steps: the two halves of a late split
# don't have enough steps left to settle.
_MAX_SCHEDULE_FRACTION = 0.5

# Trial splits are only done for the clusters of largest variance along one
# feature: this fraction of the clusters, and at least this number.
_TRIAL_FRACTION = 1 / 8
_MIN_TRIALS = 8

# Minimum number of rows of the sample in a cluster to split it.
_MIN_SPLIT_ROWS = 16

# Number of 2-means iterations of a trial split.
_TRIAL_ITERATIONS = 3


def greedy_sample_indices(n_samples, n_clusters, random_state):
    """Sorted indices of the rows of the sample, or None for all the rows."""
    size = max(_MIN_SAMPLES, _SAMPLES_PER_CLUSTER * n_clusters)
    if size >= n_samples:
        return None
    return np.sort(random_state.choice(n_samples, size=size, replace=False))


class GreedyRevival:
    """Replaces clusters by halves of other clusters when it lowers the inertia.

    Every call estimates, on a fixed sample of the data:

    - the cost of removing each cluster: the increase of the inertia when its
      rows go to their second nearest center;
    - the gain of splitting each cluster: the decrease of the inertia of a
      2-means of its rows, fitted on half of them and measured on the other
      half.

    The cheapest clusters to remove are then paired with the most profitable
    ones to split, as long as the gain is above the cost. The center of a
    removed cluster and the one of its donor are set to the two centers of the
    trial split, and they share the EMA weight of the donor.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_sample_rows, n_features)
        The sample of the data, in float32 or float64, in CSR format if
        sparse.

    n_steps : int
        Total number of steps of the fit.

    n_threads : int
        Number of OpenMP threads. The results don't depend on it. The BLAS
        should be sequential during the calls.
    """

    def __init__(self, X, n_steps, n_threads):
        if sp.issparse(X):
            if X.indices.dtype != np.int32 or X.indptr.dtype != np.int32:
                X = sp.csr_matrix(
                    (X.data, X.indices.astype(np.int32), X.indptr.astype(np.int32)),
                    shape=X.shape,
                )
        else:
            X = np.ascontiguousarray(X)
        self.X = X
        self.last_step = n_steps * _MAX_SCHEDULE_FRACTION
        self.n_threads = n_threads

    def is_active(self, step):
        """Whether clusters can be revived after `step` steps of the fit."""
        return step <= self.last_step

    def __call__(self, centers, weight_ema):
        """Revive clusters, inplace. Returns the number of revived clusters."""
        n_rows = self.X.shape[0]
        n_clusters = centers.shape[0]
        if n_clusters < 2:
            return 0

        labels = np.empty(n_rows, dtype=np.int32)
        order = np.empty(n_rows, dtype=np.intp)
        cluster_starts = np.empty(n_clusters + 1, dtype=np.intp)
        cost = np.empty(n_clusters)
        max_scatter = np.empty(n_clusters)
        feature = np.empty(n_clusters, dtype=np.intp)
        greedy_stats = _greedy_stats_sparse if sp.issparse(self.X) else _greedy_stats
        greedy_stats(
            self.X,
            np.ascontiguousarray(centers, dtype=self.X.dtype),
            labels,
            order,
            cluster_starts,
            cost,
            max_scatter,
            feature,
            self.n_threads,
        )
        counts = np.diff(cluster_starts)
        std = np.sqrt(max_scatter / np.maximum(counts, 1))

        # Trial splits of the clusters with the largest decrease of the
        # inertia for a Gaussian cut at its mean along `feature`, which is
        # proportional to `max_scatter`.
        n_trials = max(_MIN_TRIALS, int(n_clusters * _TRIAL_FRACTION))
        trials = np.argsort(-max_scatter, kind="stable")[:n_trials]
        trials = trials[counts[trials] >= _MIN_SPLIT_ROWS]
        trial_gains = np.empty(len(trials))
        split_centers = np.empty((len(trials), 2, centers.shape[1]))
        if sp.issparse(self.X):
            trial_splits = _trial_splits_sparse
            X_args = (self.X.data, self.X.indices, self.X.indptr, self.X.shape[1])
        else:
            trial_splits = _trial_splits
            X_args = (self.X,)
        trial_splits(
            *X_args,
            order,
            cluster_starts,
            trials,
            feature,
            std,
            _TRIAL_ITERATIONS,
            trial_gains,
            split_centers,
            self.n_threads,
        )
        gain = np.zeros(n_clusters)
        gain[trials] = trial_gains
        trial_of = {j: t for t, j in enumerate(trials)}

        removed, donors = _pair(cost, gain)
        for i, j in zip(removed, donors):
            centers[i], centers[j] = split_centers[trial_of[j]]
        weight_ema[donors] /= 2
        weight_ema[removed] = weight_ema[donors]
        return len(removed)


def _pair(cost, gain):
    """Pairs of clusters to remove and to split, as two arrays of indices.

    The cheapest clusters to remove go with the most profitable ones to split,
    as long as the gain is above the cost. A cluster is in one pair at most.
    """
    donors = np.argsort(-gain, kind="stable")
    used = set()
    removed_indices, donor_indices = [], []
    position = 0
    for i in np.argsort(cost, kind="stable"):
        if i in used:
            continue
        while position < len(donors) and (
            donors[position] in used or donors[position] == i
        ):
            position += 1
        if position == len(donors):
            break
        j = donors[position]
        if gain[j] <= _MARGIN * cost[i]:
            break
        removed_indices.append(i)
        donor_indices.append(j)
        used.update((i, j))
        position += 1
    return (
        np.array(removed_indices, dtype=np.intp),
        np.array(donor_indices, dtype=np.intp),
    )
