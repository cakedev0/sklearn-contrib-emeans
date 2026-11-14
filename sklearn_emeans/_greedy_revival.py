"""Greedy revival of the clusters of e-means (`revival="greedy"`)."""

# SPDX-License-Identifier: BSD-3-Clause

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import row_norms

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

# Rows per task of the thread pool.
_CHUNK_ROWS = 4096


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
        The sample of the data, in CSR format if sparse.

    n_steps : int
        Total number of steps of the fit.

    n_threads : int
        Number of threads of the pool, to stop with `close`. The results don't
        depend on it. The BLAS should be sequential during the calls.
    """

    def __init__(self, X, n_steps, n_threads):
        self.X = X
        self.sparse = sp.issparse(X)
        self.x_squared_norms = row_norms(X, squared=True).astype(X.dtype, copy=False)
        # The dense sums are centered, for their accuracy in float32.
        self.mean = (
            None
            if self.sparse
            else X.mean(axis=0, dtype=np.float64).astype(X.dtype, copy=False)
        )
        self.last_step = n_steps * _MAX_SCHEDULE_FRACTION
        self.pool = ThreadPoolExecutor(n_threads)

    def close(self):
        """Stop the threads."""
        self.pool.shutdown()

    def is_active(self, step):
        """Whether clusters can be revived after `step` steps of the fit."""
        return step <= self.last_step

    def __call__(self, centers, weight_ema):
        """Revive clusters, inplace. Returns the number of revived clusters."""
        n_clusters = centers.shape[0]
        if n_clusters < 2:
            return 0
        labels, cost, counts, variance_gain, feature, std = self._cluster_stats(centers)

        n_trials = max(_MIN_TRIALS, int(n_clusters * _TRIAL_FRACTION))
        trials = [
            j
            for j in np.argsort(-variance_gain, kind="stable")[:n_trials]
            if counts[j] >= _MIN_SPLIT_ROWS
        ]

        def trial_split(j):
            rows = self.X[np.flatnonzero(labels == j)]
            return _trial_split(rows, feature[j], std[j])

        gain = np.zeros(n_clusters)
        halves = {}
        for j, (trial_gain, trial_centers) in zip(
            trials, self.pool.map(trial_split, trials)
        ):
            gain[j] = trial_gain
            halves[j] = trial_centers

        removed, donors = _pair(cost, gain)
        for i, j in zip(removed, donors):
            centers[i], centers[j] = halves[j]
        weight_ema[donors] /= 2
        weight_ema[removed] = weight_ema[donors]
        return len(removed)

    def _cluster_stats(self, centers):
        """Labels of the sample and statistics of each cluster.

        Returns the labels, the removal costs, the numbers of rows and, for
        the feature of largest variance of each cluster: the gain of a split
        along it for Gaussian data, its index and its standard deviation.
        """
        n_rows = self.X.shape[0]
        n_clusters = centers.shape[0]
        centers = np.ascontiguousarray(centers, dtype=self.X.dtype)
        centers_squared_norms = np.einsum("ij,ij->i", centers, centers)
        starts = range(0, n_rows, _CHUNK_ROWS)

        def chunk_stats(start):
            return self._chunk_stats(
                slice(start, min(start + _CHUNK_ROWS, n_rows)),
                centers,
                centers_squared_norms,
            )

        chunks = list(self.pool.map(chunk_stats, starts))
        labels = np.concatenate([chunk[0] for chunk in chunks])
        cost = sum(chunk[1] for chunk in chunks)
        counts = np.bincount(labels, minlength=n_clusters)
        safe_counts = np.maximum(counts, 1)

        if self.sparse:
            # Sums and sums of squares per cluster and feature, as two sparse
            # matrices of same structure.
            one_hot = sp.csr_matrix(
                (np.ones(n_rows), (labels, np.arange(n_rows))),
                shape=(n_clusters, n_rows),
            )
            X_squared = self.X.copy()
            X_squared.data **= 2
            sums = (one_hot @ self.X).tocsr()
            squared_sums = (one_hot @ X_squared).tocsr()
            sums.sort_indices()
            squared_sums.sort_indices()
            row_of = np.repeat(np.arange(n_clusters), np.diff(sums.indptr))
            # count * variance
            scatter = squared_sums.data - sums.data**2 / safe_counts[row_of]
            max_scatter = np.zeros(n_clusters)
            feature = np.zeros(n_clusters, dtype=np.intp)
            for j in np.flatnonzero(np.diff(sums.indptr)):
                start, stop = sums.indptr[j], sums.indptr[j + 1]
                best = start + scatter[start:stop].argmax()
                max_scatter[j] = scatter[best]
                feature[j] = sums.indices[best]
        else:
            sums = sum(chunk[2] for chunk in chunks)
            squared_sums = sum(chunk[3] for chunk in chunks)
            scatter = squared_sums - sums**2 / safe_counts[:, None]
            feature = scatter.argmax(axis=1)
            max_scatter = scatter[np.arange(n_clusters), feature]

        max_scatter = np.maximum(max_scatter, 0)
        # Decrease of the inertia of a Gaussian cut at its mean.
        variance_gain = (2 / np.pi) * max_scatter
        return (
            labels,
            cost,
            counts,
            variance_gain,
            feature,
            np.sqrt(max_scatter / safe_counts),
        )

    def _chunk_stats(self, rows, centers, centers_squared_norms):
        """Statistics of a chunk of rows, in a thread of the pool."""
        X = self.X[rows]
        n_rows = X.shape[0]
        arange = np.arange(n_rows)
        distances = X @ centers.T
        if self.sparse and not isinstance(distances, np.ndarray):
            distances = distances.toarray()
        distances *= -2
        distances += self.x_squared_norms[rows, None]
        distances += centers_squared_norms
        labels = distances.argmin(axis=1)
        nearest = distances[arange, labels]
        distances[arange, labels] = np.inf
        # Removal cost: the rows go to their second nearest center.
        cost = np.bincount(
            labels,
            weights=(distances.min(axis=1) - nearest).astype(np.float64),
            minlength=centers.shape[0],
        )
        if self.sparse:
            return labels, cost
        # Sums and sums of squares per cluster and feature, with a GEMM on
        # the one-hot encoding of the labels.
        one_hot = distances
        one_hot.fill(0)
        one_hot[arange, labels] = 1
        X = X - self.mean
        sums = (one_hot.T @ X).astype(np.float64)
        np.multiply(X, X, out=X)
        squared_sums = (one_hot.T @ X).astype(np.float64)
        return labels, cost, sums, squared_sums


def _mean(X):
    """Mean row of a dense or sparse matrix, in float64."""
    return np.asarray(X.mean(axis=0), dtype=np.float64).ravel()


def _trial_split(X, feature, std):
    """2-means of the rows of a cluster, on a half of them.

    The 2-means starts from the mean, moved by one standard deviation in both
    directions along the feature of largest variance. Returns the decrease of
    the inertia of the cluster, estimated on the other half of the rows, and
    the two centers.
    """
    X = X.astype(np.float64)
    fitted, held_out = X[::2], X[1::2]
    mean = _mean(fitted)
    centers = np.stack([mean, mean])
    centers[0, feature] += std
    centers[1, feature] -= std

    def products(X):
        # Scalar products with the two centers, and whether the rows are
        # nearer to the second one.
        dots = np.asarray(X @ centers.T)
        squared_norms = np.einsum("ij,ij->i", centers, centers)
        second = dots[:, 1] - dots[:, 0] > (squared_norms[1] - squared_norms[0]) / 2
        return dots, squared_norms, second

    for _ in range(_TRIAL_ITERATIONS):
        _, _, second = products(fitted)
        if second.all() or not second.any():
            break
        centers[0] = _mean(fitted[np.flatnonzero(~second)])
        centers[1] = _mean(fitted[np.flatnonzero(second)])

    # Squared distances up to the squared norms of the rows, which cancel out
    # in the gain.
    dots, squared_norms, second = products(held_out)
    split = np.where(
        second, squared_norms[1] - 2 * dots[:, 1], squared_norms[0] - 2 * dots[:, 0]
    ).sum()
    whole = (mean @ mean - 2 * np.asarray(held_out @ mean)).sum()
    gain = (whole - split) * X.shape[0] / held_out.shape[0]
    return gain, centers


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
