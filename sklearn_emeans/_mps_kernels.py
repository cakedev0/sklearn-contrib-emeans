"""E-means steps with fused Metal kernels, for PyTorch tensors on Apple GPUs.

On low-dimensional data the batches are small, and the steps written with
array operations are bound by the launch of a few tens of GPU kernels, not by
the computations. Here a step is two kernels, that PyTorch compiles at run
time with `torch.mps.compile_shader` (no build step):

- `assign_accumulate` gathers a row of the batch, finds its nearest center,
  and adds the row to the sum of its cluster;
- `ema` does the EMA update of the centers from these sums.

The distances to the centers are computed with a plain loop over the features,
which only beats the matrix product of PyTorch on low-dimensional data: see
`MAX_FEATURES`. The kernels are compiled once per number of features.
"""

# SPDX-License-Identifier: BSD-3-Clause

import warnings
from functools import cache

# On an M2 with PyTorch 2.14, the kernels are 1.8x to 5x faster than the array
# operations up to 32 features. At 64 features they are still faster for 256
# clusters but not for 1024, and from 96 features they are 5x to 20x slower.
MAX_FEATURES = 32

# D is the number of features.
_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

// Metal only has integer atomics: add floats with a compare-and-swap loop on
// their bits.
inline void atomic_add(device atomic_uint* cell, float value) {
    uint expected = atomic_load_explicit(cell, memory_order_relaxed);
    while (true) {
        const uint desired = as_type<uint>(as_type<float>(expected) + value);
        if (atomic_compare_exchange_weak_explicit(
                cell, &expected, desired, memory_order_relaxed,
                memory_order_relaxed))
            break;
    }
}

// One thread per row of the batch, X[rows[start + i]]: nearest center from
// the squared distances, with ties going to the lowest index, then the row is
// added to the sum of its center (float32) and counted (uint32).
kernel void assign_accumulate(
        device atomic_uint* sums, device atomic_uint* counts,
        constant float* X, constant long* rows, constant float* C,
        constant uint& start, constant uint& k,
        uint i [[thread_position_in_grid]]) {
    const ulong r = (ulong)rows[start + i] * D;
    float x[D];
    for (uint f = 0; f < D; ++f) x[f] = X[r + f];
    float best = INFINITY;
    uint best_j = 0;
    for (uint j = 0; j < k; ++j) {
        float dist = 0.f;
        for (uint f = 0; f < D; ++f) {
            const float diff = x[f] - C[j * D + f];
            dist += diff * diff;
        }
        if (dist < best) { best = dist; best_j = j; }
    }
    for (uint f = 0; f < D; ++f) atomic_add(&sums[best_j * D + f], x[f]);
    atomic_fetch_add_explicit(&counts[best_j], 1u, memory_order_relaxed);
}

// One thread per element of the centers: EMA update of the center from the
// sum and the number of its batch rows,
//     c <- ((1 - step_size) w c + step_size sum) / w'
// with w the EMA weight and w' = (1 - step_size) w + step_size count the new
// one. A center without rows keeps its position. Resets the sums for the next
// step and, with with_epoch != 0, adds them first to the sums of the epoch.
kernel void ema(
        device float* C, device float* new_weight, constant float* weight,
        device float* sums, constant uint* counts, device float* epoch_sums,
        device float* epoch_counts, constant float& step_size,
        constant uint& with_epoch, uint t [[thread_position_in_grid]]) {
    const uint j = t / D;
    const float count = (float)counts[j];
    const float retained = (1.f - step_size) * weight[j];
    const float updated = retained + step_size * count;
    if (t % D == 0) {
        new_weight[j] = updated;
        if (with_epoch) epoch_counts[j] += count;
    }
    if (count > 0.f) {
        const float sum = sums[t];
        C[t] = (retained / updated) * C[t] + (step_size / updated) * sum;
        if (with_epoch) epoch_sums[t] += sum;
        sums[t] = 0.f;
    }
}
"""


@cache
def _library(n_features):
    """Compiled kernels for a number of features, None without them.

    Without them, the steps fall back to array operations: silently for a
    PyTorch without `compile_shader`, and with a warning (once per number of
    features) if the source doesn't compile.
    """
    import torch

    if not hasattr(torch.mps, "compile_shader"):
        return None
    try:
        return torch.mps.compile_shader(f"#define D {n_features}\n" + _SOURCE)
    except Exception as exc:
        warnings.warn(
            "The Metal kernels of sklearn_emeans could not be compiled: the fit "
            "falls back to array operations, which are slower on "
            f"low-dimensional data. Please report this error: {exc!r}",
            RuntimeWarning,
        )
        return None


def supports(X):
    """Whether the kernels can run on X, a PyTorch tensor on an Apple GPU."""
    import torch

    return (
        X.dtype == torch.float32
        and X.ndim == 2
        and 0 < X.shape[1] <= MAX_FEATURES
        and X.shape[0] < 2**32
        and _library(X.shape[1]) is not None
    )


def emeans_steps(
    X,
    rows,
    batch_size,
    centers,
    weight_ema,
    step_sizes,
    epoch_sums,
    epoch_counts,
    first_step,
    revival_period,
    revive,
):
    """Same as `_array_api.emeans_steps`, for a float32 tensor X on MPS.

    `weight_ema`, `epoch_sums` and `epoch_counts` are float32 tensors: Apple
    GPUs have no float64.
    """
    import torch

    library = _library(X.shape[1])
    X = X.contiguous()
    rows = rows.contiguous()
    # The kernels update the centers inplace, and the weights from one buffer
    # to the other.
    centers = centers.clone(memory_format=torch.contiguous_format)
    weight_ema = weight_ema.clone()
    new_weight_ema = torch.empty_like(weight_ema)
    n_clusters, n_features = centers.shape
    n_rows = rows.shape[0]
    sums = torch.zeros_like(centers)
    counts = torch.empty(n_clusters, dtype=torch.int32, device=X.device)
    with_epoch = epoch_sums is not None
    if with_epoch:
        epoch_sums = epoch_sums.contiguous()
    unused = sums  # not accessed by the kernel

    for s, step_size in enumerate(step_sizes):
        start = s * batch_size
        counts.zero_()
        library.assign_accumulate(
            sums,
            counts,
            X,
            rows,
            centers,
            start,
            n_clusters,
            threads=min(batch_size, n_rows - start),
        )
        library.ema(
            centers,
            new_weight_ema,
            weight_ema,
            sums,
            counts,
            epoch_sums if with_epoch else unused,
            epoch_counts if with_epoch else unused,
            float(step_size),
            int(with_epoch),
            threads=n_clusters * n_features,
        )
        weight_ema, new_weight_ema = new_weight_ema, weight_ema
        if revival_period > 0 and (first_step + s + 1) % revival_period == 0:
            centers, weight_ema = revive(centers, weight_ema)
            centers = centers.contiguous()
    return centers, weight_ema, epoch_sums, epoch_counts
