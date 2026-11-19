"""Two-level assignment on CUDA devices, with Triton kernels.

Same assignment as `_emeans_steps_two_level` of `_kernels.pyx` (see the
comments there): a row is compared to the coarse centers of the cells, then to
the centers of the cells it probes. The cells are refreshed with PyTorch
operations, and the rows are assigned by four kernels:

1. `_coarse_kernel` compares a block of rows to all the cells, selects the
   cells that each row probes and counts the (row, cell) pairs of each cell.
2. `_scatter_kernel` groups the pairs by cell, from the prefix sum of the
   counts.
3. `_fine_kernel` runs one program per tile of pairs of a cell, which goes
   through the centers of the cell and keeps the nearest one of each pair.
4. `_reduce_kernel` takes the nearest center of each row among its pairs.

The kernels don't read the data and the centers themselves but a float16
copy of them, scaled by a power of two so that the largest absolute value of
the data is about 1024 (`PRECISION`): the dot products run on the tensor cores
and the rows, which are read once per probed cell, take half the memory
bandwidth. The products are accumulated in float32. Rounding the inputs to 11
bits only matters for the rows that are almost as close to two centers. The
sums of the rows of each cluster and the inertia are not computed here, and
are not rounded.

The layout of the kernels, the packed keys of the coarse kernel and the fine
kernel are adapted from o-means (`contrib/o_means/triton_two_level.py` of the
faiss fork of owlgebra-ai), which is under the MIT license:

    Copyright (c) Meta Platforms, Inc. and affiliates.

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the
    "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to permit
    persons to whom the Software is furnished to do so, subject to the
    following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
    OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
    NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
    DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
    USE OR OTHER DEALINGS IN THE SOFTWARE.

This module imports PyTorch and Triton: it is only imported when the
two-level assignment runs on a CUDA device.
"""

# SPDX-License-Identifier: BSD-3-Clause AND MIT

import math

import torch
import triton
import triton.language as tl

# Inputs of the dot products:
# - "float16": scaled float16 copies of the rows and of the centers.
# - "tf32": float32 inputs, rounded to 10 bits of mantissa by the tensor
#   cores. 1.3x to 1.5x slower.
# - "ieee": float32 products, without the tensor cores. 2.5x to 4x slower.
PRECISION = "float16"

# With PRECISION = "float16", the data is scaled by a power of two that brings
# its largest absolute value to this power of two, or just under. float16
# overflows at 65504 and loses precision under 6e-5.
_FLOAT16_MAX_ABS = 1024.0

# Lower bound of the EMA weight of a center in the mean of its cell, as
# MIN_CELL_WEIGHT of `_kernels.pyx`.
_MIN_CELL_WEIGHT = 1e-12

# Numbers of rows per program of the coarse kernel, tried in this order. A
# program needs shared memory in proportion to its rows times the number of
# cells, and GPUs have 48 kB to 227 kB of it per program. When no block fits,
# the probes are selected with PyTorch operations instead.
_COARSE_BLOCKS = (32, 16)

# Number of pairs of a cell and of centers that a program of the fine kernel
# compares with one product, and number of features per product.
_PAIRS_TILE = 128
_CENTERS_BLOCK = 128
_FEATURES_BLOCK = 32

_FLOAT32_MAX = 3.4028234663852886e38


@triton.jit(do_not_specialize=["n_rows"])
def _coarse_kernel(
    x_ptr,
    cells_ptr,
    cell_norms_ptr,
    probes_ptr,
    counts_ptr,
    n_rows,
    n_features,
    x_stride,
    cells_stride,
    ratio2,
    PROBE_ALL: tl.constexpr,
    MAX_PROBES: tl.constexpr,
    N_CELLS: tl.constexpr,
    BLOCK: tl.constexpr,
    FEATURES_BLOCK: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    """Cells probed by each row: `probes` (n_rows, MAX_PROBES), -1 padded.

    `counts` gets the number of rows that probe each cell. N_CELLS is the
    number of cells rounded up to a power of two: the squared norm of the
    padding and of the empty cells is infinite.
    """
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row_mask = r < n_rows
    x_base = x_ptr + r[:, None].to(tl.int64) * x_stride
    cell = tl.arange(0, N_CELLS)
    cells_base = cells_ptr + cell[:, None].to(tl.int64) * cells_stride
    dots = tl.zeros((BLOCK, N_CELLS), tl.float32)
    x_norms = tl.zeros((BLOCK,), tl.float32)
    for first in range(0, n_features, FEATURES_BLOCK):
        col = first + tl.arange(0, FEATURES_BLOCK)
        col_mask = col < n_features
        x = tl.load(
            x_base + col[None, :], mask=row_mask[:, None] & col_mask[None, :], other=0.0
        )
        g = tl.load(cells_base + col[None, :], mask=col_mask[None, :], other=0.0)
        dots = tl.dot(
            x,
            tl.trans(g),
            acc=dots,
            input_precision=DOT_PRECISION,
            out_dtype=tl.float32,
        )
        x = x.to(tl.float32)
        x_norms += tl.sum(x * x, axis=1)
    # ||g||² - 2 x.g: the squared distance without the ||x||².
    scores = tl.load(cell_norms_ptr + cell)[None, :] - 2.0 * dots

    # Order-preserving int32 image of the scores with the cell in the low
    # bits: each probe is one min, and the ties go to the lowest cell.
    bits = scores.to(tl.int32, bitcast=True)
    keys = tl.where(bits < 0, bits ^ 2147483647, bits)
    keys = (keys & (-N_CELLS)) | cell[None, :]
    if PROBE_ALL:
        keys = tl.where(scores < float("inf"), keys, 2147483647)
    else:
        # ||x - g||² <= ratio² ||x - g_nearest||², and the nearest cell.
        min_scores = tl.min(scores, axis=1)
        nearest = tl.maximum(x_norms + min_scores, 0.0)
        thresholds = tl.maximum(ratio2 * nearest - x_norms, min_scores)
        thresholds = tl.minimum(thresholds, 3.4028234663852886e38)
        keys = tl.where(scores <= thresholds[:, None], keys, 2147483647)

    one = tl.full((BLOCK,), 1, tl.int32)
    probes_base = probes_ptr + r.to(tl.int64) * MAX_PROBES
    for s in tl.static_range(MAX_PROBES):
        min_keys = tl.min(keys, axis=1)
        valid = (min_keys != 2147483647) & row_mask
        probe = min_keys & (N_CELLS - 1)
        keys = tl.where(keys == min_keys[:, None], 2147483647, keys)
        tl.store(probes_base + s, tl.where(valid, probe, -1), mask=row_mask)
        tl.atomic_add(counts_ptr + probe, one, mask=valid, sem="relaxed")


@triton.jit(do_not_specialize=["n_rows"])
def _scatter_kernel(
    probes_ptr,
    cursors_ptr,
    pairs_ptr,
    n_rows,
    MAX_PROBES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Pairs `row * MAX_PROBES + s` of each cell, from `cursors[cell]` on.

    The order of the pairs of a cell depends on the scheduling. The labels
    don't depend on it.
    """
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row_mask = r < n_rows
    one = tl.full((BLOCK,), 1, tl.int32)
    for s in tl.static_range(MAX_PROBES):
        probe = tl.load(
            probes_ptr + r.to(tl.int64) * MAX_PROBES + s, mask=row_mask, other=-1
        )
        valid = probe >= 0
        probe = tl.where(valid, probe, 0)
        position = tl.atomic_add(cursors_ptr + probe, one, mask=valid, sem="relaxed")
        tl.store(pairs_ptr + position, r * MAX_PROBES + s, mask=valid)


@triton.jit
def _fine_kernel(
    x_ptr,
    centers_ptr,
    center_norms_ptr,
    order_ptr,
    cell_starts_ptr,
    pair_starts_ptr,
    tile_cells_ptr,
    tile_starts_ptr,
    pairs_ptr,
    pair_scores_ptr,
    pair_labels_ptr,
    n_features,
    x_stride,
    centers_stride,
    MAX_PROBES: tl.constexpr,
    TILE: tl.constexpr,
    CENTERS_BLOCK: tl.constexpr,
    FEATURES_BLOCK: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    """Nearest center of each pair of a tile of pairs of a cell, in its cell.

    The centers are sorted by cell, then by index: the centers of the cell c
    are the rows `cell_starts[c]` to `cell_starts[c + 1]` of `centers`, and
    `order` gives their indices.
    """
    tile = tl.program_id(0)
    c = tl.load(tile_cells_ptr + tile)
    q = tl.load(tile_starts_ptr + tile) + tl.arange(0, TILE)
    pair_mask = q < tl.load(pair_starts_ptr + c + 1)
    pairs = tl.load(pairs_ptr + q, mask=pair_mask, other=0)
    rows = (pairs // MAX_PROBES).to(tl.int64)
    x_base = x_ptr + rows[:, None] * x_stride
    first_center = tl.load(cell_starts_ptr + c)
    last_center = tl.load(cell_starts_ptr + c + 1)
    best = tl.full((TILE,), float("inf"), tl.float32)
    labels = tl.zeros((TILE,), tl.int32)
    for first in range(first_center, last_center, CENTERS_BLOCK):
        j = first + tl.arange(0, CENTERS_BLOCK)
        center_mask = j < last_center
        centers_base = centers_ptr + j[:, None].to(tl.int64) * centers_stride
        dots = tl.zeros((TILE, CENTERS_BLOCK), tl.float32)
        for first_col in range(0, n_features, FEATURES_BLOCK):
            col = first_col + tl.arange(0, FEATURES_BLOCK)
            col_mask = col < n_features
            x = tl.load(
                x_base + col[None, :],
                mask=pair_mask[:, None] & col_mask[None, :],
                other=0.0,
            )
            centers = tl.load(
                centers_base + col[None, :],
                mask=center_mask[:, None] & col_mask[None, :],
                other=0.0,
            )
            dots = tl.dot(
                x,
                tl.trans(centers),
                acc=dots,
                input_precision=DOT_PRECISION,
                out_dtype=tl.float32,
            )
        norms = tl.load(center_norms_ptr + j, mask=center_mask, other=float("inf"))
        scores = norms[None, :] - 2.0 * dots
        # On ties, the lowest position: the lowest index of the cell.
        block_best, block_labels = tl.min(
            scores, axis=1, return_indices=True, return_indices_tie_break_left=True
        )
        better = block_best < best
        best = tl.where(better, block_best, best)
        labels = tl.where(better, block_labels.to(tl.int32) + first, labels)
    labels = tl.load(order_ptr + labels, mask=pair_mask, other=0)
    tl.store(pair_scores_ptr + pairs, best, mask=pair_mask)
    tl.store(pair_labels_ptr + pairs, labels, mask=pair_mask)


@triton.jit(do_not_specialize=["n_rows"])
def _reduce_kernel(
    probes_ptr,
    pair_scores_ptr,
    pair_labels_ptr,
    labels_ptr,
    n_rows,
    MAX_PROBES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Nearest center of each row among its pairs, the lowest index on ties."""
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row_mask = r < n_rows
    base = r.to(tl.int64) * MAX_PROBES
    best = tl.load(pair_scores_ptr + base, mask=row_mask, other=float("inf"))
    labels = tl.load(pair_labels_ptr + base, mask=row_mask, other=0)
    for s in tl.static_range(1, MAX_PROBES):
        valid = tl.load(probes_ptr + base + s, mask=row_mask, other=-1) >= 0
        scores = tl.load(pair_scores_ptr + base + s, mask=valid, other=float("inf"))
        pair_labels = tl.load(pair_labels_ptr + base + s, mask=valid, other=0)
        better = valid & ((scores < best) | ((scores == best) & (pair_labels < labels)))
        best = tl.where(better, scores, best)
        labels = tl.where(better, pair_labels, labels)
    tl.store(labels_ptr + r, labels.to(tl.int64), mask=row_mask)


class TwoLevelAssigner:
    """Cells of the centers and two-level assignment of rows to the centers.

    Parameters
    ----------
    cells : tensor of shape (n_cells, n_features)
        Initial coarse centers.

    cell_of_center : int64 tensor of shape (n_clusters,)
        Initial cell of each center. Modified inplace by `move_to_cells_of`.

    max_probes : int
        Maximum number of cells that a row probes.

    ratio : float
        A row probes its nearest cell and the cells whose distance to the row
        is at most `ratio` times the distance of the nearest one.

    max_rows : int
        Number of rows assigned by one launch of the kernels: it gives the
        size of the buffers.

    max_abs : float
        Largest absolute value of the data, for the scale of the float16
        copies.
    """

    def __init__(self, cells, cell_of_center, max_probes, ratio, max_rows, max_abs):
        device = cells.device
        n_cells = cells.shape[0]
        self.n_cells = n_cells
        self.cells = cells
        self.cell_of_center = cell_of_center
        self.max_probes = max(1, min(max_probes, n_cells))
        self.ratio = ratio
        self.max_rows = max_rows
        self.device = device
        self.precision = PRECISION
        self.dtype = torch.float32
        self.scale = 1.0
        if PRECISION == "float16":
            self.dtype = torch.float16
            if 0 < max_abs < math.inf:
                self.scale = 2.0 ** math.floor(math.log2(_FLOAT16_MAX_ABS / max_abs))
        # The kernels read the cells padded to a power of two.
        self.n_padded_cells = max(16, triton.next_power_of_2(n_cells))
        self.coarse_blocks = list(_COARSE_BLOCKS)
        self.padded_cells = torch.zeros(
            (self.n_padded_cells, cells.shape[1]), dtype=self.dtype, device=device
        )
        self.cell_norms = torch.full(
            (self.n_padded_cells,), torch.inf, dtype=torch.float32, device=device
        )
        self.cell_starts = torch.zeros(n_cells + 1, dtype=torch.int32, device=device)
        self.pair_starts = torch.zeros(n_cells + 1, dtype=torch.int32, device=device)
        n_pairs = max_rows * self.max_probes
        self.probes = torch.empty(
            (max_rows, self.max_probes), dtype=torch.int32, device=device
        )
        self.counts = torch.zeros(self.n_padded_cells, dtype=torch.int32, device=device)
        self.pairs = torch.empty(n_pairs, dtype=torch.int32, device=device)
        self.pair_scores = torch.empty(n_pairs, dtype=torch.float32, device=device)
        self.pair_labels = torch.empty(n_pairs, dtype=torch.int32, device=device)
        self.cell_indices = torch.arange(n_cells, dtype=torch.int32, device=device)

    def _kernel_copy(self, array):
        """Copy of rows in the dtype and the scale that the kernels read."""
        if self.scale != 1.0:
            array = array * self.scale
        return array.to(self.dtype).contiguous()

    def refresh(self, centers, weight_ema):
        """Make the cells follow the centers, as `_refresh_cells`.

        Each coarse center moves to the mean of the centers of its cell,
        weighted by their EMA weights (the one of an empty cell stays), then
        each center goes to the cell of its nearest coarse center.
        """
        n_cells = self.n_cells
        weights = weight_ema.to(torch.float64).clamp_min(_MIN_CELL_WEIGHT)
        sums = torch.zeros(
            (n_cells, centers.shape[1]), dtype=torch.float64, device=self.device
        ).index_add_(0, self.cell_of_center, centers * weights[:, None])
        totals = torch.zeros(
            n_cells, dtype=torch.float64, device=self.device
        ).index_add_(0, self.cell_of_center, weights)
        nonempty = totals > 0
        self.cells = torch.where(
            nonempty[:, None],
            (sums / torch.where(nonempty, totals, 1.0)[:, None]).to(centers.dtype),
            self.cells,
        )

        half_norms = 0.5 * self.cells.square().sum(1)
        chunk_size = max(1, 2**24 // n_cells)
        cell_of_center = self.cell_of_center
        for start in range(0, centers.shape[0], chunk_size):
            cell_of_center[start : start + chunk_size] = torch.nn.functional.linear(
                centers[start : start + chunk_size], self.cells, -half_norms
            ).argmax(1)

        # Centers sorted by cell, then by index.
        order = torch.argsort(cell_of_center, stable=True)
        sizes = torch.bincount(cell_of_center, minlength=n_cells)
        self.cell_starts[1:] = torch.cumsum(sizes, 0)
        self.order = order.to(torch.int32)
        self.sorted_centers = self._kernel_copy(centers[order])
        self.sorted_norms = self.sorted_centers.to(torch.float32).square().sum(1)
        self.padded_cells[:n_cells] = self._kernel_copy(self.cells)
        # An empty cell is never probed.
        self.cell_norms[:n_cells] = torch.where(
            sizes > 0,
            self.padded_cells[:n_cells].to(torch.float32).square().sum(1),
            torch.inf,
        )

    def move_to_cells_of(self, indices, donor_indices):
        """Move revived centers to the cells of their donors."""
        self.cell_of_center[indices] = self.cell_of_center[donor_indices]

    def assign(self, X):
        """Label (int64) of each row of X, once the cells are refreshed."""
        n_rows = X.shape[0]
        if n_rows <= self.max_rows:
            return self._assign(X)
        return torch.cat(
            [
                self._assign(X[start : start + self.max_rows])
                for start in range(0, n_rows, self.max_rows)
            ]
        )

    def _assign(self, X):
        X = self._kernel_copy(X)
        n_rows, n_features = X.shape
        n_cells, max_probes = self.n_cells, self.max_probes
        self._coarse(X)

        # Pairs grouped by cell, and tiles of pairs of a cell.
        counts = self.counts[:n_cells]
        self.pair_starts[1:] = torch.cumsum(counts, 0)
        cursors = self.pair_starts[:n_cells].clone()
        grid = (triton.cdiv(n_rows, 256),)
        _scatter_kernel[grid](
            self.probes,
            cursors,
            self.pairs,
            n_rows,
            MAX_PROBES=max_probes,
            BLOCK=256,
            num_warps=4,
        )
        n_tiles = torch.div(
            counts + _PAIRS_TILE - 1, _PAIRS_TILE, rounding_mode="floor"
        )
        tile_cells = torch.repeat_interleave(self.cell_indices, n_tiles)
        tile_cells_long = tile_cells.to(torch.int64)
        first_tiles = torch.cumsum(n_tiles, 0) - n_tiles
        tile_starts = (
            self.pair_starts[tile_cells_long]
            + (
                torch.arange(tile_cells.shape[0], device=self.device)
                - first_tiles[tile_cells_long]
            )
            * _PAIRS_TILE
        ).to(torch.int32)

        labels = torch.empty(n_rows, dtype=torch.int64, device=self.device)
        _fine_kernel[(tile_cells.shape[0],)](
            X,
            self.sorted_centers,
            self.sorted_norms,
            self.order,
            self.cell_starts,
            self.pair_starts,
            tile_cells,
            tile_starts,
            self.pairs,
            self.pair_scores,
            self.pair_labels,
            n_features,
            X.stride(0),
            self.sorted_centers.stride(0),
            MAX_PROBES=max_probes,
            TILE=_PAIRS_TILE,
            CENTERS_BLOCK=_CENTERS_BLOCK,
            FEATURES_BLOCK=_FEATURES_BLOCK,
            DOT_PRECISION=self._dot_precision(),
            num_warps=4,
            num_stages=2,
        )
        _reduce_kernel[grid](
            self.probes,
            self.pair_scores,
            self.pair_labels,
            labels,
            n_rows,
            MAX_PROBES=max_probes,
            BLOCK=256,
            num_warps=4,
        )
        return labels

    def _dot_precision(self):
        # float16 inputs always go to the tensor cores.
        return "ieee" if self.precision == "ieee" else "tf32"

    def _coarse(self, X):
        n_rows = X.shape[0]
        while self.coarse_blocks:
            block = self.coarse_blocks[0]
            self.counts.zero_()
            try:
                # One stage: the software pipelining of Triton keeps a copy of
                # the operands of the product in shared memory per extra
                # stage, and all the cells are one operand. With the default
                # of 3 stages, 512 cells don't fit in the 99 kB of most GPUs.
                _coarse_kernel[(triton.cdiv(n_rows, block),)](
                    X,
                    self.padded_cells,
                    self.cell_norms,
                    self.probes,
                    self.counts,
                    n_rows,
                    X.shape[1],
                    X.stride(0),
                    self.padded_cells.stride(0),
                    self.ratio**2 if math.isfinite(self.ratio) else 0.0,
                    PROBE_ALL=not math.isfinite(self.ratio),
                    MAX_PROBES=self.max_probes,
                    N_CELLS=self.n_padded_cells,
                    BLOCK=block,
                    FEATURES_BLOCK=_FEATURES_BLOCK,
                    DOT_PRECISION=self._dot_precision(),
                    num_warps=8,
                    num_stages=1,
                )
                return
            except triton.runtime.errors.OutOfResources:
                # Raised before the launch: nothing ran.
                self.coarse_blocks.pop(0)
        self._coarse_fallback(X)

    def _coarse_fallback(self, X):
        """Same as `_coarse_kernel` with PyTorch operations, for many cells."""
        n_rows = X.shape[0]
        n_cells, max_probes = self.n_cells, self.max_probes
        cells = self.padded_cells[:n_cells].to(torch.float32)
        cell_norms = self.cell_norms[:n_cells]
        chunk_size = max(1, 2**24 // n_cells)
        for start in range(0, n_rows, chunk_size):
            rows = X[start : start + chunk_size].to(torch.float32)
            scores = torch.addmm(cell_norms[None, :], rows, cells.T, alpha=-2.0)
            scores, probes = torch.topk(
                scores, max_probes, dim=1, largest=False, sorted=True
            )
            if math.isfinite(self.ratio):
                x_norms = rows.square().sum(1, keepdim=True)
                nearest = (x_norms + scores[:, :1]).clamp_min(0.0)
                thresholds = torch.maximum(
                    self.ratio**2 * nearest - x_norms, scores[:, :1]
                ).clamp_max(_FLOAT32_MAX)
                valid = scores <= thresholds
            else:
                valid = scores < torch.inf
            self.probes[start : start + rows.shape[0]] = torch.where(valid, probes, -1)
        probes = self.probes[:n_rows].reshape(-1)
        self.counts[:n_cells] = torch.bincount(probes[probes >= 0], minlength=n_cells)
