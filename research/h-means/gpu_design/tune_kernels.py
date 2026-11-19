"""Block sizes of the Triton kernels: time of the coarse and fine kernels.

python tune_kernels.py DATA.npy K
"""

import itertools
import sys

import numpy as np
import torch
import triton_proto as tp
from bench_assign import BATCH, DEV, make_problem, timed

X = torch.from_numpy(np.load(sys.argv[1], mmap_mode="r")[:].copy()).to(DEV)
k = int(sys.argv[2])
C, G, member = make_problem(X, k)
Xb = X[:BATCH].contiguous()
cells = tp.Cells(C, G, member, torch.float32)
ws = tp.Workspace(BATCH, cells, 16, DEV)
r2 = 1.15**2
print("k", k, "cells", cells.g, flush=True)
for block, warps, stages in itertools.product((16, 32, 64), (4, 8), (1, 2, 3)):
    try:
        R, t = timed(
            lambda: tp.coarse(
                Xb,
                cells,
                ws,
                r2,
                dynamic=False,
                block_b=block,
                num_warps=warps,
                num_stages=stages,
            )
        )
        print(
            f"coarse block {block} warps {warps} stages {stages}: {t:6.1f} ms",
            flush=True,
        )
    except Exception as exc:
        print(
            f"coarse block {block} warps {warps} stages {stages}: {repr(exc)[:60]}",
            flush=True,
        )
R = tp.coarse(Xb, cells, ws, r2, dynamic=False)
for tile, block_k, warps, stages in itertools.product(
    (32, 64, 128), (32, 64, 128), (4, 8), (1, 2, 3)
):
    try:
        _, t = timed(
            lambda: tp.fine(
                Xb,
                cells,
                ws,
                R,
                block_t=tile,
                block_k=block_k,
                num_warps=warps,
                num_stages=stages,
            ),
            reps=3,
        )
        print(
            f"fine tile {tile} centers {block_k} warps {warps} stages {stages}: {t:6.1f} ms",
            flush=True,
        )
    except Exception as exc:
        print(
            f"fine tile {tile} centers {block_k} warps {warps} stages {stages}: {repr(exc)[:60]}",
            flush=True,
        )
