"""Time of each part of a two-level fit on a GPU, with a synchronization
after each part (so the total is a bit above the one of a normal fit).

    python prof_gpu.py DATA K [N]
"""

import collections
import os
import sys
import time

import numpy as np
import sklearn
import torch

from sklearn_emeans import EMeans, _array_api, _two_level_triton

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp  # noqa: E402

TIMES = collections.defaultdict(float)


def timed(name, function):
    def wrapper(*args, **kwargs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = function(*args, **kwargs)
        torch.cuda.synchronize()
        TIMES[name] += time.perf_counter() - t0
        return out

    return wrapper


class TimedKernel:
    """A Triton kernel whose launches are timed, with a synchronization."""

    def __init__(self, name, kernel):
        self.name, self.kernel = name, kernel

    def __getitem__(self, grid):
        return timed(self.name, self.kernel[grid])


for kernel in ("_scatter_kernel", "_fine_kernel", "_reduce_kernel"):
    setattr(
        _two_level_triton,
        kernel,
        TimedKernel("  " + kernel, getattr(_two_level_triton, kernel)),
    )
PROBES = [0, 0]


def count_probes(function):
    def wrapper(self, X):
        out = function(self, X)
        PROBES[0] += int((self.probes[: X.shape[0]] >= 0).sum())
        PROBES[1] += X.shape[0]
        return out

    return wrapper


Assigner = _two_level_triton.TwoLevelAssigner
Assigner._assign = count_probes(Assigner._assign)
Assigner._kernel_copy = timed("  float16 copies", Assigner._kernel_copy)
Assigner.refresh = timed("refresh cells", Assigner.refresh)
Assigner._coarse = timed("  _coarse_kernel", Assigner._coarse)
Assigner._assign = timed("assign (with coarse)", Assigner._assign)
_array_api._cluster_sums = timed("sums of the batch", _array_api._cluster_sums)
_array_api._ema_update = timed("EMA update", _array_api._ema_update)
_array_api.revive_dead_clusters = timed("revival", _array_api.revive_dead_clusters)
_array_api.labels_inertia = timed(
    "labels pass (with assign)", _array_api.labels_inertia
)
_array_api.two_level_assigner = timed("init assigner", _array_api.two_level_assigner)
EMeans._init_cells_array_api = timed("init cells", EMeans._init_cells_array_api)
EMeans._init_centers_array_api = timed("init centers", EMeans._init_centers_array_api)
_array_api.index_array = timed("indices to the device", _array_api.index_array)


n = int(sys.argv[3]) if len(sys.argv) > 3 else None
if os.path.exists(sys.argv[1]):
    X = np.load(sys.argv[1], mmap_mode="r")[:n].copy()
else:
    X = exp.load(sys.argv[1], n)
X = torch.from_numpy(X).cuda()
k = int(sys.argv[2])
with sklearn.config_context(array_api_dispatch=True):
    for rep in range(2):
        TIMES.clear()
        PROBES[:] = [0, 0]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        est = EMeans(n_clusters=k, assignment="two-level", random_state=0).fit(X)
        torch.cuda.synchronize()
        total = time.perf_counter() - t0
print(f"n={X.shape[0]} k={k} batch_size={est.batch_size_} total {total:.3f} s")
for name, t in sorted(TIMES.items(), key=lambda item: -item[1]):
    print(f"  {name:28s} {t:7.3f} s")
print(f"  probes per row, with the labels pass: {PROBES[0] / PROBES[1]:.2f}")
rng = np.random.RandomState(0)
t0 = time.perf_counter()
rng.permutation(X.shape[0])
print(f"  one permutation on the host: {time.perf_counter() - t0:.3f} s")
