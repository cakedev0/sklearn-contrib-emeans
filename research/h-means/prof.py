import sys

import torch

sys.path.insert(0, ".")
import exp
import hkm

X = torch.from_numpy(exp.load("deep10M")).cuda()
ev = torch.arange(100000, device="cuda")
hkm.run(X[:20000], {"method": "ivf", "k": 64, "T": 1}, ev[:1000])
from torch.profiler import ProfilerActivity, profile

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    r = hkm.run(X, {"method": "ivf", "k": 10000, "p": 4, "T": 1}, ev)
print(r["t_fit"], r["work"])
print(
    prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=18, max_name_column_width=40
    )
)
