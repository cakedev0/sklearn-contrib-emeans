import cProfile
import pstats
import sys

import torch

sys.path.insert(0, ".")
import exp
import hkm

torch.set_num_threads(64)
hkm.use_kernels(64)
X = torch.from_numpy(exp.load(sys.argv[1]))
ev = torch.arange(100000)
k = int(sys.argv[2])
hkm.run(X[:20000], {"method": "ivf", "k": 64, "T": 1}, ev[:1000])
for cfg in (
    {"method": "ivf", "k": k, "p": 1},
    {"method": "ivf", "k": k, "p": 4},
    {"method": "ivf", "k": k, "p": 16, "ratio": 1.15},
    {"method": "graph", "k": k, "m": 32, "graph_period": 5},
):
    pr = cProfile.Profile()
    pr.enable()
    r = hkm.run(X, cfg, ev)
    pr.disable()
    print(cfg, r["t_fit"], r["batch_size"], r["n_steps"], r["tm"])
    pstats.Stats(pr).sort_stats("tottime").print_stats(6)
