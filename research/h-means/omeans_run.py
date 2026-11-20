"""faiss e-means (dense) vs o-means (two-level) on the GPU, same metrics as exp.py."""

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp
import hkm
from faiss.contrib.e_means import Emeans
from faiss.contrib.o_means import Omeans

data, out = sys.argv[1], sys.argv[2]
ks = [int(a) for a in sys.argv[3].split(",")]
torch.backends.cuda.matmul.allow_tf32 = False
N = int(sys.argv[4]) if len(sys.argv) > 4 else None
X = torch.from_numpy(exp.load(data, N)).cuda()
n, d = X.shape
perm = torch.randperm(n, generator=torch.Generator().manual_seed(4242))
done = set()
if os.path.exists(out):
    done = {
        (r["data"] + str(r.get("n")), json.dumps(r["cfg"], sort_keys=True))
        for r in map(json.loads, open(out))
    }
for k in ks:
    g = max(2, round(k**0.5))
    cfgs = [dict(method="faiss-e", k=k, T=T) for T in (5, 2, 1)]
    cfgs += [dict(method="omeans", k=k, T=T) for T in (5, 2, 1)]
    cfgs += [
        dict(method="omeans", k=k, T=5, nlist=g, nprobe=p)
        for p in (1, 2, 4, 8)
        if p <= g
    ]
    cfgs += [
        dict(method="omeans", k=k, T=5, nlist=2 * g, nprobe=p)
        for p in (4, 8, 16)
        if p <= g
    ]
    n_eval = 1_000_000 if k <= 20000 else 200_000
    rows = perm[: min(n, n_eval)].cuda()
    init = hkm.random_init(X, k, 0)
    for cfg in cfgs:
        if (data + str(n), json.dumps(cfg, sort_keys=True)) in done:
            continue
        best, res = float("inf"), None
        try:
            for rep in range(2):
                if cfg["method"] == "faiss-e":
                    m = Emeans(d, k, nepochs=cfg["T"], batch_law="throughput", seed=0)
                else:
                    m = Omeans(
                        d,
                        k,
                        nepochs=cfg["T"],
                        batch_law="throughput",
                        seed=0,
                        nlist=cfg.get("nlist"),
                        nprobe=cfg.get("nprobe"),
                    )
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                C, labels = m.train(X, init)[:2]
                torch.cuda.synchronize()
                best = min(best, time.perf_counter() - t0)
            res = dict(t_fit=best, n=n, d=d)
            if cfg["method"] == "omeans":
                res.update(
                    nlist=m.recipe["nlist"],
                    nprobe=m.recipe["nprobe"],
                    work=cfg["T"]
                    * (m.recipe["nlist"] + m.recipe["nprobe"] * k / m.recipe["nlist"])
                    / k,
                )
            else:
                res["work"] = cfg["T"]
            res["batch_size"] = (
                m.recipe["batch_size"] if getattr(m, "recipe", None) else None
            )
            res.update(
                hkm.evaluate(
                    X, C.float(), labels.long() if labels is not None else None, rows
                )
            )
        except Exception as exc:
            import traceback

            traceback.print_exc()
            res = {"error": repr(exc)}
        res.update(cfg=cfg, data=data, device="cuda", host=os.uname().nodename)
        open(out, "a").write(json.dumps(res) + "\n")
        print(
            cfg,
            {
                a: (round(b, 5) if isinstance(b, float) else b)
                for a, b in res.items()
                if a
                in (
                    "t_fit",
                    "work",
                    "inertia_exact",
                    "inertia_labels",
                    "recall",
                    "error",
                    "nlist",
                    "nprobe",
                )
            },
            flush=True,
        )
