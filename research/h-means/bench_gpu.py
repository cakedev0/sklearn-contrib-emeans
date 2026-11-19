"""EMeans with assignment="two-level" on a GPU, against the dense e-means.

    python bench_gpu.py DATA OUT.jsonl K[,K...] [N] [--methods ...] [--reps 2]

DATA is a name that `exp.load` knows or the path of a .npy file.

Methods, all with 5 epochs and a random init:

- `emeans`: `EMeans` with the exact assignment, on a PyTorch CUDA tensor.
- `emeans-2l`: `EMeans(assignment="two-level")`.
- `emeans-2l-tf32`, `emeans-2l-ieee`: same with the other precisions of the
  kernels (`_two_level_triton.PRECISION`) in place of float16.
- `faiss-e`, `omeans-p4`, `omeans-p8`: the dense e-means and o-means of the
  owlgebra-ai/faiss fork, o-means with round(sqrt(K)) cells and 4 or 8 probes.
  They need the fork on PYTHONPATH.

Each line of OUT has the best fit time of `--reps` runs and the metrics of
`hkm.evaluate` (same rows as `exp.py` and `omeans_run.py`): the exact inertia
of the centers and the inertia of the returned labels.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import sklearn
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp  # noqa: E402
import hkm  # noqa: E402

METHODS = "emeans,emeans-2l,faiss-e,omeans-p4,omeans-p8"


def fit(method, X, k, n_epochs, init):
    """Returns (centers, labels, extra results)."""
    if method.startswith("emeans"):
        from sklearn_emeans import EMeans, _emeans

        precision = method.rsplit("-", 1)[1] if method.count("-") == 2 else None
        if precision is not None:
            from sklearn_emeans import _two_level_triton

            default_precision = _two_level_triton.PRECISION
            _two_level_triton.PRECISION = precision
        est = EMeans(
            n_clusters=k,
            n_epochs=n_epochs,
            assignment="two-level" if "-2l" in method else "exact",
            random_state=0,
        )
        with sklearn.config_context(array_api_dispatch=True):
            est.fit(X)
        if precision is not None:
            _two_level_triton.PRECISION = default_precision
        extra = dict(batch_size=est.batch_size_, inertia_attr=est.inertia_ / len(X))
        if "-2l" in method:
            extra.update(
                max_probes=_emeans._TWO_LEVEL_MAX_PROBES, ratio=_emeans._TWO_LEVEL_RATIO
            )
        return est.cluster_centers_, est.labels_.long(), extra
    if method == "faiss-e":
        from faiss.contrib.e_means import Emeans

        model = Emeans(X.shape[1], k, nepochs=n_epochs, batch_law="throughput", seed=0)
        centers, labels = model.train(X, init)[:2]
        return centers.float(), labels.long(), {}
    from faiss.contrib.o_means import Omeans

    model = Omeans(
        X.shape[1],
        k,
        nepochs=n_epochs,
        batch_law="throughput",
        seed=0,
        nlist=max(2, round(k**0.5)),
        nprobe=int(method.split("-p")[1]),
    )
    centers, labels = model.train(X, init)[:2]
    extra = dict(nlist=model.recipe["nlist"], nprobe=model.recipe["nprobe"])
    return centers.float(), labels.long(), extra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("out")
    parser.add_argument("ks")
    parser.add_argument("n", nargs="?", type=int, default=None)
    parser.add_argument("--methods", default=METHODS)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--epochs", type=float, default=5)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False

    if os.path.exists(args.data):
        X = torch.from_numpy(np.load(args.data, mmap_mode="r")[: args.n].copy()).cuda()
    else:
        X = torch.from_numpy(exp.load(args.data, args.n)).cuda()
    n, d = X.shape
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(4242))
    for k in map(int, args.ks.split(",")):
        rows = perm[: min(n, 1_000_000 if k <= 20000 else 200_000)].cuda()
        init = hkm.random_init(X, k, 0)
        for method in args.methods.split(","):
            res = dict(method=method, data=args.data, n=n, d=d, k=k, T=args.epochs)
            try:
                times = []
                for _ in range(args.reps):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    centers, labels, extra = fit(method, X, k, args.epochs, init)
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - t0)
                res.update(t_fit=min(times), times=times, **extra)
                res.update(hkm.evaluate(X, centers, labels, rows))
                del centers, labels
            except Exception as exc:
                import traceback

                traceback.print_exc()
                res["error"] = repr(exc)[:300]
            torch.cuda.empty_cache()
            res["host"] = os.uname().nodename
            # A GPU that throttles gives timings that can't be compared.
            res["gpu_temperature"], res["gpu_clock_mhz"] = (
                subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=temperature.gpu,clocks.gr",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                )
                .stdout.strip()
                .split(", ")
            )
            with open(args.out, "a") as f:
                f.write(json.dumps(res) + "\n")
            print(
                {
                    key: round(value, 5) if isinstance(value, float) else value
                    for key, value in res.items()
                    if key
                    in (
                        "method",
                        "n",
                        "k",
                        "t_fit",
                        "inertia_exact",
                        "inertia_labels",
                        "recall",
                        "error",
                        "gpu_temperature",
                        "gpu_clock_mhz",
                    )
                },
                flush=True,
            )


if __name__ == "__main__":
    main()
