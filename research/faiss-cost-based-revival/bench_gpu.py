"""faiss PR 5915 on GPU (cost-based revival) vs sklearn-contrib-emeans on GPU.

Runs in an existing PyTorch environment, with the faiss wheel, the PR's
`contrib/e_means` and sklearn-contrib-emeans 0.4.1 in `overlay/` and `pr/`.
The inertia of every run is recomputed from the final centers in float64.
"""

import argparse
import json
import os
import sys
import time

os.environ["SCIPY_ARRAY_API"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{HERE}/overlay")
# The environment has an editable install of an older sklearn_emeans.
sys.meta_path[:] = [
    f
    for f in sys.meta_path
    if "mesonpy" not in type(f).__module__
    and "editable" not in type(f).__module__.lower()
]

import faiss
import faiss.contrib
import numpy as np
import sklearn
import torch

faiss.contrib.__path__.insert(0, f"{HERE}/pr/contrib")
from faiss.contrib.e_means import Emeans, QuantizedEmeans  # noqa: E402

import sklearn_emeans  # noqa: E402
from sklearn_emeans import EMeans, _array_api  # noqa: E402

sklearn.set_config(array_api_dispatch=True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")


def load(name):
    if os.path.isdir(f"{HERE}/data/{name}"):
        return np.load(f"{HERE}/data/{name}/train.npy"), np.load(
            f"{HERE}/data/{name}/test.npy"
        )
    return np.load(f"{HERE}/npy/{name}.npy"), None


def inertia_per_sample(X, centers):
    centers = torch.as_tensor(centers, device=X.device).double()
    total = 0.0
    counts = torch.zeros(len(centers), dtype=torch.int64, device=X.device)
    chunk = max(1024, (1 << 27) // len(centers))
    for start in range(0, len(X), chunk):
        rows = X[start : start + chunk].double()
        labels = torch.cdist(rows, centers).argmin(dim=1)
        total += float(((rows - centers[labels]) ** 2).sum())
        counts += torch.bincount(labels, minlength=len(centers))
    return total / len(X), int((counts == 0).sum())


def sync():
    torch.cuda.synchronize()


def kmeans_parallel_init(X, k, seed):
    xp = _array_api.get_namespace(X)
    centers = _array_api.kmeans_parallel(
        xp, X, k, np.random.RandomState(seed), os.cpu_count()
    )
    return torch.as_tensor(centers, dtype=torch.float32, device=X.device)


def run_faiss(cls, X, k, n_epochs, seed, init=None, **kwargs):
    em = cls(X.shape[1], k, nepochs=n_epochs, seed=seed, **kwargs)
    sync()
    tic = time.perf_counter()
    init_centers = None if init is None else kmeans_parallel_init(X, k, seed)
    centers, _ = em.train(X, init_centers)
    sync()
    elapsed = time.perf_counter() - tic
    recipe = em.recipe or {}
    extra = {
        "n_splits": int(em.n_splits),
        "batch_size": int(recipe["batch_size"]),
        "backend": str(recipe.get("assignment_backend")),
        "handover": str(recipe.get("closing_handover")),
    }
    return centers, elapsed, extra


def run_emeans(X, k, n_epochs, seed, **kwargs):
    em = EMeans(n_clusters=k, n_epochs=n_epochs, random_state=seed, **kwargs)
    sync()
    tic = time.perf_counter()
    em.fit(X)
    sync()
    elapsed = time.perf_counter() - tic
    return em.cluster_centers_, elapsed, {"batch_size": int(em.batch_size_)}


VARIANTS = {
    "faiss-pr Emeans cost-based (BF16)": lambda *a: run_faiss(Emeans, *a),
    "faiss-pr Emeans mass 0.1 (BF16)": lambda *a: run_faiss(
        Emeans, *a, unused_threshold=0.1
    ),
    "faiss-pr QuantizedEmeans mass 0.1 (FP32)": lambda *a: run_faiss(
        QuantizedEmeans, *a, quantizer=None
    ),
    "faiss-pr Emeans cost-based (BF16), k-means||": lambda *a: run_faiss(
        Emeans, *a, init="k-means||"
    ),
    "emeans revival=0.1": lambda *a: run_emeans(*a),
    "emeans greedy": lambda *a: run_emeans(*a, revival="greedy"),
    "emeans k-means||": lambda *a: run_emeans(*a, init="k-means||"),
    "emeans greedy, k-means||": lambda *a: run_emeans(
        *a, init="k-means||", revival="greedy"
    ),
}

parser = argparse.ArgumentParser()
parser.add_argument("--datasets", nargs="+", required=True)
parser.add_argument("--k", nargs="+", type=int, required=True)
parser.add_argument("--epochs", nargs="+", type=float, required=True)
parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
parser.add_argument("--out", default="results_gpu.jsonl")
args = parser.parse_args()

print(
    "versions",
    faiss.__version__,
    sklearn_emeans.__version__,
    sklearn_emeans.__file__,
    torch.__version__,
    torch.cuda.get_device_name(0),
    file=sys.stderr,
    flush=True,
)
with open(args.out, "a") as out:
    for dataset in args.datasets:
        X_host, X_test = load(dataset)
        X = torch.from_numpy(np.ascontiguousarray(X_host)).cuda()
        X_test = None if X_test is None else torch.from_numpy(X_test).cuda()
        print(dataset, tuple(X.shape), file=sys.stderr, flush=True)
        for k in args.k:
            for n_epochs in args.epochs:
                n_epochs = int(n_epochs) if float(n_epochs).is_integer() else n_epochs
                # Untimed fits: the kernels compile at their first run.
                for run in VARIANTS.values():
                    run(X, k, n_epochs, 100)
                for seed in args.seeds:
                    for name, run in VARIANTS.items():
                        centers, elapsed, extra = run(X, k, n_epochs, seed)
                        inertia, n_empty = inertia_per_sample(X, centers)
                        record = {
                            "dataset": dataset,
                            "k": k,
                            "n_epochs": n_epochs,
                            "seed": seed,
                            "variant": name,
                            "fit_s": elapsed,
                            "inertia_per_sample": inertia,
                            "n_empty": n_empty,
                            **extra,
                        }
                        if X_test is not None:
                            record["test_error"] = inertia_per_sample(X_test, centers)[
                                0
                            ]
                        out.write(json.dumps(record) + "\n")
                        out.flush()
                print(dataset, k, n_epochs, "done", file=sys.stderr, flush=True)
