"""faiss PR 5915 (cost-based revival) vs sklearn-contrib-emeans revivals.

Same data, budgets and seeds as configs/kmeans_pareto.py. The inertia of every
run is recomputed here from the final centers, in float64.
"""

import argparse
import json
import os
import sys
import time

import faiss
import numpy as np
import pandas as pd
from faiss.contrib.e_means import NumpyEmeans
from sklearn.metrics import pairwise_distances_argmin
from threadpoolctl import threadpool_limits

from sklearn_emeans import EMeans
from sklearn_emeans._kmeans_parallel import _kmeans_parallel

CACHE = os.path.expanduser("~/scikit-learn-benchmarks/data_cache")
N_THREADS = 24


THREAD_DATA = os.path.expanduser("~/work/faiss-pr5915/data")


def load(name):
    """Training rows and held-out rows (None for the Pareto benchmark datasets)."""
    if os.path.isdir(f"{THREAD_DATA}/{name}"):
        train = np.ascontiguousarray(np.load(f"{THREAD_DATA}/{name}/train.npy"))
        return train, np.load(f"{THREAD_DATA}/{name}/test.npy")
    X = pd.read_parquet(f"{CACHE}/{name}_x.parq").to_numpy(dtype=np.float32)
    return np.ascontiguousarray(X), None


def inertia_per_sample(X, centers):
    centers = np.ascontiguousarray(centers, dtype=np.float32)
    total = 0.0
    counts = np.zeros(len(centers), dtype=np.int64)
    for start in range(0, len(X), 1 << 18):
        chunk = X[start : start + (1 << 18)]
        labels = pairwise_distances_argmin(chunk, centers)
        diff = chunk.astype(np.float64) - centers[labels].astype(np.float64)
        total += float(np.einsum("ij,ij->", diff, diff))
        counts += np.bincount(labels, minlength=len(centers))
    return total / len(X), int((counts == 0).sum())


def kmeans_parallel_init(X, k, seed):
    with threadpool_limits(limits=1, user_api="blas"):
        return _kmeans_parallel(X, k, np.random.RandomState(seed), N_THREADS)


def run_faiss(X, k, n_epochs, seed, init=None, **kwargs):
    em = NumpyEmeans(X.shape[1], k, nepochs=n_epochs, seed=seed, **kwargs)
    tic = time.perf_counter()
    init_centers = None if init is None else kmeans_parallel_init(X, k, seed)
    with threadpool_limits(limits=N_THREADS):
        centers, _ = em.train(X, init_centers)
    elapsed = time.perf_counter() - tic
    recipe = em.recipe or {}
    extra = {"n_splits": int(em.n_splits), "batch_size": int(recipe["batch_size"])}
    if getattr(em, "closing_handover_stats", None):
        extra["handover"] = em.closing_handover_stats
    return centers, elapsed, extra


def run_emeans(X, k, n_epochs, seed, **kwargs):
    em = EMeans(n_clusters=k, n_epochs=n_epochs, random_state=seed, **kwargs)
    tic = time.perf_counter()
    em.fit(X)
    elapsed = time.perf_counter() - tic
    return em.cluster_centers_, elapsed, {"batch_size": int(em.batch_size_)}


VARIANTS = {
    "faiss-pr cost-based": lambda *a: run_faiss(*a),
    "faiss-pr mass 0.1": lambda *a: run_faiss(*a, unused_threshold=0.1),
    "faiss-pr cost-based, k-means||": lambda *a: run_faiss(*a, init="k-means||"),
    "emeans revival=0.1": lambda *a: run_emeans(*a),
    "emeans greedy": lambda *a: run_emeans(*a, revival="greedy"),
    "emeans k-means||": lambda *a: run_emeans(*a, init="k-means||"),
    "emeans greedy, k-means||": lambda *a: run_emeans(
        *a, init="k-means||", revival="greedy"
    ),
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--datasets",
    nargs="+",
    default=["road_network_km_x10", "year_prediction_msd_standardized", "susy"],
)
parser.add_argument("--k", nargs="+", type=int, default=[64, 256, 1024])
parser.add_argument("--epochs", nargs="+", type=float, default=[2, 5, 10])
parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
parser.add_argument("--out", default="results/results.jsonl")
args = parser.parse_args()

faiss.omp_set_num_threads(N_THREADS)
with open(args.out, "a") as out:
    for dataset in args.datasets:
        X, X_test = load(dataset)
        print(dataset, X.shape, file=sys.stderr, flush=True)
        # Warm-up of both libraries.
        run_faiss(X[:100_000], 64, 1, 0)
        run_emeans(X[:100_000], 64, 1, 0)
        from threadpoolctl import threadpool_info

        for m in threadpool_info():
            print(
                m["user_api"],
                m["internal_api"],
                m.get("threading_layer"),
                m["num_threads"],
                m["filepath"],
                file=sys.stderr,
                flush=True,
            )
        for k in args.k:
            for n_epochs in args.epochs:
                n_epochs = int(n_epochs) if float(n_epochs).is_integer() else n_epochs
                for seed in args.seeds:
                    for name in args.variants:
                        run = VARIANTS[name]
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
