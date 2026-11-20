"""Time `EMeans` with the exact and the two-level assignment on one dataset.

Appends one JSON line per fit, with the same keys as `exp.py` (`t_fit`,
`inertia_exact`, `inertia_labels`, `recall`), so that `show_cpu.py` can compare
them with the runs of the prototype.

    OMP_NUM_THREADS=64 python bench_cpu.py --data deep10M --k 10000 --out res.jsonl
"""

import argparse
import ctypes
import json
import os
import resource
import time

import numpy as np
import sklearn
from sklearn_emeans._kernels import _labels_inertia
from threadpoolctl import threadpool_info, threadpool_limits

from sklearn_emeans import EMeans, _emeans

DIRS = [
    os.path.expanduser("~/work/hkm/data"),
    os.path.expanduser("~/work/hmeans-cpu/data"),
    os.path.expanduser("~/real-emb-data"),
]
DEEP_N_SAMPLES = {
    "deep10M": 10_000_000,
    "deep100M": 100_000_000,
    "deep500M": 500_000_000,
}


def interleave(X):
    """Spread the pages of `X` over the NUMA nodes, before it is written.

    `numactl --interleave=all` does it for all the memory of the process, also
    for the work buffers of each thread, which are better on its own node.
    """
    with open("/sys/devices/system/node/has_memory") as f:
        nodes = 0
        for part in f.read().strip().split(","):
            first, _, last = part.partition("-")
            for node in range(int(first), int(last or first) + 1):
                nodes |= 1 << node
    page = os.sysconf("SC_PAGESIZE")
    start = -(-X.ctypes.data // page) * page
    length = (X.ctypes.data + X.nbytes - start) // page * page
    mbind = ctypes.CDLL("libnuma.so.1", use_errno=True).mbind
    mbind.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int]
    mbind.argtypes += [ctypes.POINTER(ctypes.c_ulong), ctypes.c_ulong, ctypes.c_uint]
    MPOL_INTERLEAVE = 3
    if mbind(start, length, MPOL_INTERLEAVE, ctypes.c_ulong(nodes), 65, 0) != 0:
        raise OSError(ctypes.get_errno(), "mbind failed")


def read_in_memory(
    path, n_samples, n_features, offset, keep_page_cache=False, interleaved=False
):
    """Read float32 rows in anonymous memory, without keeping the page cache.

    For data that takes most of the memory of a socket: the memory policy of
    the process then decides where the data is, and the page cache of the file
    does not take the room of the work buffers. `keep_page_cache` is for many
    short runs on a file that is much smaller than the memory.
    """
    X = np.empty((n_samples, n_features), dtype=np.float32)
    if interleaved:
        interleave(X)
    buffer = memoryview(X).cast("B")
    chunk = 2**30
    fd = os.open(path, os.O_RDONLY)
    try:
        done = 0
        while done < len(buffer):
            n_read = os.preadv(fd, [buffer[done : done + chunk]], offset + done)
            if n_read == 0:
                raise EOFError(path)
            if not keep_page_cache:
                os.posix_fadvise(fd, offset + done, n_read, os.POSIX_FADV_DONTNEED)
            done += n_read
    finally:
        os.close(fd)
    return X


def load(name, n=None, in_memory=False, keep_page_cache=False, interleaved=False):
    for directory in DIRS:
        if name.startswith("deep"):
            # `.fbin` has a header of 8 bytes, `.f32` is the raw float32 rows.
            path = os.path.join(directory, name + ".fbin")
            offset = 8
            if not os.path.exists(path):
                path = os.path.join(directory, name + ".f32")
                offset = 0
            if os.path.exists(path):
                n_all = DEEP_N_SAMPLES[name]
                if in_memory:
                    n_read = n_all if n is None else n
                    return read_in_memory(
                        path, n_read, 96, offset, keep_page_cache, interleaved
                    )
                X = np.memmap(
                    path, dtype=np.float32, mode="r", offset=offset, shape=(n_all, 96)
                )
                break
        else:
            path = os.path.join(directory, name + ".npy")
            if os.path.exists(path):
                X = np.load(path, mmap_mode="r")
                break
    else:
        raise FileNotFoundError(name)
    return np.ascontiguousarray(X if n is None else X[:n], dtype=np.float32)


def memory_per_numa_node():
    """Memory of this process on each NUMA node, in GB (Linux only)."""
    try:
        with open("/proc/self/numa_maps") as f:
            lines = f.readlines()
    except OSError:
        return None
    usage = {}
    for line in lines:
        fields = line.split()
        page_kb = 4
        for field in fields:
            if field.startswith("kernelpagesize_kB="):
                page_kb = int(field.split("=")[1])
        for field in fields:
            if (
                field[0] == "N"
                and "=" in field
                and field[1 : field.index("=")].isdigit()
            ):
                node, n_pages = field[1:].split("=")
                usage[int(node)] = usage.get(int(node), 0) + int(n_pages) * page_kb
    return [round(usage.get(node, 0) / 2**20, 1) for node in range(max(usage) + 1)]


PHASES = {}


def time_phases():
    """Add the wall time of the calls that `fit` makes to `PHASES`.

    What is left of the fit time is the sequential Python code, mostly the
    permutation of each epoch.
    """

    def timed(name, function):
        def wrapper(*args, **kwargs):
            tic = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                PHASES[name] = PHASES.get(name, 0) + time.perf_counter() - tic

        return wrapper

    for name, attribute in [
        ("validate", "validate_data"),
        ("init_cells", "_init_cells"),
        ("steps", "_emeans_steps"),
        ("steps", "_emeans_steps_two_level"),
        ("labels", "_labels_inertia_threadpool_limit"),
        ("labels", "_labels_inertia_two_level"),
    ]:
        setattr(_emeans, attribute, timed(name, getattr(_emeans, attribute)))
    EMeans._init_centers = timed("init_centers", EMeans._init_centers)


def environment():
    """The thread pools and the variables that configure them."""
    prefixes = ("OMP_", "GOMP_", "KMP_", "MKL_", "OPENBLAS_")
    return {
        "env": {k: v for k, v in os.environ.items() if k.startswith(prefixes)},
        "threadpools": [
            " ".join(
                str(info.get(key, ""))
                for key in ("user_api", "prefix", "version", "threading_layer")
            ).strip()
            for info in threadpool_info()
        ],
        "n_cpus": len(os.sched_getaffinity(0)),
    }


def evaluate(X_eval, centers, labels, n_threads):
    """Exact inertia of the centers and inertia of the labels, per sample."""
    # The exact assignment needs 256 * n_clusters distances per thread: 5 GB
    # with 5M clusters. Keep them under 80 GB in total.
    n_threads = max(1, min(n_threads, int(80e9 // (256 * 4 * len(centers)))))
    exact = np.empty(len(X_eval), dtype=np.int32)
    with threadpool_limits(limits=1, user_api="blas"):
        inertia_exact = _labels_inertia(X_eval, centers, exact, n_threads)
    inertia_labels = 0.0
    for start in range(0, len(X_eval), 2**16):
        diff = X_eval[start : start + 2**16].astype(np.float64)
        diff -= centers[labels[start : start + 2**16]]
        inertia_labels += (diff**2).sum()
    return {
        "inertia_exact": inertia_exact / len(X_eval),
        "inertia_labels": inertia_labels / len(X_eval),
        "recall": float(np.mean(exact == labels)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int, nargs="+", required=True)
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--init", default="random")
    parser.add_argument("--assignment", nargs="+", default=["exact", "two-level"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-probes", type=int)
    parser.add_argument("--ratio", type=float)
    parser.add_argument("--min-clusters", type=int)
    parser.add_argument(
        "--assume-finite",
        action="store_true",
        help="skip the check of the input of scikit-learn, as the prototype",
    )
    parser.add_argument(
        "--in-memory",
        action="store_true",
        help="read the data in anonymous memory (DEEP only), not as a memory map",
    )
    parser.add_argument(
        "--keep-page-cache",
        action="store_true",
        help="with --in-memory, don't drop the page cache of the file",
    )
    parser.add_argument(
        "--interleave-data",
        action="store_true",
        help="with --in-memory, spread the data (and only it) over the NUMA nodes",
    )
    parser.add_argument("--tag", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    # The private constants of the two-level assignment, to explore them.
    if args.max_probes is not None:
        _emeans._TWO_LEVEL_MAX_PROBES = args.max_probes
    if args.ratio is not None:
        _emeans._TWO_LEVEL_RATIO = args.ratio
    if args.min_clusters is not None:
        _emeans._TWO_LEVEL_MIN_CLUSTERS = args.min_clusters

    X = load(
        args.data, args.n, args.in_memory, args.keep_page_cache, args.interleave_data
    )
    numa_gb_data = memory_per_numa_node()
    n_samples = X.shape[0]
    n_threads = _emeans._openmp_effective_n_threads()
    print(f"{args.data}: {X.shape}, {n_threads} threads", flush=True)
    # Warm-up of the thread pools.
    EMeans(n_clusters=64, n_epochs=1, random_state=0).fit(X[:20000])
    time_phases()
    # Only the first million is kept: the permutation of 500M rows is 4 GB.
    permutation = np.random.RandomState(4242).permutation(n_samples)[:1_000_000]
    permutation = permutation.copy()

    for k in args.k:
        eval_rows = np.sort(permutation[: 1_000_000 if k <= 20000 else 200_000])
        X_eval = X[eval_rows]
        for assignment in args.assignment:
            for _ in range(args.repeat):
                emeans = EMeans(
                    n_clusters=k,
                    n_epochs=args.epochs,
                    init=args.init,
                    assignment=assignment,
                    random_state=args.seed,
                )
                PHASES.clear()
                tic = time.perf_counter()
                with sklearn.config_context(assume_finite=args.assume_finite):
                    emeans.fit(X)
                t_fit = time.perf_counter() - tic
                result = {
                    "data": args.data,
                    "n": n_samples,
                    "d": X.shape[1],
                    "k": k,
                    "method": "EMeans " + assignment,
                    "T": args.epochs,
                    "init": args.init,
                    "seed": args.seed,
                    "tag": args.tag,
                    "max_probes": _emeans._TWO_LEVEL_MAX_PROBES,
                    "ratio": _emeans._TWO_LEVEL_RATIO,
                    "threads": n_threads,
                    # The threads are capped by this limit of OpenBLAS.
                    "openblas_max_threads": _emeans._openblas_max_threads(),
                    "slot": os.environ.get("BENCHLOCK_SLOT"),
                    "assume_finite": args.assume_finite,
                    # Below this number of clusters, "two-level" is "exact".
                    "min_clusters": _emeans._TWO_LEVEL_MIN_CLUSTERS,
                    "in_memory": args.in_memory,
                    "interleave_data": args.interleave_data,
                    # Right after the load: where the data is.
                    "numa_gb_data": numa_gb_data,
                    "t_fit": t_fit,
                    "phases": {
                        **{name: round(t, 2) for name, t in PHASES.items()},
                        "other": round(t_fit - sum(PHASES.values()), 2),
                    },
                    **environment(),
                    "batch_size": emeans.batch_size_,
                    "inertia_fit": emeans.inertia_ / n_samples,
                    "max_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    / 2**20,
                    # After the fit, so without its work buffers.
                    "numa_gb": memory_per_numa_node(),
                }
                result.update(
                    evaluate(
                        X_eval,
                        emeans.cluster_centers_,
                        emeans.labels_[eval_rows],
                        n_threads,
                    )
                )
                with open(args.out, "a") as f:
                    f.write(json.dumps(result) + "\n")
                print(
                    f"k={k} {assignment} {args.tag}: {t_fit:.2f} s, "
                    f"centers {result['inertia_exact']:.6g}, "
                    f"labels {result['inertia_labels']:.6g}, "
                    f"recall {result['recall']:.4f}, phases {result['phases']}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
