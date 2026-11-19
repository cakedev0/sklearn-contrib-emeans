"""Run a list of configurations on one dataset, appending JSON lines."""

import argparse
import json
import os
import sys
import time
import urllib.request

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hkm  # noqa: E402

DIRS = [
    os.path.expanduser("~/work/hkm/data"),
    os.path.expanduser("~/real-emb-data"),
]
DEEP_URL = (
    "https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.10M.fbin"
)


def load(name, n=None):
    if name.startswith("deep"):
        n_all = 100_000_000 if name == "deep100M" else 10_000_000
        for d in DIRS:
            path = os.path.join(d, name + ".fbin")
            if os.path.exists(path):
                X = np.memmap(
                    path, dtype=np.float32, mode="r", offset=8, shape=(n_all, 96)
                )
                break
        else:
            n_get = n or n_all
            total = n_get * 96 * 4
            buf = np.empty(total, dtype=np.uint8)
            step = 1 << 28
            for off in range(0, total, step):
                end = min(off + step, total)
                for attempt in range(20):
                    try:
                        req = urllib.request.Request(
                            DEEP_URL,
                            headers={"Range": f"bytes={8 + off}-{8 + end - 1}"},
                        )
                        with urllib.request.urlopen(req, timeout=60) as r:
                            data = r.read()
                        if len(data) != end - off:
                            raise OSError("short read")
                        buf[off:end] = np.frombuffer(data, dtype=np.uint8)
                        break
                    except Exception as exc:  # stalled or dropped connection
                        print("download retry", off, repr(exc), flush=True)
                else:
                    raise OSError("download failed")
            X = buf.view(np.float32).reshape(-1, 96)
    else:
        for d in DIRS:
            path = os.path.join(d, name + ".npy")
            if os.path.exists(path):
                X = np.load(path, mmap_mode="r")
                break
        else:
            raise FileNotFoundError(name)
    if n is not None:
        X = X[:n]
    return np.ascontiguousarray(X, dtype=np.float32)


def run_emeans(Xt, cfg, eval_rows):
    """The real EMeans, as a timing anchor."""
    import sklearn

    from sklearn_emeans import EMeans

    est = EMeans(
        n_clusters=cfg["k"], n_epochs=cfg.get("T", 5), random_state=cfg.get("seed", 0)
    )
    hkm.sync(Xt.device)
    t0 = time.perf_counter()
    if Xt.device.type == "cuda":
        with sklearn.config_context(array_api_dispatch=True):
            est.fit(Xt)
        C, labels = est.cluster_centers_, est.labels_.long()
    else:
        est.fit(Xt.numpy())
        C = torch.from_numpy(est.cluster_centers_)
        labels = torch.from_numpy(est.labels_).long()
    hkm.sync(Xt.device)
    res = dict(cfg)
    res.update(
        n=Xt.shape[0],
        d=Xt.shape[1],
        t_fit=time.perf_counter() - t0,
        work=cfg.get("T", 5) + 1,
        batch_size=est.batch_size_,
    )
    res.update(hkm.evaluate(Xt, C, labels, eval_rows))
    return res


def run_superemeans(Xt, cfg, eval_rows, n_threads):
    from superkmeans import SuperEMeans

    X = Xt.numpy()
    est = SuperEMeans(
        n_clusters=cfg["k"],
        dimensionality=X.shape[1],
        epochs=cfg.get("T", 5),
        n_threads=n_threads,
        seed=cfg.get("seed", 0),
        pruning=cfg.get("pruning", True),
    )
    t0 = time.perf_counter()
    C = est.train(X)
    t1 = time.perf_counter()
    labels = est.assign_training_points(X, C)
    t2 = time.perf_counter()
    res = dict(cfg)
    res.update(
        n=X.shape[0],
        d=X.shape[1],
        t_fit=t2 - t0,
        t_train=t1 - t0,
        work=cfg.get("T", 5) + 1,
        batch_size=est.batch_size_,
    )
    res.update(
        hkm.evaluate(
            Xt,
            torch.from_numpy(np.ascontiguousarray(C)),
            torch.from_numpy(labels.astype(np.int64)),
            eval_rows,
        )
    )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--configs", required=True, help="JSON file: list of dicts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-kernels", action="store_true")
    ap.add_argument("--ivf-kernel", default="ivf_assign")
    ap.add_argument("--host", default=os.uname().nodename)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.device == "cpu" and not args.no_kernels:
        hkm.use_kernels(args.threads or torch.get_num_threads())
        hkm.KERN["ivf"] = args.ivf_kernel
    with open(args.configs) as f:
        configs = json.load(f)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                r = json.loads(line)
                done.add(
                    (
                        r["data"],
                        r["n"] if "n" in r else None,
                        json.dumps(r["cfg"], sort_keys=True),
                    )
                )
    X = load(args.data, args.n)
    Xt = torch.from_numpy(X).to(args.device)
    n = Xt.shape[0]
    print(f"{args.data}: {tuple(Xt.shape)} on {args.device}", flush=True)
    gen = torch.Generator().manual_seed(4242)
    perm = torch.randperm(n, generator=gen)
    # warm-up
    hkm.run(
        Xt[:20000],
        {"method": "ivf", "k": 64, "T": 1},
        torch.arange(1000, device=Xt.device),
    )
    for cfg in configs:
        key = json.dumps(cfg, sort_keys=True)
        if (args.data, n, key) in done:
            continue
        n_eval = cfg.get("n_eval") or (1_000_000 if cfg["k"] <= 20000 else 200_000)
        eval_rows = perm[: min(n, n_eval)].to(Xt.device)
        try:
            if cfg["method"] == "emeans":
                res = run_emeans(Xt, cfg, eval_rows)
            elif cfg["method"] == "superemeans":
                res = run_superemeans(Xt, cfg, eval_rows, torch.get_num_threads())
            else:
                res = hkm.run(Xt, cfg, eval_rows)
        except Exception as exc:  # keep going, record the failure
            import traceback

            traceback.print_exc()
            res = {"error": repr(exc)}
            if args.device == "cuda":
                torch.cuda.empty_cache()
        res.update(
            cfg=cfg,
            data=args.data,
            device=args.device,
            host=args.host,
            threads=torch.get_num_threads(),
        )
        with open(args.out, "a") as f:
            f.write(json.dumps(res) + "\n")
        short = {
            k: (round(v, 5) if isinstance(v, float) else v)
            for k, v in res.items()
            if k
            in (
                "t_fit",
                "work",
                "inertia_exact",
                "inertia_labels",
                "recall",
                "size_cv",
                "error",
                "max_group",
                "frac_fallback",
                "n_revived",
            )
        }
        print(json.dumps(cfg), "->", json.dumps(short), flush=True)


if __name__ == "__main__":
    main()
