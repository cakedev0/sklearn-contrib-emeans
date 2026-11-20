"""Turn the JSON lines of results/ into CSV tables.

Run from this directory: ``python summarize.py``. It writes

- ``results/all_runs.csv``: one row per run, with the inertia relative to the
  flat reference with 5 epochs of the same file, dataset, n and k;
- ``results/scaling.csv``: fit time against n at 1,000 samples per cluster
  (one Xeon socket), with the exponent between consecutive sizes.
"""

import csv
import glob
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCES = ("emeans", "flat", "faiss-e")  # by order of preference
SKIP = ("method", "k", "seed", "diag", "T")


def config_name(cfg):
    extra = [f"{key}={val}" for key, val in cfg.items() if key not in SKIP]
    return cfg["method"] + ("(" + ",".join(extra) + ")" if extra else "")


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def all_runs():
    out = []
    for path in sorted(glob.glob(os.path.join(HERE, "results", "*", "*.jsonl"))):
        runs = load(path)
        refs = {}
        for method in reversed(REFERENCES):
            for r in runs:
                cfg = r["cfg"]
                if (
                    cfg["method"] == method
                    and cfg.get("T", 5) == 5
                    and "error" not in r
                ):
                    refs[r["data"], r["n"], cfg["k"]] = r
        for r in runs:
            cfg = r["cfg"]
            row = {
                "machine": os.path.basename(os.path.dirname(path)),
                "file": os.path.basename(path),
                "data": r["data"],
                "n": r.get("n"),
                "k": cfg["k"],
                "method": cfg["method"],
                "config": config_name(cfg),
                "n_epochs": cfg.get("T", 5),
                "seed": cfg.get("seed", 0),
            }
            if "error" in r:
                row["error"] = r["error"]
                out.append(row)
                continue
            ref = refs.get((r["data"], r["n"], cfg["k"]))
            row.update(
                t_fit=round(r["t_fit"], 3),
                work=round(r["work"], 4),
                inertia_centers=r["inertia_exact"],
                inertia_labels=r["inertia_labels"],
                recall=round(r["recall"], 4),
                size_cv=round(r["size_cv"], 3),
            )
            if ref is not None:
                base = ref["inertia_exact"]
                row.update(
                    reference=config_name(ref["cfg"]),
                    loss_centers_pct=round(100 * (r["inertia_exact"] / base - 1), 3),
                    loss_labels_pct=round(100 * (r["inertia_labels"] / base - 1), 3),
                    speedup=round(ref["t_fit"] / r["t_fit"], 2),
                )
            out.append(row)
    return out


def write(name, rows):
    fields = []
    for row in rows:
        fields += [key for key in row if key not in fields]
    with open(os.path.join(HERE, "results", name), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def scaling(rows):
    rows = [
        r
        for r in rows
        if r["machine"] == "xeon"
        and r["file"] in ("v4_scaling.jsonl", "v4_final_100m.jsonl")
        and "error" not in r
    ]
    out = []
    for config in dict.fromkeys((r["config"], r["n_epochs"]) for r in rows):
        runs = sorted(
            (r for r in rows if (r["config"], r["n_epochs"]) == config),
            key=lambda r: r["n"],
        )
        previous = None
        for r in runs:
            row = {key: r.get(key) for key in ("config", "n_epochs", "n", "k")}
            row.update(
                t_fit=r["t_fit"],
                work=r["work"],
                loss_centers_pct=r.get("loss_centers_pct"),
                loss_labels_pct=r.get("loss_labels_pct"),
            )
            if previous is not None:
                row["time_exponent"] = round(
                    math.log(r["t_fit"] / previous["t_fit"])
                    / math.log(r["n"] / previous["n"]),
                    2,
                )
            out.append(row)
            previous = r
    return out


if __name__ == "__main__":
    runs = all_runs()
    write("all_runs.csv", runs)
    write("scaling.csv", scaling(runs))
    print(len(runs), "runs")
