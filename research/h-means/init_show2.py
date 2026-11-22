"""Tables of the init study with `EMeans(assignment="two-level")`.

Usage: python init_show2.py results/xeon/init/emeans.jsonl [inertia_labels]
"""

import json
import sys
from collections import defaultdict

import numpy as np

path = sys.argv[1]
metric = sys.argv[2] if len(sys.argv) > 2 else "inertia_exact"
runs = defaultdict(dict)  # (data, n, k) -> (init, epochs) -> seed -> record
for line in open(path):
    r = json.loads(line)
    runs[r["data"], r["n"], r["k"]].setdefault((r["init"], r["epochs"]), {})[
        r["seed"]
    ] = r

cols = list(runs)
names = [f"{d} {k}" for d, n, k in cols]
rows = []
for col in cols:
    for key in runs[col]:
        if key not in rows:
            rows.append(key)


def cell(col, key, what):
    ref = runs[col].get(("random", 5), {})
    cur = runs[col].get(key, {})
    seeds = sorted(set(ref) & set(cur))
    if not seeds:
        return ""
    if what == "rel":
        rel = [cur[s][metric] / ref[s][metric] - 1 for s in seeds]
        return f"{100 * np.median(rel):+.2f}%"
    if what == "sd":
        rel = [cur[s][metric] / ref[s][metric] - 1 for s in seeds]
        return f"{100 * min(rel):+.1f} to {100 * max(rel):+.1f}"
    if what == "t_init":
        # The init of the string inits is inside the fit: use the difference
        # with the fit of random.
        if key[0] == "k-means||":
            t = [cur[s]["t_fit"] - ref[s]["t_fit"] for s in seeds]
        else:
            t = [cur[s]["t_init"] for s in seeds]
        return f"{np.median(t):.2f}"
    return f"{np.median([cur[s]['t_init'] + cur[s]['t_fit'] for s in seeds]):.2f}"


def table(title, what, keys):
    print(f"\n{title}\n")
    print("| init | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for key in keys:
        label = key[0] if key[1] == 5 else f"{key[0]}, {key[1]} epochs"
        print(f"| `{label}` | " + " | ".join(cell(c, key, what) for c in cols) + " |")


inits = [key for key in rows if key != ("random", 5)]
table(f"{metric} against random with 5 epochs (median over the seeds)", "rel", inits)
table("range over the seeds (%)", "sd", inits)
table("time of the init (s, median)", "t_init", [k for k in rows if k[1] == 5])
table("time of the init and the fit (s, median)", "t_total", rows)
print("\nseeds:", {n: len(runs[c][("random", 5)]) for n, c in zip(names, cols)})
