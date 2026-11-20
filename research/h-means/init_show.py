"""Table of the init study: inertia of the labels relative to the random init
with 5 epochs of the same fit, mean over the seeds."""

import collections
import json
import sys

runs = [json.loads(line) for path in sys.argv[1:] for line in open(path)]
groups = collections.defaultdict(list)
for r in runs:
    groups[r["data"], r["n"], r["k"], r["fit"], r["init"], r["T"]].append(r)


def mean(rs, key):
    vals = [r[key] for r in rs if key in r]
    return sum(vals) / len(vals) if vals else float("nan")


cases = dict.fromkeys(key[:4] for key in groups)
for case in cases:
    ref = mean(groups[(*case, "random", 5)], "inertia_labels")
    epochs = sorted({key[5] for key in groups if key[:4] == case})
    inits = dict.fromkeys(key[4] for key in groups if key[:4] == case)
    n_seeds = len(groups[(*case, "random", 5)])
    print(f"\n== {case[0]} n={case[1]} k={case[2]} fit={case[3]} ({n_seeds} seeds)")
    print(
        f"{'init':14s} {'t_init':>7s} {'init%':>7s} "
        + " ".join(f"{'T=' + str(T):>8s}" for T in epochs)
        + "   t_fit(T=5)"
    )
    for init in inits:
        rs5 = groups[(*case, init, epochs[-1])]
        cells = " ".join(
            f"{100 * (mean(groups[(*case, init, T)], 'inertia_labels') / ref - 1):+8.3f}"
            for T in epochs
        )
        t = mean(rs5, "t_fit") if init != "k-means||" else mean(rs5, "t_total")
        print(
            f"{init:14s} {mean(rs5, 't_init'):7.2f} {100 * (mean(rs5, 'init_inertia') / ref - 1):+7.1f} {cells}   {t:.2f}"
        )
