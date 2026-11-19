"""Table of the runs of bench_gpu.py: python show_gpu.py results/l4/gpu_main.jsonl

The loss is the inertia of the labels against the one of `emeans` (exact
assignment) on the same data and number of clusters.
"""

import json
import sys

runs = [json.loads(line) for line in open(sys.argv[1])]
reference = {
    (r["data"], r["n"], r["k"]): r["inertia_labels"]
    for r in runs
    if r["method"] == "emeans" and "error" not in r
}
print(
    f"{'data':13s} {'n':>9s} {'k':>7s} {'method':15s} {'fit (s)':>8s} {'loss':>8s}"
    f" {'centers':>8s} {'recall':>7s} {'temp':>4s}"
)
for r in sorted(runs, key=lambda r: (r["data"], r["n"], r["k"])):
    if "error" in r:
        print(
            f"{r['data']:13s} {r['n']:9d} {r['k']:7d} {r['method']:15s} {r['error'][:60]}"
        )
        continue
    ref = reference.get((r["data"], r["n"], r["k"]))
    loss = f"{100 * (r['inertia_labels'] / ref - 1):+.2f}%" if ref else ""
    centers = f"{100 * (r['inertia_exact'] / ref - 1):+.2f}%" if ref else ""
    print(
        f"{r['data']:13s} {r['n']:9d} {r['k']:7d} {r['method']:15s} {r['t_fit']:8.3f}"
        f" {loss:>8s} {centers:>8s} {r['recall']:7.4f} {r['gpu_temperature']:>4s}"
    )
