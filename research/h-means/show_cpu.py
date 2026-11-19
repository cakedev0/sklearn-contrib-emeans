"""Print a table of the runs of `bench_cpu.py` and of `exp.py`.

    python show_cpu.py res.jsonl [proto.jsonl ...]

The loss is the inertia of the labels against the one of `EMeans exact` on the
same data, number of clusters and number of samples. With several runs of a
configuration, the time is the median.
"""

import json
import statistics
import sys
from collections import defaultdict


def main():
    runs = defaultdict(list)
    for path in sys.argv[1:]:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if "error" in r:
                    continue
                method = r["method"]
                if method == "ivf":
                    method = "prototype"
                elif r.get("tag"):
                    method += f" ({r['tag']})"
                runs[(r["data"], r["n"], r["k"], method)].append(r)

    print("| data | n | k | method | fit (s) | centers | labels | recall |")
    print("|---|---|---|---|---|---|---|---|")
    for (data, n, k, method), rs in runs.items():
        reference = runs.get((data, n, k, "EMeans exact"))
        best = rs[0]
        t_fit = statistics.median(r["t_fit"] for r in rs)
        if reference is None:
            centers, labels = "", ""
        else:
            ref = reference[0]["inertia_labels"]
            centers = f"{100 * (best['inertia_exact'] / ref - 1):+.2f}%"
            labels = f"{100 * (best['inertia_labels'] / ref - 1):+.2f}%"
        print(
            f"| {data} | {n:,} | {k:,} | {method} | {t_fit:.3g} | "
            f"{centers} | {labels} | {best['recall']:.4f} |"
        )


if __name__ == "__main__":
    main()
