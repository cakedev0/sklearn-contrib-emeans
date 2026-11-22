"""Markdown tables of the runs: the mean of the seeds of each case.

`python analyze.py` reads results/results.jsonl (CPU), `python analyze.py gpu`
reads results/results_gpu.jsonl.
"""

import collections
import json
import sys

import numpy as np

GPU = sys.argv[1:] == ["gpu"]
if GPU:
    PATH = "results/results_gpu.jsonl"
    VARIANTS = {
        "faiss-pr Emeans cost-based (BF16)": "PR cost",
        "faiss-pr Emeans mass 0.1 (BF16)": "PR mass",
        "faiss-pr QuantizedEmeans mass 0.1 (FP32)": "PR mass, FP32",
        "faiss-pr Emeans cost-based (BF16), k-means||": "PR cost, km\\|\\|",
        "emeans revival=0.1": "EM 0.1",
        "emeans greedy": "EM greedy",
        "emeans k-means||": "EM km\\|\\|",
        "emeans greedy, k-means||": "EM greedy, km\\|\\|",
    }
    PR_COST, PR_MASS = (
        "faiss-pr Emeans cost-based (BF16)",
        "faiss-pr Emeans mass 0.1 (BF16)",
    )
else:
    PATH = "results/results.jsonl"
    VARIANTS = {
        "faiss-pr cost-based": "PR cost",
        "faiss-pr mass 0.1": "PR mass",
        "faiss-pr cost-based, k-means||": "PR cost, km\\|\\|",
        "emeans revival=0.1": "EM 0.1",
        "emeans greedy": "EM greedy",
        "emeans k-means||": "EM km\\|\\|",
        "emeans greedy, k-means||": "EM greedy, km\\|\\|",
    }
    PR_COST, PR_MASS = "faiss-pr cost-based", "faiss-pr mass 0.1"

runs = collections.defaultdict(list)
with open(PATH) as f:
    for line in f:
        r = json.loads(line)
        runs[r["dataset"], r["k"], r["n_epochs"], r["variant"]].append(r)
cases = sorted({key[:3] for key in runs})


def mean(case, variant, metric):
    return np.mean([r[metric] for r in runs[(*case, variant)]])


def table(title, metric, relative):
    print(f"\n**{title}**\n")
    print("| dataset | k | epochs | " + " | ".join(VARIANTS.values()) + " |")
    print("| --- | --- | --- |" + " --- |" * len(VARIANTS))
    for case in cases:
        if metric not in runs[(*case, PR_COST)][0]:
            continue
        means = [mean(case, v, metric) for v in VARIANTS]
        if relative:
            cells = [f"{100 * (m / min(means) - 1):.1f}" for m in means]
        else:
            cells = [f"{m:.3g}" for m in means]
        print("| " + " | ".join([*map(str, case), *cells]) + " |")


def ratios():
    """Fit time of each revival over the mass-based one of the same library."""
    print("\n**Fit time ratio: median over the datasets and budgets (min to max)**\n")
    print("| k | PR cost / PR mass | EM greedy / EM 0.1 |")
    print("| --- | --- | --- |")
    pairs = [(PR_COST, PR_MASS), ("emeans greedy", "emeans revival=0.1")]
    for k in sorted({case[1] for case in cases}):
        cells = []
        for num, den in pairs:
            values = [
                mean(case, num, "fit_s") / mean(case, den, "fit_s")
                for case in cases
                if case[1] == k
            ]
            cells.append(
                f"{np.median(values):.2f} ({min(values):.2f} to {max(values):.2f})"
            )
        print(f"| {k} | " + " | ".join(cells) + " |")


table(
    "Training inertia, % above the best variant of the row", "inertia_per_sample", True
)
table("Held-out error, % above the best variant of the row", "test_error", True)
table("Training inertia per sample", "inertia_per_sample", False)
table("Held-out error per sample", "test_error", False)
table("Centers that are nearest to no training row", "n_empty", False)
table("Fit time (s)", "fit_s", False)
ratios()
