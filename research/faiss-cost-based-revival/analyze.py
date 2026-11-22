"""Markdown tables of results/results.jsonl: the mean of the seeds of each case."""

import collections
import json

import numpy as np

VARIANTS = {
    "faiss-pr cost-based": "PR cost",
    "faiss-pr mass 0.1": "PR mass",
    "faiss-pr cost-based, k-means||": "PR cost, km\\|\\|",
    "emeans revival=0.1": "EM 0.1",
    "emeans greedy": "EM greedy",
    "emeans k-means||": "EM km\\|\\|",
    "emeans greedy, k-means||": "EM greedy, km\\|\\|",
}

runs = collections.defaultdict(list)
with open("results/results.jsonl") as f:
    for line in f:
        r = json.loads(line)
        runs[r["dataset"], r["k"], r["n_epochs"], r["variant"]].append(r)
cases = sorted({key[:3] for key in runs})


def table(title, metric, relative):
    print(f"\n**{title}**\n")
    print("| dataset | k | epochs | " + " | ".join(VARIANTS.values()) + " |")
    print("| --- | --- | --- |" + " --- |" * len(VARIANTS))
    for case in cases:
        if metric not in runs[(*case, "faiss-pr cost-based")][0]:
            continue
        means = [np.mean([r[metric] for r in runs[(*case, v)]]) for v in VARIANTS]
        if relative:
            cells = [f"{100 * (m / min(means) - 1):.1f}" for m in means]
        else:
            cells = [f"{m:.3g}" for m in means]
        print("| " + " | ".join([*map(str, case), *cells]) + " |")


table(
    "Training inertia, % above the best variant of the row", "inertia_per_sample", True
)
table("Held-out error, % above the best variant of the row", "test_error", True)
table("Training inertia per sample", "inertia_per_sample", False)
table("Held-out error per sample", "test_error", False)
table("Centers that are nearest to no training row", "n_empty", False)
table("Fit time (s)", "fit_s", False)
