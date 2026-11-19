"""Table of results relative to the flat T=5 reference of the same data/k/seed/device."""

import collections
import json
import sys

rows = [json.loads(l) for f in sys.argv[1:] if not f.startswith("-") for l in open(f)]
rows = [r for r in rows if "error" not in r]


def name(c):
    m = c["method"]
    extra = {k: v for k, v in c.items() if k not in ("method", "k", "seed", "diag")}
    return m + (
        "(" + ",".join(f"{k}={v}" for k, v in extra.items()) + ")" if extra else ""
    )


groups = collections.defaultdict(list)
for r in rows:
    groups[
        (r["data"], r["n"], r["cfg"]["k"], r["device"], r["cfg"].get("seed", 0))
    ].append(r)
for key, rs in sorted(groups.items(), key=str):
    ref = [r for r in rs if r["cfg"]["method"] == "flat" and r["cfg"].get("T", 5) == 5]
    ref = ref[0] if ref else None
    print(
        f"\n== {key[0]} n={key[1]} k={key[2]} {key[3]} seed={key[4]}"
        + (
            f"  flat: inertia={ref['inertia_exact']:.5g} t={ref['t_fit']:.2f}s"
            if ref
            else ""
        )
    )
    print(
        f"{'method':38s} {'time':>8s} {'speedup':>7s} {'work':>7s} {'d_exact%':>8s} {'d_lab%':>7s} {'recall':>6s} {'cv':>5s} {'small':>6s} {'maxg':>5s}"
    )
    for r in rs:
        de = (
            100 * (r["inertia_exact"] / ref["inertia_exact"] - 1)
            if ref
            else float("nan")
        )
        dl = (
            100 * (r.get("inertia_labels", float("nan")) / ref["inertia_exact"] - 1)
            if ref
            else float("nan")
        )
        sp = ref["t_fit"] / r["t_fit"] if ref else float("nan")
        print(
            f"{name(r['cfg']):38s} {r['t_fit']:8.2f} {sp:7.2f} {r['work']:7.3f} {de:8.3f} {dl:7.3f} {r.get('recall', 0):6.3f} {r.get('size_cv', 0):5.2f} {r.get('frac_small', 0):6.3f} {r.get('max_group', 0):5d}"
        )
