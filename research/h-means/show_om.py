import collections
import json
import sys

rows = [json.loads(l) for l in open(sys.argv[1])]
groups = collections.defaultdict(list)
for r in rows:
    if "error" in r:
        print("ERR", r["data"], r["cfg"], r["error"][:100])
        continue
    groups[(r["data"], r["cfg"]["k"], r["n"])].append(r)
for key, rs in groups.items():
    ref = [r for r in rs if r["cfg"]["method"] == "faiss-e" and r["cfg"]["T"] == 5][0]
    print(
        f"\n== {key[0]} n={key[2]} k={key[1]}  dense e-means T=5: {ref['t_fit']:.3f}s inertia {ref['inertia_exact']:.5g}"
    )
    print(
        f"{'method':34s} {'time':>7s} {'speedup':>7s} {'work':>6s} {'d_exact%':>8s} {'d_lab%':>7s} {'recall':>6s}"
    )
    for r in rs:
        c = r["cfg"]
        nm = (
            c["method"]
            + f"(T={c['T']}"
            + (
                f",nlist={r.get('nlist')},nprobe={r.get('nprobe')}"
                if c["method"] == "omeans"
                else ""
            )
            + ")"
        )
        print(
            f"{nm:34s} {r['t_fit']:7.3f} {ref['t_fit'] / r['t_fit']:7.2f} {r['work']:6.3f} {100 * (r['inertia_exact'] / ref['inertia_exact'] - 1):8.3f} {100 * (r.get('inertia_labels', float('nan')) / ref['inertia_exact'] - 1):7.3f} {r.get('recall', 0):6.3f}"
        )
