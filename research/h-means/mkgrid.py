import json
import sys


def grid(k, seeds=(0,), flat=True, full=True):
    out = []
    for seed in seeds:
        b = {"k": k, "seed": seed}
        if flat:
            out += [dict(b, method="flat", T=T) for T in (5, 2, 1)]
        out.append(dict(b, method="h2"))
        out += [dict(b, method="ivf", p=p) for p in (1, 2, 4, 8)]
        out += [dict(b, method="graph", m=m) for m in (8, 16, 32, 64)]
        if full:
            out += [
                dict(b, method="ivf", p=4, coarse=c) for c in ("frozen", "reassign")
            ]
            out.append(dict(b, method="ivf", p=4, init="h2"))
            out.append(dict(b, method="graph", m=32, init="h2"))
            out.append(dict(b, method="h2", top_n=max(20000, 256 * round(k**0.5))))
    return out


ks = [int(a) for a in sys.argv[2].split(",")]
flat = "noflat" not in sys.argv
cfgs = [c for k in ks for c in grid(k, flat=flat)]
json.dump(cfgs, open(sys.argv[1], "w"))
print(len(cfgs))
