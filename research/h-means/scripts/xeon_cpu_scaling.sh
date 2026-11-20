#!/bin/bash
# Scaling at 1,000 samples per cluster on DEEP: EMeans exact, the prototype
# (adaptive, with its ratio of 1.15 and with 1.2) and EMeans two-level. Run
# under benchlock, on one socket:
#   benchlock --tag "h-means cpu scaling" s1 -- ./xeon_cpu_scaling.sh
export OMP_NUM_THREADS=64
HMEANS=${HMEANS:-$HOME/work/hmeans-cpu}
cd $HMEANS
export PATH=$PWD/venv/bin:$PATH
OUT=res/scaling.jsonl
for n in 1 3 10 30; do
  k=${n}000
  python bench_cpu.py --data deep100M --n ${n}000000 --k $k --assignment two-level \
    --min-clusters 1 --repeat 3 --out $OUT
  proto="\"k\": $k, \"seed\": 0, \"method\": \"ivf\", \"p\": 16, \"p_final\": 16"
  echo "[{$proto, \"ratio\": 1.15}, {$proto, \"ratio\": 1.2}]" > grid_scaling.json
  (cd ~/work/hkm/v4 && ../venv/bin/python exp.py --data deep100M --n ${n}000000 \
    --device cpu --threads 64 --configs $HMEANS/grid_scaling.json \
    --out $HMEANS/res/scaling_proto.jsonl)
  python bench_cpu.py --data deep100M --n ${n}000000 --k $k --assignment exact --out $OUT
done
