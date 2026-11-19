#!/bin/bash
# EMeans exact, the prototype (adaptive) and EMeans two-level on the five main
# cases. Run under benchlock, on one socket:
#   benchlock --tag "h-means cpu main" s1 -- ./xeon_cpu_main.sh
# HMEANS is a directory with a venv where this branch is installed, bench_cpu.py
# and res/. The prototype as benchmarked is in ~/work/hkm/v4.
export OMP_NUM_THREADS=64
HMEANS=${HMEANS:-$HOME/work/hmeans-cpu}
cd $HMEANS
export PATH=$PWD/venv/bin:$PATH
OUT=res/main.jsonl
for data in sift1m glove200 wolt-clip512 deep10M; do
  ks="10000"
  [ $data = deep10M ] && ks="10000 100000"
  for k in $ks; do
    python bench_cpu.py --data $data --k $k --assignment exact --out $OUT
    echo "[{\"k\": $k, \"seed\": 0, \"method\": \"ivf\", \"p\": 16, \"ratio\": 1.15, \"p_final\": 16}]" > grid_main.json
    (cd ~/work/hkm/v4 && ../venv/bin/python exp.py --data $data --device cpu \
      --threads 64 --configs $HMEANS/grid_main.json --out $HMEANS/res/main_proto.jsonl)
    python bench_cpu.py --data $data --k $k --assignment two-level --repeat 3 --out $OUT
  done
done
