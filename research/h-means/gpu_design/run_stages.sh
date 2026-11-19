#!/bin/bash
# Triton option with one pipelining stage in the coarse kernel (the default
# of 3 needs too much shared memory from 512 cells), against 3 stages and
# blocks of 16 rows where they fit.
cd "$(dirname "$0")"
PY=$HOME/work/omeans/venv/bin/python
DATA=${DATA:-../data/deep2M.npy}
for k in 10000 100000; do
  COARSE_STAGES=1 $PY -u bench_assign.py $DATA $k --methods triton --out design_stages1.jsonl > log_${k}_stages1.txt 2>&1
done
COARSE_STAGES=3 $PY -u bench_assign.py $DATA 10000 --methods triton --out design_stages3.jsonl > log_10000_stages3.txt 2>&1
COARSE_STAGES=1 COARSE_BLOCK=16 $PY -u bench_assign.py $DATA 100000 --methods triton --out design_block16.jsonl > log_100000_block16.txt 2>&1
