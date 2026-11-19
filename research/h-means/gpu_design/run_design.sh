#!/bin/bash
# Design benchmark of the two-level assignment on one GPU:
#   benchlock --tag "h-means gpu design bench" all -- ./run_design.sh
cd "$(dirname "$0")"
export PYTHONPATH=$HOME/work/omeans/fork
PY=$HOME/work/omeans/venv/bin/python
DATA=${DATA:-../data/deep2M.npy}
$PY -u bench_assign.py $DATA 10000 --out design.jsonl > log_10k.txt 2>&1
$PY -u bench_assign.py $DATA 100000 --out design.jsonl > log_100k.txt 2>&1
