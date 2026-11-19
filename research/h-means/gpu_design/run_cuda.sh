#!/bin/bash
# The CUDA C option alone, then everything at k=1M.
cd "$(dirname "$0")"
export PYTHONPATH=$HOME/work/omeans/fork
PY=$HOME/work/omeans/venv/bin/python
DATA=${DATA:-../data/deep2M.npy}
$PY -u bench_assign.py $DATA 10000 --methods cuda --out design.jsonl > log_10k_cuda.txt 2>&1
$PY -u bench_assign.py $DATA 100000 --methods cuda --out design.jsonl > log_100k_cuda.txt 2>&1
$PY -u bench_assign.py $DATA 1000000 --methods omeans,triton,cuda --out design.jsonl > log_1m.txt 2>&1
