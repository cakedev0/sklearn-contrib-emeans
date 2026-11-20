#!/bin/bash
export OMP_NUM_THREADS=64
PY=../venv/bin/python
INITS="random,stratified,kpp,kpp-t4,kpp-inertia,h2"
for d in sift1m wolt-clip512 glove200; do
  $PY init_exp.py --data $d --k 10000 --inits "$INITS,k-means||" --seeds 0,1,2 --out res/init.jsonl
done
$PY init_exp.py --data deep10M --k 10000 --inits "$INITS,k-means||" --seeds 0,1 --out res/init.jsonl
$PY init_exp.py --data deep10M --k 100000 --inits "$INITS" --fits two-level --seeds 0 --out res/init.jsonl
for d in sift1m wolt-clip512; do
  $PY init_exp.py --data $d --k 1000 --inits "$INITS,k-means||" --seeds 0,1,2 --out res/init.jsonl
done
