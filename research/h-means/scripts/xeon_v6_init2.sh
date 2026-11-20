#!/bin/bash
export OMP_NUM_THREADS=64
PY=../venv/bin/python
NEW="stratified-inertia,stratified-half,sub-10,sub-25"
run() {  # data k seeds
  $PY init_exp.py --data $1 --k $2 --inits "$NEW" --fits two-level --seeds $3 --out res/init2.jsonl
  $PY init_exp.py --data $1 --k $2 --inits random --epochs 6,7,8,10 --fits two-level --seeds $3 --out res/init2.jsonl
}
run wolt-clip512 10000 0,1,2
run sift1m 10000 0,1,2
run glove200 10000 0,1,2
run wolt-clip512 1000 0,1,2
run deep10M 10000 0,1
run deep10M 100000 0
