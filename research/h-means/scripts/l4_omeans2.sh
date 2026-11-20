#!/bin/bash
PY=~/work/omeans/venv/bin/python
export PYTHONPATH=~/work/omeans/fork
$PY omeans_run.py deep10M res/om_gpu.jsonl 1000,10000,100000
for d in sift1m glove200 wolt-clip512; do $PY omeans_run.py $d res/om_gpu.jsonl 100000; done
