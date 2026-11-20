#!/bin/bash
PY=~/work/omeans/venv/bin/python
export PYTHONPATH=~/work/omeans/fork
# 100 points per cluster, n = 1M, 3M, 10M
$PY omeans_run.py deep10M res/om_scale.jsonl 10000 1000000
$PY omeans_run.py deep10M res/om_scale.jsonl 30000 3000000
$PY omeans_run.py deep10M res/om_scale.jsonl 100000 10000000
# 1000 points per cluster
$PY omeans_run.py deep10M res/om_scale.jsonl 1000 1000000
$PY omeans_run.py deep10M res/om_scale.jsonl 3000 3000000
$PY omeans_run.py deep10M res/om_scale.jsonl 10000 10000000
