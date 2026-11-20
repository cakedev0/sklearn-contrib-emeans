#!/bin/bash
PY=~/work/emeans-array-api/.pixi/envs/default/bin/python
for d in sift1m glove200 wolt-clip512; do
  $PY exp.py --data $d --device cuda --configs main_1m.json --out res/main_gpu.jsonl
done
$PY exp.py --data deep10M --device cuda --configs main_10m.json --out res/main_gpu.jsonl
