#!/bin/bash
PY=~/work/emeans-array-api/.pixi/envs/default/bin/python
for d in sift1m wolt-clip512 glove200; do $PY exp.py --data $d --device cuda --configs v3_adapt.json --out res/adapt.jsonl; done
$PY exp.py --data deep10M --device cuda --configs v3_adapt.json --out res/adapt.jsonl
