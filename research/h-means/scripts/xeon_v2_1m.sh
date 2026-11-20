#!/bin/bash
export OMP_NUM_THREADS=64
for d in sift1m glove200 wolt-clip512; do
  ../venv/bin/python exp.py --data $d --device cpu --threads 64 --configs v2_1m.json --out res/v2_cpu_1m.jsonl
done
