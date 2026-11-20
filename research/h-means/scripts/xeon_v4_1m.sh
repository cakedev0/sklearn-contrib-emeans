#!/bin/bash
export OMP_NUM_THREADS=64
for d in sift1m glove200 wolt-clip512; do
  ../venv/bin/python exp.py --data $d --device cpu --threads 64 --configs f_1m.json --out res/f_cpu.jsonl
done
