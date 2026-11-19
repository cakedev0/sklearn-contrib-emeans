#!/bin/bash
export OMP_NUM_THREADS=64
for n in 1 3 10 30; do
  ../venv/bin/python exp.py --data deep100M --n ${n}000000 --device cpu --threads 64 --configs scale_${n}m.json --out res/scale.jsonl
done
