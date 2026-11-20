#!/bin/bash
export OMP_NUM_THREADS=64
for d in sift1m glove200 wolt-clip512; do
  venv/bin/python exp.py --data $d --device cpu --threads 64 --configs main_1m.json --out res/main_cpu.jsonl
done
venv/bin/python exp.py --data deep10M --device cpu --threads 64 --configs main_10m.json --out res/main_cpu.jsonl
