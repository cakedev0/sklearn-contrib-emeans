#!/bin/bash
export OMP_NUM_THREADS=64
../venv/bin/python exp.py --data deep10M --device cpu --threads 64 --configs f_10m.json --out res/f_cpu.jsonl
until [ -f ../data/dl100.done ]; do sleep 30; done
../venv/bin/python exp.py --data deep100M --device cpu --threads 64 --configs f_100m.json --out res/f_cpu100.jsonl
