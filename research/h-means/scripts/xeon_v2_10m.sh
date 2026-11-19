#!/bin/bash
export OMP_NUM_THREADS=64
../venv/bin/python exp.py --data deep10M --device cpu --threads 64 --configs v2_10m.json --out res/v2_cpu_10m.jsonl
../venv/bin/python exp.py --data deep10M --device cpu --threads 64 --configs v2_10m_big.json --out res/v2_cpu_10m.jsonl
