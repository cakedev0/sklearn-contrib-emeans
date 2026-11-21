#!/bin/bash
# EMeans(assignment="two-level") on the L4 against the exact EMeans, the dense
# e-means of faiss and o-means. Run from research/h-means under the lock:
#   benchlock --tag "h-means gpu main" all -- scripts/l4_gpu_main.sh
# PY is the Python with PyTorch, Triton, sklearn_emeans (this branch) and
# faiss.contrib of the owlgebra-ai/faiss fork. SCIPY_ARRAY_API=1 must be set.
OUT=${OUT:-results/l4/gpu_main_v2.jsonl}
FAST=emeans-2l,faiss-e,omeans-p4,omeans-p8,omeans
for data in sift1m glove200 wolt-clip512; do
  $PY bench_gpu.py $data $OUT 10000 --methods emeans,$FAST
done
# Scaling at 100 samples per cluster, and DEEP 10M at k=10k.
$PY bench_gpu.py deep10M $OUT 10000 1000000 --methods emeans,$FAST,emeans-2l-tf32,emeans-2l-ieee
$PY bench_gpu.py deep10M $OUT 30000 3000000 --methods emeans,$FAST
$PY bench_gpu.py deep10M $OUT 10000,100000 --methods $FAST,emeans-2l-tf32
# 10 samples per cluster: o-means fails there on the L4 (shared memory).
$PY bench_gpu.py deep10M $OUT 1000000 --methods emeans-2l,omeans-p8,omeans --reps 1
# The exact fits are long: one run each.
$PY bench_gpu.py deep10M $OUT 10000,100000 --methods emeans --reps 1
