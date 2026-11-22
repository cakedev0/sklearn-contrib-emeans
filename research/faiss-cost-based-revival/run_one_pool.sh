#!/bin/bash
# The two faiss variants again with the OpenMP build of OpenBLAS: one thread pool.
cd ~/work/faiss-pr5915
P=".pixi/envs/default/bin/python"
V=(--variants "faiss-pr cost-based" "faiss-pr mass 0.1" --out results_one_pool.jsonl)
$P bench.py --datasets road_network_km_x10 year_prediction_msd_standardized susy --k 64 256 1024 --epochs 2 5 10 "${V[@]}"
$P bench.py --datasets shuttle_z power_z --k 256 1024 --epochs 2 5 "${V[@]}"
$P bench.py --datasets geonames census_z kdd_z --k 256 1024 4096 --epochs 2 5 "${V[@]}"
echo ALL_DONE
