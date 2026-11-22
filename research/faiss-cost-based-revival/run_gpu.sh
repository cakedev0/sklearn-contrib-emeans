#!/bin/bash
cd ~/work/faiss-pr5915-gpu
P=~/work/emeans-array-api/.pixi/envs/default/bin/python
$P bench_gpu.py --datasets road_network_km_x10 year_prediction_msd_standardized --k 64 256 1024 --epochs 2 5 10
$P bench_gpu.py --datasets shuttle_z power_z --k 256 1024 --epochs 2 5
$P bench_gpu.py --datasets geonames census_z kdd_z --k 256 1024 4096 --epochs 2 5
echo ALL_DONE
