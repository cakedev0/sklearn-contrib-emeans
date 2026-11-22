#!/bin/bash
cd ~/work/faiss-pr5915
P=".pixi/envs/default/bin/python"
export PATH=$PWD/.pixi/envs/default/bin:$PATH
$P bench.py --datasets road_network_km_x10 year_prediction_msd_standardized susy --k 64 256 1024 --epochs 2 5 10
$P bench.py --datasets shuttle_z power_z --k 256 1024 --epochs 2 5
$P bench.py --datasets geonames census_z --k 256 1024 4096 --epochs 2 5
until grep -q PREP_DONE prep.log; do sleep 20; done
$P bench.py --datasets kdd_z --k 256 1024 4096 --epochs 2 5
echo ALL_DONE
