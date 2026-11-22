#!/bin/bash
# Greedy k-means++ inside the groups (4 trials per step), where the init matters.
# Run as: benchlock --tag "emeans init study" s1 -- bash xeon_v7_init2.sh
export OMP_NUM_THREADS=64
run() {  # data k seeds
    venv/bin/python init_exp2.py --data "$1" --k "$2" --seeds "$3" \
        --inits random,kpp-t4,kpp-inertia-t4 --out res/init.jsonl
}
for a in "clip 10000" "clip 1024"; do
    run $a 0,1,2
done
for a in "rcv1-lsa 1024" "rcv1-lsa 4096" "covtype 1024" "kddcup99 1024" \
         "kblobs-64 1024" "zipfblobs-64 1024" "kblobs-64 10000"; do
    run $a 0,1,2,3,4
done
