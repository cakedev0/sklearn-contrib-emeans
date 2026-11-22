#!/bin/bash
# Init study with EMeans(assignment="two-level"), one Xeon socket, 64 threads.
# Run as: benchlock --tag "emeans init study" s1 -- bash xeon_v7_init.sh
export OMP_NUM_THREADS=64
INITS='random,stratified,stratified-inertia,kpp,kpp-inertia,sub-10,k-means||'
run() {  # data k seeds
    venv/bin/python init_exp2.py --data "$1" --k "$2" --seeds "$3" \
        --inits "$INITS" --random-epochs 6,8,10 --out res/init.jsonl
}
for a in "sift 10000" "glove 10000" "clip 10000" "clip 1024"; do
    run $a 0,1,2
done
for a in "rcv1-lsa 1024" "rcv1-lsa 4096" "covtype 1024" "kddcup99 1024" \
         "kblobs-64 1024" "zipfblobs-64 1024" "kblobs-64 10000"; do
    run $a 0,1,2,3,4
done
run deep10M 10000 0,1
run deep10M 100000 0
