#!/bin/bash
# Grid of n_samples x samples per cluster (100k, 10k, 1k, 100) on DEEP, with
# EMeans(assignment="two-level"), which is the exact assignment below 1024
# clusters. EMeans exact is run as a reference where n_samples * n_clusters is
# at most 1e12 (less than about 10 minutes). The arguments are the numbers of
# samples, in millions. Run under benchlock, 64 threads on one socket:
#   benchlock --tag "h-means cpu grid" s1 -- ./xeon_cpu_grid.sh 1 5 20 100
#   benchlock --tag "h-means cpu grid 500M" all -- ./xeon_cpu_grid.sh 500
# 500M samples (192 GB, about 230 GB with the buffers of the fit) don't fit in
# the memory of one socket for sure: this run needs the `all` slot. It binds
# the threads to the second socket and takes the memory of the first one only
# when the second one is full. Its data is read in anonymous memory, the other
# runs map the file.
export OMP_NUM_THREADS=64
HMEANS=${HMEANS:-$HOME/work/hmeans-cpu}
cd $HMEANS
export PATH=$PWD/venv/bin:$PATH
OUT=${OUT:-res/grid.jsonl}
for m in "$@"; do
  n=${m}000000
  data=deep100M; run=""; flags=""; repeat=3
  [ $m -ge 100 ] && repeat=1
  if [ $m -gt 100 ]; then
    data=deep500M
    run="numactl --cpunodebind=4-7 --preferred-many=4-7"
    flags="--in-memory"
    # Free the page cache of the other datasets, to leave the memory of the
    # second socket to the data.
    python - <<'P'
import glob, os
for path in glob.glob(os.path.expanduser("~/work/hkm/data/*.[fn][bp][iy]*")):
    fd = os.open(path, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
P
  fi
  ks=""; ref=""
  for size in 100000 10000 1000 100; do
    k=$((n / size))
    ks="$ks $k"
    [ $k -ge 1024 ] && [ $((m * k)) -le 1000000 ] && ref="$ref $k"
  done
  $run python bench_cpu.py --data $data --n $n --k $ks --assignment two-level \
    --repeat $repeat $flags --tag grid --out $OUT
  [ -n "$ref" ] && $run python bench_cpu.py --data $data --n $n --k $ref \
    --assignment exact $flags --tag grid --out $OUT
done
