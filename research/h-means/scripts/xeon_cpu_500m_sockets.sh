#!/bin/bash
# 500M samples of DEEP with 50k clusters: 64 threads on one socket against
# 128 threads on the two sockets. The data is 192 GB, more than what one
# socket surely has free, so this needs the whole machine:
#   benchlock --tag "h-means 500M one or two sockets" all -- ./xeon_cpu_500m_sockets.sh
# EMeans does not use more threads than the compiled limit of OpenBLAS, which
# is 64 in the wheels of SciPy: there, the second run also has 64 threads.
# ENV_BIN is the bin directory of the environment. With the OpenBLAS of
# conda-forge (limit of 128 threads):
#   ENV_BIN=$PWD/cf/.pixi/envs/default/bin OUT=res/grid_500m_sockets_cf.jsonl
HMEANS=${HMEANS:-$HOME/work/hmeans-cpu}
cd $HMEANS
export PATH=${ENV_BIN:-$PWD/venv/bin}:$PATH
OUT=${OUT:-res/grid_500m_sockets.jsonl}
# Free the page cache of the other datasets, to leave the memory of the second
# socket to the data.
python - <<'P'
import glob, os
for path in glob.glob(os.path.expanduser("~/work/hkm/data/*.[fn][bp][iy]*")):
    fd = os.open(path, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
P
bench="python bench_cpu.py --data deep500M --k 50000 --assignment two-level --in-memory --out $OUT"
# One socket: threads on the second socket, memory there as long as it has
# some free.
OMP_NUM_THREADS=64 numactl --cpunodebind=4-7 --preferred-many=4-7 \
  $bench --tag "one socket"
# Two sockets: memory spread evenly over the NUMA nodes.
OMP_NUM_THREADS=128 numactl --interleave=all $bench --tag "two sockets"
