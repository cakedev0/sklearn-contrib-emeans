#!/bin/bash
# Fit time against the BLAS, the number of threads, the placement on the
# sockets and the OpenMP settings. One process per line of the file given as
# argument:
#
#     <env> <threads> <placement> [VAR=value ...]
#
# - env: pypi (OpenBLAS of the wheels, at most 64 threads, libgomp), cf
#   (OpenBLAS of conda-forge, at most 128, libgomp), cfl (the same with the
#   OpenMP runtime of LLVM), mkl (MKL of conda-forge, no limit, OpenMP runtime
#   of LLVM);
# - placement: one (threads and data on socket 1), two (threads on both
#   sockets, all the memory of the process interleaved over the NUMA nodes),
#   two-data (threads on both sockets, only the data interleaved), two-local
#   (threads on both sockets, default policy of Linux for all the memory).
#
# Quick probes, on 100M samples:
#
#     benchlock --tag "h-means threads probe" all -- \
#         ./xeon_cpu_threads_probe.sh probes_1.txt
#
# On the 500M samples (the page cache of the file must not stay):
#
#     DATA=deep500M K=50000 FLAGS=--in-memory REPEAT=2 OUT=res/threads_500m.jsonl \
#         benchlock --tag "h-means threads 500M" all -- \
#         ./xeon_cpu_threads_probe.sh threads_500m.txt
set -u
cd ~/work/hmeans-cpu
DATA=${DATA:-deep100M}
K=${K:-10000}
FLAGS=${FLAGS:---in-memory --keep-page-cache}
REPEAT=${REPEAT:-1}
OUT=${OUT:-res/threads_probe.jsonl}

if [[ "$FLAGS" != *--keep-page-cache* ]]; then
    # Free the page cache of the other datasets: the data needs the memory.
    venv/bin/python - <<'P'
import glob, os
for path in glob.glob(os.path.expanduser("~/work/hkm/data/*.[fn][bp][iy]*")):
    fd = os.open(path, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
P
fi

while read -r env threads placement settings; do
    case "$env" in
        "" | "#"*) continue ;;
        pypi) python=venv/bin/python ;;
        *) python=$env/.pixi/envs/default/bin/python ;;
    esac
    more=""
    case "$placement" in
        one) bind="numactl --cpunodebind=4-7 --preferred-many=4-7" ;;
        two) bind="numactl --interleave=all" ;;
        two-data) bind="" && more="--interleave-data" ;;
        two-local) bind="" ;;
        *) echo "unknown placement $placement" && exit 1 ;;
    esac
    echo "== $env $threads $placement $settings"
    # shellcheck disable=SC2086
    env $settings OMP_NUM_THREADS="$threads" $bind $python bench_cpu.py \
        --data "$DATA" --k "$K" --assignment two-level $FLAGS $more \
        --repeat "$REPEAT" --tag "$env $placement $settings" --out "$OUT" \
        < /dev/null
done < "$1"
