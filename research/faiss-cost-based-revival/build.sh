#!/bin/bash
set -euxo pipefail
cd ~/work/faiss-pr5915
if [ ! -d faiss ]; then
  git clone --filter=blob:none https://github.com/facebookresearch/faiss.git
fi
cd faiss
git fetch origin pull/5915/head
git checkout -q ba7781afb44d5c3a7f6993616e9e642e52256941
git log --oneline -1
cmake -B build -G Ninja -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=ON \
  -DFAISS_OPT_LEVEL=avx2 -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release -DFAISS_ENABLE_C_API=OFF -DBLA_VENDOR=OpenBLAS \
  -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DPython_EXECUTABLE=$CONDA_PREFIX/bin/python
cmake --build build -j 24 --target faiss faiss_avx2 swigfaiss swigfaiss_avx2
cd build/faiss/python
python -m pip install --no-deps --no-build-isolation .
cd ~
python -c "
import faiss, sklearn_emeans
print(faiss.__version__, faiss.__file__, sklearn_emeans.__version__)
print(hasattr(faiss, 'accumulate_removal_stats'), faiss.get_compile_options())
from faiss.contrib.e_means import NumpyEmeans
"
