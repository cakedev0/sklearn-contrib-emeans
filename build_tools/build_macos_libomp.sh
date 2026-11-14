#!/bin/bash
# Build the LLVM OpenMP runtime to vendor in the macOS wheels.
#
# usage: build_macos_libomp.sh <llvm version> <install prefix>
#
# The libomp builds of conda-forge, Homebrew and the PyTorch wheels export weak
# symbols (C++ template instantiations of the internals of the runtime). dyld
# binds the weak symbols of all the images of a process to a single
# definition, so two such libomp loaded together call into each other: with
# the PyTorch wheel, it ends in a segfault or a deadlock, see
# https://github.com/cakedev0/sklearn-contrib-emeans/issues/15. This build
# keeps the C++ symbols private. The OpenMP entry points are C symbols.

set -euxo pipefail

VERSION="$1"
PREFIX="$2"
WORK_DIR="$(mktemp -d)"

git clone --quiet --depth 1 --branch "llvmorg-$VERSION" --filter=blob:none --sparse \
    https://github.com/llvm/llvm-project.git "$WORK_DIR/llvm-project"
git -C "$WORK_DIR/llvm-project" sparse-checkout set \
    openmp runtimes cmake llvm/cmake llvm/utils/llvm-lit third-party

cmake -S "$WORK_DIR/llvm-project/runtimes" -B "$WORK_DIR/build" \
    -DLLVM_ENABLE_RUNTIMES=openmp \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=/usr/bin/clang \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
    -DLIBOMP_OMPT_SUPPORT=OFF \
    -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-unexported_symbol,__Z*"
cmake --build "$WORK_DIR/build" --target omp --parallel 4

mkdir -p "$PREFIX/lib" "$PREFIX/include"
cp "$WORK_DIR/build/openmp/runtime/src/libomp.dylib" "$PREFIX/lib/"
cp "$WORK_DIR/build/openmp/runtime/src/omp.h" "$PREFIX/include/"
rm -rf "$WORK_DIR"

bash "$(dirname "$0")/check_macos_libomp.sh" "$PREFIX/lib/libomp.dylib"
