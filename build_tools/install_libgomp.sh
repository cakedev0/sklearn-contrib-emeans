#!/bin/bash
# Install the libgomp (OpenMP runtime of GCC) of conda-forge in $1. Run in the
# manylinux image by cibuildwheel's before-all, so that auditwheel vendors
# this libgomp in the Linux wheels.
#
# The only shared libgomp of the manylinux_2_28 image is the one of AlmaLinux 8
# (GCC 8.5): its gcc-toolset-14 links to it. Since GCC 14, libgomp no longer
# makes the idle threads spin on x86 CPUs with performance and efficiency
# cores, where the spinning makes the fit times unstable (random fits are up
# to 5x slower), see https://github.com/cakedev0/sklearn-contrib-emeans/issues/8.
# A libgomp newer than the compiler is fine, not an older one.

set -euxo pipefail

PREFIX="$1"
LIBGOMP_VERSION=16.2.0

case "$(uname -m)" in
    x86_64) PLATFORM=linux-64 ;;
    aarch64) PLATFORM=linux-aarch64 ;;
    *) echo "Unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac

WORK_DIR="$(mktemp -d)"
curl -fsSL --retry 3 "https://micro.mamba.pm/api/micromamba/$PLATFORM/latest" \
    | tar -xj -C "$WORK_DIR" bin/micromamba
"$WORK_DIR/bin/micromamba" create --yes --prefix "$PREFIX" \
    --channel conda-forge "libgomp=$LIBGOMP_VERSION"
rm -rf "$WORK_DIR"
# The package only has libgomp.so.1.0.0: on conda-forge, libgomp.so.1 comes
# from another package (_openmp_mutex).
ln -s libgomp.so.1.0.0 "$PREFIX/lib/libgomp.so.1"
ls -l "$PREFIX/lib"
