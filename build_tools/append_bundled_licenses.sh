#!/bin/bash
# Append the licenses of the libraries vendored in the wheels (OpenMP runtime)
# to LICENSE. Run once per wheel build job, by cibuildwheel's before-all.

set -euxo pipefail

PROJECT_DIR="$1"
case "$(uname -s)" in
    Linux*) OS=linux ;;
    Darwin*) OS=macos ;;
    *) OS=windows ;;
esac

printf '\n----\n\n' >> "$PROJECT_DIR/LICENSE"
cat "$PROJECT_DIR/build_tools/LICENSE_$OS.txt" >> "$PROJECT_DIR/LICENSE"
