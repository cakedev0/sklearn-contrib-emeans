#!/bin/bash
# Check that a libomp.dylib, or the one vendored in each of the given wheels,
# exports no weak symbol (see build_macos_libomp.sh).
#
# usage: check_macos_libomp.sh <libomp.dylib or wheel>...

set -euo pipefail

check() {
    # grep reads all the output of nm: stopping early (grep -q) breaks the pipe
    # and fails the script.
    local weak
    weak="$(nm -gUm "$1" | grep "weak external" || true)"
    if [[ -n "$weak" ]]; then
        echo "$2 exports weak symbols:"
        echo "$weak"
        exit 1
    fi
    # The symbol that libomp 11 lacks, as a check that nm lists the exports.
    nm -gU "$1" | grep "___kmpc_dispatch_deinit" > /dev/null
    echo "$2: no weak symbol exported"
}

for path in "$@"; do
    if [[ "$path" == *.whl ]]; then
        tmp="$(mktemp -d)"
        unzip -q "$path" "sklearn_emeans/.dylibs/libomp.dylib" -d "$tmp"
        check "$tmp/sklearn_emeans/.dylibs/libomp.dylib" "$(basename "$path")"
        rm -rf "$tmp"
    else
        check "$path" "$path"
    fi
done
