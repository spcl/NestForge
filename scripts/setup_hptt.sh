#!/usr/bin/env bash
# Build HPTT (tensor transpose) into PREFIX, the HPTT_ROOT DaCe's HPTT environment reads (PREFIX/include,
# PREFIX/lib/libhptt.so). Without it DaCe refuses to lower a top-level TensorTranspose kernel to its pure loop.
#
# The commit is pinned, as in the bench's container images; the portable 'scalar' target runs on any CPU.
# Idempotent: an existing PREFIX/lib/libhptt.so is kept (CI caches PREFIX).
#
# Usage: scripts/setup_hptt.sh PREFIX
set -euo pipefail

PREFIX="${1:?usage: scripts/setup_hptt.sh PREFIX}"
REPO="https://github.com/springer13/hptt.git"
REF="942538649b51ff14403a0c73a35d9825eab2d7de"

if [ ! -e "$PREFIX/lib/libhptt.so" ]; then
    src="$PREFIX.build"  # on disk beside PREFIX: /tmp may be memory-backed
    rm -rf "$src"
    git init -q "$src"
    git -C "$src" fetch -q --depth 1 "$REPO" "$REF"
    git -C "$src" checkout -q FETCH_HEAD
    make -C "$src" scalar CXX=g++ CXX_FLAGS="-O3 -std=c++11 -DNDEBUG -fPIC" -j"$(nproc)" >&2
    mkdir -p "$PREFIX/include" "$PREFIX/lib"
    cp "$src"/include/*.h "$PREFIX/include/"
    cp "$src/lib/libhptt.so" "$PREFIX/lib/"
    rm -rf "$src"
fi
echo "HPTT_ROOT=$PREFIX"
