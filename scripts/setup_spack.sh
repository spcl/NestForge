#!/usr/bin/env bash
# nest-forge spack setup: register compilers + build the COMPILER x LIBRARY matrix (userspace, no sudo).
# Run scripts/setup_apt.sh first for the base toolchain spack registers + bootstraps from.
#
# Phases:
#   1. clone/reuse spack at $SPACK_ROOT and source it;
#   2. `spack compiler find` -- register the PATH compilers (and any vendor compilers whose env is
#      loaded: `source /opt/intel/oneapi/setvars.sh`, or the nvhpc modulefile, before running this);
#   3. optionally BUILD extra compilers (--spack-compilers; hours) and register them;
#   4. build each library in SPACK_MATRIX_LIBS with each compiler in SPACK_MATRIX_COMPILERS -- the
#      compiler x library matrix the arena sweeps.
#
# Idempotent (spack skips already-installed specs). Nothing here needs root.
#
# Usage: scripts/setup_spack.sh [--spack-compilers] [-h]
#
# Tunables (env vars):
#   SPACK_ROOT=~/spack                        where spack is cloned
#   SPACK_MATRIX_COMPILERS="gcc clang"        compiler specs the matrix is built against
#   SPACK_MATRIX_LIBS="openblas sleef fftw"   libraries built for each compiler
#   SPACK_BUILD_COMPILERS="gcc@13 llvm@17"    compilers to build when --spack-compilers is given
set -euo pipefail

SPACK_ROOT="${SPACK_ROOT:-$HOME/spack}"
SPACK_MATRIX_COMPILERS="${SPACK_MATRIX_COMPILERS:-gcc clang}"
SPACK_MATRIX_LIBS="${SPACK_MATRIX_LIBS:-openblas sleef fftw}"
SPACK_BUILD_COMPILERS="${SPACK_BUILD_COMPILERS:-gcc@13 llvm@17}"
DO_SPACK_COMPILERS=0

log()  { printf '\033[1;32m[spack]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[spack:warn]\033[0m %s\n' "$*" >&2; }
usage() { sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --spack-compilers) DO_SPACK_COMPILERS=1 ;;
    -h|--help) usage 0 ;;
    *) warn "unknown arg: $1"; usage 1 ;;
  esac
  shift
done

if [ ! -d "$SPACK_ROOT" ]; then
  log "cloning spack into $SPACK_ROOT"
  git clone --depth=1 https://github.com/spack/spack.git "$SPACK_ROOT"
else
  log "reusing spack at $SPACK_ROOT"
fi
# shellcheck disable=SC1091
. "$SPACK_ROOT/share/spack/setup-env.sh"

# Put vendor compilers on PATH so `spack compiler find` registers them too: oneAPI needs its setvars,
# nvhpc just needs its compilers/bin on PATH.
source_vendor_env() {
  if [ -f /opt/intel/oneapi/setvars.sh ]; then
    log "sourcing Intel oneAPI setvars.sh (so icx/icpx/ifx get registered)"
    set +u; # shellcheck disable=SC1091
    . /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || warn "oneAPI setvars.sh returned nonzero"; set -u
  fi
  local nv
  nv=$(ls -d /opt/nvidia/hpc_sdk/Linux_*/*/compilers/bin 2>/dev/null | sort -V | tail -1 || true)
  if [ -n "${nv:-}" ] && [ -d "$nv" ]; then
    log "adding nvhpc to PATH: $nv"
    export PATH="$nv:$PATH"
  fi
}
source_vendor_env

log "registering compilers on PATH (gcc/clang/gfortran + any vendor ones just sourced)"
spack compiler find || warn "spack compiler find found nothing new"

if [ "$DO_SPACK_COMPILERS" -eq 1 ]; then
  for c in $SPACK_BUILD_COMPILERS; do
    log "building compiler $c (slow)"
    spack install "$c" || warn "spack install $c failed; continuing"
    spack load "$c" 2>/dev/null && spack compiler find || true
  done
fi

log "building the COMPILER x LIBRARY matrix"
log "  compilers: $SPACK_MATRIX_COMPILERS"
log "  libraries: $SPACK_MATRIX_LIBS"
for cc in $SPACK_MATRIX_COMPILERS; do
  if ! spack compiler info "$cc" >/dev/null 2>&1; then
    warn "no compiler '$cc' registered in spack; skipping its column"
    continue
  fi
  for lib in $SPACK_MATRIX_LIBS; do
    log "spack install $lib %$cc"
    spack install "$lib %$cc" || warn "failed: $lib %$cc (continuing the matrix)"
  done
done
log "done. 'spack find' lists the matrix; 'spack load <lib> %<cc>' to use one."
