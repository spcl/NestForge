#!/usr/bin/env bash
# nest-forge environment setup for an Ubuntu machine.
#
# Two phases:
#   1. apt  -- the system toolchain nest-forge's owned build + arena need: compilers (gcc/clang), the
#              OpenMP runtimes (libomp/libgomp), the fast linkers (lld/gold/mold) + LTO archivers
#              (gcc-ar/llvm-ar), the vector-math libraries (SLEEF, glibc libmvec), BLAS/LAPACK, and the
#              python/build tooling. Vendor toolchains (Intel oneAPI = icx/icpx/ifx + libiomp5 + SVML;
#              NVIDIA HPC SDK = nvc/nvc++ + libnvomp) are behind flags (big downloads, own apt repos).
#   2. spack -- register the apt compilers, optionally build extra compilers, then build the
#              COMPILER x LIBRARY matrix (each library built with each compiler) so the arena can sweep
#              real, differently-built libraries.
#
# Everything is idempotent: installed packages / cloned repos / registered compilers are detected and
# skipped. apt needs sudo; spack runs in userspace. Nothing here is destructive.
#
# Usage:
#   scripts/setup_env.sh [--apt] [--spack] [--all] [--oneapi] [--nvhpc] [--spack-compilers] [-h]
#
#   (no phase given)   == --apt          just the apt phase (safe, fast)
#   --all              == --apt --spack
#   --oneapi/--nvhpc   add the vendor apt repos + toolchains (implies the apt phase)
#   --spack-compilers  in the spack phase, also BUILD extra compilers (gcc/llvm) -- hours, off by default
#
# Tunables (env vars, with defaults):
#   SPACK_ROOT=~/spack                 where spack is cloned
#   SPACK_MATRIX_COMPILERS="gcc clang" spack compiler specs the library matrix is built against
#   SPACK_MATRIX_LIBS="openblas sleef fftw"   libraries built for each compiler
#   SPACK_BUILD_COMPILERS="gcc@13 llvm@17"    compilers to build when --spack-compilers is given
#   NVHPC_PKG=nvhpc                    NVIDIA HPC SDK apt package (e.g. nvhpc-24-3 to pin)
set -euo pipefail

# --- config -----------------------------------------------------------------------------------------
SPACK_ROOT="${SPACK_ROOT:-$HOME/spack}"
SPACK_MATRIX_COMPILERS="${SPACK_MATRIX_COMPILERS:-gcc clang}"
SPACK_MATRIX_LIBS="${SPACK_MATRIX_LIBS:-openblas sleef fftw}"
SPACK_BUILD_COMPILERS="${SPACK_BUILD_COMPILERS:-gcc@13 llvm@17}"
NVHPC_PKG="${NVHPC_PKG:-nvhpc}"

DO_APT=0 DO_SPACK=0 DO_ONEAPI=0 DO_NVHPC=0 DO_SPACK_COMPILERS=0

log()  { printf '\033[1;32m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[setup:warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[setup:err]\033[0m %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

usage() { sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

# --- args -------------------------------------------------------------------------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    --apt) DO_APT=1 ;;
    --spack) DO_SPACK=1 ;;
    --all) DO_APT=1; DO_SPACK=1 ;;
    --oneapi) DO_ONEAPI=1; DO_APT=1 ;;
    --nvhpc) DO_NVHPC=1; DO_APT=1 ;;
    --spack-compilers) DO_SPACK_COMPILERS=1; DO_SPACK=1 ;;
    -h|--help) usage 0 ;;
    *) warn "unknown arg: $1"; usage 1 ;;
  esac
  shift
done
# default phase: apt only.
if [ "$DO_APT" -eq 0 ] && [ "$DO_SPACK" -eq 0 ]; then DO_APT=1; fi

SUDO=""
if [ "$(id -u)" -ne 0 ]; then have sudo || die "need root or sudo for apt"; SUDO="sudo"; fi

# --- apt helpers ------------------------------------------------------------------------------------
APT_UPDATED=0
apt_update_once() { [ "$APT_UPDATED" -eq 1 ] || { $SUDO apt-get update -y; APT_UPDATED=1; }; }

# Install packages, but do not abort if some are unavailable on this release -- install what exists,
# warn about the rest (flang / mold / a specific libsleef name differ across Ubuntu versions).
apt_install() {
  apt_update_once
  local ok=() miss=() p
  for p in "$@"; do
    if apt-cache show "$p" >/dev/null 2>&1; then ok+=("$p"); else miss+=("$p"); fi
  done
  [ "${#miss[@]}" -eq 0 ] || warn "not in apt on this release, skipping: ${miss[*]}"
  [ "${#ok[@]}" -eq 0 ] || DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y --no-install-recommends "${ok[@]}"
}

phase_apt() {
  have lsb_release && [ "$(lsb_release -is 2>/dev/null)" = "Ubuntu" ] || warn "not detected as Ubuntu; apt names may differ"
  log "apt: core build + python tooling"
  apt_install build-essential cmake ninja-build git pkg-config ca-certificates curl wget \
              python3 python3-pip python3-venv python3-dev

  log "apt: compilers (gcc/g++/gfortran, clang/llvm) + fortran"
  apt_install gcc g++ gfortran clang clang-tools llvm llvm-dev
  apt_install flang        # optional: LLVM Fortran, not on every release

  log "apt: OpenMP runtimes (libomp = LLVM/clang; libgomp ships with gcc)"
  apt_install libomp-dev libgomp1

  log "apt: fast linkers (lld/gold/mold) + LTO archivers (gcc-ar via binutils, llvm-ar via llvm)"
  apt_install lld binutils binutils-gold mold

  log "apt: vector-math libraries (SLEEF; glibc libmvec is part of libc6, already present)"
  apt_install libsleef-dev

  log "apt: BLAS/LAPACK (arena BLAS axis is a TODO, but install the libs now)"
  apt_install libopenblas-dev liblapack-dev libblis-dev libfftw3-dev

  [ "$DO_ONEAPI" -eq 1 ] && phase_oneapi
  [ "$DO_NVHPC" -eq 1 ] && phase_nvhpc

  log "apt phase done. Verify: gcc/g++/clang/gfortran on PATH; libomp/libgomp/libsleef installed."
}

# --- Intel oneAPI (icx/icpx/ifx + libiomp5 + SVML + MKL) --------------------------------------------
phase_oneapi() {
  log "Intel oneAPI: adding apt repo"
  local key=/usr/share/keyrings/oneapi-archive-keyring.gpg
  if [ ! -f "$key" ]; then
    wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
      | gpg --dearmor | $SUDO tee "$key" >/dev/null
    echo "deb [signed-by=$key] https://apt.repos.intel.com/oneapi all main" \
      | $SUDO tee /etc/apt/sources.list.d/oneAPI.list >/dev/null
    APT_UPDATED=0
  fi
  apt_install intel-oneapi-compiler-dpcpp-cpp intel-oneapi-compiler-fortran \
              intel-oneapi-openmp intel-oneapi-mkl intel-oneapi-mkl-devel
  warn "oneAPI needs env: 'source /opt/intel/oneapi/setvars.sh' to put icx/icpx/ifx on PATH (+ libiomp5, SVML)."
}

# --- NVIDIA HPC SDK (nvc/nvc++/nvfortran + libnvomp) ------------------------------------------------
phase_nvhpc() {
  log "NVIDIA HPC SDK: adding apt repo"
  local key=/usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg
  if [ ! -f "$key" ]; then
    curl -fsSL https://developer.download.nvidia.com/hpc-sdk/ubuntu/DEB-GPG-KEY-NVIDIA-HPC-SDK \
      | $SUDO gpg --dearmor -o "$key"
    echo "deb [signed-by=$key] https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /" \
      | $SUDO tee /etc/apt/sources.list.d/nvhpc.list >/dev/null
    APT_UPDATED=0
  fi
  apt_install "$NVHPC_PKG"
  warn "nvhpc installs under /opt/nvidia/hpc_sdk; add its compilers/bin to PATH (or load its modulefile)."
}

# --- spack: register compilers + build the compiler x library matrix --------------------------------
phase_spack() {
  if [ ! -d "$SPACK_ROOT" ]; then
    log "spack: cloning into $SPACK_ROOT"
    git clone --depth=1 https://github.com/spack/spack.git "$SPACK_ROOT"
  else
    log "spack: already at $SPACK_ROOT (leaving as-is)"
  fi
  # shellcheck disable=SC1091
  . "$SPACK_ROOT/share/spack/setup-env.sh"

  log "spack: registering compilers on PATH (gcc/clang/gfortran, and vendor ones if their env is loaded)"
  spack compiler find || warn "spack compiler find found nothing new"

  if [ "$DO_SPACK_COMPILERS" -eq 1 ]; then
    local c
    for c in $SPACK_BUILD_COMPILERS; do
      log "spack: building compiler $c (slow)"
      spack install "$c" || warn "spack install $c failed; continuing"
      # make the freshly built compiler usable as a matrix compiler
      spack load "$c" 2>/dev/null && spack compiler find || true
    done
  fi

  log "spack: building the COMPILER x LIBRARY matrix"
  log "  compilers: $SPACK_MATRIX_COMPILERS"
  log "  libraries: $SPACK_MATRIX_LIBS"
  local cc lib spec
  for cc in $SPACK_MATRIX_COMPILERS; do
    if ! spack compiler info "$cc" >/dev/null 2>&1; then
      warn "spack has no compiler '$cc' registered; skipping its column"
      continue
    fi
    for lib in $SPACK_MATRIX_LIBS; do
      spec="$lib %$cc"
      log "spack install $spec"
      spack install "$spec" || warn "failed: $spec (continuing the matrix)"
    done
  done
  log "spack phase done. 'spack find' lists the matrix; 'spack load <lib> %<cc>' to use one."
}

# --- run --------------------------------------------------------------------------------------------
[ "$DO_APT" -eq 1 ] && phase_apt
[ "$DO_SPACK" -eq 1 ] && phase_spack
log "all requested phases complete."
