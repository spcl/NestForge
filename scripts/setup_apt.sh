#!/usr/bin/env bash
# nest-forge apt setup for an Ubuntu machine (system toolchain only; spack is scripts/setup_spack.sh).
#
# Installs what nest-forge's owned build + arena need from apt: compilers (gcc/clang/gfortran, +flang),
# the OpenMP runtimes (libomp/libgomp), the fast linkers (lld/gold/mold) + LTO archivers
# (gcc-ar/llvm-ar), the vector-math libraries (SLEEF; glibc libmvec is already in libc6), BLAS/LAPACK,
# and the python/build tooling. Vendor toolchains are behind flags (own apt repos + gpg keys):
#   --oneapi  Intel oneAPI  -- icx/icpx/ifx + libiomp5 + SVML + MKL
#   --nvhpc   NVIDIA HPC SDK -- nvc/nvc++/nvfortran + libnvomp
#
# Assumes sudo privileges (uses sudo directly, or runs as-is when already root). Idempotent: present
# packages / repos / gpg keys are detected and skipped. Nothing here is destructive.
#
# Usage: scripts/setup_apt.sh [--oneapi] [--nvhpc] [-h]
#
# Tunables: NVHPC_PKG=nvhpc   (pin e.g. nvhpc-24-3)
set -euo pipefail

NVHPC_PKG="${NVHPC_PKG:-nvhpc}"
DO_ONEAPI=0 DO_NVHPC=0

log()  { printf '\033[1;32m[apt]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[apt:warn]\033[0m %s\n' "$*" >&2; }
usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --oneapi) DO_ONEAPI=1 ;;
    --nvhpc) DO_NVHPC=1 ;;
    -h|--help) usage 0 ;;
    *) warn "unknown arg: $1"; usage 1 ;;
  esac
  shift
done

# Assume sudo: use it unless we are already root.
SUDO=sudo
[ "$(id -u)" -eq 0 ] && SUDO=""

APT_UPDATED=0
apt_update_once() { [ "$APT_UPDATED" -eq 1 ] || { $SUDO apt-get update -y; APT_UPDATED=1; }; }

# Install what exists on this release; warn + skip the rest (flang / mold / a libsleef name vary).
apt_install() {
  apt_update_once
  local ok=() miss=() p
  for p in "$@"; do
    if apt-cache show "$p" >/dev/null 2>&1; then ok+=("$p"); else miss+=("$p"); fi
  done
  [ "${#miss[@]}" -eq 0 ] || warn "not in apt on this release, skipping: ${miss[*]}"
  [ "${#ok[@]}" -eq 0 ] || DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y --no-install-recommends "${ok[@]}"
}

phase_oneapi() {
  log "Intel oneAPI: apt repo + gpg key"
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
  if [ -f /opt/intel/oneapi/setvars.sh ]; then
    log "sourcing oneAPI setvars.sh (puts icx/icpx/ifx + libiomp5/SVML on PATH for the rest of this run)"
    set +u; # shellcheck disable=SC1091
    . /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || warn "setvars.sh returned nonzero"; set -u
  fi
  warn "oneAPI env does NOT persist past this script -- add 'source /opt/intel/oneapi/setvars.sh' to your shell rc."
}

phase_nvhpc() {
  log "NVIDIA HPC SDK: apt repo + gpg key"
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

have_ubuntu() { command -v lsb_release >/dev/null 2>&1 && [ "$(lsb_release -is 2>/dev/null)" = "Ubuntu" ]; }
have_ubuntu || warn "not detected as Ubuntu; apt package names may differ"

log "core build + python tooling"
apt_install build-essential cmake ninja-build git pkg-config ca-certificates curl wget gnupg \
            python3 python3-pip python3-venv python3-dev

log "compilers: gcc/g++/gfortran, clang/llvm, flang"
apt_install gcc g++ gfortran clang clang-tools llvm llvm-dev
apt_install flang        # optional: LLVM Fortran, not on every release

log "OpenMP runtimes: libomp (LLVM/clang), libgomp (ships with gcc)"
apt_install libomp-dev libgomp1

log "fast linkers (lld/gold/mold) + LTO archivers (gcc-ar via binutils, llvm-ar via llvm)"
apt_install lld binutils binutils-gold mold

log "vector-math libraries: SLEEF (glibc libmvec is part of libc6, already present)"
apt_install libsleef-dev

log "BLAS/LAPACK (arena BLAS axis is a TODO; install the libs now)"
apt_install libopenblas-dev liblapack-dev libblis-dev libfftw3-dev

[ "$DO_ONEAPI" -eq 1 ] && phase_oneapi
[ "$DO_NVHPC" -eq 1 ] && phase_nvhpc

log "done. gcc/g++/clang/gfortran on PATH; libomp/libgomp/libsleef + linkers installed."
