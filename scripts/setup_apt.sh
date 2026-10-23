#!/usr/bin/env bash
# nest-forge apt setup for an Ubuntu machine (system toolchain only).
#
# Installs what nest-forge's owned build + arena need from apt: compilers (gcc/clang/gfortran, +flang),
# the OpenMP runtimes (libomp/libgomp), BLAS/LAPACK, and the python/build tooling. Intel oneAPI is
# behind a flag (its own apt repo + gpg key):
#   --oneapi  Intel oneAPI  -- icx/icpx/ifx + libiomp5 + SVML + MKL
#   --no-flang  skip flang (CI: no test runs it, and its 45 MB download stalls on slow mirrors)
#   --archives DIR  keep downloaded .deb files in DIR (CI caches DIR, so a slow mirror is never hit twice)
#
# Assumes sudo privileges (uses sudo directly, or runs as-is when already root). Idempotent: present
# packages / repos / gpg keys are detected and skipped. Nothing here is destructive.
#
# Usage: scripts/setup_apt.sh [--oneapi] [--no-flang] [--archives DIR] [-h]
set -euo pipefail

DO_ONEAPI=0
DO_FLANG=1
ARCHIVES=""

log()  { printf '\033[1;32m[apt]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[apt:warn]\033[0m %s\n' "$*" >&2; }
# --help prints the leading comment block
usage() { awk 'NR==1{next} /^#/{sub(/^# ?/,"");print;next}{exit}' "$0"; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --oneapi) DO_ONEAPI=1 ;;
    --no-flang) DO_FLANG=0 ;;
    --archives) ARCHIVES="$2"; shift ;;
    -h|--help) usage 0 ;;
    *) warn "unknown arg: $1"; usage 1 ;;
  esac
  shift
done

SUDO=sudo
[ "$(id -u)" -eq 0 ] && SUDO=""

APT_UPDATED=0
# A stalled mirror must fail and retry, not hang the job: CI once waited an hour on one Release file.
APT_NET=(-o Acquire::Retries=3 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30)
if [ -n "$ARCHIVES" ]; then mkdir -p "$ARCHIVES/partial"; APT_NET+=(-o "Dir::Cache::Archives=$ARCHIVES"); fi
apt_update_once() { [ "$APT_UPDATED" -eq 1 ] || { $SUDO apt-get "${APT_NET[@]}" update -y; APT_UPDATED=1; }; }

# Install what exists on this release; warn + skip the rest (flang's package name varies).
apt_install() {
  apt_update_once
  local ok=() miss=() p
  for p in "$@"; do
    if apt-cache show "$p" >/dev/null 2>&1; then ok+=("$p"); else miss+=("$p"); fi
  done
  [ "${#miss[@]}" -eq 0 ] || warn "not in apt on this release, skipping: ${miss[*]}"
  # pass DEBIAN_FRONTEND through sudo via env: sudo's env_reset strips an exported one, and debconf may then prompt
  [ "${#ok[@]}" -eq 0 ] || $SUDO env DEBIAN_FRONTEND=noninteractive apt-get "${APT_NET[@]}" install -y --no-install-recommends "${ok[@]}"
}

# Install a signing key only once it is fully fetched: a truncated key would pass the `[ ! -f "$key" ]` guard forever.
install_gpg_key() {  # <url> <dest-keyring>
  local url="$1" dest="$2" tmp
  tmp=$(mktemp)
  if curl -fsSL "$url" | gpg --dearmor >"$tmp" 2>/dev/null && [ -s "$tmp" ] && $SUDO install -m 0644 "$tmp" "$dest"; then
    rm -f "$tmp"; return 0
  fi
  rm -f "$tmp"; warn "failed to fetch/dearmor/install signing key from $url"; return 1
}

phase_oneapi() {
  log "Intel oneAPI: apt repo + gpg key"
  local key=/usr/share/keyrings/oneapi-archive-keyring.gpg
  if [ ! -f "$key" ]; then
    install_gpg_key https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB "$key" \
      || { warn "skipping oneAPI repo (no key)"; return 1; }
    echo "deb [signed-by=$key] https://apt.repos.intel.com/oneapi all main" \
      | $SUDO tee /etc/apt/sources.list.d/oneAPI.list >/dev/null
    APT_UPDATED=0
  fi
  apt_install intel-oneapi-compiler-dpcpp-cpp intel-oneapi-compiler-fortran \
              intel-oneapi-openmp intel-oneapi-mkl intel-oneapi-mkl-devel
  warn "add 'source /opt/intel/oneapi/setvars.sh' to your shell rc to use icx/icpx/ifx"
}

have_ubuntu() { command -v lsb_release >/dev/null 2>&1 && [ "$(lsb_release -is 2>/dev/null)" = "Ubuntu" ]; }
have_ubuntu || warn "not detected as Ubuntu; apt package names may differ"

log "core build + python tooling"
apt_install build-essential cmake ninja-build git pkg-config ca-certificates curl wget gnupg \
            python3 python3-pip python3-venv python3-dev

log "compilers: gcc/g++/gfortran, clang/llvm, flang"
apt_install gcc g++ gfortran clang clang-tools llvm llvm-dev
[ "$DO_FLANG" -eq 0 ] || apt_install flang  # optional: LLVM Fortran, not on every release

# GCC 13's vectorizer mislowers masked selects (DaCe refuses to trust it), so gcc/g++/c++ point at GCC 14 or newer.
MIN_GCC=14
ALT_PRIORITY=100  # above the distro g++ entry of the c++ alternative
if [ "$(gcc -dumpversion 2>/dev/null | cut -d. -f1)" -lt "$MIN_GCC" ] 2>/dev/null; then
  apt_install "gcc-$MIN_GCC" "g++-$MIN_GCC"
  $SUDO update-alternatives --install /usr/bin/gcc gcc "/usr/bin/gcc-$MIN_GCC" "$ALT_PRIORITY" \
    --slave /usr/bin/g++ g++ "/usr/bin/g++-$MIN_GCC"
  $SUDO update-alternatives --install /usr/bin/c++ c++ "/usr/bin/g++-$MIN_GCC" "$ALT_PRIORITY"
fi

log "OpenMP runtimes: libomp (LLVM/clang), libgomp (ships with gcc)"
apt_install libomp-dev libgomp1

log "BLAS/LAPACK (LAPACKE: the C interface CPF renders vendor LAPACK calls against)"
apt_install libopenblas-dev liblapack-dev liblapacke-dev libblis-dev libfftw3-dev

[ "$DO_ONEAPI" -eq 1 ] && { phase_oneapi || warn "oneAPI setup incomplete"; }

# the CI cache step saves DIR as the invoking user, so hand it back from root
[ -z "$ARCHIVES" ] || $SUDO chown -R "${SUDO_USER:-$(id -un)}" "$ARCHIVES"

log "done. gcc/g++/clang/gfortran on PATH; libomp/libgomp installed."
