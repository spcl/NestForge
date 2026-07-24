#!/usr/bin/env bash
# nest-forge dev runner: run tests / scripts / inline python with the correct env EVERY time, so the
# fiddly incantation lives in one place. It always sets:
#   * DACE_default_build_folder -> an isolated on-disk build dir (NOT /tmp, which is tmpfs/RAM)
#   * OMPI_MCA_rmaps_base_oversubscribe=1 -> MPI anti-hang for dace scripts
#   * PYTHONPATH -> the repo root (Work/dace is the EXTENDED dace; nestforge imports resolve)
#
# Usage:
#   scripts/dev.sh test  [PYTEST_ARGS...]     pytest, single worker (-n1) -- safe for compile tests
#   scripts/dev.sh test -jN [PYTEST_ARGS...]  pytest with N xdist workers (-n N)
#   scripts/dev.sh sweep [PYTEST_ARGS...]     2-pass: -n4, then re-run the failures at -n1 so a real
#                                             failure is told apart from a parallel-build .so collision
#                                             (no args -> the whole suite)
#   scripts/dev.sh run SCRIPT.py [ARGS...]    python SCRIPT.py with the env
#   scripts/dev.sh py -c 'CODE'               inline python with the env (any python args)
#   scripts/dev.sh env                        print the env this would export, then exit
#
# Options (before the subcommand):
#   --fresh       wipe the build folder first (clean .dacecache between runs)
#   --build DIR   build folder (default: $NF_BUILD or ~/.cache/dace_sweep/wfdbg/nfdev)
#   -h|--help     this help
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${NF_BUILD:-$HOME/.cache/dace_sweep/wfdbg/nfdev}"
FRESH=0

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) awk 'NR==1{next} /^#/{sub(/^# ?/,"");print;next}{exit}' "$0"; exit 0 ;;
    --fresh) FRESH=1; shift ;;
    --build) BUILD="$2"; shift 2 ;;
    *) break ;;
  esac
done

[ "$FRESH" = 1 ] && rm -rf "$BUILD"
mkdir -p "$BUILD"
export DACE_default_build_folder="$BUILD"
export OMPI_MCA_rmaps_base_oversubscribe=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

cmd="${1:-}"; shift || true
case "$cmd" in
  env)
    printf 'DACE_default_build_folder=%s\nOMPI_MCA_rmaps_base_oversubscribe=1\nPYTHONPATH=%s\n' "$BUILD" "$PYTHONPATH" ;;
  run) exec python3 "$@" ;;
  py)  exec python3 "$@" ;;
  test)
    workers="-n1"
    case "${1:-}" in -j*) workers="-n${1#-j}"; shift ;; esac
    exec python3 -m pytest -p no:cacheprovider "$workers" "$@" ;;
  sweep)
    out="$(mktemp -d)"
    echo "=== PASS 1: -n4 ==="
    python3 -m pytest -q -p no:cacheprovider -n4 --no-header "$@" > "$out/p1.txt" 2>&1 || true
    tail -1 "$out/p1.txt"
    grep -oE '^(FAILED|ERROR) [^ ]+' "$out/p1.txt" | awk '{print $2}' | sort -u > "$out/f1.txt"
    n=$(wc -l < "$out/f1.txt")
    echo "pass-1 failures: $n"
    if [ "$n" -gt 0 ]; then
      echo "=== PASS 2: re-run those $n at -n1 (filter build collisions) ==="
      rm -rf "$BUILD"; mkdir -p "$BUILD"
      python3 -m pytest -q -p no:cacheprovider -n1 --no-header $(cat "$out/f1.txt") > "$out/p2.txt" 2>&1 || true
      tail -1 "$out/p2.txt"
      grep -oE '^(FAILED|ERROR) [^ ]+' "$out/p2.txt" | awk '{print $2}' | sort -u > "$out/f2.txt"
      echo "=== REAL failures (survive -n1) ==="; cat "$out/f2.txt"
      echo "=== flakes (failed -n4, passed -n1) ==="; comm -23 "$out/f1.txt" "$out/f2.txt"
      [ -s "$out/f2.txt" ] && exit 1 || exit 0
    fi
    echo "ALL GREEN" ;;
  ""|*) awk 'NR==1{next} /^#/{sub(/^# ?/,"");print;next}{exit}' "$0"; exit 1 ;;
esac
