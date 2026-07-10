#!/usr/bin/env bash
# Format every nest-forge source: Python via yapf (pyproject [tool.yapf], 120 cols) and C/C++ via
# clang-format (.clang-format, 160 cols). Default rewrites in place; --check only reports and exits
# nonzero if anything is unformatted (the CI gate).
#
# Usage: scripts/format.sh [--check] [-h]
set -euo pipefail

MODE=fix
case "${1:-}" in
  --check) MODE=check ;;
  -h|--help) sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  "") ;;
  *) echo "unknown arg: $1" >&2; exit 1 ;;
esac

cd "$(git rev-parse --show-toplevel)"

# Resolve tools: prefer the PATH binary, fall back to the pip module (clang-format/yapf ship wheels).
YAPF="yapf"; command -v yapf >/dev/null 2>&1 || YAPF="python -m yapf"
CF="clang-format"; command -v clang-format >/dev/null 2>&1 || CF="python -m clang_format"
$YAPF --version >/dev/null 2>&1 || { echo "yapf not available (pip install yapf)" >&2; exit 2; }
$CF --version >/dev/null 2>&1 || { echo "clang-format not available (apt install clang-format / pip install clang-format)" >&2; exit 2; }

# shellcheck disable=SC2207
PY=($(git ls-files '*.py'))
# shellcheck disable=SC2207
CC=($(git ls-files '*.c' '*.cc' '*.cpp' '*.cxx' '*.h' '*.hpp' '*.hxx'))

rc=0
if [ "${#PY[@]}" -gt 0 ]; then
  if [ "$MODE" = fix ]; then
    $YAPF -i "${PY[@]}"; echo "yapf: formatted ${#PY[@]} python files"
  else
    d=$($YAPF -d "${PY[@]}") || true
    if [ -n "$d" ]; then echo "$d"; echo "python NOT formatted (run scripts/format.sh)"; rc=1
    else echo "yapf: ${#PY[@]} python files OK"; fi
  fi
fi

if [ "${#CC[@]}" -gt 0 ]; then
  if [ "$MODE" = fix ]; then
    $CF -i "${CC[@]}"; echo "clang-format: formatted ${#CC[@]} C/C++ files"
  else
    if $CF --dry-run --Werror "${CC[@]}"; then echo "clang-format: ${#CC[@]} C/C++ files OK"
    else echo "C/C++ NOT formatted (run scripts/format.sh)"; rc=1; fi
  fi
else
  echo "clang-format: no tracked C/C++ files (config ready for emitted C++)"
fi

exit "$rc"
