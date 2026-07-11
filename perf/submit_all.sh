#!/bin/bash
# Submit the WHOLE nest-forge daint pipeline. perf/daint_all.sh already runs every phase (full matrix ->
# cross-language -> static-lib overhead -> plots) in ONE job, so "submit everything" is a single sbatch.
# This wrapper additionally gates the big run behind the smoke job: the full run is queued with an
# `afterok` dependency on the smoke, so a broken pipeline is caught in ~40 min instead of after 24 h.
#
# Usage:
#   bash perf/submit_all.sh                 # smoke first, then the full run only if the smoke succeeds
#   SMOKE=0 bash perf/submit_all.sh         # full run only (skip the smoke gate)
#   REPS=5 COMPILERS=gcc bash perf/submit_all.sh   # any daint_all.sh env knob passes straight through
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"

if [ "${SMOKE:-1}" = "1" ]; then
  smoke_id="$(sbatch --parsable "$here/daint_all_smoke.sh")"
  echo "[submit_all] smoke job: $smoke_id (40 min)"
  full_id="$(sbatch --parsable --dependency="afterok:$smoke_id" "$here/daint_all.sh")"
  echo "[submit_all] full job:  $full_id (starts only if the smoke succeeds)"
else
  full_id="$(sbatch --parsable "$here/daint_all.sh")"
  echo "[submit_all] full job:  $full_id"
fi
