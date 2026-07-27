#!/bin/bash
# Submit the tsvc2-vs-tsvc2.5 CORPUS COMPARISON: phase 1 (the full axis matrix) over both corpora, with
# the multi-dim tile-op vectorizer lane on, and nothing else. A thin wrapper over perf/daint_all.sh -- the
# sweep, the rank self-partitioning and the merge all already live there.
#
# What this compares, and what it does NOT: tsvc2 (151 kernels) and tsvc2.5 (65) share ZERO kernel keys,
# so there is no per-kernel A/B to run. The comparison is between two kernel POPULATIONS -- tsvc2.5 is the
# irregular/conditional extension set (argmax, ext_break_*, cond_reduce_*, scatter/gather) where the
# classic TSVC assumptions stop holding. tables.md's `### per corpus` section is the read-off; the pooled
# headline geomeans above it describe neither corpus.
#
# Both corpora are imported from the hpcagent_bench submodule (nestforge/tsvc.py: tsvc_corpus.py and
# tsvc_2_5_corpus.py), so `git submodule update --init` must have run. `foundation` (245 kernels) is a
# superset of both, but sweeping it loses the tsvc2/tsvc2.5 provenance the comparison is keyed on -- hence
# --corpora tsvc2 tsvc2_5 rather than the foundation default.
#
# Usage:
#   bash perf/submit_tsvc2_vs_2_5.sh                     # smoke gate, then the comparison
#   SMOKE=0 bash perf/submit_tsvc2_vs_2_5.sh             # straight to the comparison
#   VECTORIZE=0 bash perf/submit_tsvc2_vs_2_5.sh         # drop the vectorizer lane (much shorter)
#   REPS=11 COMPILERS="gcc clang" bash perf/submit_tsvc2_vs_2_5.sh   # any daint_all.sh knob passes through
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/.." && pwd)"   # resolved on the LOGIN node; SLURM copies the batch script to its spool
                                 # dir, so the job cannot re-derive it (see daint_all.sh's resolve_repo).

# Phase 1 only: the corpus comparison reads tsvc_full's tables, and the XL / overhead / call-overhead
# phases neither produce nor consume a per-corpus number.
common="ALL,NF_REPO=$repo,CORPORA=tsvc2 tsvc2_5,RUN_FULL=1,RUN_CROSSLANG=0,RUN_OVERHEAD=0,RUN_CALLOVERHEAD=0"
common="$common,VECTORIZE=${VECTORIZE:-1},OUT_FULL=$repo/perf_results/tsvc2_vs_2_5"

if [ "${SMOKE:-1}" = "1" ]; then
  # The smoke defaults to CORPORA=tsvc2; point it at BOTH so a tsvc2.5-only import break is caught in the
  # 40-minute job rather than after the comparison has burned its wall clock.
  smoke_id="$(sbatch --parsable --export="ALL,NF_REPO=$repo,CORPORA=tsvc2 tsvc2_5" "$here/daint_all_smoke.sh")"
  echo "[tsvc2-vs-2.5] smoke job: $smoke_id (40 min)"
  job_id="$(sbatch --parsable --export="$common" --dependency="afterok:$smoke_id" "$here/daint_all.sh")"
  echo "[tsvc2-vs-2.5] comparison: $job_id (starts only if the smoke succeeds)"
else
  job_id="$(sbatch --parsable --export="$common" "$here/daint_all.sh")"
  echo "[tsvc2-vs-2.5] comparison: $job_id"
fi
echo "[tsvc2-vs-2.5] results -> $repo/perf_results/tsvc2_vs_2_5/tables.md (read the '### per corpus' section)"
