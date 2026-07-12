#!/bin/bash
#SBATCH --job-name=nf-all-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:40:00
#SBATCH --partition=normal
#SBATCH --account=g34
#SBATCH --output=nf_all_smoke_%j.out
#SBATCH --error=nf_all_smoke_%j.err
# No hardcoded --chdir: the repo root is resolved at run time (below) from this script's location, so
# results land in <this-repo>/perf_results/ whatever the clone is named. Override with NF_REPO=/path.
#
# Fast pre-flight for daint_all.sh: one node, one rank, a handful of kernels, ONE compiler
# (gcc), the SMALL `S` preset, tiny reps. It walks all four phases end-to-end so a submitter
# validates the whole path in minutes BEFORE the big run:
#   PHASE 1  tsvc_full          -- toolchains load, emit+validate+time, the DaCe-cpp baseline
#                                  builds without a build-dir collision, the strict gate passes.
#   PHASE 2  crosslang_xl       -- cross-language translate+run (c + fortran).
#   PHASE 3  staticlib_overhead -- monolithic vs external `.a` compile.
#   PHASE 4  plot_overhead.py + plot_winners.py (rank 0).
# Same code path + same preamble as daint_all.sh, just tiny --only / preset / reps.
#
# LANGUAGES defaults to "c fortran" so PHASE 2 (crosslang_xl accepts only c/fortran) stays green.
# Add c++ (LANGUAGES="c c++ fortran") to also smoke the phase-1 C++ recompile lane -- phase 2 then
# errors on c++ and is skipped (guarded), which is fine for a pre-flight.
#
# Submit with:   sbatch perf/daint_all_smoke.sh
#   ONLY="s000 s112 s1115" COMPILERS="gcc clang" sbatch perf/daint_all_smoke.sh
#   RUN_OVERHEAD=0 RUN_PLOTS=0 sbatch perf/daint_all_smoke.sh    # just phases 1-2

set -euo pipefail
# Repo root defaults to THIS script's location (<repo>/perf/daint_all_smoke.sh -> <repo>); override NF_REPO.
REPO="${NF_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
echo "[smoke] repo root: $REPO (results under $REPO/perf_results/)"

export OMP_NUM_THREADS="72"
export PYTHONUNBUFFERED=1

# dace transitively imports mpi4py; under srun's PMI it auto-inits MPI and hangs/aborts. This driver
# uses only the SLURM rank env vars (never MPI), so disable mpi4py auto-init + the UCX/OMPI hang traps.
export MPI4PY_RC_INITIALIZE=0
export MPI4PY_RC_FINALIZE=0
export UCX_VFS_ENABLE=n
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader

export PYTHONUSERBASE=/capstor/scratch/cscs/$USER/aarch64/python
export PATH=$PYTHONUSERBASE/bin:$PATH
source /capstor/scratch/cscs/$USER/aarch64/venvs/myenv/bin/activate
alias python=python3.11

spack load gcc@16.1.0
spack load llvm@22.1.5
module load nvhpc 2>/dev/null || echo "[all-smoke] nvhpc module not found -- nvc/nvc++/nvfortran skipped"
source /opt/intel/oneapi/setvars.sh 2>/dev/null || echo "[all-smoke] oneAPI setvars not found -- icx/icpx/ifx skipped"

export DACE_compiler_use_cache=0
export DACE_PERF_CXX_STD="${DACE_PERF_CXX_STD:-c++23}"

# --- small env-var knobs (SMALL preset -- do NOT run real PROF/XL here) ---------
ONLY="${ONLY:-s000 s1115}"                 # a handful of kernels only
CORPORA="${CORPORA:-tsvc2}"
LANGUAGES="${LANGUAGES:-c fortran}"        # keeps phase 2 green (crosslang = c/fortran only); see note
PARALLELISM="${PARALLELISM:-both}"
OPT_MODES="${OPT_MODES:-baseline canonicalize}"
COST_MODELS="${COST_MODELS:-default cheap no-vec}"
FP_MODES="${FP_MODES:-default-fp no-fast-errno}"
PROFILE_PRESET="${PROFILE_PRESET:-S}"      # SMALL
XL_PRESET="${XL_PRESET:-S}"                # SMALL
COMPILERS="${COMPILERS:-gcc}"              # ONE compiler
REPS="${REPS:-3}"
XL_REPS="${XL_REPS:-3}"
OVERHEAD_CXX="${OVERHEAD_CXX:-g++}"
OVERHEAD_REPS="${OVERHEAD_REPS:-2}"
CALLOVERHEAD_CC="${CALLOVERHEAD_CC:-gcc}"
CALLOVERHEAD_INNER="${CALLOVERHEAD_INNER:-500}"
CALLOVERHEAD_REPS="${CALLOVERHEAD_REPS:-3}"
CALLOVERHEAD_PRESET="${CALLOVERHEAD_PRESET:-S}"
COMPILE_JOBS="${COMPILE_JOBS:-8}"

RUN_FULL="${RUN_FULL:-1}"
RUN_CROSSLANG="${RUN_CROSSLANG:-1}"
RUN_OVERHEAD="${RUN_OVERHEAD:-1}"
RUN_CALLOVERHEAD="${RUN_CALLOVERHEAD:-1}"
RUN_PLOTS="${RUN_PLOTS:-1}"

OUT_FULL="${OUT_FULL:-perf_results/all_smoke/tsvc_full}"
OUT_XL="${OUT_XL:-perf_results/all_smoke/crosslang_xl}"
OUT_OVERHEAD="${OUT_OVERHEAD:-perf_results/all_smoke/staticlib_overhead}"
OUT_CALLOVERHEAD="${OUT_CALLOVERHEAD:-perf_results/all_smoke/calloverhead}"

# --- PHASE 1: full axis matrix (nestforge.perf.tsvc_full) ----------------------
run_full () {
  srun --cpu-bind=cores bash -c '
    export DACE_default_build_folder="${TMPDIR:-/tmp}/nf_dace_smoke_${SLURM_JOB_ID:-0}_${SLURM_PROCID:-0}"
    python3 -m nestforge.perf.tsvc_full \
      --corpora '"$CORPORA"' --only '"$ONLY"' --languages '"$LANGUAGES"' --opt-modes '"$OPT_MODES"' \
      --parallelism "'"$PARALLELISM"'" --cost-models '"$COST_MODELS"' --fp-modes '"$FP_MODES"' \
      --profile-preset "'"$PROFILE_PRESET"'" --compilers "'"$COMPILERS"'" --reps "'"$REPS"'" \
      --compile-jobs "'"$COMPILE_JOBS"'" --out "'"$OUT_FULL"'"
  ' || echo "[all-smoke] phase 1 (tsvc_full) sweep failed (partial results kept)"
  python3 -m nestforge.perf.tsvc_full --tables-only --out "$OUT_FULL" \
    || echo "[all-smoke] phase 1 (tsvc_full) tables failed"
}

# --- PHASE 2: cross-language XL (nestforge.perf.crosslang_xl) -------------------
run_crosslang () {
  srun --cpu-bind=cores bash -c '
    export DACE_default_build_folder="${TMPDIR:-/tmp}/nf_dace_smoke_${SLURM_JOB_ID:-0}_${SLURM_PROCID:-0}"
    python3 -m nestforge.perf.crosslang_xl \
      --corpora '"$CORPORA"' --only '"$ONLY"' --languages '"$LANGUAGES"' --preset "'"$XL_PRESET"'" \
      --compilers "'"$COMPILERS"'" --reps "'"$XL_REPS"'" --out "'"$OUT_XL"'"
  ' || echo "[all-smoke] phase 2 (crosslang_xl) sweep failed (partial results kept)"
  python3 -m nestforge.perf.crosslang_xl --tables-only --out "$OUT_XL" \
    || echo "[all-smoke] phase 2 (crosslang_xl) tables failed"
}

# --- PHASE 3: static-lib compile overhead (nestforge.perf.staticlib_overhead) --
run_overhead () {
  srun --cpu-bind=cores bash -c '
    export DACE_default_build_folder="${TMPDIR:-/tmp}/nf_dace_smoke_${SLURM_JOB_ID:-0}_${SLURM_PROCID:-0}"
    python3 -m nestforge.perf.staticlib_overhead \
      --only '"$ONLY"' --compiler "'"$OVERHEAD_CXX"'" --reps "'"$OVERHEAD_REPS"'" --out "'"$OUT_OVERHEAD"'"
  ' || echo "[all-smoke] phase 3 (staticlib_overhead) sweep failed (partial results kept)"
  python3 -m nestforge.perf.staticlib_overhead --tables-only --out "$OUT_OVERHEAD" \
    || echo "[all-smoke] phase 3 (staticlib_overhead) tables failed"
}

# --- PHASE 4: runtime call overhead (nestforge.perf.calloverhead) --------------
run_calloverhead () {
  srun --cpu-bind=cores bash -c '
    export DACE_default_build_folder="${TMPDIR:-/tmp}/nf_dace_smoke_${SLURM_JOB_ID:-0}_${SLURM_PROCID:-0}"
    python3 -m nestforge.perf.calloverhead \
      --only '"$ONLY"' --compiler "'"$CALLOVERHEAD_CC"'" --preset "'"$CALLOVERHEAD_PRESET"'" \
      --inner "'"$CALLOVERHEAD_INNER"'" --reps "'"$CALLOVERHEAD_REPS"'" --out "'"$OUT_CALLOVERHEAD"'"
  ' || echo "[all-smoke] phase 4 (calloverhead) sweep failed (partial results kept)"
  python3 -m nestforge.perf.calloverhead --tables-only --out "$OUT_CALLOVERHEAD" \
    || echo "[all-smoke] phase 4 (calloverhead) tables failed"
}

# --- PHASE 5: plots (rank 0) ---------------------------------------------------
run_plots () {
  python3 perf/plot_overhead.py --results-dir "$OUT_OVERHEAD" \
    || echo "[all-smoke] phase 5 plot_overhead failed"
  python3 perf/plot_calloverhead.py --results-dir "$OUT_CALLOVERHEAD" \
    || echo "[all-smoke] phase 5 plot_calloverhead failed"
  python3 perf/plot_winners.py --results-dir "$OUT_FULL" \
    || echo "[all-smoke] phase 5 plot_winners failed"
}

# --- run the enabled phases in sequence ----------------------------------------
if [ "$RUN_FULL" = "1" ]; then
  echo "[all-smoke] === PHASE 1: full axis matrix (tsvc_full) -> $OUT_FULL ==="
  run_full
fi
if [ "$RUN_CROSSLANG" = "1" ]; then
  echo "[all-smoke] === PHASE 2: cross-language XL (crosslang_xl) -> $OUT_XL ==="
  run_crosslang
fi
if [ "$RUN_OVERHEAD" = "1" ]; then
  echo "[all-smoke] === PHASE 3: static-lib overhead (staticlib_overhead) -> $OUT_OVERHEAD ==="
  run_overhead
fi
if [ "$RUN_CALLOVERHEAD" = "1" ]; then
  echo "[all-smoke] === PHASE 4: runtime call overhead (calloverhead) -> $OUT_CALLOVERHEAD ==="
  run_calloverhead
fi
if [ "$RUN_PLOTS" = "1" ]; then
  echo "[all-smoke] === PHASE 5: plots (rank 0) ==="
  run_plots
fi
echo "[all-smoke] done."
