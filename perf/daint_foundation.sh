#!/bin/bash
#SBATCH --job-name=nf-foundation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4      # 4 ranks/node -- one per GH200 Grace socket (node = 4x72-core Grace)
#SBATCH --cpus-per-task=72
#SBATCH --time=12:00:00          # phase 1 dominates; trim with LANGUAGES / COST_MODELS / FP_MODES,
                                 # cut kernels with LIMIT, or turn a phase off (RUN_*=0).
#SBATCH --partition=normal
#SBATCH --account=g34
#SBATCH --output=nf_foundation_%j.out
#SBATCH --error=nf_foundation_%j.err
#
# The hpcagent_bench FOUNDATION track (245 kernels) swept with all variants, on CSCS Alps/daint
# (GH200, aarch64). Sibling of perf/daint_all.sh, which sweeps the standalone tsvc2 / tsvc2_5
# corpus modules instead; since HPCAgent-Bench 642ef538 foundation is a true SUPERSET of both, so
# this script alone covers everything those two do plus the 29 non-TSVC foundation microkernels.
#
# Two independent phases, each guarded (`|| echo`, partial results kept) and each toggleable:
#
#   PHASE 1  RUN_FULL=1   full axis matrix -> nestforge.perf.tsvc_full --corpora foundation
#            Per kernel: the native _reference.cpp baseline, the DaCe-cpp baseline, and the sweep
#            over opt-mode x language x parallelism x compiler x cost-model x FP x veclib, with a
#            strict-ieee bit-exact gate. Timed median-of-N at a >L3 (memory-bound) size.
#   PHASE 2  RUN_ARENA=1  per-nest arena -> nestforge.perf.foundation_sweep
#            Each kernel lowered to its nests, each nest measured across compiler x FP-level and
#            deduped by built artifact. This is the per-nest granularity data phase 1 does not
#            produce (phase 1 measures whole kernels).
#
# Both phases self-partition across ranks via SLURM_PROCID/SLURM_NTASKS -- no MPI, no coordination
# -- and each rank gets a UNIQUE DACE_default_build_folder so concurrent ranks never share a build
# dir. Each sweep is followed by its cross-rank `--tables-only` merge, run as plain python3.
#
# Submit with:
#   sbatch perf/daint_foundation.sh                               # everything
#   LIMIT=8 REPS=3 RUN_ARENA=0 sbatch perf/daint_foundation.sh    # smoke: 8 kernels, phase 1 only
#   COMPILERS="gcc clang nvc icx" sbatch perf/daint_foundation.sh
#   ARENA_PRESET=L PROFILE_PRESET=XL sbatch perf/daint_foundation.sh   # big confirmation run
#
# NOTE: no hardcoded --chdir (SLURM copies this script to its spool dir, so the script's own path is
# NOT the clone). The repo root is resolved at run time; override with NF_REPO=/path.

set -euo pipefail

# Resolve the repo root robustly: under sbatch ${BASH_SOURCE[0]} is the spool copy, not the clone.
# Try NF_REPO, then the submit dir, then the script dir, then the standard daint clone -- first one
# that actually IS a clone wins.
resolve_repo () {
  local c
  for c in "${NF_REPO:-}" "${SLURM_SUBMIT_DIR:-}" \
           "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)" \
           "/capstor/scratch/cscs/$USER/aarch64/NestForge"; do
    [ -n "$c" ] && [ -f "$c/nestforge/__init__.py" ] && [ -d "$c/perf" ] && { echo "$c"; return 0; }
  done
  return 1
}
REPO="$(resolve_repo)" || {
  echo "[fnd] ERROR: cannot find the NestForge clone (tried NF_REPO='${NF_REPO:-}', SLURM_SUBMIT_DIR='${SLURM_SUBMIT_DIR:-}', script dir). Resubmit with NF_REPO=/path/to/clone." >&2
  exit 1
}
cd "$REPO"
# Do NOT rely on cwd: srun launches ranks with their OWN working directory (the spool dir), so every
# --out below is ABSOLUTE and PYTHONPATH pins THIS clone ahead of site-packages -- otherwise a rank
# imports a stale installed nestforge instead of the freshly pulled code.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$REPO/perf_results" || {
  echo "[fnd] ERROR: cannot create '$REPO/perf_results' (REPO='$REPO', pwd='$(pwd)', whoami='$(whoami)')." >&2
  exit 1
}
echo "[fnd] repo root: $REPO (results under $REPO/perf_results/); PYTHONPATH pinned to the clone"

export OMP_NUM_THREADS="72"        # one Grace socket's worth of cores per rank
export OMP_PROC_BIND="close"       # pin OpenMP threads within the rank's socket -- without these
export OMP_PLACES="cores"          # the timed medians drift run to run
export PYTHONUNBUFFERED=1

# dace transitively imports mpi4py; under srun's PMI it auto-inits MPI and hangs. This driver reads
# only the SLURM rank env vars (never MPI), so disable auto-init and the UCX/hwloc hang traps.
export MPI4PY_RC_INITIALIZE=0
export MPI4PY_RC_FINALIZE=0
export UCX_VFS_ENABLE=n
export HWLOC_COMPONENTS=-gl
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader

export PYTHONUSERBASE=/capstor/scratch/cscs/$USER/aarch64/python
export PATH=$PYTHONUSERBASE/bin:$PATH
source /capstor/scratch/cscs/$USER/aarch64/venvs/myenv/bin/activate

# Toolchains: loading all of them here == "all compilers", since discover_toolchains reads PATH.
spack load gcc@16.1.0
spack load llvm@22.1.5
spack load cmake                   # DaCe's CMake codegen lane needs cmake on PATH
spack load openblas 2>/dev/null || echo "[fnd] openblas not in spack -- BLAS-backed lanes fall back to naive loops"
module load nvhpc 2>/dev/null || echo "[fnd] nvhpc module not found -- nvc/nvc++/nvfortran skipped"
source /opt/intel/oneapi/setvars.sh 2>/dev/null || echo "[fnd] oneAPI setvars not found -- icx/icpx/ifx skipped"

# `spack load openblas` sets PATH but NOT LD_LIBRARY_PATH/CPATH, and the install sits off the ldconfig
# cache -- without these DaCe reports BLAS "not installed" and expands to a naive loop (~25x slower).
OPENBLAS_DIR="$(spack location -i openblas 2>/dev/null || echo "${OPENBLAS_DIR:-}")"
if [ -n "$OPENBLAS_DIR" ]; then
  export OPENBLAS_DIR
  for _d in "$OPENBLAS_DIR"/lib "$OPENBLAS_DIR"/lib64; do
    [ -d "$_d" ] && export LD_LIBRARY_PATH="$_d:${LD_LIBRARY_PATH:-}" LIBRARY_PATH="$_d:${LIBRARY_PATH:-}"
  done
  [ -d "$OPENBLAS_DIR/include" ] && export CPATH="$OPENBLAS_DIR/include:${CPATH:-}"
fi

# Multi-rank build hygiene: a shared build dir is a cmake/FindMPI hang waiting to happen.
export DACE_compiler_use_cache=0
export DACE_PERF_CXX_STD="${DACE_PERF_CXX_STD:-c++20}"

# --- env-var knobs (all ${VAR:-default}; defaults chosen for the full run) -----
CORPORA="${CORPORA:-foundation}"           # the point of this script; add "tsvc2 tsvc2_5" only if you
                                           # deliberately want the duplicate standalone-module runs
LANGUAGES="${LANGUAGES:-c c++ fortran}"
PARALLELISM="${PARALLELISM:-both}"
OPT_MODES="${OPT_MODES:-simplify-parallel canonicalize auto-opt}"
VECLIBS="${VECLIBS:-auto}"                 # auto = none + the device's winning vector-math library
COST_MODELS="${COST_MODELS:-default cheap no-vec}"
FP_MODES="${FP_MODES:-default-fp no-fast-errno}"
PROFILE_PRESET="${PROFILE_PRESET:-PROF}"   # phase 1 size (>L3, memory-bound)
COMPILERS="${COMPILERS:-auto}"             # auto = every discovered compiler; or a whitespace list
REPS="${REPS:-5}"                          # phase 1 timed runs per cell (median)
MATRIX_PRESET="${MATRIX_PRESET:-lean}"     # lean | full -- see perf/daint_all.sh on the cost x parallel
                                           # cross-product this collapses
COMPILE_JOBS="${COMPILE_JOBS:-16}"         # phase 1 bounded compile pool
ARENA_PRESET="${ARENA_PRESET:-M}"          # phase 2 manifest preset (S/M/L/XL)
ARENA_REPS="${ARENA_REPS:-15}"             # phase 2 timed runs per cell
LIMIT="${LIMIT:-}"                         # first N kernels only (smoke runs); empty = all 245

RUN_FULL="${RUN_FULL:-1}"
RUN_ARENA="${RUN_ARENA:-1}"

# ABSOLUTE ($REPO-rooted) so results are independent of the process cwd. A relative override would
# reintroduce the cwd bug, so keep these absolute.
OUT_FULL="${OUT_FULL:-$REPO/perf_results/foundation_full}"
OUT_ARENA="${OUT_ARENA:-$REPO/perf_results/foundation_arena}"

# --- PHASE 1: full axis matrix over the foundation corpus ----------------------
run_full () {
  srun --cpu-bind=verbose,cores --distribution=block:block bash -c '
    export DACE_default_build_folder="/dev/shm/nf_fnd_${SLURM_JOB_ID:-0}_${SLURM_PROCID:-0}"
    [ -w /dev/shm ] || export DACE_default_build_folder="${TMPDIR:-/tmp}/nf_fnd_${SLURM_JOB_ID:-0}_${SLURM_PROCID:-0}"
    python3 -m nestforge.perf.tsvc_full \
      --corpora '"$CORPORA"' --languages '"$LANGUAGES"' --opt-modes '"$OPT_MODES"' \
      --parallelism "'"$PARALLELISM"'" --cost-models '"$COST_MODELS"' --fp-modes '"$FP_MODES"' \
      --veclibs '"$VECLIBS"' \
      --profile-preset "'"$PROFILE_PRESET"'" --compilers "'"$COMPILERS"'" --reps "'"$REPS"'" \
      --matrix-preset "'"$MATRIX_PRESET"'" --compile-jobs "'"$COMPILE_JOBS"'" \
      '"${LIMIT:+--limit $LIMIT}"' --out "'"$OUT_FULL"'"
  ' || echo "[fnd] phase 1 (tsvc_full) sweep failed (partial results kept)"
  python3 -m nestforge.perf.tsvc_full --tables-only --out "$OUT_FULL" \
    || echo "[fnd] phase 1 (tsvc_full) tables failed"
}

# --- PHASE 2: per-nest arena over the foundation track -------------------------
run_arena () {
  srun --cpu-bind=verbose,cores --distribution=block:block bash -c '
    export DACE_default_build_folder="/dev/shm/nf_arena_${SLURM_JOB_ID:-0}_${SLURM_PROCID:-0}"
    [ -w /dev/shm ] || export DACE_default_build_folder="${TMPDIR:-/tmp}/nf_arena_${SLURM_JOB_ID:-0}_${SLURM_PROCID:-0}"
    python3 -m nestforge.perf.foundation_sweep \
      --preset "'"$ARENA_PRESET"'" --reps "'"$ARENA_REPS"'" \
      '"${LIMIT:+--limit $LIMIT}"' --out "'"$OUT_ARENA"'"
  ' || echo "[fnd] phase 2 (foundation_sweep) failed (partial results kept)"
  python3 -m nestforge.perf.foundation_sweep --tables-only --out "$OUT_ARENA" \
    || echo "[fnd] phase 2 (foundation_sweep) tables failed"
}

echo "[fnd] corpora=$CORPORA compilers=$COMPILERS profile=$PROFILE_PRESET arena=$ARENA_PRESET limit=${LIMIT:-all}"
[ "$RUN_FULL" = "1" ]  && { echo "[fnd] === PHASE 1: full axis matrix -> $OUT_FULL ==="; run_full; }
[ "$RUN_ARENA" = "1" ] && { echo "[fnd] === PHASE 2: per-nest arena -> $OUT_ARENA ==="; run_arena; }
echo "[fnd] done; results under $REPO/perf_results/"
