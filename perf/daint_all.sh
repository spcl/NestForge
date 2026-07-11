#!/bin/bash
#SBATCH --job-name=nf-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4      # 4 ranks/node -- one per GH200 Grace socket. `nproc --all` on a
#SBATCH --cpus-per-task=72       # daint compute node to confirm (GH200 node = 4x72-core Grace).
#SBATCH --time=24:00:00          # phase 1 (the full axis matrix) dominates; trim it with LANGUAGES /
                                 # PARALLELISM / COST_MODELS / FP_MODES, or turn phases off (RUN_*=0).
#SBATCH --partition=normal
#SBATCH --account=g34
#SBATCH --output=nf_all_%j.out
#SBATCH --error=nf_all_%j.err
#SBATCH --chdir=/capstor/scratch/cscs/ybudanaz/aarch64/nest-forge
#
# ONE phased nest-forge perf job on CSCS Alps/daint (GH200, aarch64). Supersedes the
# former per-job scripts (daint_tsvc_full / daint_crosslang_xl / daint_staticlib_overhead /
# daint_tsvc_arena). It runs, in sequence, four independent phases -- each guarded so one
# failing phase NEVER aborts the rest (`|| echo`, partial results kept) and each toggleable
# via a RUN_<PHASE> env var (all default on):
#
#   PHASE 1  RUN_FULL=1       full axis matrix  -> nestforge.perf.tsvc_full      (the PRIMARY job)
#            For every kernel of both corpora (tsvc2 + tsvc2.5): native original.cpp, the
#            DaCe-cpp baseline, and the nest-forge sweep over opt-mode x language x parallelism
#            x compiler x cost-model x FP, plus a strict-ieee bit-exact gate. Timed median-of-N
#            at a >L3 (memory-bound) profiling size.
#   PHASE 2  RUN_CROSSLANG=1  cross-language XL -> nestforge.perf.crosslang_xl
#            Same corpora at the XL problem size, every language x compiler family, compiled +
#            run + validated. (crosslang_xl translates via numpyto's two AOT targets and accepts
#            only `c`/`fortran`, so it uses its OWN CROSSLANG_LANGUAGES knob (default "c fortran"),
#            independent of phase 1's LANGUAGES. RUN_CROSSLANG=0 skips it.)
#   PHASE 3  RUN_OVERHEAD=1   static-lib overhead -> nestforge.perf.staticlib_overhead
#            Per kernel, own-build the DaCe SDFG monolithically vs via an external static `.a`
#            and report the modular-assembly compile overhead (external / monolithic).
#   PHASE 4  RUN_PLOTS=1      plots (rank 0)     -> perf/plot_overhead.py + perf/plot_winners.py
#            Rendered once, single-process, after the sweeps.
#
# Phases 1-3 run under `srun --cpu-bind=cores` (kernels self-partition across ranks via
# SLURM_PROCID/SLURM_NTASKS); each rank gets a UNIQUE DACE_default_build_folder so concurrent
# ranks never share a build dir (belt-and-suspenders vs a cmake/FindMPI hang). Every sweep is
# followed by its cross-rank `--tables-only` merge, run as plain python3 (rank 0). The plots in
# phase 4 also run as plain python3.
#
# Submit with:   sbatch perf/daint_all.sh                 # ALWAYS smoke first: perf/daint_all_smoke.sh
#   COMPILERS="gcc clang nvc icx" REPS=21 sbatch perf/daint_all.sh
#   RUN_FULL=1 RUN_CROSSLANG=0 RUN_OVERHEAD=0 sbatch perf/daint_all.sh   # just the primary matrix
#   PROFILE_PRESET=XL LANGUAGES="c fortran" sbatch perf/daint_all.sh     # big confirmation run

set -euo pipefail
cd /capstor/scratch/cscs/ybudanaz/aarch64/nest-forge

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
#python3.11 -m venv /capstor/scratch/cscs/$USER/aarch64/venvs/myenv  # one-time; scratch can be purged
source /capstor/scratch/cscs/$USER/aarch64/venvs/myenv/bin/activate
alias python=python3.11

# Toolchains: gcc + llvm via spack, nvhpc via module, Intel (icx/icpx/ifx) via oneAPI setvars.
# discover_toolchains picks up whatever is on PATH, so loading all of them here == "all compilers".
spack load gcc@16.1.0
spack load llvm@22.1.5
module load nvhpc 2>/dev/null || echo "[all] nvhpc module not found -- nvc/nvc++/nvfortran skipped"
source /opt/intel/oneapi/setvars.sh 2>/dev/null || echo "[all] oneAPI setvars not found -- icx/icpx/ifx skipped"

# Multi-rank build hygiene (see the cmake-hang note above).
export DACE_compiler_use_cache=0
export DACE_PERF_CXX_STD="${DACE_PERF_CXX_STD:-c++23}"

# --- env-var knobs (all ${VAR:-default}; defaults chosen for the full run) -----
CORPORA="${CORPORA:-tsvc2 tsvc2_5}"
LANGUAGES="${LANGUAGES:-c c++ fortran}"    # phase 1 (tsvc_full) languages: c / c++ / fortran
CROSSLANG_LANGUAGES="${CROSSLANG_LANGUAGES:-c fortran}"  # phase 2 (crosslang_xl) accepts only c/fortran
PARALLELISM="${PARALLELISM:-both}"
OPT_MODES="${OPT_MODES:-baseline canonicalize}"
COST_MODELS="${COST_MODELS:-default cheap no-vec}"
FP_MODES="${FP_MODES:-default-fp no-fast-errno}"
PROFILE_PRESET="${PROFILE_PRESET:-PROF}"   # phase 1 size (>L3, memory-bound)
XL_PRESET="${XL_PRESET:-XL}"               # phase 2 size
COMPILERS="${COMPILERS:-auto}"             # "auto" = every discovered compiler; or a whitespace list
REPS="${REPS:-5}"                          # phase 1 loopnest eval: median of 5 timing runs
XL_REPS="${XL_REPS:-5}"                    # phase 2 loopnest eval: median of 5 timing runs
OVERHEAD_CXX="${OVERHEAD_CXX:-g++}"        # phase 3 C++ compiler for the owned DaCe build
OVERHEAD_REPS="${OVERHEAD_REPS:-5}"        # phase 3 cold-compile reps
COMPILE_JOBS="${COMPILE_JOBS:-16}"         # phase 1 bounded compile pool

RUN_FULL="${RUN_FULL:-1}"
RUN_CROSSLANG="${RUN_CROSSLANG:-1}"
RUN_OVERHEAD="${RUN_OVERHEAD:-1}"
RUN_PLOTS="${RUN_PLOTS:-1}"

OUT_FULL="${OUT_FULL:-perf_results/tsvc_full}"
OUT_XL="${OUT_XL:-perf_results/crosslang_xl}"
OUT_OVERHEAD="${OUT_OVERHEAD:-perf_results/staticlib_overhead}"

# --- PHASE 1: full axis matrix (nestforge.perf.tsvc_full) ----------------------
# srun gives each rank a UNIQUE dace build folder (SLURM_PROCID). `|| echo` keeps a rank/compiler
# failure from aborting the table pass or the later phases.
run_full () {
  srun --cpu-bind=cores bash -c '
    export DACE_default_build_folder="${TMPDIR:-/tmp}/nf_dace_${SLURM_JOB_ID:-0}_${SLURM_PROCID:-0}"
    python3 -m nestforge.perf.tsvc_full \
      --corpora '"$CORPORA"' --languages '"$LANGUAGES"' --opt-modes '"$OPT_MODES"' \
      --parallelism "'"$PARALLELISM"'" --cost-models '"$COST_MODELS"' --fp-modes '"$FP_MODES"' \
      --profile-preset "'"$PROFILE_PRESET"'" --compilers "'"$COMPILERS"'" --reps "'"$REPS"'" \
      --compile-jobs "'"$COMPILE_JOBS"'" --out "'"$OUT_FULL"'"
  ' || echo "[all] phase 1 (tsvc_full) sweep failed (partial results kept)"
  python3 -m nestforge.perf.tsvc_full --tables-only --out "$OUT_FULL" \
    || echo "[all] phase 1 (tsvc_full) tables failed"
}

# --- PHASE 2: cross-language XL (nestforge.perf.crosslang_xl) -------------------
run_crosslang () {
  srun --cpu-bind=cores bash -c '
    export DACE_default_build_folder="${TMPDIR:-/tmp}/nf_dace_${SLURM_JOB_ID:-0}_${SLURM_PROCID:-0}"
    python3 -m nestforge.perf.crosslang_xl \
      --corpora '"$CORPORA"' --languages '"$CROSSLANG_LANGUAGES"' --preset "'"$XL_PRESET"'" \
      --compilers "'"$COMPILERS"'" --reps "'"$XL_REPS"'" --out "'"$OUT_XL"'"
  ' || echo "[all] phase 2 (crosslang_xl) sweep failed (partial results kept)"
  python3 -m nestforge.perf.crosslang_xl --tables-only --out "$OUT_XL" \
    || echo "[all] phase 2 (crosslang_xl) tables failed"
}

# --- PHASE 3: static-lib compile overhead (nestforge.perf.staticlib_overhead) --
run_overhead () {
  srun --cpu-bind=cores bash -c '
    export DACE_default_build_folder="${TMPDIR:-/tmp}/nf_dace_${SLURM_JOB_ID:-0}_${SLURM_PROCID:-0}"
    python3 -m nestforge.perf.staticlib_overhead \
      --compiler "'"$OVERHEAD_CXX"'" --reps "'"$OVERHEAD_REPS"'" --out "'"$OUT_OVERHEAD"'"
  ' || echo "[all] phase 3 (staticlib_overhead) sweep failed (partial results kept)"
  python3 -m nestforge.perf.staticlib_overhead --tables-only --out "$OUT_OVERHEAD" \
    || echo "[all] phase 3 (staticlib_overhead) tables failed"
}

# --- PHASE 4: plots (rank 0, after the sweeps) ---------------------------------
run_plots () {
  python3 perf/plot_overhead.py --results-dir "$OUT_OVERHEAD" \
    || echo "[all] phase 4 plot_overhead failed"
  python3 perf/plot_winners.py --results-dir "$OUT_FULL" \
    || echo "[all] phase 4 plot_winners failed"
}

# --- run the enabled phases in sequence ----------------------------------------
if [ "$RUN_FULL" = "1" ]; then
  echo "[all] === PHASE 1: full axis matrix (tsvc_full) -> $OUT_FULL ==="
  run_full
fi
if [ "$RUN_CROSSLANG" = "1" ]; then
  echo "[all] === PHASE 2: cross-language XL (crosslang_xl) -> $OUT_XL ==="
  run_crosslang
fi
if [ "$RUN_OVERHEAD" = "1" ]; then
  echo "[all] === PHASE 3: static-lib overhead (staticlib_overhead) -> $OUT_OVERHEAD ==="
  run_overhead
fi
if [ "$RUN_PLOTS" = "1" ]; then
  echo "[all] === PHASE 4: plots (rank 0) ==="
  run_plots
fi
echo "[all] done."
