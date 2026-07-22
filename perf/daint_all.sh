#!/bin/bash
#SBATCH --job-name=nf-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4      # 4 ranks/node -- one per GH200 Grace socket. `nproc --all` on a
#SBATCH --cpus-per-task=72       # daint compute node to confirm (GH200 node = 4x72-core Grace).
#SBATCH --time=12:00:00          # phase 1 (the full axis matrix) dominates; trim it with LANGUAGES /
                                 # PARALLELISM / COST_MODELS / FP_MODES, or turn phases off (RUN_*=0).
                                 # Lowered from 24h: per-compile NF_COMPILE_TIMEOUT now bounds a runaway
                                 # kernel, so a stuck build no longer needs the full wall clock to die.
#SBATCH --partition=normal
#SBATCH --account=g34
#SBATCH --output=nf_all_%j.out
#SBATCH --error=nf_all_%j.err
# NOTE: no hardcoded --chdir (SLURM copies this script to its spool dir, so the script's own path is NOT
# the clone). The repo root is resolved at run time (resolve_repo, below): NF_REPO, then SLURM_SUBMIT_DIR,
# then the script dir, then the standard daint clone (/capstor/scratch/cscs/$USER/aarch64/NestForge) --
# first one that IS a clone wins. Results land in <that-repo>/perf_results/. Override with NF_REPO=/path.
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
#   PHASE 3  RUN_OVERHEAD=1     static-lib COMPILE overhead -> nestforge.perf.staticlib_overhead
#            Per kernel, own-build the DaCe SDFG monolithically vs via an external static `.a`
#            and report the modular-assembly compile overhead (external / monolithic).
#   PHASE 4  RUN_CALLOVERHEAD=1 runtime CALL overhead -> nestforge.perf.calloverhead
#            The stateless emitted kernel built + TIMED three ways: inlined (#include), external
#            fat-LTO `.a` (linker inlines from the archive), external plain `.a` (out-of-line call).
#            Reports external/inline (the call cost) and external-lto/inline (~1.0 = LTO recovers it).
#   PHASE 5  RUN_PLOTS=1        plots (rank 0) -> perf/plot_overhead.py + perf/plot_calloverhead.py +
#            perf/plot_winners.py. Rendered once, single-process, after the sweeps.
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
# Resolve the repo root robustly. SLURM COPIES this batch script into its spool dir before running, so
# ${BASH_SOURCE[0]} is NOT the clone under sbatch (it is /var/spool/slurmd/...). Try NF_REPO, then the
# submit dir, then the script dir, and pick the first that actually IS the clone (has nestforge/ + perf/).
# submit_all.sh sets NF_REPO explicitly from the login node (where BASH_SOURCE is valid), so that path is
# always correct; a bare `sbatch perf/daint_all.sh` from the repo root is covered by SLURM_SUBMIT_DIR.
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
  echo "[all] ERROR: cannot find the NestForge clone (tried NF_REPO='${NF_REPO:-}', SLURM_SUBMIT_DIR='${SLURM_SUBMIT_DIR:-}', script dir). Resubmit with NF_REPO=/path/to/clone." >&2
  exit 1
}
cd "$REPO"
# Do NOT rely on cwd. Under sbatch, srun launches the ranks with their OWN working directory (the spool
# dir, not this post-`cd` cwd), so a relative `perf_results/` lands in an unwritable place and a bare
# `python -m nestforge` picks a STALE site-packages copy instead of this clone. Two cwd-proof guards:
#   * every --out and every plot path below is ABSOLUTE ($REPO/...), so results land in the clone no
#     matter which directory the process actually runs in;
#   * PYTHONPATH pins THIS clone ahead of site-packages, so `python -m nestforge.*` always imports the
#     freshly pulled code (this is what fixes `No module named nestforge.perf.calloverhead` on a box whose
#     installed nestforge predates that module). srun forwards the exported env to every rank.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
# Create the results root up front (absolute) and, if that fails, report exactly what resolve_repo picked
# and where we are -- so a bad REPO is diagnosed loudly instead of surfacing as a cryptic per-driver mkdir.
mkdir -p "$REPO/perf_results" || {
  echo "[all] ERROR: cannot create '$REPO/perf_results' (REPO='$REPO', pwd='$(pwd)', whoami='$(whoami)'). Check NF_REPO points at a writable clone." >&2
  exit 1
}
echo "[all] repo root: $REPO (results under $REPO/perf_results/); PYTHONPATH pinned to the clone"

export OMP_NUM_THREADS="72"        # one Grace socket's worth of cores per rank
export OMP_PROC_BIND="close"       # pin OpenMP threads, packed within the rank's socket (matches dace slurm_perf.sh)
export OMP_PLACES="cores"          # one OpenMP place per physical core -- without these the timed medians drift
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
spack load cmake                   # DaCe's default (CMake) codegen lane needs cmake on PATH
spack load openblas 2>/dev/null || echo "[all] openblas not in spack -- BLAS-backed lanes fall back to naive loops"
module load nvhpc 2>/dev/null || echo "[all] nvhpc module not found -- nvc/nvc++/nvfortran skipped"
source /opt/intel/oneapi/setvars.sh 2>/dev/null || echo "[all] oneAPI setvars not found -- icx/icpx/ifx skipped"

# `spack load openblas` sets PATH but NOT LD_LIBRARY_PATH/CPATH, and the install sits off the ldconfig
# cache -- so DaCe's detection needs OPENBLAS_DIR + the lib dir on LD_LIBRARY_PATH, or BLAS-backed nodes
# report "not installed" and expand to a naive loop (~25x slower). Matches dace slurm_perf.sh:84-90.
OPENBLAS_DIR="$(spack location -i openblas 2>/dev/null || echo "${OPENBLAS_DIR:-}")"
if [ -n "$OPENBLAS_DIR" ]; then
  export OPENBLAS_DIR
  for _d in "$OPENBLAS_DIR"/lib "$OPENBLAS_DIR"/lib64; do
    [ -d "$_d" ] && export LD_LIBRARY_PATH="$_d:${LD_LIBRARY_PATH:-}" LIBRARY_PATH="$_d:${LIBRARY_PATH:-}"
  done
  [ -d "$OPENBLAS_DIR/include" ] && export CPATH="$OPENBLAS_DIR/include:${CPATH:-}"
fi

# Multi-rank build hygiene (see the cmake-hang note above).
export DACE_compiler_use_cache=0
export DACE_PERF_CXX_STD="${DACE_PERF_CXX_STD:-c++20}"

# --- env-var knobs (all ${VAR:-default}; defaults chosen for the full run) -----
CORPORA="${CORPORA:-tsvc2 tsvc2_5}"
LANGUAGES="${LANGUAGES:-c c++ fortran}"    # phase 1 (tsvc_full) languages: c / c++ / fortran
CROSSLANG_LANGUAGES="${CROSSLANG_LANGUAGES:-c fortran}"  # phase 2 (crosslang_xl) accepts only c/fortran
PARALLELISM="${PARALLELISM:-both}"
OPT_MODES="${OPT_MODES:-simplify-parallel canonicalize auto-opt}"
VECLIBS="${VECLIBS:-auto}"                  # phase 1 vector-math library axis: auto (none + device winner) / none / list
COST_MODELS="${COST_MODELS:-default cheap no-vec}"
FP_MODES="${FP_MODES:-default-fp no-fast-errno}"
PROFILE_PRESET="${PROFILE_PRESET:-PROF}"   # phase 1 size (>L3, memory-bound)
XL_PRESET="${XL_PRESET:-XL}"               # phase 2 size
COMPILERS="${COMPILERS:-auto}"             # "auto" = every discovered compiler; or a whitespace list
REPS="${REPS:-5}"                          # phase 1 loopnest eval: median of 5 timing runs
XL_REPS="${XL_REPS:-5}"                    # phase 2 loopnest eval: median of 5 timing runs
OVERHEAD_CXX="${OVERHEAD_CXX:-g++}"        # phase 3 C++ compiler for the owned DaCe build
OVERHEAD_REPS="${OVERHEAD_REPS:-5}"        # phase 3 cold-compile reps
CALLOVERHEAD_CC="${CALLOVERHEAD_CC:-gcc}"  # phase 4 C compiler (inline vs external-.a vs external-lto)
CALLOVERHEAD_INNER="${CALLOVERHEAD_INNER:-4000}"   # phase 4 kernel calls per timed trampoline invocation
CALLOVERHEAD_REPS="${CALLOVERHEAD_REPS:-9}"        # phase 4 timed invocations (median)
CALLOVERHEAD_PRESET="${CALLOVERHEAD_PRESET:-M}"    # phase 4 size (small enough that call cost is visible)
COMPILE_JOBS="${COMPILE_JOBS:-16}"         # phase 1 bounded compile pool
# Phase 1 matrix size. 'lean' (default) sweeps the vectorizer cost axis (default/cheap/no-vec) only on the
# sequential single-core lane -- exactly what plot_vectorization reads -- and collapses auto-par/omp-emit to
# the compiler default, ~halving the timing cells with NO loss of single-core vectorization data. Identical-
# flag cost cells (clang/icx/nvc 'cheap'=='default') are always deduped regardless. Set 'full' for the
# exhaustive cost x parallel cross-product.
MATRIX_PRESET="${MATRIX_PRESET:-lean}"     # phase 1 matrix preset: lean | full

RUN_FULL="${RUN_FULL:-1}"
RUN_CROSSLANG="${RUN_CROSSLANG:-1}"
RUN_OVERHEAD="${RUN_OVERHEAD:-1}"
RUN_CALLOVERHEAD="${RUN_CALLOVERHEAD:-1}"
RUN_PLOTS="${RUN_PLOTS:-1}"

# ABSOLUTE ($REPO-rooted) so the result dirs are independent of the process cwd (see the PYTHONPATH note
# above). A caller may still override with an absolute OUT_*; a relative override would reintroduce the cwd
# bug, so keep these absolute.
OUT_FULL="${OUT_FULL:-$REPO/perf_results/tsvc_full}"
OUT_XL="${OUT_XL:-$REPO/perf_results/crosslang_xl}"
OUT_OVERHEAD="${OUT_OVERHEAD:-$REPO/perf_results/staticlib_overhead}"
OUT_CALLOVERHEAD="${OUT_CALLOVERHEAD:-$REPO/perf_results/calloverhead}"

# --- PHASE 1: full axis matrix (nestforge.perf.tsvc_full) ----------------------
# srun gives each rank a UNIQUE dace build folder (SLURM_PROCID). `|| echo` keeps a rank/compiler
# failure from aborting the table pass or the later phases.
run_full () {
  srun --cpu-bind=verbose,cores --distribution=block:block bash -c '
    export DACE_default_build_folder="/dev/shm/nf_dace_${SLURM_JOB_ID:-0}_${SLURM_PROCID:-0}"
    [ -w /dev/shm ] || export DACE_default_build_folder="${TMPDIR:-/tmp}/nf_dace_${SLURM_JOB_ID:-0}_${SLURM_PROCID:-0}"
    python3 -m nestforge.perf.tsvc_full \
      --corpora '"$CORPORA"' --languages '"$LANGUAGES"' --opt-modes '"$OPT_MODES"' \
      --parallelism "'"$PARALLELISM"'" --cost-models '"$COST_MODELS"' --fp-modes '"$FP_MODES"' \
      --veclibs '"$VECLIBS"' \
      --profile-preset "'"$PROFILE_PRESET"'" --compilers "'"$COMPILERS"'" --reps "'"$REPS"'" \
      --matrix-preset "'"$MATRIX_PRESET"'" --compile-jobs "'"$COMPILE_JOBS"'" --out "'"$OUT_FULL"'"
  ' || echo "[all] phase 1 (tsvc_full) sweep failed (partial results kept)"
  python3 -m nestforge.perf.tsvc_full --tables-only --out "$OUT_FULL" \
    || echo "[all] phase 1 (tsvc_full) tables failed"
  # Plot NOW, right after this phase's data exists -- so a later phase hanging never costs us the
  # phase-1 plots (rank 0, plain python3, never under srun).
  if [ "$RUN_PLOTS" = "1" ]; then
    python3 "$REPO/perf/plot_winners.py" --results-dir "$OUT_FULL" \
      || echo "[all] phase 1 plot_winners failed"
    python3 "$REPO/perf/plot_speedup_matrix.py" --results-dir "$OUT_FULL" \
      || echo "[all] phase 1 plot_speedup_matrix failed"
    # Single-core vectorization: where one compiler's vectorizer beats another's by a lot (the sequential
    # scalar-vs-vector story the averaged winners/matrix plots hide).
    python3 "$REPO/perf/plot_vectorization.py" --results-dir "$OUT_FULL" \
      || echo "[all] phase 1 plot_vectorization failed"
  fi
}

# --- PHASE 2: cross-language XL (nestforge.perf.crosslang_xl) -------------------
run_crosslang () {
  srun --cpu-bind=verbose,cores --distribution=block:block bash -c '
    export DACE_default_build_folder="/dev/shm/nf_dace_${SLURM_JOB_ID:-0}_${SLURM_PROCID:-0}"
    [ -w /dev/shm ] || export DACE_default_build_folder="${TMPDIR:-/tmp}/nf_dace_${SLURM_JOB_ID:-0}_${SLURM_PROCID:-0}"
    python3 -m nestforge.perf.crosslang_xl \
      --corpora '"$CORPORA"' --languages '"$CROSSLANG_LANGUAGES"' --preset "'"$XL_PRESET"'" \
      --compilers "'"$COMPILERS"'" --reps "'"$XL_REPS"'" --out "'"$OUT_XL"'"
  ' || echo "[all] phase 2 (crosslang_xl) sweep failed (partial results kept)"
  python3 -m nestforge.perf.crosslang_xl --tables-only --out "$OUT_XL" \
    || echo "[all] phase 2 (crosslang_xl) tables failed"
}

# --- PHASE 3: static-lib compile overhead (nestforge.perf.staticlib_overhead) --
run_overhead () {
  srun --cpu-bind=verbose,cores --distribution=block:block bash -c '
    export DACE_default_build_folder="/dev/shm/nf_dace_${SLURM_JOB_ID:-0}_${SLURM_PROCID:-0}"
    [ -w /dev/shm ] || export DACE_default_build_folder="${TMPDIR:-/tmp}/nf_dace_${SLURM_JOB_ID:-0}_${SLURM_PROCID:-0}"
    python3 -m nestforge.perf.staticlib_overhead \
      --compiler "'"$OVERHEAD_CXX"'" --reps "'"$OVERHEAD_REPS"'" --out "'"$OUT_OVERHEAD"'"
  ' || echo "[all] phase 3 (staticlib_overhead) sweep failed (partial results kept)"
  python3 -m nestforge.perf.staticlib_overhead --tables-only --out "$OUT_OVERHEAD" \
    || echo "[all] phase 3 (staticlib_overhead) tables failed"
  if [ "$RUN_PLOTS" = "1" ]; then
    python3 "$REPO/perf/plot_overhead.py" --results-dir "$OUT_OVERHEAD" \
      || echo "[all] phase 3 plot_overhead failed"
  fi
}

# --- PHASE 4: runtime call overhead (nestforge.perf.calloverhead) --------------
# The stateless emitted kernel built three ways -- inlined (#include), external fat-LTO `.a` (linker
# inlines from the archive), external plain `.a` (out-of-line call) -- and TIMED. external/inline is the
# call cost; external-lto/inline (~1.0) shows LTO recovering it.
run_calloverhead () {
  srun --cpu-bind=verbose,cores --distribution=block:block bash -c '
    export DACE_default_build_folder="/dev/shm/nf_dace_${SLURM_JOB_ID:-0}_${SLURM_PROCID:-0}"
    [ -w /dev/shm ] || export DACE_default_build_folder="${TMPDIR:-/tmp}/nf_dace_${SLURM_JOB_ID:-0}_${SLURM_PROCID:-0}"
    python3 -m nestforge.perf.calloverhead \
      --compiler "'"$CALLOVERHEAD_CC"'" --preset "'"$CALLOVERHEAD_PRESET"'" \
      --inner "'"$CALLOVERHEAD_INNER"'" --reps "'"$CALLOVERHEAD_REPS"'" --out "'"$OUT_CALLOVERHEAD"'"
  ' || echo "[all] phase 4 (calloverhead) sweep failed (partial results kept)"
  python3 -m nestforge.perf.calloverhead --tables-only --out "$OUT_CALLOVERHEAD" \
    || echo "[all] phase 4 (calloverhead) tables failed"
  if [ "$RUN_PLOTS" = "1" ]; then
    python3 "$REPO/perf/plot_calloverhead.py" --results-dir "$OUT_CALLOVERHEAD" \
      || echo "[all] phase 4 plot_calloverhead failed"
  fi
}

# --- PHASE 5: plots (rank 0, after the sweeps) ---------------------------------
run_plots () {
  python3 "$REPO/perf/plot_overhead.py" --results-dir "$OUT_OVERHEAD" \
    || echo "[all] phase 5 plot_overhead failed"
  python3 "$REPO/perf/plot_calloverhead.py" --results-dir "$OUT_CALLOVERHEAD" \
    || echo "[all] phase 5 plot_calloverhead failed"
  python3 "$REPO/perf/plot_winners.py" --results-dir "$OUT_FULL" \
    || echo "[all] phase 5 plot_winners failed"
  # The deliverable: cross-language x compiler x flag speedup matrix vs the gcc and llvm vendor defaults
  # (two tabs) + per-kernel speedup scatter over each.
  python3 "$REPO/perf/plot_speedup_matrix.py" --results-dir "$OUT_FULL" \
    || echo "[all] phase 5 plot_speedup_matrix failed"
  python3 "$REPO/perf/plot_vectorization.py" --results-dir "$OUT_FULL" \
    || echo "[all] phase 5 plot_vectorization failed"
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
  echo "[all] === PHASE 3: static-lib compile overhead (staticlib_overhead) -> $OUT_OVERHEAD ==="
  run_overhead
fi
if [ "$RUN_CALLOVERHEAD" = "1" ]; then
  echo "[all] === PHASE 4: runtime call overhead (calloverhead) -> $OUT_CALLOVERHEAD ==="
  run_calloverhead
fi
if [ "$RUN_PLOTS" = "1" ]; then
  echo "[all] === PHASE 5: plots (rank 0) ==="
  run_plots
fi
echo "[all] done."
