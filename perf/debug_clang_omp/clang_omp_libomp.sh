#!/bin/bash
#SBATCH --job-name=nf-clang-omp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --hint=nomultithread
#SBATCH --time=00:25:00
#SBATCH --partition=debug
#SBATCH --account=g34
#SBATCH --output=/capstor/scratch/cscs/ybudanaz/aarch64/NestForge/perf/debug_clang_omp/clang_omp_%j.out
#SBATCH --error=/capstor/scratch/cscs/ybudanaz/aarch64/NestForge/perf/debug_clang_omp/clang_omp_%j.err
#SBATCH --chdir=/capstor/scratch/cscs/ybudanaz/aarch64/NestForge
#
# HYPOTHESIS (debug node): clang's omp-emit cells failed ONLY because LLVM's OpenMP runtime (libomp.so)
# is off the default loader path -- NOT because clang/Polly is missing. FIX under test: flags.py now
# detects the runtime via the compiler driver and bakes -Wl,-rpath into the cell, so it loads WITHOUT
# any LD_LIBRARY_PATH help. This job runs the PATCHED code (PYTHONPATH pins this clone) with NO manual
# libomp path, and asserts the 3 kernels that fail in production (s000/s1112/s1115) now validate.
#   PART 1: independent mechanism control -- a trivial clang -fopenmp exe, with vs without the rpath.
#   PART 2: the real deliverable -- tsvc_full clang omp-emit, and the baked RUNPATH proof (readelf).

set -uo pipefail
REPO=/capstor/scratch/cscs/ybudanaz/aarch64/NestForge
cd "$REPO"

# --- env mirror of perf/daint_all.sh (deliberately NO libomp / LD_LIBRARY_PATH handling) ----------
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"   # pin THIS (patched) clone
export OMP_NUM_THREADS=16
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export PYTHONUNBUFFERED=1
export MPI4PY_RC_INITIALIZE=0
export MPI4PY_RC_FINALIZE=0
export UCX_VFS_ENABLE=n
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader
export PYTHONUSERBASE=/capstor/scratch/cscs/$USER/aarch64/python
export PATH=$PYTHONUSERBASE/bin:$PATH
source /capstor/scratch/cscs/$USER/aarch64/venvs/myenv/bin/activate
spack load gcc@16.1.0
spack load llvm@22.1.5
export DACE_compiler_use_cache=0
export DACE_PERF_CXX_STD="${DACE_PERF_CXX_STD:-c++20}"

CC="$(command -v clang)"
LIBOMP="$(clang -print-file-name=libomp.so 2>/dev/null)"
LIBOMP_DIR="$(dirname "$LIBOMP")"
echo "[dbg] clang            = $CC"
echo "[dbg] libomp.so        = $LIBOMP"
echo "[dbg] LD_LIBRARY_PATH  = ${LD_LIBRARY_PATH:-<empty>}   (note: libomp dir is NOT added by hand)"

echo "############### PART 1: independent mechanism control (trivial clang -fopenmp exe) ###############"
TMP="$(mktemp -d)"; printf '#include <omp.h>\n#include <stdio.h>\nint main(){printf("threads=%d\\n",omp_get_max_threads());return 0;}\n' > "$TMP/t.c"
clang -fopenmp "$TMP/t.c" -o "$TMP/no_rpath.out" 2>/dev/null && echo "[dbg] built no_rpath.out"
clang -fopenmp -L"$LIBOMP_DIR" -Wl,-rpath,"$LIBOMP_DIR" "$TMP/t.c" -o "$TMP/rpath.out" 2>/dev/null && echo "[dbg] built rpath.out"
echo -n "[dbg] no_rpath.out (loader has no libomp dir) : "; ( env -u LD_LIBRARY_PATH "$TMP/no_rpath.out" && echo "LOADED (unexpected)" ) 2>&1 | head -1
echo -n "[dbg] rpath.out    (rpath baked)              : "; ( env -u LD_LIBRARY_PATH "$TMP/rpath.out"    || echo "FAILED (unexpected)" ) 2>&1 | head -1

echo "################### PART 2: PATCHED tsvc_full -- clang omp-emit, no manual path ###################"
RD="$REPO/perf/debug_clang_omp/results"; rm -rf "$RD"; mkdir -p "$RD"
export DACE_default_build_folder="${TMPDIR:-/tmp}/nf_dbg_${SLURM_JOB_ID:-0}"
python3 -m nestforge.perf.tsvc_full --compilers clang --languages c --parallelism omp-emit \
    --corpora tsvc2 --only s000 s1112 s1115 --no-gate --reps 3 --out "$RD" || echo "[dbg] sweep returned nonzero"

echo "################### PROOF: a built clang omp .so carries a RUNPATH to libomp ###################"
SO="$(find "$DACE_default_build_folder" -name '*.so' 2>/dev/null | head -1)"
if [ -n "$SO" ]; then echo "[dbg] inspecting $SO"; readelf -d "$SO" 2>/dev/null | grep -Ei 'RPATH|RUNPATH|NEEDED.*omp' || echo "[dbg] (no rpath/omp NEEDED found)"; else echo "[dbg] no built .so found to inspect"; fi

echo "############################### SUMMARY: clang omp-emit cell status ###############################"
python3 - "$RD" <<'PY'
import json, glob, os, collections, sys
ok = collections.Counter(); errs = collections.Counter()
for f in glob.glob(os.path.join(sys.argv[1], '*.json')):
    k = json.load(open(f))
    if 'skipped' in k:
        continue
    for c in (k.get('cells') or []):
        if c.get('role') == 'timing' and c.get('compiler') == 'clang' and c.get('parallel') == 'omp-emit':
            ok[c.get('ok')] += 1
            if c.get('ok') is not True:
                errs[str(c.get('error') or c.get('reason') or '')[:80]] += 1
print(f"clang omp-emit cells -> ok-counts {dict(ok)}   (expect ok=True > 0, no libomp errors)")
for m, n in errs.most_common(5):
    print(f"    fail x{n}: {m!r}")
PY
echo "DEBUG_DONE"
