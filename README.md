# nest-forge

Extract loop-/map-nests from a DaCe SDFG, re-emit each as a standalone numpy reference + YAML config,
farm them out to OptArena's translator to produce C / C++ / Fortran variants, compile each across a
compiler × flag × FP-mode matrix, benchmark against generated data, pick the best per nest, link winners
into the full program, and compare against baselines. A DaCe backend competes in the same arena.

Everything lives here and plugs into DaCe through its external-transformation registry; DaCe itself
stays unmodified.

## Quick start

```bash
# 1. clone with the OptArena submodule (SSH; needs read access to spcl/OptArena)
git clone git@github.com:spcl/NestForge.git && cd NestForge
git submodule update --init --recursive           # pulls external/optarena over SSH

# 2. DaCe `extended` as a sibling checkout, then the editable deps + test/format tools
git clone -b extended git@github.com:spcl/dace.git ../dace   # or: git -C ../dace switch extended
pip install -r requirements-dev.txt               # -e ../dace, -e external/optarena, pytest, yapf, ...

# 3. run the unit suite (must pass with zero skips)
pytest -m "not integration"
```

## Submit the benchmark jobs (CSCS Alps / daint)

The whole benchmark is **one phased SLURM job** (`perf/daint_all.sh`). `submit_all.sh` gates the full run
behind a quick smoke, so a broken pipeline fails in ~40 min instead of after 24 h:

```bash
bash perf/submit_all.sh            # smoke (40 min) -> full run only if the smoke succeeds
SMOKE=0 bash perf/submit_all.sh    # full run only (skip the smoke gate)
REPS=5 COMPILERS=gcc bash perf/submit_all.sh   # any daint_all.sh knob passes straight through
```

`perf/daint_all.sh` phases (each toggled by `RUN_<PHASE>=0|1`, all on by default):

1. **full matrix** (`nestforge.perf.tsvc_full`) — every TSVC kernel (tsvc2 + tsvc2.5) swept over
   opt-mode `{baseline, canonicalize}` × language `{c, c++, fortran}` × `{sequential, auto-par}` ×
   compiler × cost-model `{default, cheap, no-vec}` × FP `{default-fp, no-fast-errno}`, plus a
   strict-ieee correctness gate. Median-of-5 timing at the `PROF` size (working set > L3 → memory-bound);
   compared against the native `.cpp` baseline and the DaCe-cpp lane.
2. **cross-language XL** (`nestforge.perf.crosslang_xl`) — the same kernels at the XL problem size.
3. **static-lib overhead** (`nestforge.perf.staticlib_overhead`) — monolithic vs external `.a` compile time.
4. **plots** (rank 0) — `perf/plot_overhead.py` (single-SDFG vs static-lib) and `perf/plot_winners.py`
   (single-compiler geomean winner vs nest-forge per-kernel best, plus a per-kernel winner table).

Knobs (all `${VAR:-default}`): `COMPILERS` (auto), `REPS` (5), `PROFILE_PRESET` (PROF), `LANGUAGES`,
`CROSSLANG_LANGUAGES`, `COST_MODELS`, `FP_MODES`, `RUN_FULL`/`RUN_CROSSLANG`/`RUN_OVERHEAD`/`RUN_PLOTS`.
Results land under `perf_results/`; merge the per-rank tables with `--tables-only`, e.g.
`python -m nestforge.perf.tsvc_full --tables-only --out perf_results/tsvc_full`. Full matrix, sizing
rationale, multi-rank partitioning, and the cmake-hang mitigation are in `perf/README_tsvc_full.md`.

## Layout
```
nestforge/
  extract.py      extract_nest_to_sdfg(parent_sdfg, node) -> (standalone_sdfg, Boundary)
  strategies.py   Strategy = (SDFG) -> [(parent_sdfg, node)]; `outer` default + registry
  emit_numpy.py   sdfg_to_numpy / nest_to_numpy -> C-style python/numpy kernel (no allocation)
  emit_libnode.py library-node -> numpy op (MatMul/Dot/Reduce/...), in-place writes
  emit_yaml.py    OptArena BenchSpec manifest (symbols, array shapes/dtypes)
  translator.py   NATIVE: numpy -> C/C++/Fortran translator (over the optarena submodule)
  corpus.py       NATIVE: npbench/polybench kernel corpus (over the optarena submodule)
  libnode.py      ExternalCall LibraryNode + ExpandDaceReference / ExpandExternCall
  pass_lower.py   LowerNestsToExternalCall(strategy=skip-taskloops)
  build.py        owned DaCe build (generate + compile + link ourselves; bind_program timing)
  isolation.py    run_isolated: run a compiled kernel in a forked child (segfault/OOM-safe)
  arena.py        compiler discovery + compiler×flag×FP-mode sweep + winner + report
  perf/
    flags.py            shared flag matrix: FP-precision ladder, cost models, auto-par, C-ABI C++
    tsvc.py             (nestforge/tsvc.py) TSVC corpus adapter + preset sizing
    tsvc_full.py        the full-matrix job (3 lanes + the axis sweep, median-of-N, multi-rank)
    crosslang_xl.py     cross-compiler × cross-language job at a fixed preset
    tsvc_arena.py       per-kernel three-column arena (native / default / flag-matrix winner)
    staticlib_overhead.py   monolithic vs external static-lib compile-time overhead
perf/               daint sbatch (daint_all.sh + smoke + submit_all.sh) + plot_*.py + README
```

## Dependencies
- **DaCe — the `extended` branch, installed editable** from a sibling checkout (`../dace`). The PyPI
  `dace` wheel lacks the extended-only passes nest-forge uses (e.g.
  `dace.transformation.interstate.expand_nested_sdfg_inputs`). `requirements-dev.txt` pins `-e ../dace`.
- **OptArena — the `external/optarena` git submodule** (`git@github.com:spcl/OptArena.git`, currently
  private, pulled over SSH). `git submodule update --init --recursive` then `pip install -e external/optarena`.
  nest-forge surfaces exactly two of its pieces as native APIs: `nestforge.translator` (numpy → C/C++/Fortran)
  and `nestforge.corpus` (the npbench/polybench kernel corpus).
- **Toolchain** — two idempotent setup scripts (`--help` each): `scripts/setup_apt.sh` (apt system
  toolchain: gcc/clang/gfortran, libomp/libgomp, linkers, BLAS; `--oneapi`/`--nvhpc` add the vendor repos)
  and `scripts/setup_spack.sh` (the spack compiler × library matrix, userspace).
- **Formatting** — yapf for Python (120 cols) + clang-format for C/C++ (160 cols); `scripts/format.sh`
  rewrites in place, `--check` is the gate.
- **CI** (`.github/workflows/ci.yml`) — format gate → toolchain → editable DaCe + OptArena → the unit set
  (`-m "not integration"`) with zero-skip enforcement. **Currently disabled** (manual dispatch only): it
  needs the private OptArena submodule via an `OPTARENA_DEPLOY_KEY` secret. Re-enable by adding that secret
  and restoring the `push`/`pull_request` triggers. No key is ever stored in the repo.

## Design docs
- `DESIGN.md` — emitter contract, cross-cutting concerns, refinement plan.
- `BUILD.md` — nest-forge owning its build (generate + compile + link ourselves, manual init/finalize,
  `<chrono>` timing, maximal-LTO static-lib inlining).
- `PARALLEL.md` — parallel-region handling: compile intent, the single-runtime + driver-owned-init link
  contract, stability under parallelism.
- `PREDICTIVE.md` — profile-based + offline-predictive modes (compiler ranking, FP safety).
- `docs/FP_PRECISION_LEVELS.md` — the FP-precision ladder swept by the arena (per gcc/llvm/nvidia/intel,
  C + Fortran), verified against real compilers.
- `docs/FP_RISK.md` — static classifier for when fast-math / a parallel reduction is numerically dangerous.
- `docs/OPT_RECORDS.md` — emitting + parsing GCC/LLVM/Intel/NVIDIA optimization records for the predictive mode.
- `docs/GPU_EXTENSION_FUTURE.md` — a (not-yet-implemented) sketch of emitting device kernels.

## Status
CPU path is end-to-end: extract → strategy → numpy + OptArena manifest → translate to C/C++/Fortran →
compile across the compiler × flag × FP-mode matrix → validate vs the numpy oracle (strict-ieee is
bit-exact) → median-of-N timing (fork-isolated) → winner → `ExternalCall` libnode linking the winning
`.so` into the whole SDFG → per-nest report. The TSVC compiler-arena (`nestforge/perf`) and its phased
daint job exercise this at scale across both TSVC corpora.

Emitter coverage spans C-style pre-allocated buffers, `LoopRegion` + `ConditionalBlock` control flow,
nested-SDFG-in-map inlining (via `ExpandNestedSDFGInputs`), library nodes (MatMul/Dot/Reduce/Solve/
Cholesky/…), WCR reductions, and loop-variable-sized scratch widened to a caller-allocatable bound;
emission is read-only and refuses nests it cannot soundly express. `examples/demo_fma.py` shows
ieee-strict bit-exact vs fast-math FMA rounding.

Next: nested map-in-map; expose hidden layer-config symbols for the ML kernels; SQLite result tracking;
GPU targets.
```
