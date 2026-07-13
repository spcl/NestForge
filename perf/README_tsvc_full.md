# TSVC full-matrix job (`tsvc_full`)

One comprehensive CSCS Alps/daint (GH200, aarch64) job that, for **every** kernel of **both** TSVC
corpora (`tsvc2` = 151 kernels + `tsvc2_5` = 65 kernels), measures three lanes and a large compiler/flag
sweep, timing at a memory-bound size with **median-of-N** reps.

- Driver: [`nestforge/perf/tsvc_full.py`](../nestforge/perf/tsvc_full.py)
- Shared flags: [`nestforge/perf/flags.py`](../nestforge/perf/flags.py) (reduced-FP / auto-par / C++ axes)
- Sbatch: **one phased job** [`perf/daint_all.sh`](daint_all.sh) runs this matrix as PHASE 1 plus three
  more phases (see [How to run](#how-to-run)); [`perf/daint_all_smoke.sh`](daint_all_smoke.sh) is its
  pre-flight. (These supersede the former per-job `daint_tsvc_full` / `daint_crosslang_xl` /
  `daint_staticlib_overhead` / `daint_tsvc_arena` scripts.)
- Tests: [`tests/test_tsvc_full.py`](../tests/test_tsvc_full.py)

## Lanes (columns per kernel)

| # | lane | what | compiler | FP |
|---|------|------|----------|----|
| 1 | **native original.cpp** | `<key>_original.cpp` scalar reference loop | one C++ compiler | `-O3 -march=native` default (auto-vec reference) |
| 2 | **DaCe-cpp baseline** | DaCe's OWN C++ codegen of the **extracted-nest standalone SDFG** (owned direct-compile, **no cmake**) | one C++ compiler | `-O3` + **strict-ieee** (`-ffp-contract=off`) |
| 3 | **nest-forge sweep** | the extracted compute nest → numpyto C/C++/Fortran, compiled across the full axis matrix | every discovered family | reduced 2-rung FP + a strict gate |

**Lane 2 is the speedup baseline** every lane-3 cell divides by (`speedup = DaCe-cpp median / nest median`).
It uses the **extracted-nest standalone SDFG**, not the whole-kernel SDFG, so it does the *same work* the
nest-forge lanes do — apples-to-apples timing. This matters for the many multi-level kernels the strategy
peels to an **inner** nest (a leaked outer index fixed to 0, e.g. `s1115`): the whole-kernel SDFG would
compute *all* rows (~`LEN`× more work) and inflate the speedup meaninglessly. The median time is reported
**always** (identical iteration space = a fair baseline). The strict-ieee cross-check bit-matches for most
kernels; for **loop-carried-state recurrences** (`s111`/`s112`) DaCe's raw codegen and numpyto lower the
promoted-state boundary contract differently, so those baselines are **flagged `†`** in the table but
their *timing* stays representative. (The real correctness guarantee is the lane-3 strict-ieee **gate**,
which is bit-exact for every language.)

## Axis matrix (lane 3), per kernel

| axis | values | notes |
|------|--------|-------|
| opt-mode | `baseline`, `canonicalize` | pre-split SDFG optimization; **emit-time** axis (changes the source) |
| language | `c`, `c++`, `fortran` | numpyto has no C++ target → **C++ = the emitted C recompiled by a C++ frontend** |
| parallelization | `sequential`, `auto-par`, `omp-emit` | single-core; compiler auto-parallelizer of plain loops; OUR `#pragma omp` source |
| compiler | `gcc`, `clang`, `nvhpc`, `intel` | whatever `discover_toolchains` finds on PATH/spack |
| cost-model | `default`, `cheap`, `no-vec` | the shared vectorizer cost axis (single-core scalar-vs-vector) |
| FP mode | `default-fp`, `no-fast-errno` | **reduced 2-rung** timing axis (not the 4-level ladder) |

Plus a **strict-ieee correctness GATE** cell per `(opt-mode, language, compiler)` — sequential, default
cost — that must be **bit-exact** (`maxdiff == 0`) vs the numpy fp64 oracle. (Auto-par reorders
reductions, so bit-exactness is only asserted for the sequential gate.)

**Cell count per kernel** (before dedup), per compiler family:
`2 opt × 3 lang × (3 par × 3 cost × 2 fp  +  1 gate)` = timing `2×3×3×3×2 = 108`, gate `2×3 = 6`,
minus:
- **cell dedup** (always on): identical flag sets are the SAME measurement, so they are timed **once**,
  not once per label — clang/icx/nvc `cheap` ≡ `default`, nvidia `assume-finite` ≡ `contract-fma`, … So a
  family with no `cheap` knob contributes one cost cell on the vectorized lane, not two;
- **`--matrix-preset`** (`lean` = the daint default, `full` = exhaustive). `lean` sweeps the vectorizer
  **cost axis only on the `sequential` single-core lane** — precisely what `plot_vectorization` reads —
  and collapses `auto-par`/`omp-emit` to the compiler default. Measured on the daint toolset this is
  **~50% fewer timing runs** (24% from dedup alone) with **no loss of single-core vectorization data**.
  Override with `MATRIX_PRESET=full sbatch perf/daint_all.sh` (or `--matrix-preset full`);
- **clang/flang auto-par** is unsupported (recorded as `unsupported`, not compiled);
- families with no compiler for a language (recorded, not compiled).

### The three per-family design decisions

- **C++ = recompiled C, kept C-ABI.** numpyto emits no C++ target. The C++ lane compiles a generated
  wrapper `<key>_cxxwrap.cpp` that `#include`s the emitted `.c` inside `extern "C" {}` — without it the
  C++ frontend name-mangles `<key>_fp64` and ctypes (and any whole-program static-`.a` link) cannot find
  it. C++ also needs `-Drestrict=__restrict__` (C++ has no `restrict` keyword) and, on **g++ only**, a
  `__builtin_complex` compound-literal shim (g++ lacks the builtin in C++ mode; clang/nvc/icx have it).
  Verified locally that `nm` shows an **unmangled** `T <key>_fp64` for g++/clang++/nvc++.
- **Auto-par support** (verified on gcc 15 / clang 21 / nvc 26.3): gcc `-ftree-parallelize-loops=N
  -fopenmp` (genuinely emits `GOMP_parallel`), nvc `-Mconcur`, icx `-qopenmp -parallel` (best-effort — a
  rejecting compile is recorded as an error cell). **clang/flang have no plain-loop auto-parallelizer**
  (the emitted source carries no OpenMP pragmas, Polly is not guaranteed) → recorded `unsupported`.
- **Reduced FP** (`flags.REDUCED_FP_MODES`): `default-fp` = the vendor default at `-O3` (fast on intel);
  `no-fast-errno` = FMA contraction + `-fno-math-errno`, no reassociation (nvc has no `-fno-math-errno`,
  so its rung is exactly `contract-fma`). Fortran drops `-fno-math-errno` and gfortran gains
  `-fno-frontend-optimize`.

## Sizing

| purpose | preset | size | why |
|---------|--------|------|-----|
| correctness | `M` (cap) | LEN_1D = 32768 | the pure-Python O(N) oracle is slow; the `.so` is size-agnostic so small-validate + large-time exercises the same code |
| **profiling / timing** | `PROF` | LEN_1D = 2²⁴ = 16.7 M → **128 MiB/array** | one fp64 array clearly exceeds the **GH200 Grace L3 (~114 MB/socket)** → realistic **memory-bound** regime, but smaller than XL (whose alloc/first-touch dominates) |
| submitted confirmation | `XL` | LEN_1D = 268 M (~2 GiB/array) | `PROFILE_PRESET=XL` (phase 1); phase 2 (`crosslang_xl`) also runs the corpora at XL |

`PROF` is defined in `tsvc._PRESET` for `LEN_1D`/`LEN_2D`/`LEN_3D` (each ≈128 MiB/array). The intent (per
the plan): **profile at a size that does not fit L3, submit the confirmation at XL.**

## Timing method + speed strategy

- **Median of N** individually-timed reps (`--reps`, default 11), plus min / p25 / p75 / mean — robust to
  OS jitter. (Not a single mean over the whole loop.)
- Each cell is compiled **once**; its `.so` is reused for validate + every timing rep.
- Identical flag sets are **deduped** to a single compile.
- Per-cell **compiles run in a bounded thread pool** (`--compile-jobs`) — compilation, not the timed run,
  is the bottleneck. Timed runs stay strictly **sequential** (one kernel at a time, fork-isolated) so
  they don't contend for memory bandwidth.
- A fast **VALIDATE** pass (compile + one small run) precedes the **TIMING** pass; **only validated cells
  are timed**, so broken cells never waste timing budget.
- A language whose numpyto emit fails is skipped (no wasted compiles).
- Every compiled-kernel execution runs in a **forked child** (`run_isolated`) so a segfault / OOM /
  runaway kills only the child, never the rank.

## Multi-rank + the cmake-hang mitigation

Kernels **self-partition** across ranks via `SLURM_PROCID`/`SLURM_NTASKS` (round-robin `my_slice`); the
union of slices is every kernel exactly once (unit-tested). The final `--tables-only` pass re-scans the
whole results tree across all ranks.

The DaCe-cpp lane compiles **directly (no cmake)** into a per-kernel `mkdtemp`, so concurrent ranks never
share a build dir. The sbatch **additionally** pins a rank-unique `DACE_default_build_folder`
(`…_${SLURM_PROCID}`) and sets `DACE_compiler_use_cache=0` (belt-and-suspenders against a shared
`.dacecache`), and keeps the mpi4py/UCX anti-hang env (`OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader
UCX_VFS_ENABLE=n MPI4PY_RC_INITIALIZE=0`) so DaCe's transitive `mpi4py` import can never stall MPI_Init /
FindMPI under srun's PMI.

## How to run

Everything runs from **one phased sbatch job**, [`perf/daint_all.sh`](daint_all.sh). It executes four
independent phases in sequence; each is guarded (`|| echo`, partial results kept) so one failing phase
never aborts the rest, and each is toggleable via a `RUN_<PHASE>` env var (all default **on**):

| phase | `RUN_*` | driver | what |
|-------|---------|--------|------|
| **1** | `RUN_FULL` | `nestforge.perf.tsvc_full` | the full axis matrix documented above (**the primary job**) → `OUT_FULL` |
| **2** | `RUN_CROSSLANG` | `nestforge.perf.crosslang_xl` | cross-language XL: both corpora at the XL size, every language × compiler → `OUT_XL` |
| **3** | `RUN_OVERHEAD` | `nestforge.perf.staticlib_overhead` | static-lib compile overhead (external `.a` / monolithic) → `OUT_OVERHEAD` |
| **4** | `RUN_PLOTS` | `plot_winners.py`, `plot_speedup_matrix.py`, `plot_vectorization.py`, `plot_overhead.py`, `plot_calloverhead.py` | winners + speedup matrix + **single-core vectorization** + overhead plots (rank 0; also rendered per-phase so a later hang never costs earlier plots) |

Phases 1–3 run under `srun --cpu-bind=cores` (kernels self-partition across ranks) with a rank-unique
`DACE_default_build_folder`; each sweep is followed by its cross-rank `--tables-only` merge (plain
`python3`, rank 0). Phase 4 also runs as plain `python3`.

```bash
# 1) pre-flight (1 node, 1 rank, few kernels, gcc, small `S` preset) -- ALWAYS run this first.
#    Walks ALL FOUR phases in minutes.
sbatch perf/daint_all_smoke.sh

# 2) the full phased job (4 ranks/node); PHASE 1 is at the profiling size (>L3)
sbatch perf/daint_all.sh

# variants (env-var knobs, all have defaults):
COMPILERS="gcc clang nvc icx" REPS=21 sbatch perf/daint_all.sh
RUN_FULL=1 RUN_CROSSLANG=0 RUN_OVERHEAD=0 sbatch perf/daint_all.sh          # just the primary matrix
PROFILE_PRESET=XL LANGUAGES="c fortran" sbatch perf/daint_all.sh           # big confirmation run
RUN_CROSSLANG=0 RUN_OVERHEAD=0 RUN_PLOTS=0 sbatch perf/daint_all.sh        # equivalent of the old tsvc_full job

# local (do NOT run real PROF/XL locally -- use a few M elements):
PYTHONPATH=. python -m nestforge.perf.tsvc_full --only s000 --corpora tsvc2 \
  --compilers gcc --profile-preset S --reps 3 --out perf_results/tsvc_full
PYTHONPATH=. python -m nestforge.perf.tsvc_full --tables-only --out perf_results/tsvc_full
```

**Env-var knobs** (`daint_all.sh`, all `${VAR:-default}`):

- **phases:** `RUN_FULL`, `RUN_CROSSLANG`, `RUN_OVERHEAD`, `RUN_PLOTS` (default `1`)
- **shared:** `CORPORA` (`tsvc2 tsvc2_5`), `LANGUAGES` (`c c++ fortran`), `COMPILERS` (`auto`),
  `DACE_PERF_CXX_STD` (`c++23`)
- **phase 1:** `PARALLELISM` (`both`), `OPT_MODES` (`baseline canonicalize`), `COST_MODELS`
  (`default cheap no-vec`), `FP_MODES` (`default-fp no-fast-errno`), `PROFILE_PRESET` (`PROF`),
  `REPS` (`11`), `COMPILE_JOBS` (`16`), `OUT_FULL` (`perf_results/tsvc_full`)
- **phase 2:** `XL_PRESET` (`XL`), `XL_REPS` (`20`), `OUT_XL` (`perf_results/crosslang_xl`)
- **phase 3:** `OVERHEAD_CXX` (`g++`), `OVERHEAD_REPS` (`5`), `OUT_OVERHEAD` (`perf_results/staticlib_overhead`)

> **Phase 2 caveat:** `crosslang_xl` translates via numpyto's two AOT targets and accepts only `c` and
> `fortran` for `--languages`. The shared `LANGUAGES` default includes `c++` (needed by phase 1), so with
> the default, **phase 2 errors on `c++` and is skipped** (guarded — phases 3–4 still run). For a green
> phase 2 set `LANGUAGES="c fortran"`, or `RUN_CROSSLANG=0` to skip it. The smoke defaults to
> `LANGUAGES="c fortran"` so all four phases stay green.

## Results

Per-kernel JSON at `--out` (default `perf_results/tsvc_full/`, named `<corpus>_<key>.json`), merged by
`--tables-only` into `tables.md`:
- a **strict-ieee bit-exact gate** verdict (PASS / a table of any failures),
- one row per `(kernel, opt-mode, language, compiler)`: native, DaCe-cpp, best nest cell + its config,
  maxdiff, and the **best-nest / DaCe-cpp speedup** (with a geomean),
- an **unsupported/skipped** summary (clang auto-par, missing Fortran compilers, …) — recorded, never
  silently dropped.

## What only daint can confirm

- **nvhpc** (nvc/nvc++/nvfortran) and **Intel oneAPI** (icx/icpx/ifx) toolchains, and whether icx accepts
  `-qopenmp -parallel` (auto-par) or reports it as an error cell.
- **XL memory**: 268 M-element arrays (~2 GiB each) fit a Grace socket only within the rank budget; drop
  ranks / `LANGUAGES` if a many-array kernel OOMs (the fork isolates the OOM to one child).
- The aarch64 `-march=native` / spack `gcc@16.1.0` + `llvm@22.1.5` builds (local dev box is x86_64).
