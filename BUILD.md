# nest-forge — owning the build, the timing, and the linking

nest-forge should not depend on `dace.compile()` for the arena. It should generate code, compile, link,
time, and call everything itself — one consistent compiler + flag set across the DaCe-backend
competitor and the offloaded nests, so the arena's numbers are a fair codegen-vs-codegen comparison and
the static node libraries can be inlined into the driver. Substrate under `PREDICTIVE.md` (the
flag/compiler sweep), `PARALLEL.md` (the OpenMP link contract), and the overhead baseline
(`scripts/overhead_baseline.py`).

## 1. Own the compile (don't call `dace.compile`)

`dace.compile()` runs DaCe's own build system and returns a `CompiledSDFG` whose Python `__call__`
re-marshals every argument (ms of overhead unrelated to the generated code). The arena wants raw C,
built our way:

1. **Generate the source, not the binary.** `code_objects = sdfg.generate_code()` returns
   `CodeObject`s (the `.cpp`/`.h` DaCe would have compiled). Write them out; do not let DaCe build them.
2. **Find DaCe's runtime headers.** The generated code `#include`s DaCe's runtime
   (`dace/runtime/include/...`). Locate it once: `Path(dace.__file__).parent / "runtime" / "include"`
   (and `dace.Config.get('compiler', 'cpu', 'args')` for the flags DaCe itself would use, as a
   reference). Compile with `-I<that>` (plus any library-node environment includes, §5).
3. **Compile + link ourselves.** `g++ -std=c++14 -O3 -march=native -I<dace-include> ... prog.cpp
   -shared -fPIC -o dace.so` — one compiler + flag set, ours, matching the offloaded-nest build so the
   competition is fair. This is the same `languages.build_kernel_lib_commands` shape optarena already
   uses for the offload C; unify both under one builder in nest-forge.
4. **Call via ctypes/cffi (§4), handling init/finalize (§2).**

The offload side is *already* owned this way: `scripts/overhead_baseline.py` and `arena.py` build the
translator's C with `gcc` directly and call it via `ctypes` — so "own the build" means giving the DaCe
competitor the same treatment, not inventing a new pipeline.

## 2. init / finalize are manual

A DaCe-generated `.so` does **not** auto-initialize. It exposes three C-linkage entry points for an
SDFG named `N`:

- `N_t *__dace_init_N(<symbols/args>)` — allocates the SDFG state (persistent transients, library-node
  handles: BLAS/MKL contexts, streams, MPI). Returns an opaque state handle.
- `void __program_N(N_t *handle, <args>)` — the kernel. Called once per invocation.
- `int __dace_exit_N(N_t *handle)` — frees the state.

So a raw call is `handle = lib.__dace_init_N(...)`, then `lib.__program_N(handle, ...)` per run, then
`lib.__dace_exit_N(handle)` at teardown. `CompiledSDFG` does exactly this internally (`_init` / `_cfunc`
/ `_exit`); owning the build means replicating those three calls ourselves. For a pure-compute nest the
init/exit are near-empty but must still be called — `__program_N`'s signature takes the handle. Time
only `__program_N` (init/exit are one-time setup, not per-invocation).

## 3. The timing transformation (own the timer, not `sdfg.instrument`)

Rather than DaCe's `sdfg.instrument = Timer` (whose report units/nesting are awkward and which ties us
to DaCe's build), insert the timer ourselves as a **DaCe transformation** that wraps the kernel in a
start block and an end block:

- **Global code**: add `#include <chrono>` (and a `static double __nf_timer_ms;` or an output buffer)
  via `sdfg.append_global_code("#include <chrono>\n...")` so the header is present in the TU.
- **Start block**: a new block *before* the kernel's entry, holding one tasklet whose C++ body is
  `auto __nf_t0 = std::chrono::high_resolution_clock::now();` (stored in a state field / global).
- **End block**: a new block *after* the kernel, one tasklet:
  `__nf_timer_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now()
  - __nf_t0).count();` written into a caller-visible output buffer we read back.

Critical construction rule (learned the hard way — see the memory note): **never set `start_block`
manually.** Add the start block through the CFG add API with `is_start_block=True` (or insert it and
re-point the edges via the API); a post-hoc `sdfg.start_block = id` is silently ignored, orphans the old
start, and later dominator computations `KeyError`. The transformation adds the two blocks and rewires
the entry/exit edges through the add API, then returns the SDFG.

Two timing variants the user asked for — **with and without library nodes**:

- **with libnodes**: time the SDFG as-is (MatMul/Dot/Reduce stay as library nodes → the BLAS/MKL path).
- **without libnodes**: `sdfg.expand_library_nodes()` first, then time the pure-loop form.

The gap between the two is exactly the value of the library-node (BLAS) implementation vs the expanded
loops — a useful signal for the arena's BLAS-backend link axis.

For the **offload** side the analogue is a tiny C++ `main`/harness that brackets `__program_N(...)` (or
the extern-C offload entry) with the same `<chrono>` calls, so both sides are measured in-C, identically,
with no Python or marshaling in the loop. One `time_kernel(entry, reps)` harness serves both.

## 4. ctypes vs cffi

- **ctypes** (stdlib, no build step) is what the arena and the overhead script use now. It needs the C
  signature; nest-forge currently recovers the ABI order by regex over the generated `void N_fp64(...)`
  because the translator reorders arguments vs the manifest.
- **cffi API-mode** compiles a small wrapper against the generated **header**, so it parses the real
  signature — killing the brittle regex ABI parsing and giving lower per-call overhead. Worth adopting
  once the owned builder emits/keeps the header; until then ctypes is fine.

## 5. Linking — maximal LTO to inline the static node libs

The whole-program step links each nest's winner `.a` into the driver. To let the linker inline the
offloaded kernels across the archive boundary:

- Compile node-lib objects **and** the driver with `-flto`; link with `-flto` so the LTO plugin runs at
  link time and can inline `__program_N` / the extern-C offload entry into the driver.
- Archive with **`gcc-ar`/`llvm-ar`**, not plain `ar` — plain `ar` drops the GIMPLE/bitcode LTO sections,
  so `-flto` at link finds nothing to inline.
- Add `-ffunction-sections -fdata-sections -Wl,--gc-sections` (dead-strip) and
  `-fno-semantic-interposition` (allow inlining of exported symbols).
- For the library-node environments (BLAS/MKL/MPI) pull their `-I`/`-L`/`-l` from the DaCe environment
  (`@dace.library.environment`) so the owned link line matches what `dace.compile` would have used.

**The tension to make explicit** (and expose as a mode, not a silent default): LTO-inlining the
offloaded nests into the driver *defeats* the per-nest independent-compiler premise of the arena — you
cannot both compile nest A with icx and nest B with nvc **and** LTO-inline both into one gcc driver
(LTO needs one compiler's bitcode). So there are two link modes:

- **per-nest best-compiler** (the arena default): each nest is an independently compiled `.a` (its own
  winning compiler/flags), linked as opaque objects; no cross-boundary inlining, maximum per-nest
  specialization.
- **monolithic best-inline**: one compiler for all nests + `-flto`, so the driver inlines everything;
  best call overhead + cross-nest optimization, at the cost of per-nest compiler choice.

The arena should build and report both, since which wins is workload-dependent (call-overhead-bound
kernels favor LTO-inline; compute-bound kernels favor per-nest specialization).

## 6. The overhead baseline, on this substrate

`scripts/overhead_baseline.py` today establishes **correctness** (offload matches the numpy oracle and
DaCe's own codegen, per nest) and times the offloaded C. The **fair overhead ratio** (offload vs
in-process DaCe) needs §1–§3: build the DaCe competitor ourselves, bracket both with the same C++
`<chrono>` harness, and compare `__program_N`-only times. Once the owned builder + timing-transformation
land, the script drops `dace.compile` for the timing path and reports the ratio directly.

## 7. How the OpenMP offload plugs in (translator comment feature)

`PARALLEL.md` needs a parallel nest to emit `#pragma omp parallel for`. The translator route (spec'd
with the user): nest-forge emits a **directive comment** in the numpy source over the parallel loop, and
the optarena translator maps it one-to-one to the target's directive. This becomes a general
**"translate comments"** feature in the translator:

- **Comment capture**: Python's `ast` *discards* comments, so the translator frontend must capture them
  with `tokenize` (COMMENT tokens carry `(row, col)`), building `{lineno -> comment}` and associating a
  comment with the statement it precedes.
- **Plain comment** `# text` → C `// text` (or `/* text */`), Fortran `! text`.
- **Directive comment** `# omp parallel for` → C `#pragma omp parallel for`, Fortran `!$OMP PARALLEL DO`
  (note `for` → `DO`; Fortran's OMP sentinel is `!$OMP`, not `#pragma`).
- **Fortran restrictions** (the user's flag): emit free-form `!` / `!$OMP` (the emitter produces `.f90`
  free-form), keep directive lines within the 132-column free-form limit, and pair a loop directive with
  its `!$OMP END PARALLEL DO` where required. Fixed-form column-1 sentinels are not needed if the emitter
  is free-form-only, but the comment-type must still be `!`, never `//`/`#`.
- **Where**: `numpyto_c/emit.py::_emit_for` (line ~192) and `numpyto_fortran/emit.py::_emit_for`
  (line ~270) gain a "directive/comment attached to this loop" hook fed by the frontend's tokenize map.
- **PR + unit tests** (to optarena): a `# omp parallel for` over a loop → `#pragma omp parallel for`
  (C) / `!$OMP PARALLEL DO` (Fortran); a plain `# note` → `// note` / `! note`; a comment on a
  non-loop statement; a comment that would exceed the Fortran line limit (must wrap/skip, not emit an
  illegal line). (Note: a standing directive records "optarena: push to main, not PRs"; confirm the
  push/PR mechanics before anything outward-facing.)

nest-forge then emits the directive comment for a nest whose `Boundary.nest_parallelism == "parallel"`
and `parent_is_parallel == False` (`PARALLEL.md` §2), and the arena adds `-fopenmp` + the
`OMP_NUM_THREADS` axis + the single-mandated-runtime driver-init link contract (`PARALLEL.md` §3).

## 8. Reading optimization reports

Predictive-compiling (`PREDICTIVE.md` §2A) parses each compiler's optimization report to rank
compiler×flags without running. `docs/OPT_RECORDS.md` is the per-compiler reference (GCC text +
gzip-JSON, LLVM YAML/bitstream, Intel `.optrpt`, NVIDIA `-Minfo`). The planned `nestforge/opt_report.py`:
compile-only with the report flags for the discovered compiler, parse into the normalized record
`{file, line, col, pass, status, vector_width?, interleave?, unroll?, reason?, estimated_speedup?, raw}`
(YAML load for LLVM/icx, line-regex for GCC/NVIDIA/classic-Intel), then score each loop with nest-forge's
known trip counts (`Σ width·trips/latency`, penalize missed-vec/spills/remainder) to rank compilers and
profile only the top-k.

## Summary of the owned pipeline

generate code (`sdfg.generate_code`) → compile+link ourselves (one compiler/flags, DaCe headers on the
`-I` line, LTO for the whole-program inline mode) → call via ctypes/cffi with manual
`__dace_init`/`__program`/`__dace_exit` → time with our own `<chrono>` start/end timing-transformation
(added via the CFG add API, never a manual `start_block`), with- and without-libnode variants → feed the
same fair numbers into the arena's compiler×flag×FP-mode×thread-count sweep.
