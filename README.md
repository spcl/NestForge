# nest-forge

Extract loop-/map-nests from a DaCe SDFG, re-emit each as a standalone numpy reference + YAML
config, farm them out to OptArena's translator to produce C/C++/Fortran variants, compile each
across a compiler × flag × FP-mode matrix, benchmark against generated data, pick the best per
nest, link winners into the full program, and compare against baselines. A DaCe backend competes
in the same arena. Results are tracked FP-Arena-style for the `extended` track.

Everything lives here and plugs into DaCe through its external-transformation registry; DaCe
itself stays unmodified.

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
  arena.py        compiler discovery + compiler×flag×FP-mode sweep + winner + report
```

## Design docs
- `DESIGN.md` — emitter audit (contract, cross-cutting concerns, overfit verdict, refinement plan).
- `PREDICTIVE.md` — profile-based + offline-predictive modes (compiler ranking; FP-safety).
- `PARALLEL.md` — parallel-region handling: compile intent (parent-parallel → single-thread; outer →
  OpenMP), the single-mandated-runtime + driver-owned-init link contract, and stability under parallelism.
- `docs/FP_RISK.md` — static classifier for when fast-math / a parallel reduction is numerically
  dangerous: 7-sub-flag taxonomy, condition-number theory, static-vs-dynamic tool survey, the R1–R16
  rule table, and cross-language operator semantics (numpy-default + fortran-to-sdfg).
- `docs/OPT_RECORDS.md` — emitting + parsing optimization records for GCC / LLVM / Intel / NVIDIA, and
  the normalized schema + trip-count-weighted ranking the predictive mode uses.
- `BUILD.md` — nest-forge owning its build: generate code + compile + link ourselves (not
  `dace.compile`), manual init/finalize, a C++ `<chrono>` timing-transformation (with/without libnodes),
  and maximal-LTO static-lib inlining. `scripts/overhead_baseline.py` is the correctness+timing first cut.

## Deps
- DaCe (`/home/primrose/Work/dace`, branch `extended`).
- OptArena — vendored as the `external/optarena` git submodule (`github.com/spcl/OptArena`). Resolve
  and install it with:
  ```
  git submodule update --init --recursive
  pip install -e external/optarena
  ```
  nest-forge surfaces exactly two of its pieces as native, first-class APIs and reaches no other
  optarena internals:
  - `nestforge.translator` — the numpy -> C/C++/Fortran translator (`translate`, `BenchSpec`);
  - `nestforge.corpus` — the npbench/polybench kernel corpus (`iter_dace_kernels`, `CorpusKernel`).

## Status
**M0 done** (CPU, C, single MapEntry nest): extract → outer strategy → numpy + OptArena manifest
→ translate to C → compile across gcc/clang × {ieee-strict, fast-but-ieee, fast-math} → validate
vs numpy oracle → time → winner per FP mode → `ExternalCall` libnode (`DaceReference` +
`ExternCall` linking the winning `.so` into the whole SDFG program) → per-nest report.

Try it: `python examples/demo_fma.py` (shows ieee-strict bit-exact vs fast-math FMA rounding).

**M1 in progress:** real npbench/polybench corpus (`corpus.py`, 55 dace kernels); library-node
emission (MatMul/Dot/Reduce/Transpose/TensorTranspose/Solve/Cholesky → numpy / numpy.linalg);
`LoopRegion` + `ConditionalBlock` (`if`/`elif`/`else`) control-flow emission; **nested-SDFG-in-map**
inlining (via DaCe's `ExpandNestedSDFGInputs` to widen the nest to full arrays, then emit its body in
place — a masked `np.where` becomes `if I[j,k]: Z[j,k] = ...`); access-node data copies (scalar
`s = A[i]` and sub-array `B[:] = A[k]`), inter-state assignments (indirect indices hoisted onto
edges), `dace.<cast>` → `np.<cast>` and bare math intrinsic (`sqrt` → `np.sqrt`) normalization;
`skip-taskloops` (default) & `innermost` strategies; **C-style emission** — the kernel allocates
nothing, every array (inputs, outputs, `__return`, scratch transients) is a pre-allocated buffer
parameter written in place; **WCR-reduction tasklet** emission (`hist[bin] += w` → an augmented
assignment); **max-size loop scratch** — a transient sized by loop variables (`A_0` shaped `[j]`, a
shrinking `[M-i-1]`, a growing `[i+1]`) is widened per dimension to the extent's maximum over the loop
range (upper bound where the dimension increases in the variable, lower bound where it decreases —
monotonicity from the sign of the constant derivative) so it stays a caller-allocated parameter
addressed by the original `0:extent` slice; a non-monotone `R**(K-i-1)` (slope `R**i·log R`, unknown
sign) is left alone; BLAS discovery. Corpus census (extended DaCe, *runnable-eligible* — parses **and**
every scratch buffer is sizable from the kernel's own symbols): 45 emit / 7 unsupported / 3 frontend
build-fail. Validated against numpy: `ConditionalBlock` nussinov (bit-exact), contour_integral &
scattering_self_energies (fp roundoff); nested-SDFG mandelbrot1 & nbody (bit-exact); WCR azimint_naive
& azimint_hist (bit-exact — the latter nests three deep: get_bin_edges / compute_bin / histogram);
loop-scratch trisolv, lu, covariance & syrk (bit-exact / fp roundoff); Cholesky & TensorTranspose
library nodes (bit-exact). The 7 unsupported: 1 nested map-in-map (cholesky2); 3 with a scratch extent
that is not a function of the kernel symbols — a data-dependent CSR span (spmv), a dynamic length
(mandelbrot2), an FFT power `R**i` (stockham_fft); 3 with a hidden layer-config symbol not in the
signature (mlp, resnet, lenet). A size-1 array that DaCe refers to as `A[0]` but the emitter keeps as a
scalar local is reconciled in inter-state assignment strings (the `[0]` is stripped) — no emitting
kernel has a bare-write / indexed-read mismatch. Emission is read-only (nested-SDFG widening and scratch
resizing both run on a copy); guards refuse a nested-SDFG whose inter-state condition under-indexes a
multi-dim array and any scratch still unsizable after widening.

Next: nested map-in-map for cholesky2; expose hidden layer-config symbols for the ML kernels (mlp,
resnet, lenet); then wire BLAS/spack into the sweep, cost-model flag axis,
SQLite tracking, DaCe-backend competitor.
