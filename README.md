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
assignment); BLAS discovery. Corpus census (extended DaCe, *runnable-eligible* — parses **and** every
scratch buffer is sizable from the kernel's own symbols): 34 emit / 18 unsupported / 3 frontend
build-fail. Validated against numpy: `ConditionalBlock` nussinov (bit-exact), contour_integral &
scattering_self_energies (fp roundoff); nested-SDFG mandelbrot1 & nbody (bit-exact); WCR azimint_naive
(bit-exact); Cholesky & TensorTranspose library nodes (bit-exact, minimal programs). The 18 unsupported
are **17 with a loop-shaped scratch transient** (e.g. `A_0` shaped `[j]`) that a caller cannot
pre-allocate under the C-style contract (cholesky, lu, correlation, covariance, mlp, resnet,
stockham_fft, …) and 1 nested map-in-map (cholesky2). One eligible kernel still fails at runtime:
azimint_hist parses but its 3-level nested scalar return is not yet reconciled. Emission is read-only
(nested-SDFG widening runs on a copy); guards refuse a nested-SDFG whose inter-state condition
under-indexes a multi-dim array and any scratch whose shape depends on a non-argument (loop) symbol.

Next: decide the loop-shaped-transient policy (emit a loop-local `np.empty`, allocate at max size, or
keep unsupported) to unlock the 17; reconcile deep multi-level nested scalar returns (azimint_hist);
nested map-in-map for cholesky2; then wire BLAS/spack into the sweep, cost-model flag axis,
SQLite tracking, DaCe-backend competitor.
