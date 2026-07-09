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
emission (MatMul/Dot/Reduce/Transpose/Solve → numpy); `LoopRegion` + `ConditionalBlock`
(`if`/`elif`/`else`) control-flow emission; `skip-taskloops` (default) & `innermost` strategies;
**C-style emission** — the kernel allocates nothing, every array (inputs, outputs, `__return`,
scratch transients) is a pre-allocated buffer parameter written in place; BLAS discovery. Corpus
census: 47 emit / 5 unsupported (Cholesky/TensorTranspose libnodes, nested-SDFG-in-map) / 3 frontend
build-fail. Census counts *emittable-and-parseable*; the three now-parseable `ConditionalBlock`
kernels (nussinov, contour_integral, scattering_self_energies) still leave undefined names from
access-node scalar copies, nested-SDFG helper calls, and `dace.<cast>` codegen, so they are not yet
*runnable* — the next emitter gaps, tracked below.

Next: emit access-node → access-node scalar copies (`s = A[i]`) and nested-SDFG helper calls; wire
BLAS/spack into the sweep, cost-model flag axis, SQLite tracking, DaCe-backend competitor.
