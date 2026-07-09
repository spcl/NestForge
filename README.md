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
  emit_numpy.py   nest_to_numpy(standalone_sdfg) -> python/numpy kernel source
  emit_yaml.py    OptArena BenchSpec manifest (symbols, array shapes/dtypes)
  libnode.py      ExternalCall LibraryNode + ExpandDaceReference / ExpandExternCall
  pass_lower.py   LowerNestsToExternalCall(strategy=outer)
  arena.py        compiler discovery + compiler×flag×FP-mode sweep + winner + report
external/optarena git submodule (translator + data-gen + compile/time harness)
```

## Deps
- DaCe (`/home/primrose/Work/dace`, branch `extended`).
- OptArena (`external/optarena` submodule; currently the installed package is used).

## Status
**M0 done** (CPU, C, single MapEntry nest): extract → outer strategy → numpy + OptArena manifest
→ translate to C → compile across gcc/clang × {ieee-strict, fast-but-ieee, fast-math} → validate
vs numpy oracle → time → winner per FP mode → `ExternalCall` libnode (`DaceReference` +
`ExternCall` linking the winning `.so` into the whole SDFG program) → per-nest report.

Try it: `python examples/demo_fma.py` (shows ieee-strict bit-exact vs fast-math FMA rounding).

Next (M1): `LoopRegion` extraction, spack compiler discovery, cost-model flag axis, SQLite
tracking, DaCe-backend competitor, reductions/WCR in the emitter.
