---
name: phase2-offload
description: Phase 2 of the nest-forge 4-phase optimizer — decide offload granularity: which nests leave the SDFG as external library calls. Choose a named granularity (default: top-level compute nests), inspect what it selects, then externalize. Use after Phase 1 has fixed fusion granularity, before optimizing each nest.
---

# Phase 2 — offload granularity

Phase 1 fixed the *fusion* granularity. Phase 2 decides **which nests leave the SDFG as external
library calls** — the offload granularity. A granularity is a detection strategy: it selects the
nests. Default: **top-level compute nests** — the outermost nests, skipping pure map/loop scheduling
wrappers that carry no compute (`DEFAULT_GRANULARITY == "skip-taskloops"`).

## Preconditions

- **Phase 1 has run.** The fusion granularity is fixed; a fusion move cannot see inside an
  `ExternalCall`, so re-fusing after this phase means re-running Phase 1 on a fresh SDFG.
- **Input:** an SDFG at the chosen fusion granularity. `lower_nests_to_external_call` mutates it in
  place and returns the lowered nests.

## Architectural invariant — externalize BEFORE deciding offload

A nest is turned into a library call **first**; only **then** does each backend tool decide whether
its kernel is offloadable (to GPU, say). Never pre-decide offload before extraction — an offload
choice could otherwise shift the extraction underneath it. So the Phase-2 commit is
`lower_nests_to_external_call`.

## Inspect, then commit

```python
from nestforge.offload import offload_candidates, lower_nests_to_external_call, DEFAULT_GRANULARITY

for c in offload_candidates(sdfg):            # non-mutating: what the default WOULD externalize
    print(c.label, "parallel" if c.parallel else "sequential")
    # c.parent_sdfg, c.node -> the nest itself
```

`offload_candidates` is the Phase-2 analog of `enumerate_fusions`: read-only, so the agent sees the
offload set (and each nest's parallel/sequential nature — a parallel nest may carry an OpenMP scope)
before committing. Commit swaps every selected nest for an `ExternalCall`:

```python
lowered = lower_nests_to_external_call(sdfg)   # [(ExternalCall, Boundary), ...] in extraction order
```

Each `ExternalCall` stays runnable immediately via the `DaceReference` expansion, so the lowered SDFG
still validates and runs bit-exact — externalizing changes *where* compute lives, never the result.

## Choosing a granularity

```python
from nestforge.offload import strategy_names, get_strategy

strategy_names()                               # ['cfg', 'innermost', 'map', 'outer', 'skip-taskloops', 'state']
lower_nests_to_external_call(sdfg, "innermost")   # finer: every leaf nest
```

- `skip-taskloops` (default) — outermost compute nests; skips pure scheduling wrappers.
- `outer` — outermost nests, wrappers included.
- `innermost` — every leaf nest (the vectorization-style unit), across nested SDFGs too.

Register a custom granularity with `register_strategy(name, fn)` where `fn: SDFG -> [(parent, node)]`.

Coarsest granularity — the whole un-split program as one unit (no extraction):

```python
from nestforge.offload import whole_program_boundary
b = whole_program_boundary(sdfg)               # b.inputs / b.outputs = caller interface (non-transient)
```

## Linking the winner: shared vs static

Once Phase 3 has a winner, point the `ExternalCall` at its lib and expand. Two link modes, picked by
`ExternLibEnv.configure` from the path suffix:

- **`.so`** — the winner runs as a separate shared lib, loaded via rpath. Each `.so` drags its own
  OpenMP runtime.
- **`.a`** — `build_winner_archive(win, c_source, name, out_dir)` archives the winner's objects into
  `lib<name>_nest.a`; the parent `.so` pulls it in at link time. One binary, and **one libomp** — an
  archive links no runtime of its own, so the parent's is the only one. This is the static-offload
  path (`tests/test_static_offload_e2e.py`). DaCe emits the same `.a` for a whole SDFG under
  `compiler.static_archive` (native + cmake).

```python
from nestforge.arena import build_winner_archive

ext.implementation = "ExternCall"
ext.lib_path = str(build_winner_archive(win, c_source, ext.name, out))  # .a -> statically linked in
ext.symbol, ext.abi_order = win.symbol, win.abi_order
sdfg.expand_library_nodes()
```

## Guardrails

- **Never pre-decide offloadability.** Externalize first, let the backend tool decide. Deciding first
  lets an offload choice shift the extraction underneath it.
- **Take `abi_order` from the winner, never re-derive it.** The emitted signature orders parameters by
  `param_order()` (arrays sorted, then scalars), NOT by the manifest's role order. Re-deriving it
  elsewhere silently swaps pointers across the ABI.
- **Do not put several nests' objects in one `.a`.** DaCe sorts the parent's link flags and can place
  the archive before the parent objects, so `ld` pulls no member and the symbols stay unresolved until
  `dlopen` fails. Use a `.so` for a multi-nest swap.
- Externalizing changes *where* compute lives, never the result — an un-pointed `ExternalCall` still
  runs bit-exact through its `DaceReference` expansion.

## Next

Phase 2 fixes what gets externalized → **Phase 3** optimizes each externalized nest individually
(`nestforge.optimize`) → Phase 4 feeds measurements back to Phase 1 (re-fuse) or Phase 2
(re-granularize) (`nestforge.feedback`).
