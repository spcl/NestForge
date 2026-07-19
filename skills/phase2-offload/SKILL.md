---
name: phase2-offload
description: Phase 2 of the nest-forge 4-phase optimizer — decide offload granularity: which nests leave the SDFG as external library calls. Choose a named granularity (default: top-level compute nests), inspect what it selects, then externalize. Use after Phase 1 has fixed fusion granularity, before optimizing each nest.
---

# Phase 2 — offload granularity

Phase 1 fixed the *fusion* granularity. Phase 2 decides **which nests leave the SDFG as external
library calls** — the offload granularity. A granularity is a detection strategy: it selects the
nests. Default: **top-level compute nests** — the outermost nests, skipping pure map/loop scheduling
wrappers that carry no compute (`DEFAULT_GRANULARITY == "skip-taskloops"`).

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

strategy_names()                               # ['innermost', 'outer', 'skip-taskloops']
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

## Next

Phase 2 fixes what gets externalized → **Phase 3** optimizes each externalized nest individually →
Phase 4 feeds measurements back to Phase 1 (re-fuse) or Phase 2 (re-granularize).
