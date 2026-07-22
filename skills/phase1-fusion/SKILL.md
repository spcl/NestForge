---
name: phase1-fusion
description: Phase 1 of the nest-forge 4-phase optimizer — set loop/map-nest granularity by a fusion strategy (fuse everything legal), or adjust it move-by-move with the fusion/fission arms. Use when deciding how coarse or fine the kernels should be before offload.
---

# Phase 1 — fusion granularity

Granularity == fusion state. Phase 1 picks the granularity the later phases optimize at. Default:
**maximal fusion** (fuse everything legal → coarsest kernels). The agent fissions down from there.

## Preconditions

- **Runs first.** Nothing may have been externalized yet — Phase 2 replaces nests with `ExternalCall`
  nodes, and a fusion move cannot see inside one.
- **Input:** an SDFG normalized by `SymbolPropagation` (derived loop bounds rewritten back to the real
  parameters). Without it a peeled nest carries an unbindable free symbol.
- **Output:** the SAME SDFG, mutated in place. Deep-copy first if you need the original.

## One-shot: apply a named strategy

```python
from nestforge.fusion import get_fusion_strategy, fusion_strategy_names

fusion_strategy_names()                       # ['maximal-fusion']
steps = get_fusion_strategy("maximal-fusion")(sdfg)   # in place; returns step count
```

`maximal-fusion` = `LoopToMap` (loops → parallel maps where sound) + `MapFusionVertical` /
`MapFusionHorizontal` to a fixed point + `simplify`. It is the deterministic Phase-1 default and
reaches the exact fixed point the move-by-move arms below reach.

Add a strategy:

```python
from nestforge.fusion import register_fusion_strategy
register_fusion_strategy("my-strategy", lambda sdfg: ...)   # returns int step count
```

## Move-by-move: the agent surface

A strategy is a scripted policy over single moves. Same tools, one legal move at a time. Applying a
move stales the others' node references — **re-enumerate after every apply.**

Fuse (`nestforge.fusion.enumerate_fusions` / `apply_fusion`):

```python
from nestforge.fusion import enumerate_fusions, apply_fusion

while True:
    moves = enumerate_fusions(sdfg)   # every legal FusionMove right now
    if not moves:
        break
    apply_fusion(sdfg, moves[0])      # commit one; re-verifies legality before applying
```

Each `FusionMove` has `.kind` (`fuse-loops` | `fuse-map-vertical` | `fuse-map-horizontal`),
`.where` (the matched nodes), `.label()`. Only semantics-preserving moves are ever enumerated —
each is gated by the transform's own `can_be_applied_to`.

Fission — the inverse (`nestforge.fusion.fission_to_statements` / `map_fission_moves`):

```python
from dace.transformation.dataflow import MapFission

from nestforge.fusion import fission_to_statements, map_fission_moves

fission_to_statements(sdfg)           # explode whole program to statement granularity
for map_entry, nsdfg in map_fission_moves(sdfg):   # or fission ONE map at a time
    # expr_index=1 is the map-with-nested-SDFG pattern; the default 0 rejects exactly these maps.
    MapFission.apply_to(sdfg, expr_index=1, map_entry=map_entry, nested_sdfg=nsdfg)
```

Typical agent loop: `fission_to_statements` to reach the finest granularity, then `enumerate_fusions`
+ `apply_fusion` back up to the granularity that optimizes best.

## Guardrails

- **Re-enumerate after every apply.** A committed move stales every other move's node references.
  Never hold a `FusionMove` list across an `apply_fusion`.
- **Never hand-apply a transform you did not enumerate.** Enumeration is the legality gate; going
  around it is how a move becomes a miscompile instead of a refusal.
- **Deep-copy before speculating.** Moves mutate in place and there is no undo.
- Every enumerated move is value-preserving (legality-gated + fuzzed + bit-exact corpus tested), so
  you can only change granularity, never the result. Still re-validate a move SEQUENCE bit-exact
  against the un-fused numpy oracle before it competes on speed.

## Next

Phase 1 fixes granularity → **Phase 2** decides offload granularity (`nestforge.offload`) →
Phase 3 optimizes each nest (`nestforge.optimize`) → Phase 4 feeds measurements back to Phase 1
(`nestforge.feedback`).
