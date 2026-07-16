# Agentic Optimizer

## High-level design

Granularity == fusion state. More fused == coarser kernel == bigger optimization scope, fewer call boundaries. Agent picks how much to fuse.

```
canon(fusion-ready) -> regions -> [agent fuses] -> extract nests -> [agent optimizes each] -> .a
```

3 arms, compared head-to-head:

| arm | who decides |
|---|---|
| 1 | heuristics + brute-force sweep (baseline) |
| 2 | agent inside DaCe (fusion transformations) |
| 3 | agent on the complete numpy (free-form) |

All arms: validate vs numpy oracle -> time -> winner. Answers: agent beat heuristics? structured beat free-form?

Correctness non-negotiable. Structural fusion stays on SDFG (correct by construction). Numpy = oracle + rendering + arm 3. NOT the fusion surface — LLM rewriting numpy breaks loop-carried deps / reductions silently.

## Sub-design

### Canon (fusion-ready)

`StateFusion` + `LoopToMap`. Minimize states -> more co-located loops -> more adjacency. Expose parallel loops as maps.

Do NOT apply MapFusion/LoopFusion here. Those are agent moves. Fusion-READY, not fusion-BAKED.

### Arms — single-pair transformations (not passes)

| arm | class | what | status |
|---|---|---|---|
| `FuseLoops` | `interstate/fuse_loops.py` | one adjacent same-range sequential loop pair | **DONE** |
| `MapFusionVertical` | `dataflow/map_fusion_vertical.py` | producer -> consumer maps | exists |
| `MapFusionHorizontal` | `dataflow/map_fusion_horizontal.py` | sibling maps, same range | exists |
| `OTFMapFusion` | `dataflow/otf_map_fusion.py` | fuse by recompute (recompute-vs-materialize lever) | exists |
| `MapExpansion` / `LoopFission` | — | defuse, backtrack out of dead ends | exists |
| stop | — | commit granularity | — |

Passes (`LoopFusion`, `MapFusion`) = deterministic baseline + agent warm start.

**Rule: pass imports + uses the transformation.** One legality kernel, two front ends. Cannot diverge.

### Barriers

Unsupported libnode (MPI/pblas, sparse, no emitter) = hard barrier. Never offload it. Isolate in own state, offload compute before/after. `nestforge.split_unsupported` does this.

### Verification (the expensive part)

| layer | what |
|---|---|
| legality gate | `can_be_applied` — DaCe checks the deps; agent never sees illegal move |
| fuzz | random fusable SDFGs -> no crash + `validate()` + bit-exact |
| corpus | every legal site across npbench / polybench / TSVC |
| adversarial | WCR, loop-carried, aliasing, halo, mixed dtype, empty range |
| composition | random fusion SEQUENCES (what agent actually does) |
| in-loop | every committed candidate re-validated vs oracle |

### Agent

- **State**: nest list + fusability graph (which pairs fuse V/H/loop, ranges, deps) + island barriers + measured times. Compact summary, not raw SDFG.
- **Actions**: apply transform at legal site | split | stop.
- **Reward**: measured perf. Cost-model proxy mostly, real measurement periodically. Log anything capped.
- **Warm start**: `strategies.innermost` / `skip_taskloops`.
- Never run real inference on the dev box — scripted/stub only.

## Phases

| phase | what | driver |
|---|---|---|
| 1 | optimal loop fusion + loop offloading | agent in DaCe (transforms) OR agent on numpy; deterministic = LoopFusion/MapFusion passes |
| 2 | offloading granularity (which regions -> `.a`) | deterministic optimizer OR agent |
| 3 | per-kernel optimization | arena sweep OR per-nest agent |
| 4 | re-fusion after per-kernel results (feedback -> back to 1) | either |

Every phase validated vs oracle, measured against the previous. Value of each layer = a number, not an assumption.

## How to use

### FuseLoops (the agent arm)

```python
from dace.transformation.interstate.fuse_loops import FuseLoops

FuseLoops.can_be_applied_to(sdfg, first=loop_a, second=loop_b)   # legal?
FuseLoops.apply_to(sdfg, first=loop_a, second=loop_b)            # fuse that pair
sdfg.apply_transformations_repeated(FuseLoops)                   # fuse all -> fixpoint
```

### Deterministic pass (fuse everything legal)

```python
from dace.transformation.passes.canonicalize.loop_fusion import LoopFusion

n_fused = LoopFusion().apply_pass(sdfg, {})
```

### Split around unsupported (barriers + regions)

```python
from nestforge.split_unsupported import (isolate_unsupported_library_nodes,
                                         whole_program_regions, region_to_standalone)

isolate_unsupported_library_nodes(sdfg)          # MPI etc -> its own state
regions, islands = whole_program_regions(sdfg)   # externalizable regions + native islands
standalone = region_to_standalone(sdfg, regions[0], "r0")   # region -> emittable SDFG
```

### Prepare each region (numpy + manifest)

```python
from nestforge.translate import prepare_regions

prepared, islands = prepare_regions(sdfg, "kern", out_dir, sizes={"N": 64})
# one Prepared per region; islands stay native
```

### Tests

py12, `-n1`, 8GB cap (shared box).

```bash
# dace: 45 FuseLoops tests (interface + arbitrary loop/map nesting)
pytest tests/transformation/fuse_loops_test.py -q -n1

# nest-forge: split / regions / prepare_regions
pytest tests/test_split_unsupported.py tests/test_whole_program.py -q -n1
```

## Status

**DONE**
- `FuseLoops` + pass delegates to it — dace `5b24a100c`
- 45 FuseLoops tests — dace `35b2629ec`
- `split_unsupported` (isolate / regions / `region_to_standalone`) + `prepare_regions` — nest-forge

**OPEN**
- fuzz harness (random SDFG gen)
- wrap `MapFusionVertical` / `MapFusionHorizontal` as arms + their harness
- fusion-ready canon variant
- deterministic lattice search (cost-model reward)
- agent (phase 1/2), per-kernel agent (phase 3), free-form numpy arm, phase-4 feedback
