# Agentic Optimizer

## High-level design

Granularity == fusion state. More fused == coarser kernel == bigger optimization scope, fewer call boundaries. The optimizer picks how much to fuse, then how to offload, then how to tune each kernel. A deterministic optimizer OR an agent can drive each step. Both compared.

```
canon(fusion-ready) -> regions -> [P1 fuse] -> [P2 offload] -> extract -> [P3 tune each] -> .a -> [P4 re-fuse]
                                        ^__________________________________________________________|
```

Correctness non-negotiable. Structural fusion stays on the SDFG (correct by construction). Numpy = oracle + rendering + a comparison arm, NOT the fusion surface (LLM rewriting numpy breaks loop-carried deps / reductions silently). Every candidate, whoever produced it, validates bit-exact vs the un-fused numpy oracle before it competes on speed.

Three arms compared per program:

| arm | who decides |
|---|---|
| 1 | heuristics + brute-force sweep (baseline) |
| 2 | agent inside DaCe (fusion transformations) |
| 3 | agent on the complete numpy (free-form) |

## The agent contract (same in every phase)

An agent is a policy over a phase. To plug into a phase it needs 4 things:

| | what |
|---|---|
| **State** | compact structural summary — NOT the raw SDFG. Nests + fusability graph (which pairs fuse V/H/loop, ranges, deps), island barriers, + measured/estimated times. |
| **Actions** | the phase's tools (validated transformations, or an axis config, or a numpy rewrite) + `stop`. |
| **Guardrail** | every action is legality-gated (`can_be_applied`) so it is semantics-preserving; every committed candidate is re-validated bit-exact vs the oracle. The agent can never produce a wrong program — only a slow one. |
| **Reward** | measured performance. Cost-model proxy for most steps, real measurement periodically. Warm-start from the heuristics; the agent's job is to beat them. |

Agent kind: LLM with the transformations as tools + the oracle as verifier (fits optarena's agent infra). RL policy over the transform lattice is a later swap. **Never run real inference on the dev box — scripted/stub only.**

If a phase has no agent yet, its deterministic optimizer runs. Phases are independent seams — you can put an agent in one and leave the rest deterministic, and measure the delta.

## Phases

### Phase 1 — optimal loop fusion + loop offloading

**Goal.** Canonicalize to maximal fusability, then fuse to a chosen granularity; wrap each loop nest as an offloadable unit.

**Canon (fusion-ready).** `StateFusion` (+ happens-before) minimizes states -> loops co-located -> more adjacency. `LoopToMap` exposes parallel loops as maps. Do NOT run `LoopFusion` / `MapFusion` here — those are the moves. Fusion-READY, not fusion-BAKED.

**Deterministic driver.** `LoopFusion` / `MapFusion` passes (fuse everything legal).

**Agent integration.**
- *Surface A (in DaCe).* Tools = the single-pair transformations: `FuseLoops.apply_to`, `MapFusionVertical`, `MapFusionHorizontal`, `OTFMapFusion` (recompute lever), defuse (`MapExpansion`/`LoopFission`). State = the fusability graph over the canon SDFG. Each `apply` is legality-gated; the whole SDFG re-validates vs oracle. Agent walks the fusion lattice; `stop` commits.
- *Surface B (numpy).* Agent rewrites the complete numpy input directly. Validated bit-exact vs oracle. This is arm 3.

**Barriers.** An unsupported libnode (MPI/pblas, sparse, no emitter) is a hard fusion barrier — isolated into its own state, never crossed. `nestforge.split_unsupported` does this before fusion.

**Out.** A fused SDFG + its region partition.

### Phase 2 — offloading granularity

**Goal.** Decide which regions become externalized `.a` units — the coarse/fine offload boundary. (A per-nest win is only real if it beats what a whole-program compiler does for free, so whole-program is a measured baseline here.)

**Deterministic driver.** `strategies.innermost` / `skip_taskloops` + a cost model over `whole_program_regions`.

**Agent integration.** State = the region graph + per-region measured/estimated times + island barriers. Action = move a region boundary (merge two regions = a fusion; split = a defuse) or accept the current partition (`stop`). Reward = end-to-end library performance. Same guardrail: a boundary move is a legal transformation, re-validated vs oracle.

**Substrate.** `whole_program_regions` (partition) + `region_to_standalone` (region -> emittable SDFG) + `prepare_regions` (one `Prepared` per region). The agent's boundary choices change which regions exist; these functions extract whatever results.

**Out.** The chosen region set -> per-region kernels.

### Phase 3 — per-kernel optimization

**Goal.** Optimize each offloaded kernel independently.

**Deterministic driver.** The arena sweep over the DaCe axes (opt-pipeline, codegen impl, vectorization, veclib, autopar) with staged screening (characterize -> static prune -> coordinate descent -> deep sweep on the top-N).

**Agent integration.** Per-nest agent. State = the kernel + its axis space + profiling feedback. Action = pick an axis config, OR propose a numpy rewrite of that kernel. Reward = the kernel's measured time. Every candidate validated bit-exact vs the nest oracle (`strict-ieee`, forked run). Replaces the brute-force sweep with guided search.

**Out.** Per-kernel winner -> linked into the `.a`.

### Phase 4 — re-fusion after tuning (feedback)

**Goal.** Given the per-kernel results, fuse/split further where it now pays — a kernel that vectorized well may be worth fusing with a neighbour; one that didn't may be worth splitting.

**Driver.** Loops back to Phase 1's transformations, applied to the post-optimization SDFG. This is why the arms are single-pair transformations *with inverses* — the pipeline revisits granularity with measured feedback instead of committing once.

**Agent integration.** Same as Phase 1's agent, but the state now carries the Phase-3 measured per-kernel times, so the reward signal is real, not estimated. Bounded rounds (stop when a round yields no improvement).

## Verification (the correctness foundation — precedes any agent)

The agent is only as safe as the transformations it is armed with. Each arm gets:

| layer | what |
|---|---|
| legality gate | `can_be_applied` — DaCe checks the deps; the agent never sees an illegal move |
| fuzz | random fusable SDFGs -> no crash + `validate()` + bit-exact vs un-fused |
| corpus | every legal site across npbench / polybench / TSVC |
| adversarial | WCR/reductions, loop-carried, aliasing, halo, mixed dtype, empty range |
| composition | random fusion SEQUENCES (what the agent actually does) |
| in-loop | every committed candidate re-validated vs oracle |

**Rule: the pass imports + uses the transformation.** One legality kernel, two front ends -> pass and transformation cannot diverge.

## How to use

### Fuse a specific loop pair (Phase 1 agent arm)

```python
from dace.transformation.interstate.fuse_loops import FuseLoops
FuseLoops.can_be_applied_to(sdfg, first=loop_a, second=loop_b)   # legal?
FuseLoops.apply_to(sdfg, first=loop_a, second=loop_b)            # fuse that pair
sdfg.apply_transformations_repeated(FuseLoops)                   # deterministic: fuse all
```

### Deterministic fuse-everything (Phase 1 baseline)

```python
from dace.transformation.passes.canonicalize.loop_fusion import LoopFusion
n = LoopFusion().apply_pass(sdfg, {})
```

### Split around barriers + partition (Phase 1/2 substrate)

```python
from nestforge.split_unsupported import (isolate_unsupported_library_nodes,
                                         whole_program_regions, region_to_standalone)
isolate_unsupported_library_nodes(sdfg)          # MPI etc -> own state
regions, islands = whole_program_regions(sdfg)   # externalizable regions + native islands
standalone = region_to_standalone(sdfg, regions[0], "r0")
```

### Prepare each region (Phase 2 -> 3)

```python
from nestforge.translate import prepare_regions
prepared, islands = prepare_regions(sdfg, "kern", out_dir, sizes={"N": 64})
```

### Tests (shared box: py12, `-n1`, 8GB cap)

```bash
pytest tests/transformation/fuse_loops_test.py -q -n1                       # dace: 45 FuseLoops tests
pytest tests/test_split_unsupported.py tests/test_whole_program.py -q -n1   # nest-forge
```

## Status

**DONE** — `FuseLoops` + pass delegation (dace `5b24a100c`); 45 FuseLoops tests (dace `35b2629ec`); `split_unsupported` + `prepare_regions` (nest-forge).

**OPEN** — fuzz harness; `MapFusionVertical`/`Horizontal` as arms + harness; fusion-ready canon variant; `measure_whole_program_lane` (Phase 2 measured baseline); deterministic lattice search; then the agents (P1/P2, P3 per-kernel, P4 feedback) + the free-form-numpy arm.
