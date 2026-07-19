# Agentic Optimizer

## High-level design

Granularity == fusion state. DaCe first normalizes + fuses maximally (deterministic). Then an agent adjusts granularity (fuse/fission), optimizes each kernel by hand, and iterates. Correctness is a hard gate: every candidate validates bit-exact vs the un-fused numpy oracle before it competes on speed.

```
P1  dace normalize + max-fuse         (deterministic, NO agent)   -> baseline SDFG
P2  agent: dace transforms            (fuse / fission / manipulate) -> granularity
P3  agent: optimize each kernel       (from cpp | fortran | python) -> emits .cpp -> compile
P4  agent: request different fuse/fission -> back to P3 for the changed loopnests only
        ^________________________________________________________________|
```

P1 is fixed. P2–P4 are the agent. The comparison baseline is P1 + heuristic offload + brute-force sweep; the agent's job is to beat it, and the delta is measured.

## Phases

### Phase 1 — normalize + fuse (deterministic, no agent)

**What.** Loop normalization + `LoopToMap` + `LoopFusion` / `MapFusion` — fuse everything legal, via DaCe's default heuristics. No agent.

**Out.** A maximally-fused baseline SDFG — the starting granularity the agent adjusts from.

**Barriers.** An unsupported libnode (MPI/pblas, sparse, no emitter) is a hard fusion barrier — `nestforge.split_unsupported` isolates it into its own state; fusion never crosses it.

### Phase 2 — agent adjusts granularity (dace transformations)

**What.** The agent runs DaCe transformations to change granularity: **fuse** (`FuseLoops`, `MapFusionVertical`, `MapFusionHorizontal`), **fission** (`LoopFission`, `MapExpansion`), or direct SDFG edits — splitting coarse nests apart or re-fusing differently from the P1 max-fused baseline.

**Why fission matters here.** P1 already fused maximally, so the agent's main lever is fission — carve the coarse program into the kernel granularity that actually optimizes best. Fuse is for cases the heuristics missed.

**Agent integration.**
- *State*: nest + fusability/fission graph (which pairs can fuse V/H/loop, which nests can split, ranges, deps), island barriers.
- *Actions*: `FuseLoops.apply_to` / `MapFusion*` / `LoopFission` / `MapExpansion` / direct SDFG edit / `stop`.
- *Guardrail*: every transform is legality-gated (`can_be_applied`); the SDFG re-validates bit-exact vs oracle. The agent can only change granularity, never correctness.
- *Reward*: end-to-end library performance (via P3 timings), cost-model proxy between real measurements.

**Out.** The chosen loop-nest granularity -> one kernel per nest (via `whole_program_regions` / `region_to_standalone` / `prepare_regions`).

### Phase 3 — agent optimizes each kernel -> .cpp

**What.** For each kernel, the agent optimizes it starting from whichever representation it wants — the kernel is rendered as **C++**, **Fortran**, and **Python/numpy** — and produces an optimized **`.cpp`** to compile.

**Representations available** (nest-forge already emits all three):
- C++ — DaCe codegen of the nest.
- Fortran / C — optarena `numpyto` from the emitted numpy.
- Python/numpy — `sdfg_to_numpy` (also the oracle).

**Agent integration.**
- *State*: the kernel in the three representations + profiling feedback + shapes/dtypes.
- *Action*: rewrite/optimize -> emit a `.cpp`. Free-form (the agent hand-writes optimized C++), not just axis selection.
- *Guardrail*: the emitted `.cpp` is compiled, run **forked**, and validated bit-exact vs the nest oracle before it counts. A wrong rewrite is rejected, never shipped.
- *Reward*: the kernel's measured time. The deterministic baseline for comparison is the arena sweep over the DaCe axes.

**Out.** One optimized `.cpp` per kernel -> compiled -> linked into the `.a`.

### Phase 4 — agent re-fissions / re-fuses, back to Phase 3

**What.** Given the P3 kernel results, the agent requests a *different* fuse/fission (Phase-2 transforms on the post-optimization SDFG), then re-runs **Phase 3 for the changed loopnests only** — incremental, not a full redo.

**Agent integration.** Same surface as Phase 2, but the state now carries real P3 measured per-kernel times, so the fuse/fission decision is driven by measurement, not estimate — only nests touched by the new fuse/fission are re-optimized in P3. Bounded rounds; stop when a round yields no improvement. This is why the arms are single-pair transformations *with inverses*.

## The agent contract (P2–P4)

| | what |
|---|---|
| **State** | compact structural summary — NOT the raw SDFG. Nests + fuse/fission graph + island barriers + measured/estimated times. In P3 also the kernel's C++/Fortran/numpy renderings. |
| **Actions** | P2/P4: a validated DaCe transformation (fuse/fission/manipulate). P3: emit an optimized `.cpp`. Plus `stop`. |
| **Guardrail** | legality-gated + re-validated bit-exact vs oracle. The agent can only change speed, never correctness. |
| **Reward** | measured performance; cost-model proxy between real runs. Warm start from the heuristics. |

Agent kind: LLM with the transformations / codegen as tools and the oracle as verifier (fits optarena's agent infra). **Never run real inference on the dev box — scripted/stub only.**

## Verification (precedes any agent)

The agent is only as safe as the transformations + the compile-and-validate gate. Each fusion/fission arm gets:

| layer | what |
|---|---|
| legality gate | `can_be_applied` — DaCe checks the deps; the agent never sees an illegal move |
| fuzz | random fusable SDFGs -> no crash + `validate()` + bit-exact vs un-fused |
| corpus | every legal site across npbench / polybench / TSVC |
| adversarial | WCR/reductions, loop-carried, aliasing, halo, mixed dtype, empty range |
| composition | random fuse/fission SEQUENCES (what the agent actually does) |
| in-loop | every committed candidate re-validated vs oracle; every P3 `.cpp` compiled + forked + bit-exact |

**Rule: the pass imports + uses the transformation.** One legality kernel, two front ends -> pass and transformation cannot diverge.

## Implemented phase API

Each phase is now an explicit *inspect → commit* module, re-exported from the `nestforge` package top
level, with a skill file (`skills/phase*/SKILL.md`) teaching an agent to drive it. Same guardrail
throughout — every move legality-gated + re-validated bit-exact, so a phase changes only speed.

| Phase | Module | Inspect (non-mutating) | Commit | Skill | Tests |
|---|---|---|---|---|---|
| P1 fusion granularity | `nestforge.fusion` | `enumerate_fusions` | `apply_fusion` / `fission_to_statements` | `phase1-fusion` | `tests/test_fusion_strategy.py` |
| P2 offload granularity | `nestforge.offload` | `offload_candidates` | `lower_nests_to_external_call` | `phase2-offload` | `tests/test_offload_granularity.py` |
| P3 per-nest optimization | `nestforge.optimize` | `optimization_choices` | `optimize(nest, knobs)` | `phase3-optimize` | `tests/test_optimize.py` |
| P4 measurement feedback | `nestforge.feedback` | `best_outcome` / `improved` | `run_feedback_loop` | `phase4-feedback` | `tests/test_feedback.py` |

The knob bundle P3 optimizes with IS an `Optimizer` (`nestforge.optimizers`) — "each variant is an
optimizer, the agent is one more"; the P4 per-nest inner loop is `run_agent_loop`. The invariant tying
P1→P2: externalize *before* deciding offload (a nest becomes an `ExternalCall` first, each tool decides
offloadability second). The `How to use` snippets below are the lower-level primitives these facades sit on.

## How to use

### P1 deterministic fuse-everything

```python
from dace.transformation.passes.canonicalize.loop_fusion import LoopFusion
n = LoopFusion().apply_pass(sdfg, {})   # + LoopToMap / MapFusion in the canon pipeline
```

### P2/P4 agent transforms — fuse / fission a specific pair

```python
from dace.transformation.interstate.fuse_loops import FuseLoops
FuseLoops.can_be_applied_to(sdfg, first=a, second=b)   # legal?
FuseLoops.apply_to(sdfg, first=a, second=b)            # fuse
# fission: dace.transformation.passes.loop_fission / MapExpansion
```

### Split around barriers + partition into kernels (P2 -> P3)

```python
from nestforge.split_unsupported import (isolate_unsupported_library_nodes,
                                         whole_program_regions, region_to_standalone)
isolate_unsupported_library_nodes(sdfg)          # MPI etc -> own state
regions, islands = whole_program_regions(sdfg)   # externalizable kernels + native islands
standalone = region_to_standalone(sdfg, regions[0], "r0")
```

### Render each kernel (C++ / Fortran / numpy) for the P3 agent

```python
from nestforge.translate import prepare_regions, emit_sources
prepared, islands = prepare_regions(sdfg, "kern", out_dir, sizes={"N": 64})
emit_sources(prepared[0], out_dir, target="c")        # C / Fortran via numpyto
# C++ via dace codegen; numpy via sdfg_to_numpy (also the oracle)
```

### Tests (shared box: py12, `-n1`, 8GB cap)

```bash
pytest tests/transformation/fuse_loops_test.py -q -n1                       # dace: 45 FuseLoops tests
pytest tests/test_split_unsupported.py tests/test_whole_program.py -q -n1   # nest-forge
```

## Status

**DONE** — `FuseLoops` + pass delegation (dace `5b24a100c`); 45 FuseLoops tests (dace `35b2629ec`); `split_unsupported` + `prepare_regions` (nest-forge). **All four phases now have an explicit inspect→commit API + skill** (`nestforge.fusion` / `offload` / `optimize` / `feedback`; see the table above), each with a bit-exact test suite; the deterministic optimizers + `NoOpAgent` / `StubAgent` + `run_agent_loop` (`nestforge.optimizers`); `WholeProgramOptimizer` baseline lane.

**OPEN** — fuzz harness; the P3 render-all-three-representations + compile-and-validate hand-written `.cpp` gate (the facade covers axis-selection recipes, not free-form agent-authored C++ yet); a *real* (non-stub) agent behind the `AgenticOptimizer` contract for P2 granularity / P4 incremental feedback.
