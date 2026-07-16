# NestForge Agentic Optimizer — Design

## Goal

Let an agent decide two things NestForge currently decides with fixed heuristics:

1. **Loop-offloading granularity** — how coarse or fine the externalized loop nests are.
2. **Per-nest optimization** — which optimization each nest gets before it is compiled into the `.a`.

NestForge is a comparison framework, so the agent is not a replacement bolted on top — it is another
set of *arms* measured against the existing heuristic and brute-force arms, and against a free-form
agent that rewrites the whole numpy directly. The framework already knows how to validate a candidate
against a numerical oracle and time it; the agent just proposes candidates.

Correctness is non-negotiable. Every candidate — whatever produced it — is validated numerically
against the un-optimized oracle (bit-close) before it is allowed to compete on speed. The structural
path additionally is correct *by construction* (semantics-preserving transforms).

## The premise: granularity is the fusion state

The offload granularity of a program *is* its fusion state. A fused pair of loop nests is one coarser
kernel — more optimization scope, fewer call boundaries; the unfused pair is two finer kernels — more
independent parallelism decisions, smaller units. So "let the agent choose granularity" means "let the
agent choose how much to fuse."

For the agent to have real freedom, the SDFG it starts from must be in a **maximal-fusability** form:
as few states as possible and as many adjacent, legally-fusable loops as possible — but with the
fusions *not yet committed*. The fusions are the agent's moves; the canonical form is the board it
plays on.

### Maximal-fusability canonicalization (runs before the agent)

A canon *variant* that is fusion-ready, not fusion-baked (distinct from the perf canon that bakes
fusion in):

- **`StateFusion`** (+ `state_fusion_with_happens_before`) — collapse sequential states into one
  dataflow state wherever legal. Fewer states ⇒ more loops co-located in one state ⇒ more adjacency
  for map fusion. This is the "as little states as possible" the design needs.
- **`LoopToMap`** — expose every parallel loop as a map, so map fusion (not just loop fusion) applies.
- **Do not apply `MapFusion`/`LoopFusion`** here — those are the agent's actions. Everything that
  *could* fuse must be adjacent and legal-to-fuse, so the fusability graph is rich.
- **Hard barriers stay barriers.** The split-around-unsupported work (`nestforge.split_unsupported`)
  already isolates MPI/pblas/sparse/unsupported nodes into their own states; those island states are
  fusion barriers the agent may never cross. `whole_program_regions` gives the agent the legal fusion
  arena (each region) and the immovable islands between them.

Output: an SDFG where every legal fusion is an available action, minimal states, no premature
commitment — plus the region/island partition that bounds where fusion is allowed.

## The arms (structural action space)

The agent, when it operates *inside* DaCe, is armed with **single-application transformations** — each
fuses one specific pair — not the whole-program passes. Passes are the deterministic baseline; the
transformation variants are the agent's fine-grained moves (one legal site at a time, each with
`can_be_applied` + `apply`, reversible, composable).

| Arm | DaCe transformation | Meaning | Status |
|---|---|---|---|
| Map fusion, vertical | `dataflow/map_fusion_vertical.py` `MapFusionVertical` | fuse producer→consumer maps (output of A feeds B) | exists |
| Map fusion, horizontal | `dataflow/map_fusion_horizontal.py` (`MapFusionHorizontal`) | fuse two independent maps over the same range (siblings) | exists |
| Loop fusion | **`FuseLoops` — NEW** (see below) | fuse one adjacent same-range sequential loop pair | **proposed DaCe change** |
| (recompute variant) | `dataflow/otf_map_fusion.py` `OTFMapFusion` | fuse by recomputation when a shared transient can't be kept — the recompute/granularity lever | exists |
| (inverse: defuse) | `MapExpansion` / loop splitting (`LoopFission`) | back out of a fusion (dead-end recovery) | exists |
| Stop | — | commit the current granularity | — |

Each corresponding **pass** (`MapFusion` applied repeatedly, `LoopFusion`) is the *deterministic*
optimizer that fuses everything it legally can — the baseline the agent is measured against, and the
warm-start it improves on.

### `FuseLoops` transformation (proposed DaCe change — plan, not yet built)

DaCe ships loop fusion only as a **pass** (`transformation/passes/canonicalize/loop_fusion.py`,
`LoopFusion`) that fuses *every* legal consecutive same-range sequential loop pair. The agent needs a
**transformation** that fuses *one named pair*, so it can drive granularity a step at a time. The pass
already contains the complete per-pair kernel — `_same_iteration_space`, `_single_compute_state`,
`_is_doall` (never serialize a DOALL loop — leave it to `LoopToMap`), `_fusion_legal` (flow / anti /
output dependence rules via `BreakAntiDependence._dep_class`, refusing read-ahead / read-behind /
divergent output writes and any unresolved subset), and `_merge`. `FuseLoops` is a straight extraction:

- A `MultiStateTransformation` (loops are CFG blocks) with `first = PatternNode(LoopRegion)`,
  `second = PatternNode(LoopRegion)`, `expressions() = [node_path_graph(first, second)]` — matches an
  adjacent loop pair, exactly as `StateFusion` matches an adjacent state pair.
- `can_be_applied` = the pass's per-pair gates: single pure sequencing edge (no assignments, trivial
  condition), `_same_iteration_space`, both single-compute-state, neither DOALL, `_fusion_legal`.
- `apply` = the pass's `_merge`.
- **One legality kernel, two front ends.** The static helpers move to module scope (or onto the
  transformation as static methods); the `LoopFusion` pass is re-expressed to iterate
  `FuseLoops.can_be_applied_to` + `apply`. This is the critical correctness property — the pass and the
  transformation can never diverge, because they *are* the same code. No behavior change to the pass.

Open design question for review before touching DaCe: whether the shared kernel lives as module
functions in the pass file (transformation imports them) or moves onto `FuseLoops` (pass imports the
transformation). The latter makes the transformation the source of truth; the former is a smaller diff.

The recompute variant (`OTFMapFusion`) matters: vertical/horizontal fusion sometimes requires
recomputing a value instead of materializing a transient, and that recompute-vs-materialize choice is
itself a granularity decision the agent should own.

## Correctness and the test burden

This is the load-bearing part. The transforms must **never crash on valid input and never produce an
incorrect program.** The guarantee comes from a dedicated verification harness, not from trust:

1. **Legality gate** — apply only where `can_be_applied` returns true. DaCe's transforms already check
   the data dependences (a vertical fusion that would reorder a read past a write is rejected). The
   agent never sees an illegal move.
2. **Property / fuzz tests** — generate random SDFGs with fusable structure (parametrized over rank,
   iteration bounds, WCR/no-WCR, strided/unit stride, conditional writes, transient vs external
   buffers), apply each transform, and assert: (a) no exception, (b) `validate()` passes, (c) the
   pre- and post-transform SDFGs produce bit-close results on random inputs (the oracle equivalence).
3. **Corpus tests** — apply each transform at *every* legal site across npbench / polybench / TSVC,
   validate numerics each time. This is where real-world footguns surface.
4. **Adversarial cases** — the known fusion hazards, each a fixed test: reductions/WCR, loop-carried
   dependences, in-place aliasing, boundary/halo access, mixed dtypes, size-1 and empty ranges.
5. **Composition tests** — apply random *sequences* of fusions (what the agent actually does) and
   validate the end SDFG, since a fusion can invalidate a later one.
6. **Belt-and-suspenders in the loop** — every candidate granularity the agent commits to is *also*
   validated numerically against the whole-program oracle, so even a transform bug cannot ship a wrong
   library.

The matrix is large but mechanical; generating it is a good use of a multi-agent workflow (one agent
per hazard class producing the cases, an adversarial verifier confirming each actually exercises the
hazard).

## SDFG transforms vs. rewriting the numpy

The open question — should the agent fuse on the SDFG, or rewrite the original numpy directly?

**Recommendation: structural fusion is done on the SDFG; the numpy is the oracle, not the fusion
surface.** An LLM rewriting numpy to "fuse two loops" can silently break a loop-carried dependence or
turn a reduction into last-write-wins — and nothing structural catches it except the numerical oracle,
after the fact. For a compiler, correctness-by-construction (DaCe checks the dependence before fusing)
is worth far more than the LLM's fluency at rewriting array code.

The numpy earns three distinct roles instead:

- **Oracle** — the un-optimized `sdfg_to_numpy` output is the immutable ground truth every candidate
  validates against.
- **Rendering** — the readable form of each nest the per-nest agent reasons about.
- **A comparison arm** — a *free-form* agent that optimizes the complete numpy end-to-end (see below),
  measured against the structural path rather than trusted as the primary optimizer.

## Two agent roles

- **Granularity agent (global).** Sees the fusability graph and picks fusions/splits to set the nest
  granularity. Reward is the end-to-end performance of the resulting library. The current heuristics
  (`strategies.innermost`, `strategies.skip_taskloops`) become its *warm-start default policy* — the
  agent's job is to beat them, not start from scratch.
- **Per-nest optimizer agent (local).** For each chosen nest, navigate the DaCe optimization axes
  (vectorization, veclib, tiling, codegen impl, autopar) that the arena currently brute-forces — or
  propose a numpy rewrite of that nest, validated against the nest oracle. This replaces the exhaustive
  sweep with guided search; the staged screening in the main plan (characterize → prune → descend →
  deep-sweep) is the reward-cost discipline it inherits.

## Comparison arms (what NestForge measures per program)

1. **Heuristic granularity + brute-force per-nest sweep** — the current arena (baseline).
2. **Agentic granularity + agentic per-nest** — the structural agent path.
3. **Free-form whole-numpy agent** — hand the agent the *complete* numpy input and let it optimize the
   whole thing directly; validate numerically; compile.

All three go through the same validate-then-time gate and the same `.a` emission. The comparison
answers two questions the framework exists to answer: does an agent beat the heuristics, and does
structured (correct-by-construction) optimization beat free-form LLM rewriting?

## Agent loop mechanics

- **State** presented to the agent: a compact structural summary — the list of nests, the fusability
  graph (which pairs can fuse V / H / loop, their iteration spaces, sizes, data dependences), the
  island barriers, and profiling feedback (measured per-nest times) — not the raw SDFG.
- **Actions**: apply a named transform at a legal site; split; or stop. Exposed as tool calls; each
  returns the updated summary.
- **Reward**: measured performance of the compiled candidate. Real measurement is expensive, so use a
  cost-model proxy for most steps with periodic real measurement (same staged-screening discipline as
  the vectorization axis), and always record the action trace + seeds for reproducibility.
- **Agent implementation**: an LLM agent fits NestForge/OptArena's existing agent infrastructure (wire
  the agent through the scripted/stubbed interface — real inference is never run on the shared dev
  box); the transforms are its tools and the oracle is its verifier. An RL policy over the transform
  lattice is a possible later substitution once the reward signal is cheap and dense.

## Integration and reuse

The pieces already exist; the agent orchestrates them:

```
canon(fusion-ready)                      # StateFusion + LoopToMap, no fusion committed
  → whole_program_regions                # legal fusion arena + island barriers  [split_unsupported]
  → [granularity agent fuses/splits]     # the new agent, arms = the 3 transforms
  → region_to_standalone per nest        # extract each chosen nest             [split_unsupported]
  → [per-nest agent optimizes]           # axes or verified numpy rewrite
  → prepare / emit_sources → compile     # translate.py + optarena numpyto → C
  → link into .a                         # the arena
```

`region_to_standalone` + `prepare_regions` (the split-around-unsupported work) are exactly the
granularity substrate: the agent's fusion choices change which regions exist, and those functions
extract and prepare whatever granularity results, with cross-boundary transients already handled.

## Risks

- **Transform correctness** — mitigated by the verification harness above; this is the biggest cost.
- **Reward measurement cost** — cost-model proxy + selective real measurement; anything capped is
  logged, never silently dropped.
- **Search-space explosion** — the fusion lattice is large; warm-start from the heuristics, prune with
  the cost model, bound the agent's step budget.
- **Fusion dead-ends** — a fusion can block a better one; the agent needs split/defuse to back out.
- **Reproducibility** — record the action trace and seeds so a winning granularity is replayable.
- **No real inference on the dev box** — the agent runs scripted/stubbed in tests; real runs happen
  off-box.

## Phasing

The optimizer is a pipeline with a feedback loop. Each phase can be driven by a **deterministic**
optimizer or by an **agent** (inside DaCe via the transformations, or on the numpy input) — and the two
are measured against each other.

- **Phase 1 — optimal loop fusion + loop offloading.** Reach the maximal-fusability form and perform
  loop fusion + the loop-offloading mechanism. Driven either by the agent directly in DaCe (the
  transformation arms: `FuseLoops`, `MapFusionVertical`, `MapFusionHorizontal`) or by the agent
  rewriting the numpy input. The deterministic baseline is the `LoopFusion` / `MapFusion` passes (fuse
  everything legal). *Prerequisite:* the `FuseLoops` transformation + the verification harness — the
  correctness foundation and the bulk of the test burden — must exist before an agent is allowed to
  drive fusion.
- **Phase 2 — offloading granularity.** Decide how the fused program is partitioned into offloaded
  kernels (which regions become externalized `.a` units). Deterministic optimizer (the current
  `strategies` heuristics + a cost model) or the agent. Runs on the split-around-unsupported substrate
  (`whole_program_regions` / `region_to_standalone`) — the fusion state from Phase 1 defines the
  regions this phase chooses among.
- **Phase 3 — per-kernel optimization.** Optimize each offloaded kernel (the DaCe axes —
  vectorization / veclib / tiling / codegen / autopar — or a verified numpy rewrite). Deterministic
  (the arena sweep / staged screening) or the per-nest agent.
- **Phase 4 — re-fusion after per-kernel optimization (optional feedback).** Given the per-kernel
  results, fuse further where it now pays (a kernel that vectorized well may be worth fusing with a
  neighbour; one that didn't may be worth splitting). Loops back to Phase 1's transformations on the
  post-optimization SDFG. This is why the arms are single-pair transformations with inverses — the
  pipeline revisits granularity with measured feedback rather than committing once.

Every phase is validated against the numerical oracle and measured against the previous, so the value
of each layer — and of agent vs. deterministic, structural vs. free-form-numpy — is a number, not an
assumption.

### Build order

The dependency that gates everything is Phase 1's correctness foundation:

1. `FuseLoops` transformation (the proposed DaCe change) — extracted from the `LoopFusion` pass,
   sharing one legality kernel.
2. The verification harness for `FuseLoops` + `MapFusionVertical` + `MapFusionHorizontal` (legality,
   fuzz, corpus, adversarial, composition) — *no agent*.
3. Deterministic granularity search over the transformation lattice with a cost-model reward.
4. The agent drives the lattice (Phase 1/2), warm-started by the heuristics.
5. Per-nest agent (Phase 3); free-form whole-numpy arm; Phase-4 feedback.
