---
name: agent-graph
description: The map an AI agent needs to drive nest-forge's 4-phase optimizer — the SDFG graph model (regions, states, nests), the read-only inspection API (control-flow tree + per-nest read/write sets), the fusion API (can-fuse-with-reason, apply, fission), how to externalize a nest and hand back a compiled C/C++/Fortran kernel, and the two ways a nest's kernel gets evaluated (agent-authored vs framework-swept). Read this first; each phase has its own deeper skill.
---

# Driving nest-forge as an agent

nest-forge lifts a program to a DaCe SDFG, then optimizes it in four phases. You (the agent) change the
graph through a small, value-preserving API and hand compiled kernels back. You can only change
*granularity* and *which kernel runs where* — never the result. Every move is legality-gated; a bad move
is refused, not mis-applied.

The **same API drives the deterministic (non-agent) path** — `deterministic_optimizers()` and an
`AgenticOptimizer` are both just `Optimizer` subclasses proposing over these calls. Nothing here is
agent-only.

## Three units you manipulate

| unit | what it is | phase | key type |
|---|---|---|---|
| **region** | a fusion unit: a maximal loop/map-nest (recursive — a `ConditionalBlock`'s branches are separate regions) | 1 | `LoopRegion` / `MapEntry` |
| **nest** | a region externalized as a library call | 2 | `ExternalCall` + `Boundary` |
| **variant** | a compiled kernel for a nest (a `(language, compiler, flags, fp-mode)` build) | 3 | `Cell` / prebuilt lib |

## Step 0 — see the graph (read-only, safe anytime)

```python
from nestforge.introspect import describe_graph, nest_reads_writes

print(describe_graph(sdfg))
```
```
SDFG 'two_maps'
  state 'MapState'  [fusion barrier]
    map map[i] over 0:N  PARALLEL  reads=['A', 'B'] writes=['T']
    map map[i] over 0:N  PARALLEL  reads=['T'] writes=['C']
```

Indentation is control-flow nesting. Each `state` is a **fusion barrier** (see next section). Each nest
line carries its parallel/sequential nature and its **read and write array sets** — what the loop-nest
actually does, without outlining it. `nest_reads_writes(state, node)` returns `(reads, writes)` for one
nest. After a nest is externalized, the same sets live on `Boundary.inputs` / `Boundary.outputs`.

## States are control-flow dependencies — the fusion barrier

You cannot fuse everything. **A DaCe `State` is a hard sequencing boundary**: everything in a state runs,
then the next state runs. Two map-nests fuse only if they live in the *same* state; the graph orders
states as a control-flow dependency, and the arms respect it:

- **map fusion** (vertical/horizontal) is **state-local** — never crosses a state.
- **loop fusion** may cross a state boundary via the CFG (two adjacent `LoopRegion`s in one region).

So to fuse compute that currently sits in two different states, the states must first merge. **State
fusion is not yet exposed in nest-forge** (roadmap below) — today, treat a cross-state pair as
un-fusable and reshape granularity another way (fission both to a common level, or fuse at the loop
level). `can_fuse` tells you exactly when a state barrier is the blocker.

## Phase 1 — change granularity (fuse / fission)

Ask before you act. `can_fuse` returns `"yes"` or a one-line reason — the **same gate** `apply_fusion`
uses, so `"yes"` is exactly an applicable move:

```python
from nestforge.fusion import can_fuse, enumerate_fusions, apply_fusion, fission_to_statements

can_fuse(sdfg, first_map, second_map)
# "yes"
# "nests are in different states -- a State boundary is a control-flow dependency ..."
# "intermediate 'T' is a live output (non-transient); fusing would drop a result -- cannot fuse."
# "different map ranges: 0:N vs 0:M -- horizontal fusion needs the same range."
# "blocked by FuseLoops: different iteration ranges, or a loop-carried dependency between the two."
```

Worked loop — fission to the finest granularity, then fuse back up to the granularity that measures
best:

```python
fission_to_statements(sdfg)              # explode to statement granularity
while True:
    moves = enumerate_fusions(sdfg)      # every LEGAL fuse right now (all three arms)
    if not moves:
        break
    apply_fusion(sdfg, moves[0])         # commit one; re-verifies legality itself
    # applying a move STALES every other move's node refs -- re-enumerate (loop does)
```

Each `FusionMove` has `.kind` (`fuse-loops` | `fuse-map-vertical` | `fuse-map-horizontal`), `.where`
(matched nodes), `.label()`. A named strategy is just a scripted policy over these moves —
`get_fusion_strategy("maximal-fusion")(sdfg)` is the "fuse everything legal" default. Deeper:
`skills/phase1-fusion`.

## Phase 2 — externalize (hand the nest to the next phase)

```python
from nestforge.offload import offload_candidates
from nestforge.pass_lower import lower_nests_to_external_call

for c in offload_candidates(sdfg, "skip-taskloops"):   # non-mutating preview
    print(c.label, "parallel" if c.parallel else "sequential")

lowered = lower_nests_to_external_call(sdfg, "skip-taskloops")  # [(ExternalCall, Boundary), ...]
```

Each nest becomes an `ExternalCall` node that still runs immediately (a numpy-reference fallback), so the
lowered SDFG validates and runs bit-exact. Externalizing changes *where* compute lives, never the result.
Deeper: `skills/phase2-offload`.

## Phase 3 — two ways a nest's kernel is evaluated

A nest's kernel is chosen one of two ways. Both produce an `Outcome` (correctness + median µs), so they
are directly comparable.

**Mode A — agent-authored.** You supply the whole kernel: a source file *and* the compiler + flags (or a
prebuilt library). The framework wires it in and measures it **once, as given** — no search.

- **Prefer a compiled language we can call**: C, C++, or Fortran, exposed as a single `extern "C"` entry.
  The kernel is invoked from the parent through a C ABI; there is **no python-callback path today**
  (compiled-lib-only — see roadmap).
- Compile your source to `lib<name>_nest.a` (static, one libomp — see `skills/phase2-offload`) or `.so`,
  then point the node at it:
  ```python
  ext.lib_path  = "/abs/path/libmykernel_nest.a"   # .a -> statically linked into the parent .so
  ext.symbol    = "mykernel"                        # the extern-C entry
  ext.abi_order = ["A", "B", "C", "N"]              # arg order the COMPILED signature uses
  sdfg.expand_library_nodes()
  ```
  **`abi_order` is the silent-break field** — it must match the order the compiled symbol expects, *not*
  the manifest/role order. Wrong order is an undiagnosed ABI corruption, not an error. Read the emitted
  signature; never trust a guessed order.

**Mode B — framework-swept.** You supply only the code (the numpy reference, or C); the framework sweeps
`language × compiler × flags × fp-mode × variant` and picks the winner:
```python
from nestforge.translate import prepare, emit_sources
from nestforge.arena import run_arena, build_winner_archive

prep = prepare(boundary, ext.name, out_dir)
emit_sources(prep, gen_dir, target="c")     # target: "numpy" | "c" | "cpp" | "fortran"
res  = run_arena(prep, boundary, c_source, build_dir, sizes={"N": 1 << 14})
win  = res.winners["ieee-strict"]            # best correct cell for that fp-mode
```

**The comparison is the point.** Record the cost of each: Mode A's cost is the agent's own budget (tokens
+ wall-clock to author the kernel) plus its resulting time; Mode B's cost is the arena's search time plus
its winner's time. Reporting both is how the framework answers *does an agent optimizing and generating
code itself beat our exhaustive sweep, and at what cost?* Deeper: `skills/phase3-optimize`.

## Measure in full-program context (per-nest, differential)

Optimize one nest, but **always measure the full program** — swap only the nest currently under
optimization; keep every other nest at its current implementation. The parent SDFG is unchanged; only the
`ExternalCall.lib_path` of the nest under test moves. This is the honest differential: it charges the
change for its real whole-program effect (including the offload boundary), not an isolated micro-time.
**optarena supplies the program input and driver**; the framework's job is to run that full program with
the one nest swapped.

## Phase 4 — feed measurements back

```python
from nestforge.feedback import run_feedback_loop     # re-fuse (Phase 1) from measured Outcomes
```
An improvement re-enters Phase 1 (change fusion) or Phase 2 (change offload granularity) and re-measures;
it stops when a round stops paying off. Deeper: `skills/phase4-feedback`.

## Correctness contract

- Every fuse/fission move is value-preserving and legality-gated; you cannot break correctness, only
  change granularity. Still validate the SDFG bit-exact against the numpy oracle after a move sequence,
  before it competes on speed.
- A Mode-A kernel must pass the same correctness gate as a swept one before its time counts.
- Run compiled kernels forked (a segfault must not kill the driver).

## Not yet exposed (roadmap — do not call)

- **State fusion across a barrier** — no transform surfaced; a cross-state pair is un-fusable today.
- **python-callback kernels** — the pipeline is compiled-lib-only (`extern "C"` + `ctypes`); a `.a` that
  trampolines back into python is a design goal, not a current path.
- **Region memoization in Phase 4** — feedback re-measures coarsely; per-region dirty-tracking (re-optimize
  only the regions a re-fuse changed) is planned.
