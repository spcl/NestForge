# Phase 1: Shape Kernels

prev: [0 Normalize](0-normalize.md) · next: [2 Define Scopes](2-define-scopes.md)

Phase 1 decides which maps and loops fuse and which split. The scheduling agent never edits graph
nodes; it reads two views and requests moves.

- **Structure.** `Session.describe()` prints control-flow regions, states and nests with their
  iteration domains and read/write sets. `Session.kernel_source` renders a nest as NumPy, C++ or
  Fortran.
- **Cost.** `describe(metrics=True)` adds symbolic work, depth and operational intensity (OI) per
  scope. OI divides work by the bytes a scope moves under a simple cache model: a map caches
  perfectly, a loop caches nothing.

Each move is legal only when its check accepts it: a DaCe transformation's `can_be_applied`, the
per-region match of the DaCe pass it names, or, for `interchange-map-loop`, NestForge's own check. `list_moves(kind)` returns the legal moves as `{kind, labels, epoch}`. `apply_move(kind,
labels, epoch)` applies one and returns a `MoveResult` whose status is `applied`, `illegal`,
`not-implemented`, `not-found` or `stale`. Labels are the tree labels `describe()` prints, and its
first line shows the epoch.

| kind | labels | DaCe |
|---|---|---|
| `loop-fusion` | first, second loop | `LoopFusion` |
| `loop-fission` | loop | `LoopFission`'s split of one loop into independent statement groups |
| `map-fusion` | two maps | `MapFusionVertical` through an intermediate, else `MapFusionHorizontal` |
| `map-fission` | map | `MapFission` on a map whose body is one nested SDFG |
| `subgraph-fission` | map, body block | nest the body before and after the block, then `MapFission`: two maps |
| `interchange-map-map` | outer, inner map | `MapInterchange` |
| `interchange-loop-map` | loop, its one map | `MoveLoopIntoMap`: the map becomes outer |
| `interchange-map-loop` | map, its one loop | NestForge: the loop becomes outer |
| `interchange-if-loop` | if, the loop it guards | `MoveIfIntoLoop`: the guard moves into the body |
| `interchange-loop-if` | loop, its one if | `MoveLoopInvariantIfUp`: an invariant guard moves out |
| `interchange-loop-loop` | | not implemented |

`subgraph-fission` widens every transient the two halves share by the map's range. `interchange-map-loop`
is legal when the map body is exactly the loop, its bounds do not vary across map iterations, and no transient
or symbol inside the body carries a value between loop iterations; the refusal names which condition failed.
A guard always moves into a loop; it moves out only when its condition is loop-invariant.

When no move fuses two regions that should run as one, phase 2's `define_scope` makes them one kernel
instead, and the kernel written in phase 4 fuses them.

`fission_all` splits the whole program to statement granularity, loops included. The id-based calls
stay: `list_fusions` / `fuse`, `list_fissions` / `fission` and state fusion via `fuse_regions`.

| | |
|---|---|
| default | `full_fusion(sdfg, targets)`: canonicalization's `fuse` stage and the stages after it |
| hand-chosen | apply moves, then `finish_schedule(sdfg, targets)` |
| code | `nestforge/phases/schedule.py`, `nestforge/ir/introspect.py` |
