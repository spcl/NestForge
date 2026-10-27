# Stage 2: Moves

prev: [1 Canonicalize](1-canonicalize.md) · next: [3 Scopes](3-scopes.md)

Stage 2 shapes the loop nests. The agent reads the structure tree and asks for moves; it never edits the graph.
Every move is an existing DaCe transformation, checked for legality before it applies. No tiling.

`describe()` prints the tree; its first line carries the epoch. Every loop and state is a row; loop and map rows
carry `work=` and `depth=` from one pass of DaCe's work-depth analysis, and an innermost loop, or a map holding a
nested SDFG, carries `num_states=`. When that pass fails or takes over 60 s, the second line says why and the rows
carry no costs for that epoch. A state shows only `kernels=<count>` until `expand(labels, epoch)` opens it:
then its maps, nested maps and leaf-kernel bodies appear (an expanded map also shows its nested states). An
expansion lasts until the epoch changes. `list_moves(kind)` returns the legal moves as
`{kind, labels, epoch}`; map fusions list data-linked pairs and each map with the next map of its scope, never every
pair (CloudSC would list 20k), while `apply_move` still takes any pair, and `apply_move(kind, labels, epoch)` returns a `MoveResult` whose status is `applied`,
`illegal`, `not-found` or `stale`. Labels are the tree labels, unique across the whole program.
Every applied move is followed by `structural_cleanup`, the same as after canonicalization, so the states a move
leaves fusible are fused and every map body is canonical again (the bodies a fusion leaves side by side merge into
one nested SDFG); the agent never calls it. Any mutation starts a new epoch, so a move read at an old epoch
returns `stale`. `metrics(label)` gives the
symbolic work and depth of one kernel. `list_move_kinds()` (`moves.move_kinds`) is this table as records, the
agent's catalog.

| kind | labels | DaCe |
|---|---|---|
| `loop-fusion` | first, second loop | `LoopFusion` |
| `loop-fission` | loop | `LoopFission` |
| `map-fusion` | two maps | `MapFusionVertical` (in place when the first map updates the array), else `OTFMapFusion` (a window read is recomputed per element), else `MapFusionHorizontal` |
| `map-fission` | map | `MapFission`: per computation (independent statement groups, never per tasklet); `map-fusion` back what shares data |
| `subgraph-fission` | map, body block | `SubgraphFission`: per state, cut after the named block |
| `interchange-map-map` | outer, inner map | `MapInterchange` |
| `interchange-loop-map` | loop, its map | `MoveLoopIntoMap` |
| `interchange-map-loop` | map, its loop | `MapLoopInterchange` (splits a shared state around the map first) |
| `interchange-if-loop` | if, loop | `MoveIfIntoLoop` |
| `interchange-loop-if` | loop, its if | `MoveLoopInvariantIfUp` |

| | |
|---|---|
| default | `default_moves()`: canonicalization's `fuse` stage and the stages after it |
| agent | moves, then `finish_moves()` runs the stages after `fuse` but `fuse_final`, keeping the chosen maps |
| score | `guided_moves()`: the legal move with the lowest [static score](../score.md) change, while it is negative; `score_moves()` ranks without applying |
| code | `nestforge/stages/moves.py` |
