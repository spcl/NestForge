# Stage 2: Moves

prev: [1 Canonicalize](1-canonicalize.md) · next: [3 Scopes](3-scopes.md)

Stage 2 shapes the loop nests. The agent reads the structure tree and asks for moves; it never edits the graph.
Every move is an existing DaCe transformation, checked for legality before it applies. No tiling.

`describe()` prints the tree; its first line carries the epoch. `list_moves(kind)` returns the legal moves as
`{kind, labels, epoch}`, and `apply_move(kind, labels, epoch)` returns a `MoveResult` whose status is `applied`,
`illegal`, `not-implemented`, `not-found` or `stale`. Labels are the tree labels, unique across the whole program.
Any mutation starts a new epoch, so a move read at an old epoch returns `stale`. `metrics(label)` gives symbolic
work and depth (DaCe's `work_depth`).

| kind | labels | DaCe |
|---|---|---|
| `loop-fusion` | first, second loop | `LoopFusion` |
| `loop-fission` | loop | `LoopFission` |
| `map-fusion` | two maps | `MapFusionVertical`, else `MapFusionHorizontal` |
| `map-fission` | map | `MapFission` |
| `subgraph-fission` | map, body block | `SubgraphFission` |
| `interchange-map-map` | outer, inner map | `MapInterchange` |
| `interchange-loop-map` | loop, its map | `MoveLoopIntoMap` |
| `interchange-map-loop` | map, its loop | `MapLoopInterchange` |
| `interchange-if-loop` | if, loop | `MoveIfIntoLoop` |
| `interchange-loop-if` | loop, its if | `MoveLoopInvariantIfUp` |
| `interchange-loop-loop` | | not implemented |

| | |
|---|---|
| default | `default_moves()`: canonicalization's `fuse` stage and the stages after it |
| agent | moves, then `finish_moves()` runs the stages after `fuse` but `fuse_final`, keeping the chosen maps |
| code | `nestforge/stages/moves.py` |
