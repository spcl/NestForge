---
name: nestforge
description: "How to read the NestForge tree and which moves rewrite it: fusion, fission and interchange on the program's canonical parallel form."
when: "you drive the NestForge tools: read this before your first view_tree or apply_move."
applies: {explicit: true}
---

## The tree

`get_cpf` and `view_tree` print the program as a tree headed by `SDFG '<name>'  epoch=<n>`.

| row | is |
|---|---|
| `for<L>_<i>  t=lo:hi` / `while<L>_<i>` | a sequential loop |
| `if<L>_<i>` | a branch; its arms print as `when <cond>` / `else` |
| `state<L>_<i>` | one step of the program; collapsed it shows `kernels=<k>`, expanded its maps |
| `kernel<L>_<i>  [i=lo:hi]  reads=[..] writes=[..]` | a parallel map (a loop whose iterations are independent); `: <stmt>` lines are its body |

- `L` is the nesting level, `i` the position at that level. Labels are names, not positions you compute.
- `work=` counts operations, `depth=` the longest dependent chain, both as formulas in the sizes.
- `view_tree expand=[labels]` opens states and maps; `collapse=[labels]` closes them. A map whose body is a small
  program shows that program's states once expanded: those states are where `subgraph-fission` cuts.
- A state is a dataflow graph: inside it, maps are ordered only by the data they share and may run concurrently.
  Going from one state to the next is a barrier: everything in the first state finishes, and its writes are
  visible, before the next starts. Loops and branches only ever join states. So maps in different states never fuse
  (`map-fusion` needs one state); fuse the loops around them instead. Inside a map body the barrier is per
  iteration; `subgraph-fission` widens it to the whole map, every iteration of the first part before any of the
  second, with values passed between the parts becoming arrays.
- Every move bumps the epoch and relabels the tree. Pass the epoch you read the labels at; a stale one is refused,
  so read the tree again after each move.

## Moves

`list_move_kinds` lists every kind with the rows it takes, in order, and the DaCe transformation it runs.
`list_moves` lists the moves legal right now as `kind` + `labels`; `apply_move` takes one back with the epoch.
An illegal move is refused with the reason and changes nothing.

- Fusion (`loop-fusion`, `map-fusion`): one loop or map instead of two; less memory traffic, intermediates
  become scalars.
- `loop-fission`: one loop per independent statement group.
- `map-fission`: split per computation, one map per independent output group. Follow it with `map-fusion` on the
  pieces that share data.
- `subgraph-fission`: split per state, after the named state of the map's body.
- Interchange (`interchange-*`): reorder a loop, map or guard with the one directly inside it.

## Flow

1. `get_cpf`, then `view_tree` to open the states and maps worth reading (largest `work=` first).
2. `list_move_kinds` once; then `list_moves`, `apply_move`. `apply_move` returns the performance: keep a move when
   the whole-program time drops, else apply its inverse when `list_moves` offers one.
3. `run_defaults stage=moves` applies the default fusion; `stage=finish` finishes after your own moves.
4. `finalize`, then `score` / `submit` with the `library` and `workspace_bytes` it returns.

Scopes (which maps become one compiled kernel), GPU placement and the compiler configuration are not yours to
choose here: `get_performance` and `finalize` run their defaults.
