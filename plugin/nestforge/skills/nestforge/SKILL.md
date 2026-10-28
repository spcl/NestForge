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
| `for<L>_<i>  t=lo:hi` | a sequential loop: its iterations run in order (a for or a while loop) |
| `if<L>_<i>` | a branch; its arms print as `when <cond>` / `else` |
| `state<L>_<i>` | one step of the program; collapsed it shows `doalls=<k>`, expanded its doalls |
| `doall<L>_<i>  [i=lo:hi]  reads=[..] writes=[..]` | a do-all loop: its iterations are proven independent, so they may run in any order or at once; `: <stmt>` lines are its body |
| `lib_<family><L>_<i>  LIBNODE <Kind>  reads=[..] writes=[..]` | a library call: `lib_blas` (Gemm, Gemv, Dot, ...), `lib_fft`, `lib_linalg`, `lib_reduce`, `lib_scan`, ...; it runs as a vendor or library call unless you expand it |

- `L` is the nesting level, `i` the position at that level. Labels are names, not positions you compute.
- `work=` counts operations, `depth=` the longest dependent chain, both as formulas in the sizes.
- `view_tree expand=[labels]` opens states and doalls; `collapse=[labels]` closes them. A doall whose body is a small
  program shows that program's states once expanded: those states are where `subgraph-fission` cuts.
- A state is a dataflow graph: inside it, doalls are ordered only by the data they share and may run concurrently.
  Going from one state to the next is a barrier: everything in the first state finishes, and its writes are
  visible, before the next starts. Loops and branches only ever join states. So doalls in different states never
  fuse (`doall-fusion` needs one state); fuse the for loops around them instead. Inside a doall body the barrier is
  per iteration; `subgraph-fission` widens it to the whole doall, every iteration of the first part before any of
  the second, with values passed between the parts becoming arrays.
- A kernel is a scope, not a row: the part of the program NestForge compiles as one unit. By default every
  top-level doall (with what it holds) is one kernel, run on the CPU with threads or on the GPU. A top-level for
  loop stays a loop on the host that launches the kernels inside it.
- Every move bumps the epoch and relabels the tree. Pass the epoch you read the labels at; a stale one is refused,
  so read the tree again after each move.

## Moves

`list_move_kinds` lists every kind with the rows it takes, in order, and the DaCe transformation it runs.
`list_moves` lists the moves legal right now as `kind` + `labels`; `apply_move` takes one back with the epoch.
An illegal move is refused with the reason and changes nothing.

- Fusion (`for-fusion`, `doall-fusion`): one loop or map instead of two; less memory traffic, intermediates
  become scalars. `doall-fusion` fuses a producer and its consumer (in place when the producer updates the array),
  recomputes what a consumer reads of a window (a stencil) on the fly, or fuses two independent maps over one range.
- `for-fission`: one loop per independent statement group.
- `doall-fission`: split per computation, one map per independent output group. Follow it with `doall-fusion` on the
  pieces that share data.
- `subgraph-fission`: split per state, after the named state of the map's body.
- Interchange (`interchange-*`): reorder a loop, map or guard with the one directly inside it.
- Tiling (`tile-doall`, `tile-for` and their `untile-` inverses): `apply_move` with `tile_sizes` turns a loop into
  a loop over tiles holding a loop over one tile's points, so a tile's working set fits a cache. A doall takes one
  size per dimension (`[32, 32]`), a for loop one size (`[64]`); a size need not divide the range. A tiled for loop
  keeps its order; interchange then blocks a nest. `untile-doall` takes the tile doall and its point doall back to
  one, `untile-for` the tile for loop.
- `collapse-doalls` merges a doall and the doall directly inside it into one multi-dimensional doall (one parallel
  space); `expand-doall` splits one back into nested doalls.
- `expand-library`: a `lib_<family>` row becomes the plain maps and tasklets that compute it (a gemm becomes a
  zero fill and a multiply map with a `+` reduction), in canonical form, so the moves above reach inside it. Expand
  a library call when fusing it with its neighbours saves memory traffic; leave it when the vendor call is faster.
  `list_moves` offers it only on calls with a vendor-free expansion.

## Flow

1. `get_cpf`, then `view_tree` to open the states and maps worth reading (largest `work=` first).
2. `list_move_kinds` once; then `list_moves`, `apply_move`. `apply_move` returns the performance: keep a move when
   the whole-program time drops, else apply its inverse when `list_moves` offers one.
3. `run_defaults stage=moves` applies the default fusion; `stage=finish` finishes after your own moves.
4. `finalize`, then `score` / `submit` with the `library` and `workspace_bytes` it returns.

Scopes (which doalls become one compiled kernel), GPU placement and the compiler configuration are not yours to
choose here: `get_performance` and `finalize` run their defaults.
