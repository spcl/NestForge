---
name: nestforge
description: "How to read the NestForge tree (for loops, doalls, library calls) and pick the moves that rewrite it: fusion, fission, interchange, tiling, library expansion and implementation; with the full tool set, also scopes and kernel code. Read it whenever NestForge tools are in your tool list, even if you only plan to call finalize."
when: "you drive the NestForge tools: read this before your first view_tree or apply_move."
applies: {explicit: true}
---

## The tree

`get_cpf` and `view_tree` print the program as a tree headed by `SDFG '<name>'  epoch=<n>`.

| row | is |
|---|---|
| `for<L>_<i>  t=lo:hi` | a sequential loop: its iterations run in order (a for or a while loop) |
| `if<L>_<i>` | a branch; its arms print as `when <cond>` / `else` |
| `state<L>_<i>` | one step of the program; collapsed it shows `doalls=<k>` (library calls counted), expanded its doalls |
| `doall<L>_<i>  [i=lo:hi]  reads=[..] writes=[..]` | a do-all loop: its iterations are proven independent, so they may run in any order or at once; `: <stmt>` lines are its body |
| `lib_<family><L>_<i>  LIBNODE <Kind>  reads=[..] writes=[..]` | a library call: `lib_blas` (Gemm, Gemv, Dot, ...), `lib_fft`, `lib_linalg`, `lib_reduce`, `lib_scan`, ...; a black-box doall: it runs as a vendor or library call unless you expand it |

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
  top-level doall (with what it holds), a library call among them, is one kernel, run on the CPU with threads or on
  the GPU. A top-level for
  loop stays a loop on the host that launches the kernels inside it.
- Every move bumps the epoch and relabels the tree. Pass the epoch you read the labels at; a stale one is refused,
  so read the tree again after each move.

## Moves

`list_move_kinds` lists every kind with the rows it takes, in order, and the DaCe transformation it runs.
`list_moves` lists the moves legal right now as `kind` + `labels` + `epoch`, with `tile_sizes` on a tiling move (the
sizes it was checked at) and `implementation` on a library-implementation move. `apply_move` takes one back as
listed, or with your own `tile_sizes`. An illegal move is refused with the reason and changes nothing, so a refusal
costs one call, not a broken program.

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
- Choosing tile sizes: a tile's data should fit the cache it is meant for (the innermost tile the closest cache, the
  next one the cache after it) and its innermost size should be a multiple of the machine's vector width, so no
  remainder loop runs per tile. Read the sizes from the machine you run on; `get_performance` after the move tells
  you whether it paid.
- `collapse-doalls` merges a doall and the doall directly inside it into one multi-dimensional doall (one parallel
  space); `expand-doall` splits one back into nested doalls.
- `expand-library`: a `lib_<family>` row becomes the plain maps and tasklets that compute it (a gemm becomes a
  zero fill and a multiply map with a `+` reduction), in canonical form, so the moves above reach inside it. Expand
  a library call when fusing it with its neighbours saves memory traffic; leave it when the vendor call is faster.
  `list_moves` offers it only on calls with a vendor-free expansion. A library call fuses with a doall only once
  expanded: unexpanded it is a black box.
- `set-library-implementation`: a `lib_<family>` row keeps its row and is built through the `implementation` you
  pass: loops (`pure`), the host vendor library (`OpenBLAS`, `FFTW3`) or, with a GPU target, the device one (`cuBLAS`,
  `hipBLAS`, ...). `list_moves` lists one move per implementation the row may take. A host library keeps its kernel
  on the CPU, a device library puts it on the GPU.

## Scopes and kernels (when your tool list has them)

Settle the moves first: a move or a new scope rebuilds the kernels it touches, and the sources you wrote for them
are dropped.

- `set_scope labels=[...]` makes one kernel of several top-level doalls of one state (library calls among them) or of
  a straight run of blocks. Group doalls that pass an array between them: inside one kernel you can keep that array in
  registers or cache instead of memory. `unset_scope` undoes it.
- `list_kernels` shows each kernel's device, its arguments and which kernels produce them; start with the largest
  share in `get_performance`.
- `read_kernel` gives the source NestForge rendered. `write_kernel` takes a replacement with the same entry
  (`extern "C"` name and parameters, in order) in `cpp` for a CPU kernel or the dialect `read_kernel` showed (`cuda`,
  `hip`) for a GPU one. It is built and run against the kernel's oracle, and linked only when it matches, so a wrong
  kernel never reaches `finalize`.
- `tune_kernel` sweeps compilers and flags for one kernel and links the fastest correct build; run it last, on the
  source you keep.
