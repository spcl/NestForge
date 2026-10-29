---
name: nestforge
description: "How to read the NestForge tree (for loops, doalls, library calls) and pick the moves that rewrite it: fusion, fission, interchange, tiling, library expansion and implementation; with the full tool set, also scopes and kernel code. Read it whenever NestForge tools are in your tool list, even if you only plan to call finalize."
when: "you drive the NestForge tools: read this before your first view_tree or apply_move."
applies: {explicit: true}
---

## The tree

`get_cpf` and `view_tree` print the program as a tree headed by `SDFG '<name>'`.

| row | is |
|---|---|
| `for<n>  t=lo:hi` | a sequential loop: its iterations run in order (a for or a while loop) |
| `if<n>` | a branch; its arms print as `when <cond>` / `else` |
| `state<n>` | one step of the program; collapsed it shows `doalls=<k>` (library calls counted), expanded its doalls |
| `doall<n>  [i=lo:hi]  reads=[..] writes=[..]` | a do-all loop: its iterations are proven independent, so they may run in any order or at once; `: <stmt>` lines are its body |
| `lib_<family><n>  LIBNODE <Kind>  reads=[..] writes=[..]` | a library call: `lib_blas` (Gemm, Gemv, Dot, ...), `lib_fft`, `lib_linalg`, `lib_reduce`, `lib_scan`, ...; a black-box doall: it runs as a vendor or library call unless you expand it |

- A label is a row's name: it stays on the row while the row lives. Labels are names, not positions you compute.
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
  the GPU. A top-level for loop stays a loop on the host that launches the kernels inside it.
- A move retires the labels of the rows it took and names the rows it made afresh; every other row keeps its label.
  A retired label is refused as unknown, so read the tree again after each move.

## Moves

`list_move_kinds` lists every kind with the rows it takes, in order, and the DaCe transformation it runs.
`list_moves` lists the moves legal right now as `kind` + `labels`, with `tile_sizes` on a tiling move (the sizes it
was checked at) and `implementation` on a library-implementation move. `apply_move` takes one back as listed, or with
your own `tile_sizes`. An illegal move is refused with the reason and changes nothing, so a refusal costs one call, not
a broken program. Every move keeps the program's results: it changes how the loops run, never what they compute.

Each move below: the rows it takes, the transformation, the program before and after, and when it pays.

### Fusion: fewer loops, less memory traffic

**`for-fusion`** `[for, for]`, LoopFusion. Two adjacent loops over the same range become one.
```
for i in 0:N: A[i] = f(X[i])          for i in 0:N:
for i in 0:N: B[i] = g(A[i])    ->        A[i] = f(X[i]); B[i] = g(A[i])
```
Pays when the second loop reads what the first wrote: the value is still in a register or cache. Refused when an
iteration of the fused loop would read a value a later iteration of the first loop writes.

**`doall-fusion`** `[doall, doall]` of one state, MapFusionVertical / OTFMapFusion / MapFusionHorizontal.
```
doall i: T[i] = A[i] * 2               doall i:
doall i: C[i] = T[i] + 1        ->         t = A[i] * 2; C[i] = t + 1     (T is gone)
```
A producer and its consumer: the intermediate array becomes a scalar (kept as the array when the producer updates it
in place). A consumer reading a window of the producer (a stencil) recomputes those values on the fly. Two independent
doalls over one range fuse side by side. Pays by removing a whole array's write and read; a recomputing fusion pays
only while the recomputation is cheaper than the traffic.

### Fission: split a loop, one piece per computation

**`for-fission`** `[for]`, LoopFission. A loop of independent statement groups becomes one loop per group.
```
for i: A[i] = f(i); B[i] = g(i)   ->   for i: A[i] = f(i)
                                       for i: B[i] = g(i)
```
Pays when the groups want different orders, or when one group is parallel and the other is not.

**`doall-fission`** `[doall]`, MapFission. One doall per independent output group.
```
doall i: A[i] = f(i); B[i] = g(A[i])   ->   doall i: A[i] = f(i)
                                            doall i: B[i] = g(A[i])
```
Usually followed by `doall-fusion` of the pieces that share data, with different neighbors.

**`subgraph-fission`** `[doall, state]`, SubgraphFission. A doall whose body is a small program of states (expand
the doall to see them) is cut after the named state: the states up to it in one doall, the rest in a second; values
passed between the parts become arrays. Pays when the parts want different schedules.

### Interchange: reorder two nested loops

**`interchange-doall-doall`** `[outer doall, inner doall]`, MapInterchange. `doall i: doall j: B` becomes
`doall j: doall i: B`. Pays when it puts the index that walks memory contiguously innermost.

**`interchange-for-doall`** `[for, doall]`, MoveLoopIntoMap. `for t: doall i: B` becomes `doall i: for t: B`. One
parallel region instead of one per step of `t`; refused when an iteration `i` reads another's values across steps.

**`interchange-doall-for`** `[doall, for]`, MapLoopInterchange. `doall i: for t: B` becomes `for t: doall i: B`; the
loop must be the doall's whole body. Brings the loop out, where it can fuse with a neighboring loop over `t`.

**`interchange-if-for`** `[if, for]`, MoveIfIntoLoop. `if c: for i: B` becomes `for i: if c: B`, so the loop can
fuse with its neighbors. **`interchange-for-if`** `[for, if]`, MoveLoopInvariantIfUp: the inverse, for a condition
that does not change inside the loop; the test then runs once instead of once per iteration.

### Tiling: work in cache-sized blocks

**`tile-doall`** `[doall]` with `tile_sizes` (one per dimension, `[32, 32]`), MapTiling.
```
doall i in 0:N: B   ->   doall i0 in 0:N step 32:
                             doall i in i0:min(i0+32, N): B
```
**`tile-for`** `[for]` with one size (`[64]`), LoopTiling: the same for a sequential loop, same iterations in the same
order; follow it with an interchange to block a nest. A size need not divide the range. Pays when a tile's data fits
the cache it is meant for: the innermost tile the closest cache, its innermost size a multiple of the machine's
vector width. Read those sizes from the machine you run on; `get_performance` after the move tells whether it paid.
**`untile-doall`** `[tile doall, point doall]` and **`untile-for`** `[for]`, UntileLoops: the inverses.

### Shape of a doall

**`collapse-doalls`** `[outer doall, inner doall]`, MapCollapse: `doall i: doall j: B` becomes `doall (i, j): B`,
one parallel space, so short ranges still give every thread work. **`expand-doall`** `[doall]`, MapExpansion: the
inverse.

### Library calls

**`expand-library`** `[lib_<family>]`, the library's own pure expansion, then canonicalization. The call becomes the
doalls and statements that compute it (a gemm: a zero fill and a multiply doall with a `+` reduction), so the moves
above reach inside it. A library call fuses with a doall only once expanded. Expand when fusing it with its neighbors
saves memory traffic; keep it when the vendor call is faster. Offered only on calls with a vendor-free expansion.

**`set-library-implementation`** `[lib_<family>]` with `implementation`. The row and its label stay; the call is
built through loops (`pure`), the host vendor library (`OpenBLAS`, `FFTW3`) or, with a GPU target, the device one
(`cuBLAS`, `hipBLAS`, ...). `list_moves` lists one move per implementation the row may take. A host library keeps its
kernel on the CPU, a device library puts it on the GPU.

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
