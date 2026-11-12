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
| `state<n>  reads=[..] writes=[..]` | one step of the program, a dataflow graph; collapsed it also shows `doalls=<k>` (library calls counted), expanded its doalls |
| `doall<n>  [i=lo:hi]  reads=[..] writes=[..]` | a do-all loop: its iterations are proven independent, so they may run in any order or at once; `: <stmt>` lines are its body |
| `lib_<family><n>  LIBNODE <Kind> [<implementation>]  reads=[..] writes=[..]` | a library call: `lib_blas` (Gemm, Gemv, Dot, ...), `lib_fft`, `lib_linalg`, `lib_reduce`, `lib_scan`, ...; a black-box doall. `[..]` is its current implementation: by default the fastest this host has for the target (OpenBLAS, FFTW3, or the device library), never `pure` while another exists. Loops only when you choose them |

- A label is a row's name: it stays on the row while the row lives. Labels are names, not positions you compute.
- `work=` counts operations, `depth=` the longest dependent chain, both as formulas in the sizes.
- `view_tree expand=[labels]` opens a state or doall with everything below it: its doalls, nested doalls, leaf
  bodies, and the states of a doall whose body is a small program (those states are where `subgraph-fission`
  cuts). `collapse=[labels]` closes any open state or doall, also one inside a row you expanded; the row stays, its
  content goes. Expanding the outer row again opens everything below it once more. The view lasts while the row
  lives; a move keeps it on the rows it does not retire.
- A state row carries what a doall row carries: `reads=` and `writes=` (arguments, and arrays other states also
  use; a temporary of the state alone is left out) and its `work=` and `depth=`. A state that touches no array is
  its label alone.
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
The move `rollback` (`apply_move` with `labels []`, always legal; with nothing to take back it changes nothing) takes
the last move back (program, labels and `applied_moves` as before it), and again for the one before, up to 8. So
trying a move costs two calls and a measurement, never the program.

`apply_move` and `run_defaults` answer at once, without a time. `get_performance` measures the program as it stands:
each kernel in one fixed compiler configuration, so two measurements compare (`finalize`'s sweep, what `score` and
`submit` grade, is faster), and each kernel once until a move changes it. A kernel too slow for its oracle at the graded
size is checked at a smaller one and timed at the graded size; the whole program is always checked at the graded size.

### Choosing a move

1. `get_performance` first. Work on the kernel with the largest share; a move on a 2% kernel cannot pay.
2. Read that kernel's rows and pick by what you see:

| you see | try | because |
|---|---|---|
| two doalls of one state, the second reading an array the first writes | `doall-fusion` | the array becomes a scalar: one write and one read of it gone |
| the same, in different states or loops | `for-fusion` of the loops around them, then `doall-fusion` | doalls fuse only inside one state |
| `for t: doall i` with a short doall body | `interchange-for-doall` | one parallel region, not one per step |
| a doall nest whose inner parameter is not the last subscript (`A[j, i]` with `j` inner) | `interchange-doall-doall` | the last subscript is the contiguous one: it belongs innermost |
| a doall over an array far larger than the cache, read more than once | `tile-doall` | each block is reused while cached |
| a doall mixing a parallel computation with one that blocks a move | `doall-fission`, then move the pieces | each piece takes its own order |
| a `lib_` row beside doalls on the same arrays | `expand-library`, then fuse | fusion reaches inside only once expanded |
| a `doall-fusion` refused because one doall reads a slice (`path[:, k]`) of an array the other writes | `snapshot-reads`, then `doall-fusion` | the two stop touching one array |
| a loop or doall of a few constant iterations (`for k in 0:3`) | `unroll-for`, `unroll-doall` | no loop left; each copy's indices are constants the compiler folds |
| a bound or flag that is an argument you expect to take one value (`K`, a mode switch) | `propagate-config`, then move inside the specialized branch | its bounds become constants and the branches it decides drop |
| a `lib_copy` (or other `lib_` call) inside a loop whose inputs the loop never changes, or one between two loops | `lib-before`, `lib-after`, `lib-into` | it runs once, not every iteration; the loops become adjacent for `for-fusion` |
| a `lib_sync` waiting earlier than the host needs the device's results | `lib-after` past host work on other data | the host overlaps that work with the device |

3. Apply one move, or the few that belong together, then `get_performance`. Faster: keep. Not faster: `rollback`.
4. The hints of `get_performance` outrank this table, in the order it prints them: a kernel that `failed` or has a
   `wrong result` first (`rollback` the move that caused it), then `bytes spilled` (the body is too big:
   `doall-fission`), then `loop not vectorized` with the compiler's reason.

Moves with a listed inverse: `tile-doall` / `untile-doall`, `tile-for` / `untile-for`, `collapse-doalls` /
`expand-doall`, `interchange-for-doall` / `interchange-doall-for`, `interchange-if-for` / `interchange-for-if`, and
`interchange-doall-doall` with itself. Fusion, fission, unrolling, `propagate-config`, `expand-library`,
`snapshot-reads`, `doall-to-for` and `lib-before` / `lib-after` / `lib-into` (a call never moves back into a loop) have none: `rollback` is the way back.

### A worked example

`T[i] = (A[i-1] + A[i] + A[i+1]) / 3`, then `B[i] = 2 * T[i]`. Collapsed, the program is one state:
```
SDFG 'blur_scale'
`- state2  doalls=2  reads=['A', 'T'] writes=['B', 'T']  work=4*N - 8 depth=4
```
`view_tree expand=[state2]` opens it, its doalls and the one-state program each doall body is:
```
SDFG 'blur_scale'
`- state2  reads=['A', 'T'] writes=['B', 'T']  work=4*N - 8 depth=4
   |- doall0  [_loop_it_0=1:N - 1]  reads=['A'] writes=['T']  num_states=1  work=3*N - 6 depth=3
   |  : A_slice_plus_A_slice = A[_loop_it_0 - 1] + A[_loop_it_0]
   |  : A_slice_A_slice_plus_A_slice = A_slice_plus_A_slice + A[_loop_it_0 + 1]
   |  : T[_loop_it_0] = A_slice_A_slice_plus_A_slice / np.float64(3)
   |  `- state3  reads=['A'] writes=['T']  work=3 depth=3
   `- doall1  [_loop_it_1=1:N - 1]  reads=['T'] writes=['B']  num_states=1  work=N - 2 depth=1
      : B[_loop_it_1] = np.float64(2) * T[_loop_it_1]
      `- state4  reads=['T'] writes=['B']  work=1 depth=1
```
`view_tree collapse=[doall0]` would hide `doall0`'s body and keep the rest open. `doall1` reads the `T` that
`doall0` writes, in one state. `list_moves kind=doall-fusion` lists `labels=[doall0, doall1]`; `apply_move` answers
`applied`, reason `MapFusionVertical`, and the tree is
```
SDFG 'blur_scale'
`- state2  reads=['A'] writes=['B', 'T']  work=4*N - 8 depth=4
   `- doall2  [_loop_it_0=1:N - 1]  reads=['A'] writes=['B', 'T']  num_states=3  work=4*N - 8 depth=4
      : A_slice_plus_A_slice = A[_loop_it_0 - 1] + A[_loop_it_0]
      : A_slice_A_slice_plus_A_slice = A_slice_plus_A_slice + A[_loop_it_0 + 1]
      : __map_fusion_T = A_slice_A_slice_plus_A_slice / np.float64(3)
      : _loop_it_1 = _loop_it_0
      : B[_loop_it_1] = np.float64(2) * __map_fusion_T
      : T[_loop_it_0] = __map_fusion_T
      `- state5  reads=['A'] writes=['B', 'T']  work=4 depth=4
```
`doall0` and `doall1` are retired and `doall2` is new. `B` now takes `T`'s value from a scalar in the same
iteration, so the state no longer reads `T` back; `T` is still written because it is an argument (a temporary
array would be gone). `get_performance` then says whether it paid. If not, `rollback` answers
`undid doall-fusion` and the tree holds `doall0` and `doall1` again.

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
`doall j: doall i: B`. Pays when it puts the index that walks memory contiguously innermost. The inner doall may also be
the one map in the nested body of the outer (`doall j: {nested: doall k}`): the body is inlined and its one-iteration
maps dropped, then the maps swap; refused when the body holds anything beside that map (such as a copy). A
one-iteration dimension of the inner map that reads the outer parameter (`jj in j:j`, what a fission leaves) is dropped
first, and an inner map with one iteration only has nothing to swap.

**`interchange-for-doall`** `[for, doall]`, MoveLoopIntoMap. `for t: doall i: B` becomes `doall i: for t: B`. One
parallel region instead of one per step of `t`; refused when an iteration `i` reads another's values across steps. Also
for a loop whose body is several doalls over one shared axis, in several states or behind ifs: `for t: { doall i: A;
doall i: B }` becomes `doall i: for t: { A; B }` (the row names one of the doalls). Apply `doall-fusion` to the doalls
that fuse first. One parallel region for all the doalls and steps, each column's data staying in cache across `t`. A
time loop whose iterator indexes nothing only grows each column's working set, so judge it by timing.

**`interchange-doall-for`** `[doall, for]`, MapLoopInterchange. `doall i: for t: B` becomes `for t: doall i: B`; the
loop must be the doall's whole body. Brings the loop out, where it can fuse with a neighboring loop over `t`.

**`interchange-if-for`** `[if, for]`, MoveIfIntoLoop. `if c: for i: B` becomes `for i: if c: B`, so the loop can
fuse with its neighbors. **`interchange-for-if`** `[for, if]`, MoveLoopInvariantIfUp: the inverse, for a condition
that does not change inside the loop; the test then runs once instead of once per iteration.
**`interchange-doall-if`** `[doall]`, MoveMapInvariantIfUp: `doall i: if c: B` becomes `if c: doall i: B` for a condition
that does not read `i`; the test runs once, and the doall can then be reordered with the loops inside the branch. A doall
with other work around the `if` is split into one doall per branch. Only this doall's `if` moves; the guards of other
doalls stay where they are.

### Tiling: work in cache-sized blocks

**`tile-doall`** `[doall]` with `tile_sizes` (one per dimension, `[32, 32]`), MapTiling.
```
doall i in 0:N: B   ->   doall i0 in 0:N step 32:
                             doall i in i0:min(i0+32, N): B
```
**`tile-for`** `[for]` with one size (`[64]`), LoopTiling: the same for a sequential loop, same iterations in the same
order. A size need not divide the range. Pays when a tile's data fits
the cache it is meant for: the innermost tile the closest cache, its innermost size a multiple of the machine's
vector width. No tool reports the machine's caches: start from the size `list_moves` lists (32 per dimension), then
try half and double, `rollback` between tries, and keep the size `get_performance` shows fastest.
**`untile-doall`** `[tile doall, point doall]` and **`untile-for`** `[for]`, UntileLoops: the inverses.

### Shape of a doall

**`collapse-doalls`** `[outer doall, inner doall]`, MapCollapse: `doall i: doall j: B` becomes `doall (i, j): B`,
one parallel space, so short ranges still give every thread work. **`expand-doall`** `[doall]`, MapExpansion: the
inverse. **`doall-to-for`** `[doall]`, MapToForLoop: `doall k: doall j: B` becomes `for k: doall j: B`, the doalls
inside staying parallel, so a pass over levels (`expand-doall` it first) can `for-fusion` with the level scan it
feeds; its temporaries then become scalars when `doall-fusion` fuses the doalls of the merged body.
**`snapshot-reads`** `[doall]`, a doall of its own per slice: a doall that reads a slice of an array (`path[:, k]`,
`path[k, :]`: one index per dimension, either a map parameter or a value the doall does not change) which another doall of
the state writes first copies the slice into a transient, so the two doalls stop touching one array and `doall-fusion`
can merge them (Floyd-Warshall: the `outer` temporary disappears). The copy is a doall, not a plain copy, which the
cleanup after every move would fold back into a read of the array. Offered only where a slice is read and written by
different doalls of one state.

### Unrolling and specialization

**`unroll-for`** `[for]` with `max_iterations`, LoopUnroll, and **`unroll-doall`** `[doall]`, not inside another
doall of its state, with `max_iterations`, MapUnroll: one copy of the body per iteration, the loop gone.
```
for k in 0:3: B(k)   ->   B(0); B(1); B(2)
```
Only a constant trip count unrolls, and only up to `max_iterations` (the largest you accept; `list_moves` checks 16).
Pays on short loops whose copies the compiler then folds and schedules together; a long one only grows the code.

**`propagate-config`** `[]` (no row) with `config_values`, ConfigPropagation: the values you expect the scalar
arguments and free symbols to take at run time (`{"K": 3}`). The program becomes
```
if K == 3: <program with K = 3>  else: <program as it was>
```
The specialized branch runs only when the arguments equal the values; any other draw runs the generic program, so
results stay correct. Inside the specialized branch the bounds are constants, so `unroll-for` and `unroll-doall` reach
its loops, and the moves above apply there too. Offered when the program has a scalar argument or a free symbol.

### Library calls

**`lib-before`** / **`lib-after`** `[lib_<family>, state|for|if]`: the call leaves its state for a new state just
before (after) the named block, which is in the call's region or in a loop around it; never inside another block.
Every copy between two arrays is a `lib_copy` row (`b[:] = a`), so this is how a copy moves.
```
for k: { b = a; doall: c += b * k }   --lib-before [lib_copy0, for0]-->   b = a; for k: doall: c += b * k
```
Only a call whose memlets are its whole effect moves; a communication or a kernel never does. A host barrier
(`lib_sync`, a device or stream wait) moves too: it crosses no block that issues device work or touches what device
work touches, and a barrier inside a kernel stays.
Refused when a block it crosses writes what it reads or touches what it writes, when what it reads is computed (or
what it writes is read) beside it in its state on the side it leaves, when it would leave a branch, and when a loop it
leaves is not proven to run at least once (a constant trip count, or a bound the program's assumptions make positive).
Hoist a call out of a loop that never changes its inputs; sink it after a loop that never reads its output; take it
from between two loops so `for-fusion` can join them.

**`lib-into`** `[lib_<family>, state]`: the same call joins an existing state of its region or of a loop around it
instead of a new one, so no state is added between two others. It reads the array from the access node that state
last writes it to, and runs after every access there that conflicts with it when it comes from a later point (before
every one when it comes from an earlier point), ordered by empty edges. Refused for an accumulation into what it
touches, two unordered writes of one array, a block that is not a state, and, for a `lib_sync`, a state whose device
work it would run ahead of (a barrier joins only after a state's device work).

**`expand-library`** `[lib_<family>]`, the library's own pure expansion, then canonicalization. The call becomes the
doalls and statements that compute it (a gemm: a zero fill and a multiply doall with a `+` reduction), so the moves
above reach inside it. A library call fuses with a doall only once expanded. Expand when fusing it with its neighbors
saves memory traffic; keep it when the vendor call is faster. Offered only on calls with a vendor-free expansion.

**`set-library-implementation`** `[lib_<family>]` with `implementation`. The row and its label stay; the call is
built through loops (`pure`), the host vendor library (`OpenBLAS`, `FFTW3`) or, with a GPU target, the device one
(`cuBLAS`, `hipBLAS`, ...). The tree prints each call's implementation; `list_moves` lists one move per other
implementation the row may take whose library this machine has, with `current` the one it has. The default is already
the fastest, so this move is for trying another. A host library keeps its kernel on the CPU, a device library puts it on the GPU.

## Scopes and kernels (when your tool list has them)

Settle the moves first: they close at your first kernel. `next_phase` out of the scopes phase gives what no scope covers
its default scope and fixes the kernels, so the source you write for a kernel stays until a new round. `next_phase`
out of the last phase starts that round: the moves reopen and the scopes and kernel code are dropped, so `score` first.

- `set_scope labels=[...]` makes one kernel of several top-level doalls of one state (library calls among them) or of
  a straight run of blocks. Group doalls that pass an array between them: inside one kernel you can keep that array in
  registers or cache instead of memory. `unset_scope` undoes it while the kernels are not fixed.
- `list_kernels` shows each kernel's device, its arguments and which kernels produce them; start with the largest
  share in `get_performance`.
- `focus_kernel name=...` scopes your kernel work to one kernel: `write_kernel` and `tune_kernel` refuse every other
  kernel until you focus it. Reads (`list_kernels`, `read_kernel`) take any kernel.
- `read_kernel` gives the source NestForge rendered. `write_kernel` takes a replacement with the same entry
  (`extern "C"` name and parameters, in order) in `cpp` for a CPU kernel or the dialect `read_kernel` showed (`cuda`,
  `hip`) for a GPU one. It is built and run against the kernel's oracle, and linked only when it matches. A kernel
  with no oracle to match answers `unchecked`: it is linked, and only the whole program's check in `performance` says
  whether it is right.
- `tune_kernel` sweeps compilers and flags for one kernel and links the fastest correct build; run it last, on the
  source you keep.
