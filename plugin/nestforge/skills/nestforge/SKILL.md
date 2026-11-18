---
name: nestforge
description: "How to read the NestForge tree (for and while loops, doalls, library calls) and pick the moves that rewrite it: fusion, fission, interchange, tiling, hoisting, library calls, layouts, vectorization, schedules; with the full tool set, also scopes and kernel code. Read it whenever NestForge tools are in your tool list, even if you only plan to call finalize."
when: "you drive the NestForge tools: read this before your first view_tree or apply_move."
applies: {explicit: true}
---

## The tree

`get_cpf` and `view_tree` print the program as a tree headed by `SDFG '<name>'`. Each row has a label, its name in
every later call.

| row | is |
|---|---|
| `for<n>  i=lo:hi` | a loop over a known range; its iterations run in order |
| `while<n>  <cond>` | a loop that runs until its condition fails; no tiling or unrolling applies |
| `if<n>` | a branch; its arms print as `when <cond>` / `else` |
| `state<n>` | one step of the program, a dataflow graph |
| `doall<n>  [i=lo:hi]` | a loop whose iterations are proven independent: they may run in any order or at once |
| `lib_<family><n>  LIBNODE <Kind> [<implementation>]` | a library call (`lib_blas`, `lib_fft`, `lib_reduce`, `lib_copy`, ...), a doall you cannot see into |
| `: <statement>` | one statement of a doall's body |

| field | means |
|---|---|
| `reads=[..] writes=[..]` | the arrays a row reads and writes; a state leaves out a temporary only it uses |
| `doalls=<k>` | on a closed state: its doalls and library calls |
| `num_states=<k>` | the states under a loop or doall |
| `accum=(y += over i)` | the doall adds into `y` across `i`; `accum=(y +=)`: each iteration adds into its own element |
| `[<implementation>]` | how a library call is built: the fastest this host has (`OpenBLAS`, `FFTW3`, a device library), until you set another |

- **Names.** Labels (`doall3`) are never reused: a move retires the labels of the rows it took and labels the rows it
  made afresh; a retired label is refused as unknown. Arguments keep the program's names. Temporaries the compiler
  made are `tmp<n>` (one element) and `Tmp<n>` (an array); a name means one container everywhere, in Python, in C and
  in `container=`.
- **What shows.** A closed state is one line. `view_tree expand=[labels]` opens a state (its doalls as rows) or a
  doall (its statements, nested doalls, and the states of its body, where `subgraph-fission` cuts). `collapse=[labels]`
  closes one. Bodies print as Python; `c=[labels]` prints them as C (vectorized tiles, inline assembly), `python=[labels]`
  goes back. None of these changes the program, and a move keeps the view of every row it does not retire.
- **Order.** Inside a state, doalls are ordered only by the data they share. Between two states everything in the
  first finishes before the second starts, so doalls of different states never fuse: fuse the loops around them.
  Loops and branches hold states.
- **Kernels.** A kernel is a scope, not a row: by default each top-level doall (with what it holds) or library call
  is one kernel, run on CPU threads or on the GPU; a top-level loop stays on the host and launches the kernels in it.

## Calls

| call | does | changes the program |
|---|---|---|
| `view_tree` | the tree, with `expand`, `collapse`, `c`, `python` | no |
| `get_work_depth labels=[..]` | each row's work (operations) and depth (longest dependent chain) at this run's sizes; `formula=true` in the sizes | no |
| `list_move_kinds` | every kind, the rows it takes in order, the transformation it runs | no |
| `list_moves [kind=..]` | the moves legal now, as `kind` + `labels` and the arguments they were checked with | no |
| `get_performance` | each kernel's time and share, and ranked hints | no |
| `apply_move kind labels [arguments]` | one move; answers `applied` or `illegal: <reason>` at once, without the tree | yes |
| `apply_move rollback []` | takes the last move back, labels included; again for the one before, up to 8 | yes |
| `tune_move kind labels values [scope]` | a tiling move, `vectorize-doall` or `set-schedule-gpu` at 1 to 4 sizes (tile sizes, lanes, block), each measured; keeps the fastest if it beats the program, else nothing | yes |
| `run_defaults stage=moves` / `finish` | the default moves (wavefronts, then fusion) / the stages after your moves | yes |

Arguments, each meaning one thing: `tile_sizes` (tile extents), `lanes` (vector lanes), `block` (a GPU thread block),
`max_iterations`, `implementation`, `config_values`, `operation`, `container`, `isa`, `schedule`, `chunk`,
`threads`, `force`, `scope`.

`get_performance` times each kernel in one fixed build, so two measurements compare (`finalize`'s sweep, what `score`
grades, is faster). Every move keeps the program's results: it changes how loops run, never what they compute.

## How to work

1. `get_performance`; work on the kernel with the largest share (a move on a 2% kernel cannot pay).
2. Open its rows and, if you need to rank loops, `get_work_depth` them.
3. Pick a move below, `list_moves kind=...` for its labels, `apply_move`, then `get_performance`. A correct program
   more than 3% slower than the fastest one measured is rolled back to it by `get_performance` itself, so measure
   only after the moves that should pay together. To pick a tile size, `tune_move` instead of tiling, measuring and
   untiling.
4. Undo with `rollback`, never with an inverse move: an inverse makes new rows with new labels; `rollback` gives back
   the very program. After a move, open only the rows you need; do not reprint the whole tree.
5. Hints outrank this page, in their order: a `failed` or `wrong result` kernel first (`rollback`), then `bytes
   spilled` (the body is too big: `doall-fission`), then `loop not vectorized` with the compiler's reason.

| you see | try |
|---|---|
| two doalls of one state, the second reading what the first writes | `doall-fusion` |
| the same in different states or loops | `for-fusion` of the loops, then `doall-fusion` |
| `doall-fusion` refused: one doall reads a slice (`path[:, k]`) the other writes | `snapshot-reads`, then `doall-fusion` |
| `for t: doall i` with a short body | `interchange-for-doall` |
| `for t: doall i` over large rows, `t` carrying a recurrence (prefix sums, a time loop per column) | column slabs: `interchange-for-doall`, `tile-doall` the doall, `interchange-doall-for` the inner pair (`doall ii: for t: doall i`) |
| a doall nest whose inner parameter is not the last subscript (`A[j, i]`, `j` inner) | `interchange-doall-doall` |
| a doall over data far larger than the cache, reread | `tile-doall` (sizes: `tune_move`) |
| a loop whose iterations each read the previous along two axes (Gauss-Seidel, LU) | `skew-wavefront` |
| a doall mixing a parallel part with a part that blocks a move | `doall-fission` |
| a `lib_` row beside doalls on the same arrays | `inline-library`, then fuse |
| a statement that uses nothing the iterations change (`s = a * b`) | `hoist-invariant-code` |
| an `if` inside a loop whose condition the loop never changes | `hoist-invariant-guard` |
| a temporary computed before a loop that blocks a fusion | `sink-invariant-code` |
| a loop or doall of a few constant iterations | `unroll-for`, `unroll-doall` |
| an argument you expect to take one value (`K`, a mode flag) | `propagate-config` |
| a `lib_copy` or other call in a loop that never changes its inputs, or between two loops | `lib-before`, `lib-after`, `lib-into` |
| short doall ranges that leave threads idle | `flatten-doalls` |
| iterations of uneven cost, in a CPU kernel | `set-schedule-cpu` with `dynamic` or `guided` |
| a hot leaf doall whose hint says `loop not vectorized` | `vectorize-doall`, last (lanes: `tune_move`) |
| an argument read with a stride no move can change (`trans[i, j]` with `i` inner, a matrix used transposed) | `relayout`, first |

## Moves

Each line: the rows it takes, its arguments, what it does, when it pays or is refused. `list_move_kinds` gives the
DaCe transformation behind each.

**Fusion and fission**
- `for-fusion` `[for, for]`: two adjacent loops over one range become one. Pays when the second reads what the first
  wrote. Refused when an iteration would read a value a later iteration of the first loop writes.
- `doall-fusion` `[doall, doall]` of one state: a producer and its consumer become one doall; the array between them
  becomes a scalar (a stencil's window is recomputed). Two independent doalls over one range fuse side by side.
- `snapshot-reads` `[doall]`: a doall reading a slice another doall of the state writes first copies the slice, so
  the two stop sharing an array and fuse (Floyd-Warshall).
- `for-fission` `[for]`, `doall-fission` `[doall]`: one loop or doall per independent group of statements; then fuse
  the pieces with different neighbors.
- `subgraph-fission` `[doall, state]`: a doall whose body is several states (open the doall) is cut after the named
  state; values passed between the parts become arrays.

**Interchange**
- `interchange-doall-doall` `[outer, inner]`: `doall i: doall j` becomes `doall j: doall i`; put the contiguous
  index innermost. Also when the inner doall is the one map of the outer doall's body.
- `interchange-for-doall` `[for, doall]`: `for t: doall i` becomes `doall i: for t`, one parallel region instead of
  one per step; also for a loop of several doalls over one axis (fuse them first). Refused when an `i` reads another
  `i`'s values across steps.
- `interchange-doall-for` `[doall, for]`: `doall i: for t` becomes `for t: doall i`; the loop must be the whole body.

**Shape**
- `flatten-doalls` `[outer, inner]`: `doall i: doall j` becomes `doall (i, j)`; `unflatten-doall` `[doall]` splits it back.
- `doall-to-for` `[doall]`: the doall runs in order, the doalls inside stay parallel (to fuse with a loop over levels).
- `for-to-doall` `[for]`: a loop DaCe proves independent becomes a doall. `force=true` waives the check of
  conflicting writes, only when you know what DaCe cannot (`out[perm[i]]` with a permutation); the answer then warns.

**Tiling** (`tile_sizes`)
- `tile-doall` `[doall]`, one size per dimension (`[32, 32]`); `tile-for` `[for]`, one size. A size need not divide
  the range; the innermost should fit the closest cache and be a multiple of the vector width. Find it with
  `tune_move` (`values=[[32],[64],[128]]`).
- `untile-doall` `[tile doall, point doall]`, `untile-for` `[for]`: the inverses, for an earlier tiling.
- `skew-wavefront` `[for]`, optional `tile_sizes` (two extents): a nest whose iterations depend along a diagonal runs
  diagonal by diagonal, each diagonal a doall. Refused, with the reason, where the diagonals are not independent.

**Hoisting** (the row is a `for` or a `doall`)
- `hoist-invariant-code`: what no iteration changes (an expression of arguments, a load of an element the loop never
  writes) runs once, before the loop. Offered only where something hoists.
- `sink-invariant-code` with `container`: the statements before the loop that compute that temporary move into it and
  run every iteration, so the loop no longer depends on them (to fuse) and the temporary stops living across it.
- `hoist-invariant-guard`: `for i: if c: B` becomes `if c: for i: B` for a `c` the loop never changes; a doall with
  other work around the `if` splits into one doall per branch.
- `sink-invariant-guard` `[if]`: `if c: for i: B` becomes `for i: if c: B`, so the loop can fuse with its neighbors.

**Specialization**
- `propagate-config` `[]` with `config_values` (`{"K": 3}`): the program becomes `if K == 3: <specialized> else:
  <as it was>`; inside, bounds are constants, so the unrolls reach the loops. Results stay correct for any value.
- `unroll-for` `[for]`, `unroll-doall` `[doall]` with `max_iterations` (`list_moves` checks 16): one copy of the body
  per iteration; only a constant trip count. Pays on short loops.

**Library calls**
- `lib-before`, `lib-after` `[lib, block]`: the call moves to a new state just before or after the block, in its
  region or a loop around it. `lib-into` `[lib, state]`: it joins that state instead. Hoist a call out of a loop that
  never changes its inputs; take it from between two loops so they fuse. A `lib_sync` moves past host work on other
  data. Refused when a crossed block touches what it touches, or a loop it leaves may run zero times.
- `inline-library` `[lib]`: the call becomes the doalls that compute it, so the moves reach inside. Keep the call
  when the vendor library is faster.
- `set-library-implementation` `[lib]` with `implementation`: another build of the call (`pure`, a host or device
  library); `list_moves` lists those this machine has, `current` the one now. A device library puts its kernel on the GPU.

**Layout, vectorization, schedules**
- `relayout` `[]` with `operation`, first, before any other move: an array argument is stored the way the loops read
  it (`{"action": "permute", "array": "A", "order": ["i1", "i0"]}`; also `block`, `split`, `pad`, `shuffle`). The
  harness lays it out outside the timing. Dimensions are named `i0`, `i1`, ... and keep their names through a permute.
  `list_moves` offers the transpose of the first 2D argument. A `cast` (`"to": "fp32"`) applies and times, but
  `finalize` refuses it: roll it back before you submit.
- `vectorize-doall` `[doall]`, an outermost one, with `lanes` (`[8]`, innermost last) and `isa` (`auto`, `avx512`, `avx2`, `sve`, `neon`,
  `scalar`): the doall runs as SIMD tiles, with a masked tail. Last: no move takes its body apart afterwards; read it
  with `c=`.
- `set-schedule-cpu` `[doall]`, in a CPU kernel (`apply_scope_move`), with `schedule` (`static`, `dynamic`, `guided`,
  `default`) and optional `chunk` and `threads`: the OpenMP schedule of an outermost doall.
- `set-schedule-gpu` `[doall]`, in a GPU kernel: on the kernel doall `block` (`[256]`, `[32, 8]`); on a doall inside
  it `schedule`: `sequential` (each thread runs it), `static` (one iteration per thread), `dynamic` (a queue).
- `stage-loads-shared` `[doall]` with `block`, in a GPU kernel with its copies in: the tiles its threads reread go
  through shared memory. Last: placing again drops it.

## Scopes and kernels (when your tool list has them)

Settle the moves first: they close at your first kernel. `next_phase` out of the scopes phase gives what no scope covers
its default scope and fixes the kernels. `next_phase` out of the last phase starts a new round: the moves reopen and
the scopes and kernel code are dropped, so `score` first.

- `set_scope labels=[...]` makes one kernel of several top-level doalls of one state (library calls among them) or of
  a straight run of blocks; group doalls that pass an array between them. `unset_scope` undoes it until the kernels
  are fixed.
- `list_scope_moves scope=<kernel>` and `apply_scope_move scope=<kernel> ...` are `list_moves` and `apply_move` inside
  one kernel. With a GPU target, `place devices={...}` puts each kernel on `cpu` or `gpu`; `insert_copies` moves the
  copies into the program, where adjacent GPU kernels share them.
- `list_kernels` shows each kernel's device, arguments and producers. `focus_kernel name=...` picks the kernel that
  `write_kernel` and `tune_kernel` may change.
- `read_kernel` gives the kernel's source. `write_kernel` takes a replacement with the same `extern "C"` entry, in
  `cpp` or the GPU dialect `read_kernel` showed; it is linked only when it matches the kernel's oracle (`unchecked`
  when there is none: only the whole program's check says whether it is right).
- `tune_kernel` sweeps compilers and flags for one kernel and links the fastest correct build; run it last.
