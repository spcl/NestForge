# The seven stages

[Overview](../README.md). One section per stage: what it decides, its default, the agent's call.

## Stage 1: Canonicalize

DaCe canonicalization rewrites the program into the canonical parallel form (CPF): parallel loops become maps,
reductions and scans are lifted, statements are distributed. It runs once, with the target's preset, up to and
excluding the `fuse` stage, so stage 2 owns every fusion decision, then `structural_cleanup` (DaCe's
`StructuralCleanup`: states fused, dead states and symbols dropped, no nested SDFG inlined; then every innermost map
body made one nested SDFG and every reduction moved out of it: `map { nested SDFG } -> access node -wcr->`). No stage
after `fuse` runs, so nothing specializes. It is never searched.

A session's canonicalization leaves wavefronts to the moves (`Targets.wavefront`, off by default: DaCe's
`canonicalize(wavefront=False)`). A loop that carries a dependence along both axes of its nest stays a sequential
loop, and the `skew-wavefront` move, or the default moves, skew it. The NestForge agent and the LLVM-pass agent of the
study start from this program, so canonicalization does neither agent's work.

Right after it, and whenever a move makes a call without one, `set_default_implementations` gives each library call
left to automatic selection the implementation finalization would pick for it: the fastest this host has for the target,
never `pure` while another exists. The tree prints it, and it stays automatic until `set-library-implementation` pins
one.

| | |
|---|---|
| default | `Session.canonicalize()` |
| agent | none |
| code | `nestforge/stages/canonicalize.py` |

## Stage 2: Moves

Stage 2 shapes the loop nests. The agent reads the structure tree and asks for moves; it never edits the graph.
Every move is an existing DaCe transformation, checked for legality before it applies.

`describe()` prints the structure tree, the IR the agent reads. [ir.md](ir.md) is its contract: the rows, their fields
(`reads=`, `writes=`, `work=`, `depth=`), the labels, the body languages and the view calls (`expand`, `collapse`,
`set_language`), none of which changes the program. `list_moves(kind)` returns the legal moves as `{kind, labels}`. Map
fusions list data-linked pairs and each map with the next map of its scope, never every pair (CloudSC would list 20k),
while `apply_move` still takes any pair. `apply_move(kind, labels)` returns a `MoveResult` whose status is `applied`,
`illegal` or `not-found`.

Every applied move is followed by `structural_cleanup`, the same as after canonicalization, so the states a move leaves
fusible are fused and every map body is canonical again (the bodies a fusion leaves side by side merge into one nested
SDFG); the agent never calls it. A move retires the labels of the rows it took and names what it made afresh, so a label
read before it either still names the same row or returns `not-found`. `metrics(label)` gives the symbolic work and
depth of one kernel. `list_move_kinds()` (`moves.move_kinds`) is this table as records, the agent's catalog.

| kind | labels | DaCe |
|---|---|---|
| `for-fusion` | first, second loop | `LoopFusion`, then `StateFusion` of the merged body; loops that each hold a doall fuse without LoopFusion's guards against serializing a parallel loop (the doalls inside keep the parallelism) |
| `for-fission` | loop | `LoopFission` |
| `doall-fusion` | two maps | `MapFusionVertical` (in place when the first map updates the array), else `OTFMapFusion` (a window read is recomputed per element), else `MapFusionHorizontal`; when the intermediate is left a transient no one reads, `DeadDataflowElimination` drops its stores |
| `doall-fission` | map | `MapFission`: per computation (independent statement groups, never per tasklet); `doall-fusion` back what shares data |
| `subgraph-fission` | map, body block | `SubgraphFission`: per state, cut after the named block |
| `interchange-doall-doall` | outer, inner map | `MapInterchange`; the inner map may be the one map in the nested body of the outer (the body is inlined and its one-iteration maps dropped first) |
| `interchange-for-doall` | loop, its map | `MoveLoopIntoMap`; also a loop whose body is several doalls over one shared axis, in several states or behind ifs (the row names one of them): the loop goes inside one doall over that axis, `cfg_body` + `lane_axis` (DaCe's lane analysis). `doall-fusion` the ones that fuse first |
| `interchange-doall-for` | map, its loop | `MapLoopInterchange` (splits a shared state around the map first) |
| `sink-invariant-guard` | if whose branch ends in one for loop | `MoveIfIntoLoop`; a doall takes no if yet |
| `hoist-invariant-guard` | for loop whose only statement is an invariant if, or doall whose body holds an if | for loop: `MoveLoopInvariantIfUp`; doall: `MoveMapInvariantIfUp`, a whole-SDFG pass: tried on a copy, applied only if it lifts exactly one if, above this doall; a doall with other work around the if is unswitched into `if c: doall {A; B} else: doall A` |
| `tile-doall` | doall (+ `tile_sizes`, one per dimension) | `MapTiling` (`divides_evenly=False`) |
| `untile-doall` | tile doall, point doall | `UntileLoops`' map tile pairs, every tile dimension |
| `collapse-doalls` | doall, doall inside it | `MapCollapse` |
| `expand-doall` | doall of two or more dimensions | `MapExpansion`, each nested doall keeping the schedule |
| `doall-to-for` | one-dimensional outermost doall that holds a doall | `MapToForLoop`: the outer doall becomes a loop, the doalls inside stay parallel; empty states it leaves are dropped |
| `for-to-doall` | for loop (+ `force`) | `LoopToMap`; with `force`, its permissive check, which waives conflicting writes and some carried-read checks, but not a read its memlet test ties to another iteration's write, a range the body changes, or a symbol one iteration hands the next; NestForge refuses, before DaCe, a body with its own break, continue or return and a loop holding a call with an ordered side effect (communication); the answer of a forced move carries `forced` and a warning |
| `set-schedule-cpu` | outermost doall of a kernel on the CPU, only in a scope (+ `schedule`, `chunk`, `threads`) | sets the map's `omp_schedule`, `omp_chunk_size` and `omp_num_threads`, which the CPU code generator emits as `schedule(...)` and `num_threads(...)`; `list_scope_moves` gives each doall's schedule now as `current` |
| `set-schedule-gpu` | in a scope placed on the GPU: the kernel doall (+ `tile_sizes`, its block), or a doall directly inside it (+ `schedule`: `sequential`, `static`, `dynamic`) | the session keeps the choice per kernel and sets it on the placed body after every placement (offloading resets schedules): `gpu_block_size` on the kernel doall, `GPU_ThreadBlock` (static: a constant extent of at most 1024) or `GPU_ThreadBlock_Dynamic` (dynamic: one dimension) inside; the block comes from one place, as DaCe's GPU code generator requires; `guided` is refused until DaCe has it |
| `snapshot-reads` | doall that reads a slice (one index per dimension: a map parameter or a value the doall does not change) of an array another doall of its state writes | a doall per slice copies it into a transient and the doall reads the transient, so `doall-fusion` can merge the two doalls; the copy is a doall because a plain copy is folded back by the cleanup |
| `hoist-invariant-code` | for loop holding a loop-invariant statement or sub-block, or doall holding a statement no parameter of its nest changes | `LoopInvariantCodeMotion` on that one row: `hoist_loop_region` for a for loop, `hoist_map_scope` for a doall, judged against the outermost doall of a perfect nest |
| `auto-vectorize-scope` | top-level doall (+ `tile_sizes`, lanes per vectorized dimension, innermost last; + `isa`: `auto`, `avx512`, `avx2`, `sve`, `neon`, `scalar`) | `VectorizeMap`: the doall alone is nested and `VectorizeCPUMultiDim` tiles its innermost maps, a masked tile for the remainder, tile nodes expanded into the ISA's tile functions; every other doall stays scalar. `sve` takes its own vector length, so its `tile_sizes` is ignored. Judged when applied, on a copy first. A kernel holding the result has no NumPy oracle (its body is C++), so its builds are `unchecked` and the whole program checks them |
| `relayout` | none (+ `operation`, one operation of the bench's `requests` on an array argument; listed as a transpose of the two innermost dimensions of the first argument that has two) | relayout's `apply_requests` without conversion: the program works on the argument as `permute`, `block` / `unblock`, `split` / `unsplit`, `pad` / `unpad`, `shuffle`, `linear_shuffle` or `cast` lays it out; every operation so far is laid out again from the program before the first. Only before any other change. The operations are the submission's `requests` (`Final.requests`): the harness lays the arguments out before the call and back after it, and NestForge times the program the same way (`stages/layout.py`). `finalize` refuses a `cast` until the bench's entry takes per-argument precisions |
| `skew-wavefront` | for loop whose nest is a wavefront (+ `tile_sizes`, optional: two extents, along the loop and along the diagonal) | `WavefrontSkew`: DaCe derives the skew from the dependences and rewrites the nest into a sequential loop over diagonals (iterator `_skew_t_<n>`) holding a doall per diagonal, or a diagonal of tiles where the dependences allow (64 by 64 by default). A loop that is no wavefront is refused with `WavefrontSkew.plan_or_reason`'s sentence, class `dependence` |
| `tile-for` | for loop (+ `tile_sizes`, one size) | `LoopTiling`: a tile loop holding a clamped point loop |
| `untile-for` | tile for loop | `UntileLoops.untile_loop` |
| `set-library-implementation` | library call (+ `implementation`, one `list_moves` offered) | `LibraryNode.implementation`, pinned (`auto_select_implementation` off) so neither finalization nor the kernel's CPF unit picks another |
| `stage-loads-shared` | GPU kernel inside a GPU scope with its copies inserted (`list_moves(scope=...)` only) | `StageLoadsThroughSharedMemory` on that kernel: it is tiled into thread blocks of `tile_sizes` threads (`AddThreadBlockMap`; listed with 32 per dimension, at most three) and each array it only reads and reuses, with a constant footprint per thread, gets a `GPU_Shared` tile (halo included) filled by one block-collective copy and one barrier |
| `expand-library` | library call (`lib_<family>`) | `LibraryNode.expand` (`pure`, else `specialize`, until no library call is left), then canonicalization; what canonicalization lifts back (a gemm's maps into an einsum) is expanded once more |

| | |
|---|---|
| default | `default_moves()`: the wavefront skew on every nest it applies to (so the program has the canonical form's wavefronts), then canonicalization's `fuse` stage and the stages after it; a switched-off family ([move-families.md](move-families.md)) skips its part |
| agent | moves, then `finish_moves()` runs the stages after `fuse` but `fuse_final`, keeping the chosen maps |
| code | `nestforge/stages/moves.py` |

## Stage 3: Scopes

A scope is the atomic unit later stages optimize. Each scope is outlined into a standalone SDFG and replaced by an
`ExternalCall` library node (`extcall_<row>`) that carries the kernel's Python oracle and boundary, and a `parallel`
flag. `unset_scope(name)` inlines a kernel back (its standalone SDFG, kernel-local containers transient again),
dropping any placement first. `run_defaults("placement")` first gives what no scope covers its default scope; a block
holding a kernel is never taken whole, since no kernel nests another. The session keeps the program from before its first
kernel, and `run_defaults("round")` (`rewind`) goes back to it. Through the bench plugin, moves close at the first
kernel, scopes are fixed from placement on, and a new round reopens both ([phases](README.md#the-phases)).

`persist_temporaries(min_bytes)` is an option, not a scope: the program's heap transients of at least `min_bytes`
(1 MiB by default, at `sizes`) are allocated once at init instead of on every call, which saves the page faults of fresh
memory (1.26x on `fuse_physics_into_scan` at preset M). It needs `sizes`, leaves out transients whose extent names a symbol
`sizes` lacks, and changes no loop (`nestforge/build/sdfg.py`, `make_persistent`).

The default (`default_units` in `nestforge/stages/scopes.py`) walks the program's top-level blocks:

| top-level block | scopes |
|---|---|
| state | each parallel map and each library call (a black-box doall); sequential work stays in the program |
| loop holding no parallel map | the loop, one sequential kernel (placement keeps it on the CPU) |
| loop whose body holds only parallel maps and loops | each map; each loop in it opened the same way one level deeper (its maps), or the loop when it cannot open |
| loop whose body mixes in other work | the loop |
| conditional | each branch like a loop body; the conditional when a branch cannot open |

Before that, a parallel map holding more than `Targets.max_statements` statements (tasklets, default 200) is split by
`subgraph-fission`, at most twice, at the cut between its body blocks that leaves each piece a quarter of the
statements and passes the fewest containers from one piece to the other. Loops and conditionals are not split.
`set_scope(labels)` makes one scope of several top-level doalls (maps, library calls) of one state or of a straight run of
blocks of one control-flow region, at any depth (a loop and the state after it inside a time loop): the way to fuse
what no stage 2 move can, by writing the fused kernel in stage 5. The labels are one list, as the tree prints them.
It is refused, and nothing changes, when the blocks sit at different levels, when another block or a conditional edge
breaks the run, or when something outside the group runs between its parts.

`list_moves(kind, scope=name)` and `apply_move(kind, labels, ..., scope=name)` run the same moves inside one kernel's
body, its rows labeled as in the program; they change nothing outside it. Before placement they rewrite the body; once
placed, a structured move rewrites the body before placement (kept in `unplaced`) and the kernel is placed again, so the
default lowering keeps it. `stage-loads-shared` rewrites the lowered GPU body instead and needs the copies inserted;
placing again or a later structured move inside the scope drops it, so it comes last.

Scalar inputs cross the boundary by value; an array input crosses by pointer, a length-1 one included (a host one
as a pointer to the call's copy of its element).

A scope inside a sequential loop takes the loop's iterator as an argument (`_loop_it_N`). The loop defines it and
nothing declares it, so `default_scopes` declares it on the program, and stages 5 and 6 check the kernel at the middle
of the loop's range unless `sizes` gives a value.

| | |
|---|---|
| default | `default_scopes()` |
| agent | `set_scope(labels)`, `unset_scope(name)` |
| option | `persist_temporaries(min_bytes)` |
| code | `nestforge/stages/scopes.py`, `nestforge/ir/extract.py`, `nestforge/ir/loops.py`, DaCe's `ExternalCall` (`dace/libraries/standard/nodes/external_call.py`) |

## Stage 4: Placement

Stage 4 picks each kernel's device. Without a GPU target every kernel runs on the CPU. With one, the default puts
parallel kernels (a library call is a black-box doall, so its kernel is one) on the GPU and sequential ones on
the CPU. A kernel holding a library call pinned by `set-library-implementation` follows that implementation: a host
library (OpenBLAS, FFTW3) keeps it on the CPU, a device library (cuBLAS, hipBLAS, ...) puts it on the GPU. DaCe's `OffloadToAccelerator` schedules the GPU
kernels and inserts the host/device copies; kernels left on the CPU keep their operands in host memory.

A GPU kernel is lowered inside its call: DaCe's `OffloadToAccelerator` runs on the `ExternalCall`'s body with each
argument in the memory the call passes it in, so the body's maps become `GPU_Device` kernels and its temporaries
device memory, while CPU kernels and the program around the calls are unchanged.
`place(devices)` takes `{kernel: "cpu" | "gpu"}` and returns each kernel's device and the
transfers the [kernel DAG](../depends.md) implies: every argument whose producer and consumer sit on different
devices. It passes host memory, so a GPU kernel copies its arguments at its own boundary, inside the call.
`insert_copies()` runs `OffloadToAccelerator` on the program first: the GPU kernels' operands move to device memory,
the copies sit at the kernels' boundaries in the program (a value two GPU kernels pass stays on the device), and the
bodies are lowered over device memory; it returns the copies. Both start from the program before the first placement.
The default runs `place()` when nothing is placed, then `insert_copies()` when a GPU kernel still copies inside its
call; stage 5 builds GPU kernels over device pointers, so the copies are in before it.

| | |
|---|---|
| default | `place()`, then `insert_copies()` |
| agent | `place(devices)`, `insert_copies()` |
| code | `nestforge/stages/placement.py` |

## Stage 5: Kernels

Each kernel becomes a static library `lib<kernel>.a` with one `extern "C"` entry, linked into the program.

The default renders the kernel with DaCe's CPF code generator (`cpf.render`): one standalone unit with no DaCe
header or runtime, C++ with OpenMP on the CPU, CUDA on the GPU. The entry takes arrays by pointer and read-only
scalars by value. A GPU kernel is two units, as CPF renders it: `<kernel>.cu` holds the host launcher and
`<kernel>_device.cu` the `__global__` kernels and the launchers it calls; both go into `lib<kernel>.a`. The
compiler vectorizes; stage 6 sweeps its flags.

An agent reads `kernel_source(name)` (a GPU kernel's two units, split by the line `// ---- device unit ----`) and replaces it with `set_kernel_source(name, source, language)`. NestForge
builds the new source, checks it against the kernel's [Python oracle](../oracle.md) and links it only when it
matches; the status is `ok`, `refused`, `build-failed` or `wrong`. A kernel with no oracle to match (a symbol only the
running program computes, or an oracle that does not finish) is linked with status `unchecked`, and the whole program
checks it.

| | |
|---|---|
| default | `optimize_kernel(name)` |
| agent | `set_kernel_source(name, source, language)` |
| code | `nestforge/stages/kernel.py`, `nestforge/build/` |

## Stage 6: Variants

Stage 6 compiles each kernel's current source into variants, checks each against the oracle and keeps the fastest
correct one. It is deterministic, never agent-driven.

- **CPU.** Compiler on PATH (GNU, LLVM, oneAPI) × FP mode (strict-ieee, contract-fma, fast-math) × vectorizer cost
  model.
- **GPU.** Every nvcc on PATH × FP mode (strict-ieee, contract-fma), all with `-arch=native`.

Every variant is timed. A CPU variant runs in a forked child and a GPU variant in a spawned interpreter, so a crash
or hang is a recorded result, never a dead harness.
[FP modes](../fp-and-vectorization.md) lists the flags.

| | |
|---|---|
| default | `sweep(name)` |
| code | `nestforge/stages/variants.py`, `nestforge/build/arena.py`, `nestforge/build/flags.py` |

## Stage 7: Feedback

`feedback()` turns raw evidence into a few ranked hints the agent reads next turn: kernel times, oracle checks,
compiler remarks from one extra compile of the current variant (`-fopt-info-vec-missed`,
`-Rpass-missed=loop-vectorize`, `nvcc -Xptxas -v`). A fixed rule table decides:

| evidence | hint |
|---|---|
| build or run failed | fix the kernel |
| wrong result | check bounds, races, reduction order |
| kernel is half the kernel time or more | optimize it first |
| registers spilled | loop body too big, try fission |
| loop not vectorized | the reason, mapped to a move (fission, interchange, hoisting an `if`, `__restrict__`) |

Failures rank first; the report keeps at most eight hints.

`performance(inputs, reference)` gives every phase's feedback in one record, on a copy completed with the default
scopes and placement; kernels keep the libraries already linked, the rest get the default build. A served session
builds them with `codegen="first"`, one configuration, so two measurements compare and a kernel a move left unchanged
keeps its build and verdict; `finalize` keeps the sweep:

- `structured`: the whole program's time, and whether its outputs match `reference` (the bench's own reference and
  grading tolerance, when the plugin has one);
- `offloading`: each kernel's device and every transfer between memories, with bytes;
- `kernels`: each kernel's correctness, time and share, and the hints above.

It is cached until the program or a kernel library changes.

| | |
|---|---|
| default | `feedback()`, `performance(inputs, reference)` |
| code | `nestforge/stages/feedback.py`, `nestforge/stages/performance.py` |
