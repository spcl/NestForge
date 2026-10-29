# The seven stages

[Overview](../README.md). One section per stage: what it decides, its default, the agent's call.

## Stage 1: Canonicalize

DaCe canonicalization rewrites the program into the canonical parallel form (CPF): parallel loops become maps,
reductions and scans are lifted, statements are distributed. It runs once, with the target's preset, up to and
excluding the `fuse` stage, so stage 2 owns every fusion decision, then `structural_cleanup` (DaCe's
`StructuralCleanup`: states fused, dead states and symbols dropped, no nested SDFG inlined; then every innermost map
body made one nested SDFG and every reduction moved out of it: `map { nested SDFG } -> access node -wcr->`). No stage
after `fuse` runs, so nothing specializes. It is never searched.

| | |
|---|---|
| default | `Session.canonicalize()` |
| agent | none |
| code | `nestforge/stages/canonicalize.py` |

## Stage 2: Moves

Stage 2 shapes the loop nests. The agent reads the structure tree and asks for moves; it never edits the graph.
Every move is an existing DaCe transformation, checked for legality before it applies. No tiling.

`describe()` prints the tree. Every loop and state is a row; loop and map rows
carry `work=` and `depth=` from one pass of DaCe's work-depth analysis, and an innermost loop, or a map holding a
nested SDFG, carries `num_states=`. When that pass fails or takes over 60 s, the second line says why and the rows
carry no costs for that program. A state shows only `doalls=<count>` until `expand(labels)` opens it:
then its maps, nested maps and leaf-kernel bodies appear (an expanded map also shows its nested states). An
expansion lasts while its row lives. `list_moves(kind)` returns the legal moves as
`{kind, labels}`; map fusions list data-linked pairs and each map with the next map of its scope, never every
pair (CloudSC would list 20k), while `apply_move` still takes any pair, and `apply_move(kind, labels)` returns a `MoveResult` whose status is `applied`,
`illegal` or `not-found`. Labels are the tree labels, `<kind><n>`, unique across the whole program; a row keeps its
label while it lives (`nestforge.ir.names.Namer`).
Every applied move is followed by `structural_cleanup`, the same as after canonicalization, so the states a move
leaves fusible are fused and every map body is canonical again (the bodies a fusion leaves side by side merge into
one nested SDFG); the agent never calls it. A move retires the labels of the rows it took and names
what it made afresh, so a label read before it either still names the same row or returns `not-found`. `metrics(label)` gives the
symbolic work and depth of one kernel. `list_move_kinds()` (`moves.move_kinds`) is this table as records, the
agent's catalog.

| kind | labels | DaCe |
|---|---|---|
| `for-fusion` | first, second loop | `LoopFusion` |
| `for-fission` | loop | `LoopFission` |
| `doall-fusion` | two maps | `MapFusionVertical` (in place when the first map updates the array), else `OTFMapFusion` (a window read is recomputed per element), else `MapFusionHorizontal` |
| `doall-fission` | map | `MapFission`: per computation (independent statement groups, never per tasklet); `doall-fusion` back what shares data |
| `subgraph-fission` | map, body block | `SubgraphFission`: per state, cut after the named block |
| `interchange-doall-doall` | outer, inner map | `MapInterchange` |
| `interchange-for-doall` | loop, its map | `MoveLoopIntoMap` |
| `interchange-doall-for` | map, its loop | `MapLoopInterchange` (splits a shared state around the map first) |
| `interchange-if-for` | if, loop | `MoveIfIntoLoop` |
| `interchange-for-if` | loop, its if | `MoveLoopInvariantIfUp` |
| `tile-doall` | doall (+ `tile_sizes`, one per dimension) | `MapTiling` (`divides_evenly=False`) |
| `untile-doall` | tile doall, point doall | `UntileLoops`' map tile pairs, every tile dimension |
| `collapse-doalls` | doall, doall inside it | `MapCollapse` |
| `expand-doall` | doall of two or more dimensions | `MapExpansion`, each nested doall keeping the schedule |
| `tile-for` | for loop (+ `tile_sizes`, one size) | `LoopTiling`: a tile loop holding a clamped point loop |
| `untile-for` | tile for loop | `UntileLoops.untile_loop` |
| `set-library-implementation` | library call (+ `implementation`, one `list_moves` offered) | `LibraryNode.implementation`, pinned (`auto_select_implementation` off) so neither finalization nor the kernel's CPF unit picks another |
| `expand-library` | library call (`lib_<family>`) | `LibraryNode.expand` (`pure`, else `specialize`, until no library call is left), then canonicalization; what canonicalization lifts back (a gemm's maps into an einsum) is expanded once more |

| | |
|---|---|
| default | `default_moves()`: canonicalization's `fuse` stage and the stages after it |
| agent | moves, then `finish_moves()` runs the stages after `fuse` but `fuse_final`, keeping the chosen maps |
| code | `nestforge/stages/moves.py` |

## Stage 3: Scopes

A scope is the atomic unit later stages optimize. Each scope is outlined into a standalone SDFG and replaced by an
`ExternalCall` library node (`extcall_N`) that carries the kernel's Python oracle and boundary, and a `parallel`
flag. `unset_scope(name)` inlines a kernel back (its standalone SDFG, kernel-local containers transient again),
dropping any placement first.

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

Scalar inputs cross the boundary by value; a host length-1 array input is refused.

A scope inside a sequential loop takes the loop's iterator as an argument (`_loop_it_N`). The loop defines it and
nothing declares it, so `default_scopes` declares it on the program, and stages 5 and 6 check the kernel at the middle
of the loop's range unless `sizes` gives a value.

| | |
|---|---|
| default | `default_scopes()` |
| agent | `set_scope(labels)`, `unset_scope(name)` |
| code | `nestforge/stages/scopes.py`, `nestforge/ir/extract.py`, `nestforge/ir/loops.py`, DaCe's `ExternalCall` (`dace/libraries/standard/nodes/external_call.py`) |

## Stage 4: Placement

Stage 4 picks each kernel's device. Without a GPU target every kernel runs on the CPU. With one, the default puts
parallel kernels (a library call is a black-box doall, so its kernel is one) on the GPU and sequential ones on
the CPU. A kernel holding a library call pinned by `set-library-implementation` follows that implementation: a host
library (OpenBLAS, FFTW3) keeps it on the CPU, a device library (cuBLAS, hipBLAS, ...) puts it on the GPU. DaCe's `OffloadToAccelerator` schedules the GPU
kernels and inserts the host/device copies; kernels left on the CPU keep their operands in host memory.

`place(devices)` takes `{kernel: "cpu" | "gpu"}` and returns each kernel's device, the copies, and the
transfers the [kernel DAG](../depends.md) implies: every argument whose producer and consumer sit on different
devices. Calling it again starts from the program before the first placement.

| | |
|---|---|
| default | `place()` |
| agent | `place(devices)` |
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
matches; the status is `ok`, `refused`, `build-failed` or `wrong`.

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
scopes and placement; kernels keep the libraries already linked, the rest get the default build:

- `structured`: the whole program's time, and whether its outputs match `reference` (the bench's own reference and
  grading tolerance, when the plugin has one);
- `offloading`: each kernel's device and every transfer between memories, with bytes;
- `kernels`: each kernel's correctness, time and share, and the hints above.

It is cached until the program or a kernel library changes.

| | |
|---|---|
| default | `feedback()`, `performance(inputs, reference)` |
| code | `nestforge/stages/feedback.py`, `nestforge/stages/performance.py` |
