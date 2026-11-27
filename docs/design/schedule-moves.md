# Schedule moves

[Designs](README.md) · [Stages](../stages.md) · [Move families](../move-families.md)

Status: steps 1 to 3 built (`for-to-doall` with `force`, `set-schedule-cpu`, `set-schedule-gpu`); step 4 built 2026-11-24 (DaCe extended `fb6dabf60d`: `guided` and a chunk for the GPU queue, offered by `set-schedule-gpu`). Written 2026-11-16 against Tactics `7a016a5` and DaCe extended `6ed7dc19e6`. The
owner's decisions of 2026-11-16 are marked; everything marked "to verify" was not read in the code or run.

## Goal

An agent chooses how a loop runs, after it chose what the loop computes: in order or in parallel, and, for a
parallel loop, under which schedule on the device its scope is placed on. The moves are in the family `schedule`.

| Move | Row | Where | Arguments |
|---|---|---|---|
| `doall-to-for` (exists) | a doall | program and scope | |
| `for-to-doall` | a for loop | program and scope | `force` (optional) |
| `set-schedule-cpu` | a doall | a scope placed on the CPU | `schedule`, `chunk` (optional), `threads` (optional) |
| `set-schedule-gpu` | a doall | a scope placed on the GPU | `block` on the kernel row, `schedule` on a row inside it |

`doall-to-for` and `for-to-doall` work on the whole program and inside a scope. The two `set-schedule` moves work
only inside a scope, since a schedule belongs to the device the scope is placed on (owner).

## `for-to-doall`

The canonical form already turns every loop that DaCe proves parallel into a doall. A for loop remains where the
proof failed, where an earlier move made the loop sequential (`doall-to-for`), or where a move since then removed
the dependence.

1. Without `force`, the move runs DaCe's `LoopToMap` check and applies it. A refusal quotes the check's reason, with
   the class `DEPENDENCE`.
2. With `force`, the agent overrides a check it judges too cautious (owner). The move runs `LoopToMap` in its
   permissive mode, which waives conflicting writes (`out[perm[i]] = ...` with a permutation the compiler cannot
   see) and its first carried-read test; its memlet test of a read against another iteration's write still
   refuses, as do bounds the body changes and a symbol one iteration hands to the next. Permissive mode also lets a
   body with a break, continue or return through, so Tactics refuses those first, and a loop that holds a call
   with an ordered side effect (class `communication`): canonicalization orders such calls through data, a
   conflict the permissive mode would waive.
3. A forced move answers with a warning: the compiler did not prove the iterations independent, and a wrong
   program is the agent's responsibility. The page of the move says when forcing is reasonable (an indirection the
   agent knows to be a permutation) and that the judge's hidden inputs still grade the program.
4. The record marks the move as `forced`, so the statistics count forced moves and the wrong programs after them.

Open: whether the study's `decide` configuration may force. A forced move is not a compiler-checked move, so the
paper either switches forcing off in `decide` or reports forced moves apart. Open too: whether `force` should also
waive a carried read (`A[p[i]] = A[q[i]]` with disjoint `p` and `q`). DaCe's permissive mode keeps that refusal on
purpose (DaCe's `tests/passes/scatter_to_guarded_maps_test.py` pins it), so this needs a separate knob on `LoopToMap` in
DaCe extended, and then a bench pin bump, since the bench owns Tactics' DaCe pin.

## `set-schedule-cpu`

DaCe's map carries `omp_schedule` (static, dynamic, guided), `omp_chunk_size`, and `omp_num_threads`, and the CPU
code generator emits the clause. The move sets them on one doall of a scope placed on the CPU.

- `schedule`: `static`, `dynamic`, `guided`, or `default`. `chunk`: a positive integer, optional. `threads`: a
  positive integer, optional.
- The move changes no value the program computes, except the order in which a floating-point reduction adds. The
  benchmark grades every output against a reassociation floor (`eps_acc * sqrt(l)`), so every schedule is permitted
  on a reduction.
- Refused where the clause has no effect: a doall inside another parallel doall (it runs sequentially per thread);
  `threads` on a doall of a persistent run, whose team runs it (`#pragma omp for` takes the schedule but no
  `num_threads`). A doall whose scatter reductions DaCe range-privatizes writes no clause but takes `threads` as
  its team size in the region it opens itself.
- `default` leaves the schedule to DaCe, whose `ChooseOmpSchedule` makes it `guided` for uneven bodies.
- The clause reaches the built kernel: `test_a_cpu_schedule_reaches_the_pragma_of_the_built_kernel` reads the
  kernel's unit after the build (`DecideCpuParallelAxes`, `SplitInvariantMapNests`) and finds the pragma once.
- Built as listed: the schedule shows as each listed move's `current` in `list_scope_moves` (`dynamic,16
  threads=4`), since a scope's body has no tree of its own; the program tree shows none.

## `set-schedule-gpu`

A GPU kernel follows OpenMP's model (owner). The kernel doall is the parallel region, and a thread block is its
team. A doall inside the kernel is a work-sharing loop over that team.

| Row | Argument | DaCe | Meaning |
|---|---|---|---|
| the kernel doall | `block=[32, 8]` | `gpu_block_size` | threads per block, one size per dimension, at most three |
| a doall inside the kernel | `schedule=sequential` (default) | `Sequential` | each thread runs all iterations of its own instance |
| | `schedule=static` | `GPU_ThreadBlock` | the block's threads split the iterations evenly |
| | `schedule=dynamic` | `GPU_ThreadBlock_Dynamic` | the block's threads take iterations from a shared queue |
| | `schedule=guided` | `GPU_ThreadBlock_Dynamic`, `omp_schedule=Guided` | as dynamic, a round taking half of what is left of a range |
| | `chunk=c` with `dynamic` or `guided` | `omp_chunk_size` | a round takes at least `c` round widths (a block's or a warp's threads) of a range instead of all of it |

- `dynamic` exists on extended: `LowerDynamicThreadBlockMaps` lowers it to a thread-block map over a work queue,
  for HIP and CUDA (DaCe PR #2732 brings the same to `main`). `guided` and a `chunk` for the queue's batch are
  template arguments of DaCe's `dace::DynamicMap` (extended `cb4839c8a9`), read from the map's `omp_schedule` and
  `omp_chunk_size`.
- With `static` on an inner doall, the kernel doall iterates over blocks and the inner doall over the block's
  threads, DaCe's own model; so `static` takes a constant extent of at most 1024 (tile first). `dynamic` keeps the
  kernel over threads: DaCe's lowering adds the thread-block map, of `compiler.cuda.dynamic_map_block_size` threads,
  and takes one-dimensional doalls only.
- DaCe's GPU code generator takes the block from one place: the kernel's `gpu_block_size`, or its thread-block
  maps. The move refuses a `block` on a kernel whose inner doalls are shared, and the reverse; static and dynamic
  doalls do not share a kernel; dynamic and guided doalls do, as both run the queue.
- Refused: a row of a scope not placed on the GPU, a `block` of more than three sizes or 1024 threads, a doall
  deeper than directly inside the kernel doall, and a `chunk` that is negative or goes with `static` or `sequential`.

Built: `OffloadToAccelerator` resets every map under a kernel to `Sequential` (and raises on a GPU schedule it
finds there), so the choice cannot live in the body before placement. The session keeps it per kernel and label
(`Session.gpu_choices`, part of the rollback checkpoint) and sets it on the placed body after every placement.
`list_scope_moves` shows each row's choice as `current`. A structured move in the kernel checks every choice
again and drops one its body no longer takes. `rewind` drops every choice, `unset_scope` those of its kernel.
The device moves (`stage-loads-shared`, `copy-to-tma`, `set-pipeline-depth`) are kept per kernel the same way
(`Session.device_moves`) and made again on the placed body after every placement, so a later scope move keeps them;
one the body no longer takes is dropped. Staging sets a thread block of its own, so `set-schedule-gpu` and staging
refuse each other on the same kernel only.

## Later transformations

A student is building explicit GPU transformations: pipelining, warp specialization, and warp interleaving. They
replace the OpenMP-like schedule of a kernel where they apply. Each becomes its own move kind in a GPU family, with
its own legality check, and `set-schedule-gpu` stays the default model. A kernel that one of them rewrote refuses
`set-schedule-gpu` on the rows it owns. Their document was not read (the connector was not available on
2026-11-16).

## The tree

Every doall row of a scope shows its schedule (owner), after its range:

```text
kernel_0 (cpu)
|- doall3 [i=0:N]            omp dynamic,16
|  `- doall4 [j=0:i]         sequential (inside doall3)
kernel_1 (gpu)
|- doall7 [i=0:N]            kernel, block 256
|  `- doall8 [j=r[i]:r[i+1]] block dynamic
```

`list_moves` names, for each row, the schedules it may take. A row outside a scope shows no schedule.

## Tests

1. `for-to-doall` turns a loop that `doall-to-for` made back into the doall it was, by the tree's text (built).
2. `for-to-doall` on a loop with a carried dependence is refused with DaCe's reason, and the program stays (built);
   `force` does not waive it (built).
3. With `force`, a scatter through a permutation becomes a doall that computes the reference, and the answer
   carries the warning (built). The statistics count forced moves and the wrong kernels after them (built).
4. A forced move on a loop with a call of ordered side effect is refused (built).
5. `set-schedule-cpu` sets schedule and chunk, and the generated code holds `schedule(dynamic, 16)` on that loop
   and on no other, after the kernel is placed again (built). The generated text is the product here, so the test
   reads it.
6. A triangular nest computes the reference under every CPU schedule, on several threads.
7. `set-schedule-cpu` is refused on a nested doall, in a scope placed on the GPU, without a scope, and with a
   chunk under `default` (built). Test 6 (numerics on several threads) is not built: it needs a finalized build.
8. `set-schedule-gpu` sets `gpu_block_size`, and an inner `dynamic` gives a `GPU_ThreadBlock_Dynamic` map under
   the kernel, after which a `block` is refused; `static` on a symbolic extent and a chunk with `static` are refused; `guided` with `chunk=2` sets `omp_schedule` and `omp_chunk_size` on the placed map; a CPU
   kernel takes no GPU schedule (built; structure only, numerics on beverin).
9. `list_scope_moves` shows each schedule after the move and after a new placement (built for the block).

## Done

The four moves are listed and applied through the plugin's tools, the tree shows schedules, the record counts
forced moves, the skill page explains when each schedule pays (uneven iterations: triangular nests, sparse rows),
and `guided` and a chunk run on the GPU.

## Build order

1. `for-to-doall` with `force`, the warning, and the record's mark.
2. `set-schedule-cpu` and the schedule in the tree.
3. `set-schedule-gpu`: `block`, then `static` and `dynamic` on inner doalls.
4. DaCe extended: `guided` and a batch size for the GPU queue; then the move offers them (built 2026-11-24).
