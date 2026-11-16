# Schedule moves

[Designs](README.md) · [Stages](../stages.md) · [Move families](../move-families.md)

Status: step 1 (`for-to-doall` with `force`) built; steps 2 to 4 are design. Written 2026-11-16 against NestForge `7a016a5` and DaCe extended `6ed7dc19e6`. The
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
   see). It still refuses a read of what an earlier iteration wrote, bounds the body changes, and a symbol one
   iteration hands to the next. NestForge refuses, before DaCe, a loop that holds a call with an ordered side
   effect (class `communication`): canonicalization orders such calls through data, a conflict the permissive mode
   would waive.
3. A forced move answers with a warning: the compiler did not prove the iterations independent, and a wrong
   program is the agent's responsibility. The page of the move says when forcing is reasonable (an indirection the
   agent knows to be a permutation) and that the judge's hidden inputs still grade the program.
4. The record marks the move as `forced`, so the statistics count forced moves and the wrong programs after them.

Open: whether the study's `decide` configuration may force. A forced move is not a compiler-checked move, so the
paper either switches forcing off in `decide` or reports forced moves apart. Open too: whether `force` should also
waive a carried read (`A[p[i]] = A[q[i]]` with disjoint `p` and `q`). DaCe's permissive mode keeps that refusal on
purpose (`tests/passes/scatter_to_guarded_maps_test.py` pins it), so this needs a separate knob on `LoopToMap` in
DaCe extended, and then a bench pin bump, since the bench owns NestForge's DaCe pin.

## `set-schedule-cpu`

DaCe's map carries `omp_schedule` (static, dynamic, guided), `omp_chunk_size`, and `omp_num_threads`, and the CPU
code generator emits the clause. The move sets them on one doall of a scope placed on the CPU.

- `schedule`: `static`, `dynamic`, `guided`, or `default`. `chunk`: a positive integer, optional. `threads`: a
  positive integer, optional.
- The move changes no value the program computes, except the order in which a floating-point reduction adds, which
  the benchmark permits or forbids per task. Where it forbids reassociation, a doall with a reduction takes only
  `static`.
- Refused where the clause has no effect: a doall inside another parallel doall (it runs sequentially per thread),
  and a doall in a persistent parallel region or with a private range, for which code generation drops the clause.

## `set-schedule-gpu`

A GPU kernel follows OpenMP's model (owner). The kernel doall is the parallel region, and a thread block is its
team. A doall inside the kernel is a work-sharing loop over that team.

| Row | Argument | DaCe | Meaning |
|---|---|---|---|
| the kernel doall | `block=[32, 8]` | `gpu_block_size` | threads per block, one size per dimension, at most three |
| a doall inside the kernel | `schedule=sequential` (default) | `Sequential` | each thread runs all iterations of its own instance |
| | `schedule=static` | `GPU_ThreadBlock` | the block's threads split the iterations evenly |
| | `schedule=dynamic` | `GPU_ThreadBlock_Dynamic` | the block's threads take iterations from a shared queue |
| | `schedule=guided` | missing | as dynamic, with batches that shrink |

- `dynamic` exists on extended: `LowerDynamicThreadBlockMaps` lowers it to a thread-block map over a work queue,
  for HIP and CUDA (DaCe PR #2732 brings the same to `main`). `guided` and a `chunk` for the queue's batch are new
  work on DaCe extended, built there with their tests, and then offered here.
- With `static` or `dynamic` on an inner doall, the kernel doall iterates over blocks, not over threads. The move
  restructures the kernel accordingly (DaCe's `AddThreadBlockMap`), and the tree shows it.
- Refused: a row of a scope not placed on the GPU, a `block` whose product exceeds the device's limit, and an inner
  schedule on a doall that the kernel's threads do not all reach (to verify: the conditions of the lowering).

The move rewrites the body before placement and records the choice there, so a later placement keeps it. Today a
device move (`stage-loads-shared`) rewrites the placed body, and the next placement drops it; this move must not
inherit that.

## Later transformations

A student is building explicit GPU transformations: pipelining, warp specialization, and warp interleaving. They
replace the OpenMP-like schedule of a kernel where they apply. Each becomes its own move kind in a GPU family, with
its own legality check, and `set-schedule-gpu` stays the default model. A kernel that one of them rewrote refuses
`set-schedule-gpu` on the rows it owns. Their document was not read (the connector was not available on
2026-11-16).

## The tree

Every doall row of a scope shows its schedule (owner), after its range:

```
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
   and on no other. The generated text is the product here, so the test reads it.
6. A triangular nest computes the reference under every CPU schedule, on several threads.
7. `set-schedule-cpu` is refused on a nested doall and in a scope placed on the GPU.
8. `set-schedule-gpu` sets `gpu_block_size`, and an inner `dynamic` gives a `GPU_ThreadBlock_Dynamic` map under a
   kernel over blocks (structure only; numerics on beverin).
9. The tree shows each schedule after the move and after a new placement.

## Done

The four moves are listed and applied through the plugin's tools, the tree shows schedules, the record counts
forced moves, the skill page explains when each schedule pays (uneven iterations: triangular nests, sparse rows),
and `guided` on the GPU waits for DaCe.

## Build order

1. `for-to-doall` with `force`, the warning, and the record's mark.
2. `set-schedule-cpu` and the schedule in the tree.
3. `set-schedule-gpu`: `block`, then `static` and `dynamic` on inner doalls.
4. DaCe extended: `guided` and a batch size for the GPU queue; then the move offers them.
