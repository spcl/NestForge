# dbcsr

Explanation and solution record (`benchmarking_creation/AGENTS.md`) of an HPCAgent-Bench corpus kernel, copied unchanged into this folder. Slot E3 of the thesis benchmark set (one move). Written 2026-10-21 from Daint jobs 5111411, 5111452 (probe from the installed corpus) and 5111566 (this folder). Entry note: `2026-10-21-e3-dbcsr-entry` (thesis notes).

**Status:** `entry` (route, decoys and probe at this branch's pins; not timed).

## What this benchmark is

DBCSR block-sparse matrix multiply (`src/mm/dbcsr_mm_csr.F`, `dbcsr_mm_csr_multiply_low`) as a sort-merge join on the shared inner block index. A and B are lists of dense `block_size x block_size` blocks with an index array of `(row, col, block_id)` per slot; unused slots are padding with `block_id = -1`. The kernel loops over every A slot `i` and every B slot `j`, skips padded slots, skips pairs whose inner indices differ, and multiplies the matching blocks into `C`. The two slot loops are sequential host loops (no parallel map); the block product is the only map nest. Source: HPCAgent-Bench `e63f9f1`, `benchmarks/scientific_computing/sparse_linear_algebra/dbcsr`, unedited: `dbcsr_numpy.py` (oracle), `dbcsr.py` (`initialize()`, the sparse block generator), `dbcsr.yaml`, and the generated `dbcsr_dace.py`.

The scheduling problem is a loop-invariant guard: the test "is A slot `i` padding?" reads only `i`, but it sits inside the loop over B slots, so it runs `n_a * n_b` times instead of `n_a`.

## What the reference is

- **Baseline.** Stage 1, then `default_moves()`. The guard stays inside the loop over B slots; the tree is the stage 1 tree.
- **Reference.** `interchange-loop-if` on the B-slot loop and the A-slot guard, then `finish_moves()`: `for i { if a_index[i, 2] >= 0 { for j { ... } } }`.

The same pair on CPU and GPU: both offer the move and build it. Which device has a gap is not measured.

## How it was created

Taken as is from HPCAgent-Bench. In the 2026-10-16 corpus scan (DaCe `73f0a120e`, HPCAgent-Bench `26a4f0cfc`) the CPU `default_moves()` had already hoisted the guard (only the undo, `interchange-if-loop`, was left), and the GPU default left `interchange-loop-if`; the kernel was set aside then. At this branch's pins the default leaves the move on both devices (job 5111411).

## What the solution is

Labels from job 5111452, at the pins in the solution block. Not stable across versions.

1. `interchange-loop-if` on (`for1_0`, `if2_0`): the loop over B slots (`_loop_it_3`) and the guard `not a_index[_loop_it_2, 2] < 0`. Applied as `MoveLoopInvariantIfUp`.
2. `finish_moves()`. It keeps the hoist.

Tree after the moves (CPU and GPU alike; rows before `for0_2` omitted):

```
|- for0_2  _loop_it_2=0:n_block_inner*n_block_rows
|  |- state1_2
|  `- if1_0
|     `- block2_0  when not a_index[_loop_it_2, 2] < 0
|        `- for3_0  _loop_it_3=0:n_block_cols*n_block_inner  num_states=6
|           |- state4_0
|           |- state4_1
|           `- if4_0
|              `- block5_0  when not b_index[_loop_it_3, 2] < 0
|                 |- state6_0
|                 |- if6_0
|                 |  `- block7_0  when a_index_index_1 != b_index_index_0
|                 |     `- continue8_0
|                 `- state6_1  kernels=2
`- state0_2
```

## How to optimize it

Offered at each step, CPU and GPU the same (job 5111452):

| Step | Offered |
| --- | --- |
| stage 1 | `map-fission` 1, `interchange-map-map` 1, `interchange-loop-if` 1 |
| after `default_moves()` (baseline) | `map-fission` 1, `interchange-loop-if` 1 |
| after `interchange-loop-if` | `map-fission` 1, `interchange-map-map` 1, `interchange-if-loop` 1 |
| after `finish_moves()` | `map-fission` 1, `interchange-if-loop` 1 |

The hoist is the one move that matters. `interchange-if-loop` after it is the undo. `map-fission` on `kernel8_0` and `interchange-map-map` on (`kernel7_0`, `kernel8_0`) act on the block product inside `state6_1`; `default_moves()` and `finish_moves()` both remove the `interchange-map-map` offer, so it is not a real alternative. Whether the hoist pays depends on the padding, which the tree does not show: with `initialize()` (seed 0), 78% of A slots are padding at S and 83% at M, so at M the hoist skips about 8.7e8 of the 1.05e9 inner iterations.

## What the gains are

Not measured. To be timed at preset M (180 x 180 x 180 blocks of 32 x 32, density 0.25): baseline, the reference, the undo, and `map-fission` on `kernel8_0`, on CPU at 1 and 72 threads and on GPU, 5 repeats. Guess, not measured: the block products of the matched pairs stay the same and may dominate, so the gain may be small.

## CPU and GPU

Stage 1 leaves the same tree on both: the two slot loops on the host, the block product as a map nest inside them. The move is offered, applies and survives `finish_moves()` on both. On GPU the slot loops stay on the host and each matched pair still launches the block product, so the hoist removes host-side iterations only; a gap there is not expected (guess). Not measured on either device.

## What else matters

- **Legality.** No refusal on the path; the move is offered at stage 1.
- **Contiguity and stride.** Not the question here; the move changes control flow, not the access order.
- **Launches against stride.** On GPU the number of launches does not change with the hoist (one per matched pair).
- **Vectorization.** Not checked.
- **Live values and carries.** None across the moved guard.
- **Decoys.** The undo (`interchange-if-loop`); `map-fission` on the block product; both unmeasured.
- **Oracle.** Inputs from `initialize()` (sparse block structure; uniform draws would not give the padding). Baseline and reference both match the oracle exactly at preset S on CPU and GPU (jobs 5111452, 5111566).
- **Pins.** DaCe `801cafd`, NestForge `interchange-benchmarks` (code at `4df1830`), HPCAgent-Bench `e63f9f1`. At the 2026-10-16 pins the CPU default already took the move.

## Reference solution

```yaml
status: entry
device: cpu, gpu (same moves; timing per device to come)
threads: null            # to be measured at 1 and 72 on CPU
preset: M
dace: 801cafd179790a844eaf460084232a0c6452571d
nestforge: interchange-benchmarks (code at 4df1830) + benchmarking_creation/dbcsr
hpcagent_bench: e63f9f1ce04141ae766bcb2cc5164277b57fa59d
probe: Daint jobs 5111411 (route), 5111452 (route and stage 3-5 builds at S), 5111566 (this folder)
host: null
baseline:
  moves: [default_moves]
  time_ms: null
  time_min_ms: null
  time_max_ms: null
  repeats: 5
  oracle_rel_error: 0.0   # preset S, CPU and GPU
  tree: |
    for0_2  _loop_it_2=0:n_block_inner*n_block_rows
      for1_0  _loop_it_3=0:n_block_cols*n_block_inner
        if2_0  when not a_index[_loop_it_2, 2] < 0
          if4_0  when not b_index[_loop_it_3, 2] < 0
            state6_1  kernels=2
reference:
  moves:
    - kind: interchange-loop-if
      on: for1_0 (loop over B slots), if2_0 (A-slot padding guard)
    - finish_moves: true
  time_ms: null
  time_min_ms: null
  time_max_ms: null
  repeats: 5
  oracle_rel_error: 0.0   # preset S, CPU and GPU
  tree: |
    for0_2  _loop_it_2=0:n_block_inner*n_block_rows
      if1_0  when not a_index[_loop_it_2, 2] < 0
        for3_0  _loop_it_3=0:n_block_cols*n_block_inner
          if4_0  when not b_index[_loop_it_3, 2] < 0
            state6_1  kernels=2
decoys:
  - moves: [interchange-loop-if, finish_moves, interchange-if-loop]
    effect: undoes the hoist
    time_ms: null
  - moves: [map-fission kernel8_0]
    effect: splits the block-product map; not related to the guard
    time_ms: null
```
