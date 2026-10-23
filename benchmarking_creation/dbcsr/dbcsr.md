# dbcsr

Explanation and solution record (`benchmarking_creation/AGENTS.md`) of an HPCAgent-Bench corpus kernel, copied into this folder with the program, oracle and generator unchanged and presets M, L and XL changed (2026-10-23). Slot E3 of the thesis benchmark set (one move). Written 2026-10-21 from Daint jobs 5111411, 5111452 (probe from the installed corpus) and 5111566 (this folder). Entry note: `2026-10-21-e3-dbcsr-entry` (thesis notes).

**Status:** `entry`, timed 2026-10-23 at NestForge main's pins. Target: CPU, 1 thread, 2.53x (two jobs).

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

Preset M, whole-program host time, median of 5 calls after one warm-up, default stage 5 build, `-fopenmp` on CPU kernels, `OMP_PROC_BIND=close`, `OMP_PLACES=cores`. DaCe `042413d`, HPCAgent-Bench `2666c06`, NestForge `ib/on-main`. Every variant matches the oracle at S (relative error 0) and the baseline at M.

**Why the presets changed.** At the corpus sizes (180 x 180 x 180 slots, 32 x 32 blocks, density 0.25; 2026-10-23, jobs 5126512 to 5126514) the block products of the matched pairs dominate, and the hoist gains 1.04x on 1 thread, 1.02x at 72 threads and 1.02x on GPU. The hoist saves the B-slot loop for every padded A slot, so its share grows as the blocks shrink and as more slots are padding. Sweep on 1 thread (jobs 5127152 to 5127291):

| `block_size` | `density` | Baseline | Reference | Ratio | Undo |
| --- | --- | --- | --- | --- | --- |
| 32 | 0.25 | 2496.9 ms | 2398.9 ms | 1.04x | 0.98x |
| 8 | 0.25 | 243.7 ms | 154.1 ms | 1.58x | 1.00x |
| 4 | 0.25 | 207.2 ms | 115.5 ms | 1.79x | 0.86x |
| 2 | 0.25 | 201.0 ms | 107.4 ms | 1.87x | 1.00x |
| 4 | 0.1 | 46.9 ms | 30.9 ms | 1.52x | 0.98x |
| 2 | 0.1 | 46.9 ms | 19.4 ms | 2.42x | 0.81x |
| **4** | **0.05** | 18.3 ms | 7.4 ms | **2.46x** | 0.81x |

Presets M, L and XL now use 4 x 4 blocks at density 0.05. The slot counts are unchanged.

**At the new preset M** (jobs 5127395, 5127397, 5127398, 5127399):

| Build | Baseline | Reference | Ratio | Undo | `map-fission` |
| --- | --- | --- | --- | --- | --- |
| CPU 1 thread | 18.68 ms | **7.39 ms** | **2.53x** | 0.84x | 1.00x |
| CPU 1 thread, repeat | 18.77 ms | **7.41 ms** | **2.53x** | 0.98x | 1.00x |
| CPU 72 threads | 45.43 ms | 34.36 ms | 1.32x | 0.94x | 0.99x |
| GPU | 64.19 ms | see the 2026-10-23 experiment note | | | |

At 72 threads the default is 2.4x slower than on 1 thread: each matched pair opens a parallel region for a 4 x 4 product. The target is the CPU on 1 thread.

## CPU and GPU

Stage 1 leaves the same tree on both: the two slot loops on the host, the block product as a map nest inside them. The move is offered, applies and survives `finish_moves()` on both. On GPU the slot loops stay on the host and each matched pair still launches the block product, so the hoist removes host-side iterations only. Measured 2026-10-23: the gap is on CPU at 1 thread (2.53x); at 72 threads it is 1.32x, and the threaded default is slower than the serial one.

## What else matters

- **Legality.** No refusal on the path; the move is offered at stage 1.
- **Contiguity and stride.** Not the question here; the move changes control flow, not the access order.
- **Launches against stride.** On GPU the number of launches does not change with the hoist (one per matched pair).
- **Vectorization.** Not checked.
- **Live values and carries.** None across the moved guard.
- **Decoys.** The undo (`interchange-if-loop`, 0.84x and 0.98x on 1 thread); `map-fission` on the block product (1.00x on CPU; 0.24x on GPU at the corpus sizes, where it used to crash on unload, spcl/NestForge#25, fixed in DaCe `dee8dbd`).
- **Oracle.** Inputs from `initialize()` (sparse block structure; uniform draws would not give the padding). Baseline and reference both match the oracle exactly at preset S on CPU and GPU (jobs 5111452, 5111566).
- **Pins.** Route probed at DaCe `801cafd`, NestForge `4df1830`, HPCAgent-Bench `e63f9f1`; timed at DaCe `042413d`, NestForge `ib/on-main`, HPCAgent-Bench `2666c06`. At the 2026-10-16 pins the CPU default already took the move.

## Reference solution

```yaml
status: entry
device: cpu
threads: 1
preset: M    # 180 x 180 x 180 slots, 4 x 4 blocks, density 0.05
dace: 042413df2b96ed0f2ebb361c4879c7f68bad41d2
nestforge: ib/on-main (origin/main 8545a4b + interchange-benchmarks)
hpcagent_bench: 2666c06b73eaa9d51f05568e6c584aaafa98ad67
probe: Daint jobs 5111411 (route), 5111452 (route and stage 3-5 builds at S), 5111566 (this folder); timing 5127395, repeat 5127397
host: Daint GH200 node, exclusive
baseline:
  moves: [default_moves]
  time_ms: 18.68
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
  time_ms: 7.39
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
    time_ms: 22.24
  - moves: [map-fission kernel8_0]
    effect: splits the block-product map; not related to the guard
    time_ms: 18.63
```
