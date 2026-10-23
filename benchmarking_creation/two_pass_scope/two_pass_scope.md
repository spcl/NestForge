# two_pass_scope (slot M7)

Explanation and solution record (`benchmarking_creation/AGENTS.md`) of a synthetic kernel. Written 2026-10-22 from Daint job 5119028. Entry note: `2026-10-22-m7-two-pass-scope` (thesis notes).

**Status:** `entry`, re-targeted to the GPU on 2026-10-23: the scope beats the default 2.87x and 2.82x (two jobs) on GPU at preset M. On CPU it ties at 1 thread and loses at 72.

## What this benchmark is

Two passes over a grid `(N, M)`, with `M` contiguous. The producer walks columns. A carried value starts at 0; column `j` of `t` is `a * s + b`, then `s` becomes `s + t[0, j] + w[j]` and that value is stored in `carry[j]`. In the DaCe program (since 2026-10-23) the carried value is `cs[j]` of an `(M + 1,)` array and its update is one tasklet (see "How it was created"). The consumer walks rows and writes `out = t * t - a * carry[M - 1]`. The column loop cannot be a map, because the next scalar reads the column just written. The row loop can. They are separate blocks, and no move fuses across that boundary. The scheduling problem is to put both passes in one kernel. Synthetic (thesis slot M7).

## What the reference is

GPU (since 2026-10-23). On CPU the scope does not beat the default (below).

- **Baseline.** Stage 1, then `default_moves()`, then one kernel per parallel top-level map. Two kernels: the column map, launched once per column from the host loop that updates the scalar, and the collapsed consumer map.
- **Reference.** `define_scope` on the init state, the column loop and the consumer state. One kernel. No `finish_moves()`.

## How it was created

`layout_flip_fuse` was dropped the same morning: at DaCe `801cafd` the default already fuses the two maps ([[2026-10-22-m7-layout-flip-fuse]]). A prefix `acc[j] = previous + w[j]` with the map outside the carry is lifted to a scan in the same state, and the default fuses the maps (job 5118878). The carry has to read the map output, so the map stays inside the loop (this source). Job 5119028.

**2026-10-23 rewrite of the DaCe program** (the NumPy oracle is unchanged). Two changes:

1. The carried scalar update `s = s + t[0, j] + w[j]; carry[j] = s` is one explicit `dace.tasklet`. Written as two statements, the frontend lowered each to a length-1 map, and inside the scope each became a length-1 `omp for` with its own barrier (three per column).
2. The carried value is `cs[j]` of an `(M + 1,)` transient, read at `cs[j]` and written at `cs[j + 1]`. As a scalar, or as a one-element array (which simplification turns back into a scalar), it is passed to the GPU column kernel as a by-value `double` while the parent holds it on the device (`s_gpu`), and the GPU default did not compile (jobs 5127392, 5127433). With the per-column array every kernel takes it by pointer and the GPU default builds (job 5127479).

## What the solution is

Labels from job 5119028, epoch 1.

`define_scope` on `state0_0`, `for0_0`, `state0_1`. `state0_0` is the scalar init. `for0_0` is `0:M` with one map `0:N` writing `t` and reading the scalar. `state0_1` is the consumer nest writing `out`. The call returns applied, kernel `extcall_0`.

Stage 1 tree:

```
state0_0
for0_0  0:M
  kernel2_0  [0:N]  writes t
state0_1
  kernel1_0  [0:N]
    kernel2_1  [0:M]  writes out
```

After the scope the tree is one state holding `extcall_0` (reads `a`, `b`, `carry`, `t`, `w`; writes `carry`, `out`, `t`).

## How to optimize it

`list_moves` after stage 1 offers one move: `interchange-map-map` on `kernel1_0`, `kernel2_1`. That swaps the consumer so `M` is outer and `N` is inner. It leaves the column loop where it is, so the program is still two kernels. That is the decoy. `default_moves()` does not fuse the passes and leaves no further move. `map-fusion` is not offered: the nests are not in one state. Naming the two maps to `define_scope` is refused, because they are in different states; the blocks have to be named.

## What the gains are

CPU, preset M (N=4099, M=32771), host time, median of 5 after one warm-up (job 5119199, exclusive, nid005942). All three match the oracle at S and match each other at M.

- 1 thread: baseline 2972 ms, reference 1577 ms (1.89×), interchange 1804 ms. The baseline samples sit in two clumps (2474–2484 and 2972–3015).
- 72 threads: baseline 175 ms, reference 521 ms (0.34×), interchange 176 ms. The default is 3.0× faster than the scope. The interchange ties the default.

The reference is one kernel, but the scalar update is three length-1 `omp for` loops inside the sequential column loop, each with a barrier. At one thread that is cheaper than 32771 host launches. At 72 threads the barriers dominate. The consumer nest is the same in the baseline and the reference.

**2026-10-23, the rewritten program** at NestForge main's pins (DaCe `042413d`, HPCAgent-Bench `2666c06`, NestForge `ib/on-main`), preset M, median of 5 after one warm-up. All variants match the oracle at S (relative error 0).

| Build | Baseline | Scope (reference) | Ratio | Interchange (decoy) |
| --- | --- | --- | --- | --- |
| GPU (job 5127479) | 594.86 ms | **207.01 ms** | **2.87x** | 601.66 ms (0.99x) |
| GPU, repeat (job 5127630) | 604.11 ms | **214.40 ms** | **2.82x** | 607.48 ms (0.99x) |
| CPU 1 thread (job 5127631) | 1540.58 ms | 1533.30 ms | 1.00x | 1787.72 ms (0.86x) |
| CPU 72 threads (job 5127480) | 173.84 ms | 273.45 ms | 0.64x | 176.45 ms (0.99x) |

On GPU the default launches the column kernel once per column (32771 launches) and moves the carried value between host and device each time; the scope runs the whole column loop inside one kernel. On CPU the scope replaces one fork-join per column by two barriers per column (the column `omp for` and the length-1 loop DaCe still wraps the tasklet in), which costs about the same: a tie on 1 thread, a loss at 72. The comparison of outputs with the baseline at M is `nan` (the carried value overflows over 32771 columns); the oracle check at S passes.

## CPU and GPU

CPU and GPU stage 1 trees match, and the same scope applies on both. The gap is on GPU (2.87x). On CPU the default is the right schedule. Before the 2026-10-23 rewrite the GPU baseline and decoy did not compile (the parent passed the device pointer `s_gpu` into `extcall_0`'s by-value `double s`).

## What else matters

- **Legality.** `define_scope` on the two top-level maps is refused: they are in different states. The consumer interchange is legal and is not the answer.
- **Contiguity and stride.** `M` is contiguous. The consumer nest after stage 1 has `N` outer and `M` inner. The decoy reverses that.
- **Launches against stride.** The baseline launches the column kernel once per column and keeps the scalar update on the host. The reference is one kernel, and its scalar update is three length-1 parallel loops per column. The reference wins at 1 thread (1.89×) and loses at 72 (0.34×). Job 5119199.
- **Live values and carries.** The scalar is loop-carried through the column map. `carry` records it so the consumer reads an array. `chain_length` of `carry` is `M`.
- **Decoys.** `interchange-map-map` on the consumer. It builds on CPU and matches the oracle, and it leaves two kernels.
- **Oracle.** Inputs `a`, `b`, `w` uniform on `[-1, 1]`, seed 0. `t`, `out` and `carry` start at 0. Preset S, relative error 0 on CPU for all three schedules, and on GPU for the scope.
- **Pins.** DaCe `801cafd179790a844eaf460084232a0c6452571d`, NestForge `4a0c0434ae260140cb35a1f89ec9d10d70bb2e74`, HPCAgent-Bench `e63f9f1`.

## Reference solution

```yaml
status: entry
device: cpu
threads: 72
preset: M
dace: 801cafd179790a844eaf460084232a0c6452571d
nestforge: 4a0c0434ae260140cb35a1f89ec9d10d70bb2e74
hpcagent_bench: e63f9f1
probe: Daint job 5119028 (oracle at S)
timing: Daint job 5119199 (nid005942; also 1 thread, below)
host: nid005942
baseline:
  moves: [default_moves]
  time_ms: 174.6
  time_min_ms: 170.5
  time_max_ms: 176.9
  repeats: 5
  oracle_rel_error: 0.0
  time_ms_1thread: 2972.1
  tree: |
    for0_0  _loop_it_0=0:M
      kernel2_0  [0:N]  writes t
    state0_2
      kernel1_0  [0:N, 0:M]  writes out
reference:
  moves:
    - kind: define_scope
      on: state0_0, for0_0, state0_1
  time_ms: 520.9
  time_min_ms: 517.6
  time_max_ms: 523.8
  repeats: 5
  oracle_rel_error: 0.0
  time_ms_1thread: 1576.6
  tree: |
    state0_0
      extcall_0  reads a, b, carry, t, w  writes carry, out, t
```

```yaml
status: entry
device: gpu
threads: null
preset: M
dace: 042413df2b96ed0f2ebb361c4879c7f68bad41d2
nestforge: ib/on-main (origin/main 8545a4b + interchange-benchmarks), program rewritten 2026-10-23
hpcagent_bench: 2666c06b73eaa9d51f05568e6c584aaafa98ad67
timing: Daint job 5127479, repeat 5127630 (GH200)
baseline:
  moves: [default_moves]
  time_ms: 594.86
  repeats: 5
  oracle_rel_error: 0.0
reference:
  moves:
    - kind: set_scope
      on: state0_0, for0_0, state0_1
  time_ms: 207.01
  repeats: 5
  oracle_rel_error: 0.0
decoys:
  - moves: [interchange-map-map on the consumer]
    time_ms: 601.66
```
