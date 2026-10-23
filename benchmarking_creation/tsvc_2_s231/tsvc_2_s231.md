# tsvc_2_s231 (slot E1)

Explanation and reference solution (`benchmarking_creation/AGENTS.md`) of a kernel copied from HPCAgent-Bench (`loop_level_reasoning/tsvc_2_s231`). Written 2026-10-16 from Daint jobs 5069083 and 5069111. Set: `benchmark-set-15` (thesis notes). Probe: `2026-10-16-set15-stage1-probe` (thesis notes).

## What this benchmark is

TSVC `s231`: `for i: for j in 1..LEN_2D: aa[j, i] = aa[j-1, i] + bb[j, i]` on row-major `(LEN_2D, LEN_2D)` arrays. The recurrence is carried along `j` (the row index, the non-contiguous axis); `i` (the contiguous axis) is independent. The scheduling question is where the sequential `j` loop goes relative to the parallel `i` map, and the answer depends on the device and on the thread count. Source: HPCAgent-Bench (TSVC_2), unedited.

## What the reference is

- **Baseline:** stage 1, then `default_moves()`: `for j (1:LEN_2D) { map i (0:LEN_2D) }`, one parallel region per `j`.
- **Reference (CPU, 72 threads):** `interchange-loop-map`, then `finish_moves()`: `map i { for j }`, one parallel region; each thread walks a column.
- **CPU, 1 thread:** no move; the baseline is the right schedule (the interchange is 12.8x slower).
- **GPU:** no move; stage 1 already gives `map i { for j }`, and the only offered move (`interchange-map-loop`) is the decoy.

## How it was created

Taken as is from HPCAgent-Bench. Earlier probes: `2026-10-07-s231-stage1-probe` (thesis notes) (stage 1 pre-solves the textbook direction), `2026-10-07-s231-loop-orders` (thesis notes), and the E0 timings of `2026-10-14-score-validation-e0` (thesis notes) (OpenMP 32 threads: 571 to 22 ms at another pin).

## What the solution is

1. `interchange-loop-map` on (`for0_0`, `kernel2_0`): the loop over `j` into the map over `i`. Labels from the 2026-10-16 run.
2. `finish_moves()`.

Tree after the moves: `kernel1_0 [_loop_it_0=0:LEN_2D] reads=['aa','bb'] writes=['aa']` (the loop is now inside the map).

## How to optimize it

After stage 1 on CPU, `list_moves` offers exactly one move, `interchange-loop-map`, and `default_moves()` does not take it. The decision is whether to take it: it wins only when the map is parallel over many threads, and loses on one thread because the interchanged body walks `aa` and `bb` down a column (stride `LEN_2D`). On GPU, stage 1 already took it; the offered `interchange-map-loop` undoes it.

## What the gains are

Preset M (`LEN_2D = 10241`), whole-program host time (`perf_counter` around the linked call), median of 5 calls after one warm-up, default stage 5 build (stage 6 not run), `-fopenmp` added to every CPU kernel when OpenMP is on, `OMP_PROC_BIND=close`, `OMP_PLACES=cores`.

| Build | Baseline (default) | Stage 1 only | Reference | Ratio |
| --- | --- | --- | --- | --- |
| CPU serial | 55.63 ms [55.51, 55.89] | 55.63 | 710.71 ms [501.57, 774.24] | 0.08x (slower) |
| CPU OpenMP, 1 thread | 56.08 ms [55.78, 255.71] | 55.73 | 782.03 ms [780.27, 782.67] | 0.07x (slower) |
| CPU OpenMP, 72 threads | 61.65 ms [60.24, 63.83] | 63.46 | **14.06 ms [13.95, 14.34]** | **4.4x** |
| GPU | 37.44 ms [37.38, 38.79] | 37.46 | not offered (stage 1 already has it) | none |

Oracle relative error at preset S: 0 on every variant; at preset M every variant matches the baseline's output exactly.

## CPU and GPU

The gap exists only on a threaded CPU. One thread: the interchange is the trap. GPU: stage 1's order is already right; GPU times include host-device copies.

## What else matters

- **Legality:** no refusal on the path; the reference move was offered at stage 1.
- **Contiguity:** `i` is contiguous; the reference walks `j` innermost (strided). It wins by turning `LEN_2D - 1` parallel regions into one, not by stride.
- **Launches against stride:** the textbook case: fewer regions win with threads, stride loses on one thread.
- **Vectorization:** not checked on these builds.
- **Decoys:** the interchange itself on 1 thread (12.8x slower); `interchange-map-loop` on GPU.
- **Backtest:** stage 1 offers one move, so "every offered move once" is the reference; no other legal sequence exists.
- **Pins:** DaCe `73f0a120e`, NestForge `interchange-benchmarks` `40f140d`, HPCAgent-Bench `26a4f0cfc`.

## Reference solution

```yaml
device: cpu
threads: 72
preset: M
dace: 73f0a120e4f3ec38dd1fadeb27792e7b972ec324
nestforge: 40f140d
hpcagent_bench: 26a4f0cfc1540cead109dfc8d736f1c45263e390
host: nid006550 (GH200 node, 288 cores), Daint job 5069083
baseline:
  moves: [default_moves]
  time_ms: 61.65
  time_min_ms: 60.24
  time_max_ms: 63.83
  repeats: 5
  oracle_rel_error: 0.0
  tree: |
    for0_0  _loop_it_1=1:LEN_2D
      kernel2_0  [_loop_it_0=0:LEN_2D]  reads=['aa', 'bb'] writes=['aa']
reference:
  moves:
    - kind: interchange-loop-map
      on: for0_0 (loop over j), kernel2_0 (map over i)
    - finish_moves: true
  time_ms: 14.06
  time_min_ms: 13.95
  time_max_ms: 14.34
  repeats: 5
  oracle_rel_error: 0.0
  tree: |
    kernel1_0  [_loop_it_0=0:LEN_2D]  reads=['aa', 'bb'] writes=['aa']
```

```yaml
device: cpu
threads: 1
preset: M
note: no move is the reference; the interchange is the trap
baseline: {moves: [default_moves], time_ms: 55.63, time_min_ms: 55.51, time_max_ms: 55.89, repeats: 5, oracle_rel_error: 0.0}
trap: {moves: [interchange-loop-map, finish_moves], time_ms: 710.71, time_min_ms: 501.57, time_max_ms: 774.24, repeats: 5}
```

```yaml
device: gpu
preset: M
note: stage 1 already gives map i { for j }; no move; the offered interchange-map-loop is the decoy (not timed)
baseline: {moves: [default_moves], time_ms: 37.44, time_min_ms: 37.38, time_max_ms: 38.79, repeats: 5, oracle_rel_error: 0.0}
```
