# tsvc_2_s233 (slot M2)

Explanation and reference solution (`benchmarking_creation/AGENTS.md`) of a kernel copied from HPCAgent-Bench (`loop_level_reasoning/tsvc_2_s233`). Written 2026-10-16 from Daint jobs 5069084 and 5069111. Set: `benchmark-set-15` (thesis notes). Probe: `2026-10-16-set15-stage1-probe` (thesis notes).

## What this benchmark is

TSVC `s233`: one outer loop `i` holding two inner loops over `j`: `aa[j, i] = aa[j-1, i] + cc[j, i]` (carried along `j`, the row index) and `bb[j, i] = bb[j, i-1] + cc[j, i]` (carried along `i`, the column index). Row-major `(LEN_2D, LEN_2D)` arrays. The two statements want opposite loop orders, so the nest has to be split before either can be reordered. The manifest states the puzzle: "Only one of the two benefits from interchange, so the compiler must interchange selectively rather than transform the whole nest uniformly." Source: HPCAgent-Bench (TSVC_2), unedited.

## What the reference is

- **Baseline:** stage 1, then `default_moves()`: the two maps in the `for i` body are fused, `for i { map j { aa, bb } }`.
- **Reference (CPU, 72 threads):** `loop-fission`, `interchange-loop-map` on the `aa` half, `finish_moves()`: two maps, `map i { for j: aa }` and `map j { for i: bb }`, one parallel region each.
- **Reference (CPU, 1 thread):** `loop-fission`, `finish_moves()`. `finish_moves()` itself interchanges the `bb` half (its last stage runs `MoveLoopIntoMapGated`, which on CPU interchanges when that lowers the inner stride), giving `for i { map j: aa }` and `map j { for i: bb }`.
- **GPU:** no move; stage 1 and the default are already within 7% of every variant.

## How it was created

Taken as is from HPCAgent-Bench. Earlier: `2026-10-14-sdfg-moves-one-by-one` (thesis notes) (fission unlocks two interchanges), E0 in `2026-10-14-score-validation-e0` (thesis notes) (serial 460 to 90 ms, at another pin). At this pin the serial gain moved from the second interchange to the fission alone, because `finish_moves()` now makes that interchange.

## What the solution is

Threaded CPU (labels from the 2026-10-16 run):

1. `loop-fission` on `for0_0` (splits the `aa` and `bb` statements into two loops over `i`).
2. `interchange-loop-map` on (`for0_0`, `kernel2_0`), the `aa` half: the `i` loop goes into its `j` map. Not offered before step 1 (the loop holds two maps).
3. `finish_moves()`.

Tree after the moves: `kernel1_0 [_loop_it_3=8:LEN_2D] reads=['aa','cc'] writes=['aa']`, then `kernel1_1 [_loop_it_2=8:LEN_2D] reads=['bb','cc'] writes=['bb']`.

Serial CPU: step 1 then `finish_moves()`. Tree: `for0_0 { kernel2_0 (aa) }`, then `kernel1_0 (bb)`.

## How to optimize it

After stage 1, `list_moves` offers only `loop-fission`; `default_moves()` takes the other route (fusion of the two maps). After the fission, two `interchange-loop-map` moves are offered, one per half, with equal static scores; which one helps depends on the thread count:

| After `loop-fission` | Serial | 72 threads |
| --- | --- | --- |
| no further move | **best** | slower than the baseline |
| interchange the `aa` half (#0) | trap (4.5x slower than fission alone) | **best** |
| interchange the `bb` half (#1) | same as fission alone (`finish_moves()` makes it anyway) | same as fission alone |

So the benchmark tests three things: take a move that is neutral or worse on its own (the fission) to unlock a better one, pick the right one of two interchanges that look alike, and read the target.

## What the gains are

Preset M (`LEN_2D = 7169`), whole-program host time, median of 5 calls after one warm-up, default stage 5 build (stage 6 not run), `-fopenmp` on every CPU kernel when OpenMP is on, `OMP_PROC_BIND=close`, `OMP_PLACES=cores`. Times in ms, [min, max].

| Variant | CPU serial | CPU OpenMP 72 threads | GPU |
| --- | --- | --- | --- |
| baseline (default) | 264.86 [263.84, 268.43] | 36.01 [35.81, 36.59] | 29.53 [29.50, 30.80] |
| stage 1 only | 422.08 | 74.58 | 29.35 |
| `loop-fission` | **61.31 [61.10, 61.63] (4.3x)** | 43.58 | 31.51 |
| + interchange #0 (`aa`) | 276.17 [275.88, 632.92] | **9.21 [9.11, 9.64] (3.9x)** | 31.41 |
| + interchange #1 (`bb`) | 61.07 | 43.80 | 31.57 |
| + both | 275.65 | 9.17 | 30.77 |

Oracle relative error at preset S: 0 on every variant; at M every variant matches the baseline's output exactly.

## CPU and GPU

Gap on CPU at both thread counts, with different answers. No gap on GPU (all variants 29.4 to 31.6 ms, host time including copies).

## What else matters

- **Legality:** the interchanges are refused until `loop-fission` applies ("not the one map directly inside").
- **Contiguity:** `j` is the row index; the `aa` statement is carried along `j`, the `bb` statement along `i`.
- **Launches against stride:** serial, the `aa` interchange makes a strided column walk and loses; with threads, it removes `LEN_2D - 8` parallel regions and wins.
- **Vectorization:** not checked.
- **Decoys:** interchange #0 on one thread; doing nothing after the fission on 72 threads; the default's fusion.
- **Backtest:** stage 1 offers one move (`loop-fission`), so every legal sequence of up to three moves is in the table above.
- **Pins:** DaCe `73f0a120e`, NestForge `interchange-benchmarks` `40f140d`, HPCAgent-Bench `26a4f0cfc`.

## Reference solution

```yaml
device: cpu
threads: 72
preset: M
dace: 73f0a120e4f3ec38dd1fadeb27792e7b972ec324
nestforge: 40f140d
hpcagent_bench: 26a4f0cfc1540cead109dfc8d736f1c45263e390
host: nid006554 (GH200 node, 288 cores), Daint job 5069084
baseline:
  moves: [default_moves]
  time_ms: 36.01
  time_min_ms: 35.81
  time_max_ms: 36.59
  repeats: 5
  oracle_rel_error: 0.0
  tree: |
    for0_0  _loop_it_1=8:LEN_2D
      kernel2_0  [_loop_it_3=8:LEN_2D]  reads=['aa', 'bb', 'cc'] writes=['aa', 'bb']
reference:
  moves:
    - kind: loop-fission
      on: for0_0 (the i loop holding the aa and bb statements)
    - kind: interchange-loop-map
      on: for0_0 (i loop of the aa half), kernel2_0 (its j map)
    - finish_moves: true
  time_ms: 9.21
  time_min_ms: 9.11
  time_max_ms: 9.64
  repeats: 5
  oracle_rel_error: 0.0
  tree: |
    kernel1_0  [_loop_it_3=8:LEN_2D]  reads=['aa', 'cc'] writes=['aa']
    kernel1_1  [_loop_it_2=8:LEN_2D]  reads=['bb', 'cc'] writes=['bb']
```

```yaml
device: cpu
threads: 1
preset: M
dace: 73f0a120e4f3ec38dd1fadeb27792e7b972ec324
nestforge: 40f140d
hpcagent_bench: 26a4f0cfc1540cead109dfc8d736f1c45263e390
host: nid006554, Daint job 5069084
baseline:
  moves: [default_moves]
  time_ms: 264.86
  time_min_ms: 263.84
  time_max_ms: 268.43
  repeats: 5
  oracle_rel_error: 0.0
reference:
  moves:
    - kind: loop-fission
      on: for0_0
    - finish_moves: true
  time_ms: 61.31
  time_min_ms: 61.10
  time_max_ms: 61.63
  repeats: 5
  oracle_rel_error: 0.0
  tree: |
    for0_0  _loop_it_1=8:LEN_2D
      kernel2_0  [_loop_it_3=8:LEN_2D]  reads=['aa', 'cc'] writes=['aa']
    kernel1_0  [_loop_it_2=8:LEN_2D]  reads=['bb', 'cc'] writes=['bb']
```
