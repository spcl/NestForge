# tsvc_2_s233 (slot M2)

Explanation and reference solution (`benchmarking_creation/AGENTS.md`) of a kernel copied from HPCAgent-Bench (`loop_level_reasoning/tsvc_2_s233`). Written 2026-10-16 from Daint jobs 5069084 and 5069111; **rewritten 2026-10-23** from jobs 5126518 to 5126520 at NestForge main's pins, where the answer changed (the same answer was measured on 2026-10-22, jobs 5119702 to 5119715). Set: `benchmark-set-15` (thesis notes). Timing: `2026-10-23-retime-main` (thesis notes).

## What this benchmark is

TSVC `s233`: one outer loop `i` holding two inner loops over `j`: `aa[j, i] = aa[j-1, i] + cc[j, i]` (carried along `j`, the row index) and `bb[j, i] = bb[j, i-1] + cc[j, i]` (carried along `i`, the column index). Row-major `(LEN_2D, LEN_2D)` arrays. The two statements want opposite loop orders, so the nest has to be split before either can be reordered. The manifest states the puzzle: "Only one of the two benefits from interchange, so the compiler must interchange selectively rather than transform the whole nest uniformly." Source: HPCAgent-Bench (TSVC_2), unedited.

## What the reference is

- **Baseline:** stage 1, then `default_moves()`: the two maps in the `for i` body are fused, `for i { map j { aa, bb } }`.
- **Reference (CPU, 1 thread):** `loop-fission`, `interchange-loop-map` on the `bb` half, `finish_moves()`: `for i { map j: aa }`, then `map j { for i: bb }`. 5.47x.
- **Reference (CPU, 72 threads):** `loop-fission`, `interchange-loop-map` on both halves, `finish_moves()`: `map i { for j: aa }` and `map j { for i: bb }` in one state, one parallel region each. 2.30x.
- **GPU:** no move; the default is within 6% of the best variant, and every other variant is slower.

## How it was created

Taken as is from HPCAgent-Bench. Earlier: `2026-10-14-sdfg-moves-one-by-one` (thesis notes) (fission unlocks two interchanges), E0 in `2026-10-14-score-validation-e0` (thesis notes) (serial 460 to 90 ms, at another pin). At DaCe `73f0a120e` (2026-10-16) the serial gain moved from the second interchange to the fission alone, because `finish_moves()` made that interchange; at DaCe `801cafd` and `042413d` it no longer does, and the serial gain is back on the `bb` interchange.

## What the solution is

Labels from jobs 5126518 and 5126519 (main's pins). After the fission `for0_0` is the `aa` half and `for0_1` the `bb` half.

CPU, 1 thread:

1. `loop-fission` on `for0_0` (splits the `aa` and `bb` statements into two loops over `i`).
2. `interchange-loop-map` on (`for0_1`, `kernel2_1`), the `bb` half: its `i` loop goes into its `j` map. Not offered before step 1 (the loop holds two maps).
3. `finish_moves()`.

Tree after the moves: `for0_0 { state1_0 (aa) }`, then `state0_1` with the `bb` kernel.

CPU, 72 threads:

1. `loop-fission` on `for0_0`.
2. `interchange-loop-map` on (`for0_0`, `kernel2_0`), the `aa` half.
3. `interchange-loop-map` on (`for0_0`, `kernel2_0`) again: after step 2 the `bb` half's loop is relabelled `for0_0`.
4. `finish_moves()`.

Tree after the moves: `state0_1 kernels=2` (both halves as maps, no host loop).

## How to optimize it

After stage 1, `list_moves` offers only `loop-fission`; `default_moves()` fuses the two maps instead. After the fission, two `interchange-loop-map` moves are offered, one per half, with equal static scores. Which ones to take depends on the thread count:

| After `loop-fission` | 1 thread | 72 threads |
| --- | --- | --- |
| no further move | slower than the default (0.87x) | slower than the default (0.59x) |
| interchange the `aa` half (#0) | **trap** (0.46x) | 0.88x |
| interchange the `bb` half (#1) | **best (5.47x)** | 0.99x |
| interchange both | 0.84x | **best (2.30x)** |

So the benchmark tests three things: take a move that is worse on its own (the fission) to unlock better ones, tell two look-alike interchanges apart, and read the target. On 2026-10-16 (DaCe `73f0a120e`) `finish_moves()` made the `bb` interchange by itself and the answers were "fission alone" (1 thread) and "fission + `aa`" (72 threads); at today's pins it does not.

## What the gains are

Preset M (`LEN_2D = 7169`), whole-program host time, median of 5 calls after one warm-up, default stage 5 build (stage 6 not run), `-fopenmp` on every CPU kernel, `OMP_PROC_BIND=close`, `OMP_PLACES=cores`. Times in ms, [min, max]. Jobs 5126518 (1 thread), 5126519 (72 threads), 5126520 (GPU).

| Variant | CPU 1 thread | CPU 72 threads | GPU |
| --- | --- | --- | --- |
| baseline (default) | 359.05 [356.76, 368.58] | 34.24 [33.66, 35.33] | 37.45 [37.06, 38.55] |
| stage 1 only | 418.54 | 59.11 | 225.98 |
| `loop-fission` | 411.76 | 58.40 | 226.73 |
| + interchange #0 (`aa`) | 780.99 | 38.77 | 134.13 |
| + interchange #1 (`bb`) | **65.64 [64.44, 66.75] (5.47x)** | 34.49 | 132.37 |
| + both | 429.73 | **14.91 [14.11, 15.32] (2.30x)** | 39.78 |

Oracle relative error at preset S: 0 on every variant; at M every variant matches the baseline's output exactly.

## CPU and GPU

Gap on CPU at both thread counts, with different answers. No gap on GPU: the default (37.45 ms, host time including copies) is the fastest variant.

## What else matters

- **Legality:** the interchanges are refused until `loop-fission` applies ("not the one map directly inside").
- **Contiguity:** `j` is the row index; the `aa` statement is carried along `j`, the `bb` statement along `i`.
- **Launches against stride:** on 1 thread, interchanging `aa` makes a strided column walk and loses, while interchanging `bb` makes its walk contiguous and wins; with threads, interchanging both removes both sets of per-`i` parallel regions.
- **Vectorization:** not checked.
- **Decoys:** interchange #0 on 1 thread (0.46x); the fission alone (0.87x and 0.59x); one interchange at 72 threads (0.88x, 0.99x); the default's fusion.
- **Backtest:** stage 1 offers one move (`loop-fission`), so every legal sequence of up to three moves is in the table above.
- **Pins:** DaCe `042413d`, NestForge `ib/on-main` (origin/main `8545a4b` + `interchange-benchmarks`), HPCAgent-Bench `2666c06`.

## Reference solution

```yaml
device: cpu
threads: 1
preset: M
dace: 042413df2b96ed0f2ebb361c4879c7f68bad41d2
nestforge: ib/on-main (origin/main 8545a4b + interchange-benchmarks)
hpcagent_bench: 2666c06b73eaa9d51f05568e6c584aaafa98ad67
host: nid006376, Daint job 5126518
baseline:
  moves: [default_moves]
  time_ms: 359.05
  time_min_ms: 356.76
  time_max_ms: 368.58
  repeats: 5
  oracle_rel_error: 0.0
reference:
  moves:
    - kind: loop-fission
      on: for0_0 (the i loop holding the aa and bb statements)
    - kind: interchange-loop-map
      on: for0_1 (i loop of the bb half), kernel2_1 (its j map)
    - finish_moves: true
  time_ms: 65.64
  time_min_ms: 64.44
  time_max_ms: 66.75
  repeats: 5
  oracle_rel_error: 0.0
```

```yaml
device: cpu
threads: 72
preset: M
dace: 042413df2b96ed0f2ebb361c4879c7f68bad41d2
nestforge: ib/on-main (origin/main 8545a4b + interchange-benchmarks)
hpcagent_bench: 2666c06b73eaa9d51f05568e6c584aaafa98ad67
host: nid005332, Daint job 5126519
baseline:
  moves: [default_moves]
  time_ms: 34.24
  time_min_ms: 33.66
  time_max_ms: 35.33
  repeats: 5
  oracle_rel_error: 0.0
reference:
  moves:
    - kind: loop-fission
      on: for0_0
    - kind: interchange-loop-map
      on: for0_0 (aa half), kernel2_0
    - kind: interchange-loop-map
      on: for0_0 (bb half, relabelled), kernel2_0
    - finish_moves: true
  time_ms: 14.91
  time_min_ms: 14.11
  time_max_ms: 15.32
  repeats: 5
  oracle_rel_error: 0.0
```
