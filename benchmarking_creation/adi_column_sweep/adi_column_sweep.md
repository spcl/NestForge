# adi_column_sweep (slot M6, medium, GPU)

Explanation and solution record (`benchmarking_creation/AGENTS.md`) of a kernel derived from HPCAgent-Bench `scientific_computing/structured_grids/adi` (`e63f9f1`). Written 2026-10-22 from Daint jobs 5118678 (corpus `adi`), 5118695 and 5118842 (v1, dead end), 5119069 (v2, defaults and a greedy route), 5119089 and 5119090 (the route and every fusion pair, CPU and GPU), 5119102 (CPU timing), 5119212 and 5119364 (GPU timing), 5119154, 5119156 and 5119300 (fusion reproducers, the last one DaCe alone), 5119328 and 5119329 (a hand-fused copy, CPU and GPU). Experiment note: `2026-10-22-m6` (thesis notes). How to optimize it, in short: [README.md](README.md).

**Status:** `blocked` (not finished). The reachable reference gives 1.43 to 1.44x on GPU, under the 2x the `AGENTS.md` checklist requires; the intended reference, with the fusion of the column row maps, gives about 10.6x (hand-fused copy) but stage 2 cannot make that fusion: spcl/NestForge#27 (DaCe `MapFusionVertical`). An agent session on this kernel also needs a guard around work-depth in `describe()` (issue prepared, not filed). The benchmark is finished when #27 is fixed and the fused reference is re-timed from stage 1.

Slot: M6 (board package W12-M6).

## What this benchmark is

Alternating-direction implicit diffusion (Polybench ADI via NPBench), the oracle, arguments and presets of the corpus kernel with the x-dominant configuration (`b1 = 2`, `b2 = 1`, the oracle's defaults). Each of `TSTEPS` steps runs a row sweep and a column sweep, each two Thomas recurrences along `j` (forward, then backward) with `i` parallel. The row sweep's arrays (`pt`, `qt`, `v`) are indexed `[j, i]`, contiguous in `i`; the column sweep's (`p`, `q`, `u`) are indexed `[i, j]`, strided by `N` in `i`. The DaCe program is hand-written: every row slice is an explicit loop over `i`, so each sweep is one map under its `j` loop, and the column sweep's `p` and `q` recurrences are two source loops. The scheduling problem: which sweeps to turn from `for j { map i }` (one launch per `j`) into `map i { for j }` (one launch, one row per thread), which depends on the device, and whether the three column sweeps can then share one map.

## Why it is a good test

- **What the default does instead:** on GPU, `default_moves()` interchanges the row sweeps (lane maps over `i` with the `j` loop inside) and leaves the three column sweeps as `for j { map i }`, `N - 2` launches each per step (1.702 s at M). On CPU it does the reverse: the column sweeps are interchanged and the contiguous row sweeps kept, which is right on CPU.
- **What makes it hard:** the right set of interchanges flips with the device; neither half gains by itself: the column sweeps the GPU default left give 1.694 s, because from stage 1 the row sweeps are then left with their launches, and the `v` row interchange alone gives 2.716 s. Only the column interchanges together with the `v` row interchange beat the default (1.180 s, 1.44x).
- **What it covers:** a device-dependent interchange answer from a real code, and a fusion that only becomes possible after the interchanges and is worth about 10x on GPU (measured on a hand-fused copy), but which stage 2 cannot make at the current pins.
- **What a wrong decision looks like:** the column interchanges only; the row interchanges on CPU; following the CPU default's choice on GPU.

## What the reference is

- **Baseline (GPU):** stage 1, then `default_moves()`.
- **Reference (GPU, reachable):** `interchange-loop-map` on the `p`, `q` and `u` column sweeps and on the `v` row sweep, then `finish_moves()`.
- **Reference (GPU, intended, blocked):** the same, then `map-fusion` of the `p`, `q` and `u` row maps. Measured only on a hand-fused copy of the program (scratch kernel `adi_column_sweep_fused`, not in this folder), whose GPU default reaches 0.160 s at M.
- **CPU:** no move beats the default; the CPU pair is the default for both.

## How it was created

1. Corpus `adi` (job 5118678): stage 1 leaves every sweep as 12 elementwise maps (101 `map-fusion` offered, no interchange); only the default's fusions make the interchanges legal, so the agent would need about 11 fusions per sweep that the default makes anyway. The CPU and GPU defaults were the same tree.
2. v1, explicit loops over `i` with the column sweep's `p` and `q` in one loop (job 5118695): stage 1 splits that loop's body into a `p` map and a `q` map that `map-fusion` refuses and `loop-fission` cannot separate, so the column sweep's interchange is unreachable.
3. v2, this program (jobs 5118842, 5119069): `p` and `q` as two loops; every sweep is one map under its loop and stage 1 offers all four `interchange-loop-map`.

## What the solution is

1. `interchange-loop-map` on the column sweep writing `p` (`for1_2`, `kernel3_3` from stage 1).
2. `interchange-loop-map` on the column sweep writing `q` (again `for1_2`, `kernel3_3`: labels move).
3. `interchange-loop-map` on the column sweep writing `u` (again `for1_2`, `kernel3_3`).
4. `interchange-loop-map` on the row sweep writing `v`.
5. `finish_moves()`.

Labels from jobs 5119089, 5119090 and 5119212; the timing script names each move by the array its map writes (`routes-v2.json`, `@p`, `@q`, `@u`, `@v`). The forward row sweep (`pt`, `qt`) is two maps under one loop, so it has no interchange; the GPU default makes it one lane map, the agent cannot.

## How to optimize it

After stage 1 on both devices `list_moves` offers `interchange-loop-map` on the four single-map sweeps (`v`, `p`, `q`, `u`) and `map-fission` on the two boundary maps; on GPU also `map-fusion` of the split boundary maps. `loop-fission` is refused on every loop.

| Decision (GPU unless noted) | Preset M | Evidence |
| --- | --- | --- |
| stage 1, no move | 3.247 s | 5119212 |
| `default_moves()` | 1.702 s | 5119212 |
| the three column interchanges | 1.694 s | 5119212 |
| the `v` row interchange alone | 2.716 s | 5119364 |
| the column interchanges and the `v` row interchange | **1.180 s**; repeat 1.196 s (default on that node 1.706 s) | 5119212, 5119364 |
| the same with the column row maps fused (hand-fused copy, its default) | 0.160 s | 5119329 |
| CPU: `default_moves()` | 0.519 s | 5119102 |
| CPU: the three column interchanges | 0.549 s | 5119102 |
| CPU: the `v` row interchange alone (from stage 1) | 0.770 s | 5119102 |
| CPU: stage 1, no move | 1.003 s | 5119102 |
| CPU: hand-fused copy, its default | 0.525 s | 5119328 |

## What the gains are

Preset M (`TSTEPS 25`, `N 1600`). GPU: one GPU of a Daint GH200 node, `nvcc 12.9` `strict-ieee`; CPU: 72 threads (`OMP_PROC_BIND=close`, `OMP_PLACES=cores`), `g++` `strict-ieee` with `-fopenmp`. Whole-program host time (host-device copies included on GPU), mean of 10 calls after one warm-up; min and max not recorded. Every variant matches the oracle (rtol 1e-8, atol 1e-10) at S and M.

| Build | Preset S | Preset M |
| --- | --- | --- |
| GPU default | 19.6 ms | 1.702 s |
| GPU reference (4 interchanges) | 13.9 ms | 1.180 s |
| GPU hand-fused copy, default | 0.99 ms | 0.160 s |
| CPU default | 7.15 ms | 0.519 s |

Reachable ratio on GPU: 1.44x at M (1.41x at S); a second node gave 1.43x (1.196 s against 1.706 s, job 5119364). Below the checklist's 2x. Each half alone is no better than the default: the column interchanges 1.694 s, the `v` row interchange 2.716 s. Intended ratio, if the fusion were legal in stage 2: about 10.6x at M (the hand-fused copy under the default, so it also contains whatever the default adds; not the agent's own route).

## CPU and GPU

Stage 1 is the same on both devices except that the GPU keeps the boundary maps split. The defaults split by device: CPU interchanges the strided column sweeps and keeps the contiguous row sweeps; GPU interchanges the row sweeps and keeps the column sweeps. The gap is GPU only. On CPU no move, and not even the hand-fused column map, beats the default.

## What else matters

- **Fusion refused (blocks the full answer):** after the interchanges and `finish_moves()`, `map-fusion` p->q and q->u is refused on CPU and GPU with `blocked by MapFusionVertical on 'p' (or 'q'): shape or dependency mismatch`. Reproduced with DaCe alone (job 5119300): `MapFusionVertical` finds the pair and applies 0 times, also with `permissive=True`, whether the consumer reads `p[i, j - 1]` or `p[i, j]`. The producer is a recurrence that reads its own intermediate. Thesis notes: `2026-10-22-mapfusion-refuses-recurrence-rows` (NestForge side, draft issue) and `2026-10-22-dace-issue-draft-mapfusionvertical-recurrence` (DaCe draft).
- **Work-depth crash (blocks an agent session):** `Session.describe()` runs DaCe work-depth on every call. After `default_moves()` and after the interchanges it raises a SymPy `RecursionError` (or runs for over 25 minutes at recursion limit 30000). All numbers here use a harness-only guard that skips the costs after 60 s (`m6_guard.py`, thesis notes). Thesis note: `2026-10-22-describe-work-depth-unguarded`.
- **Memlets:** the route uses `MoveLoopIntoMap`, like spcl/NestForge#24; re-propagating memlets changed nothing here (CPU `columns_prop` 0.549 s).
- **Contiguity:** the column arrays are strided in `i` in both orders, so the interchange changes launches and per-thread locality, not coalescing.
- **Oracle:** NumPy oracle of the corpus kernel; inputs from `initialize()` (`ramp` scenario), cast to float64.
- **Pins:** DaCe `801cafd`, HPCAgent-Bench `e63f9f1`, NestForge `bench/m6` from `interchange-benchmarks` `4a0c043`.

## Reference solution

```yaml
status: blocked   # spcl/NestForge#27; reachable 1.44x < 2x
device: gpu
threads: 1
preset: M
dace: 801cafd
nestforge: 4a0c043 (bench/m6)
hpcagent_bench: e63f9f1
probe: Daint jobs 5119089, 5119090 (route), 5119212 (timing)
harness: m6_guard.py (work-depth skipped after 60 s)
baseline:
  moves: [default_moves]
  time_ms: 1702.0
  time_min_ms: null
  time_max_ms: null
  repeats: 10
  oracle_rel_error: "matches (rtol 1e-8); max error not recorded"
  tree: |
    row sweeps as lane maps over i with the j loop inside; column sweeps p, q, u as for j { map i }
reference:
  moves:
    - kind: interchange-loop-map
      on: the column sweep writing p
    - kind: interchange-loop-map
      on: the column sweep writing q
    - kind: interchange-loop-map
      on: the column sweep writing u
    - kind: interchange-loop-map
      on: the row sweep writing v
    - finish_moves: true
  time_ms: 1180.1
  time_min_ms: null
  time_max_ms: null
  repeats: 10
  oracle_rel_error: "matches (rtol 1e-8); max error not recorded"
  tree: |
    map i { for j } for the v, p, q, u sweeps; the pt/qt row sweep stays for j { map i; map i }
blocked:
  moves: [map-fusion (p, q), map-fusion (pq, u)]
  reason: DaCe MapFusionVertical refuses (strict and permissive)
  hand_fused_default_time_ms: 160.2
alternatives:
  - moves: [the three column interchanges]
    time_ms: 1694.1
decoys:
  - moves: [stage 1, no move]
    time_ms: 3246.5
  - moves: [the v row interchange alone]
    time_ms: 2715.9
```
