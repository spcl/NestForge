# conv3d_box (hard on CPU, easy on GPU)

Explanation and solution record (`benchmarking_creation/AGENTS.md`) of a kernel derived from HPCAgent-Bench `scientific_computing/structured_grids/conv_3d` (`2666c06`). Written 2026-10-23 from Daint jobs 5127348 (move walk from stage 1), 5127530 and 5127531 (builds against the oracle), 5127637 (quickstart), 5127841 (replay of every variant through the manifest), 5127866, 5127867 and 5127868 (timing). Experiment note: `2026-10-23-h2-contenders` (thesis notes). How to optimize it, in short: [README.md](README.md).

**Status:** `entry`, timed on CPU (1 and 72 threads) and GPU.

Versions: DaCe `extended` `042413d`, HPCAgent-Bench `2666c06`, NestForge `interchange-benchmarks` (`7c6a69d` plus this kernel), Daint `venv-f9081bd`.

## What this benchmark is

A non-separable 3-D box stencil, `out[i, j, k] = sum_{di, dj, dk} w[di, dj, dk] * padded[i + di, j + dj, k + dk]` with `K = 5` taps per axis (125 taps), on an edge-padded grid. The corpus kernel writes it tap-outer in NumPy; the DaCe program here is written output-stationary (one grid point, zeroed, three tap loops inside), as a direct C or GPU port writes it. Preset M: `N 257, K 5`.

## Why it is a good test

- **On CPU the best known route is 5 moves** with two fission unlocks: the `k` map goes inside the `di` and `dj` tap loops, the `j` map inside `di`. Each fission is offered only before its pushes, and each fission alone gains nothing (0.99x).
- **The answer depends on the device.** GPU: one `subgraph-fission` (34x) or `map-fission` (17x). The CPU route is a GPU trap: its prefixes keep part of the GPU gain, the full 6-move route is slower than the default (0.84x), and the 5-move route gives a wrong GPU result at preset M (see below).
- **What the default does instead:** `default_moves()` collapses `i, j, k` into one 3-D map with `for di { for dj { map dk (reduce) } }` inside. On GPU each grid point is a 256-thread block sharing the 5-term `dk` reduction (block reduction, `__syncthreads()`); on CPU every point does 125 scattered reads.
- **Tempting wrong moves:** stopping after the `k` pushes (2 moves, 1.53x against 1.89x on 1 thread); the swaps `i`/`j` and `j`/`k` (0.96x to 1.00x); walk sequences that interleave swaps (0.99x).

## What the reference is

- **Baseline:** stage 1, then `default_moves()`.
- **CPU (1 and 72 threads):** the 5 moves below, then `finish_moves()`: `map i { init; for di { map j { for dj { map k { map dk (reduce) } } } } }`. A sixth move (`j` inside `dj`) is offered; its time is within the noise of the 5-move route (1 thread 845.5 ms, range 792.7 to 846.4, against 794.1 ms, range 793.9 to 811.5).
- **GPU:** one `subgraph-fission` of the `k` map's body, then `finish_moves()`.

## How it was created

From `conv_3d`'s sum, written as the output-stationary loop nest with a zero init (without the init, the perfect nest is lifted to a library node, as `seissol_volume`'s first version was). Probed as a hard-tier (H2) candidate on 2026-10-23.

## What the solution is

CPU:

1. `map-fission` on `kernel3_0` (the `k` map).
2. `interchange-map-loop` on (`kernel4_1`, `for5_0`): `k` inside `di`.
3. `map-fission` on `kernel2_0` (the `j` map).
4. `interchange-map-loop` on (`kernel3_1`, `for4_0`): `j` inside `di`.
5. `interchange-map-loop` on (`kernel6_0`, `for7_0`): `k` inside `dj`.
6. `finish_moves()`.

GPU: `subgraph-fission` on (`kernel3_0`, `state4_0`), `finish_moves()`. Labels from jobs 5127348 and 5127866; they are not stable across versions.

## How to optimize it

Preset M, median of 5 calls after one warm-up, whole-program host time (GPU includes copies). Ratio is default time over variant time. The 72-thread default is noisy (25.75 to 33.11 ms over the 5 calls).

| Variant | CPU 1 thread | CPU 72 threads | GPU |
| --- | --- | --- | --- |
| baseline (default) | 1497.15 ms | 27.85 ms | 347.28 ms |
| stage 1, `finish_moves()` | 1497.93 ms (1.00x) | 27.54 ms (1.01x) | 347.38 ms (1.00x) |
| `map-fission` alone | 1506.60 ms (0.99x) | 28.26 ms (0.99x) | 20.97 ms (16.6x) |
| `subgraph-fission` alone | 1521.04 ms (0.98x) | 28.08 ms (0.99x) | **10.33 ms (33.6x)** |
| moves 1 and 2 (`k` inside `di`) | 979.82 ms (1.53x) | 19.67 ms (1.42x) | 19.94 ms (17.4x) |
| moves 1 to 5 (CPU reference) | **794.06 ms (1.89x)** | **16.22 ms (1.72x)** | 18.83 ms, **wrong at M** (see below) |
| 6 moves (also `j` inside `dj`) | 845.54 ms (1.77x) | 17.36 ms (1.60x) | 412.36 ms (0.84x) |
| swap `i`, `j` | 1503.65 ms (1.00x) | 28.35 ms (0.98x) | 347.04 ms (1.00x) |
| swap `j`, `k` | 1565.79 ms (0.96x) | 28.71 ms (0.97x) | 347.18 ms (1.00x) |
| walk sequence 2 (fission, push, swap, fission, `interchange-loop-map`) | 1516.11 ms (0.99x) | 26.38 ms (1.06x) | 349.57 ms (0.99x) |
| walk sequence 3 (fission, push, `interchange-loop-map`, swap, fission) | 1515.09 ms (0.99x) | 27.32 ms (1.02x) | 349.64 ms (0.99x) |

The 5-move CPU reference is walk sequence 1 of the timing run (the reference's first 5 moves).

## What the gains are

CPU 1 thread 1497.15 ms against **794.06 ms (1.89x)**; CPU 72 threads 27.85 ms against **16.22 ms (1.72x)** (default range 25.75 to 33.11 ms, so 1.6x to 2.0x); GPU 347.28 ms against **10.33 ms (33.6x)**. Every timed variant matches the oracle at preset S (relative error at most 1.3e-15) and, on CPU, the baseline's output at M (0.0).

**Wrong GPU result at preset M.** The 5-move CPU route built for GPU (job 5127868) matches the oracle at S but differs from the baseline at M by a relative error of 5.9e-2; on CPU the same schedule is exact. Every other GPU variant agrees with the baseline at M (at most 1.7e-15). A size-dependent GPU code generation bug, as in `conv2d_bias_relu`'s walk sequence 2; not yet reproduced or filed. The backtest must check the output at the timed size.

## CPU and GPU

CPU wants the 5-move route; GPU wants one fission. The CPU route on GPU is slower than the fission alone at every length, slower than the default at 6 moves, and wrong at 5.

## What else matters

- The manifest declares no small symbols: with K declared small (`public.scales`), stage 1 builds a different nest and the route is not offered (jobs 5127775 to 5127781), as for H1, H3 and M8 (`a6c787e`).
- Same mechanism as `conv2d_bias_relu` and `seissol_volume` (a fission unlocks pushing the output maps inside the summed loops); here the route needs both fissions to pay, which makes it the only one of the three with a 5-move answer.

## Reference solution

```yaml
status: entry
preset: M    # N 257, K 5
dace: 042413df2b96ed0f2ebb361c4879c7f68bad41d2
nestforge: interchange-benchmarks (7c6a69d) plus this kernel
hpcagent_bench: 2666c06b73eaa9d51f05568e6c584aaafa98ad67
host: Daint GH200 node, exclusive
references:
  - device: cpu
    threads: 1
    timing: job 5127866
    baseline:
      moves: [default_moves]
      time_ms: 1497.15   # min 1496.84, max 1497.52
      repeats: 5
      oracle_rel_error: 0.0   # preset S
      tree: |
        map (i, j, k) { out = 0; for di { for dj { map dk (reduce) } } }
    reference:
      moves:
        - kind: map-fission
          on: kernel3_0, the k map (init, sum)
        - kind: interchange-map-loop
          on: kernel4_1, for5_0 (k sum map inside di)
        - kind: map-fission
          on: kernel2_0, the j map
        - kind: interchange-map-loop
          on: kernel3_1, for4_0 (j inside di)
        - kind: interchange-map-loop
          on: kernel6_0, for7_0 (k inside dj)
        - finish_moves: true
      time_ms: 794.06    # min 793.88, max 811.53
      repeats: 5
      oracle_rel_error: 0.0
      tree: |
        map i { init; for di { map j { for dj { map k { map dk (reduce) } } } } }
  - device: cpu
    threads: 72
    timing: job 5127867
    baseline: {moves: [default_moves], time_ms: 27.85, repeats: 5, oracle_rel_error: 0.0}   # min 25.75, max 33.11
    reference: {moves: [the 5 CPU moves, finish_moves], time_ms: 16.22, repeats: 5, oracle_rel_error: 0.0}   # min 15.42, max 17.99
  - device: gpu
    timing: job 5127868
    baseline:
      moves: [default_moves]
      time_ms: 347.28    # min 347.05, max 347.58
      repeats: 5
      oracle_rel_error: 1.3e-15
      tree: |
        map (i, j, k) { ... }   # kernel: grid (N, N, N), 256 threads per point sharing the 5-term dk reduction
    reference:
      moves: [subgraph-fission kernel3_0 state4_0, finish_moves]
      time_ms: 10.33     # min 9.88, max 10.58
      repeats: 5
      oracle_rel_error: 0.0
decoys:
  - moves: [moves 1 and 2]
    time_ms: {cpu_1: 979.82, cpu_72: 19.67, gpu: 19.94}
  - moves: [the 6-move route]
    time_ms: {cpu_1: 845.54, cpu_72: 17.36, gpu: 412.36}
  - moves: [the 5-move route on GPU]
    note: wrong output at preset M (relative error 5.9e-2), correct at S
backtest:
  default_moves: {cpu_1: 1497.15, cpu_72: 27.85, gpu: 347.28}
  each_stage1_move_alone: see the table above (4 moves)
  walk_sequences: see the table above (sequences from job 5127348's walk)
```
