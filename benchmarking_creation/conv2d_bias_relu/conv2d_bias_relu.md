# conv2d_bias_relu (medium, CPU and GPU)

Explanation and solution record (`benchmarking_creation/AGENTS.md`) of a kernel derived from the HPCAgent-Bench `machine_learning` conv2d family (`2666c06`). Written 2026-10-23 from Daint jobs 5126537 (move walk from stage 1), 5126691 and 5126748 (route replay), 5126804 and 5126805 (builds against the oracle), 5127637 (quickstart), 5127839 (replay of every variant through the manifest), 5127860, 5127861, 5127862, 5128277 and 5128278 (timing). Experiment note: `2026-10-23-h2-contenders` (thesis notes). How to optimize it, in short: [README.md](README.md).

**Status:** `entry`, timed on CPU (1 and 72 threads) and GPU.

Versions: DaCe `extended` `042413d`, HPCAgent-Bench `2666c06`, NestForge `interchange-benchmarks` (`7c6a69d` plus this kernel), Daint `venv-f9081bd`.

## What this benchmark is

One convolution layer, `out[n, co, oh, ow] = relu(b[co] + sum_{ci, kh, kw} x[n, ci, oh + kh, ow + kw] * wt[co, ci, kh, kw])`, stride 1, input already padded. The DaCe program is written output-stationary, the way a direct or thread-per-output convolution is written: for each output element, the bias init, three summed loops accumulating into `out` in place, then ReLU. `n, co, oh, ow` are parallel; `ci, kh, kw` are reductions. Preset M: `N 8, CI 64, CO 64, H 57, W 57, KH 3, KW 3`.

Why this shape survives stage 1: only the innermost summed loop (`kw`) becomes a reduction map; `ci` and `kh` add into `out` in memory and stay loops. The init and the ReLU around the sums stop the frontend from lifting the nest to a library node.

## Why it is a good test

- **The answer depends on the device, and on CPU on the thread count.** On GPU one `map-fission` is the whole answer (108x). On CPU one thread it is the fission plus two interchanges (1.98x). On CPU 72 threads it is those three moves plus a map swap (14.9x); each half alone gives 2.0x and 7.6x.
- **What the default does instead:** `default_moves()` makes `map (n, co, oh, ow) { init; for ci { for kh { map kw (reduce) } }; relu }`. On CPU the generated kernel has one `#pragma omp parallel for`, over `n` (8 iterations), so 64 of 72 threads idle. On GPU each output element is one 256-thread block whose threads share the 3-tap `kw` reduction, with a block reduction and two `__syncthreads()` per `ci`, `kh` step: 253 of 256 threads idle.
- **What makes it hard:** none of the interchanges is offered from stage 1. `map-fission` of the `ow` map makes the first `interchange-map-loop` legal, and on CPU the fission alone is slower than the default (0.83x). On 72 threads two unrelated decisions combine: the loop order (`ow` inside the sums) and the parallel axis (`co` outside `n`).
- **Tempting wrong moves:** the swaps of `co`/`oh` and `oh`/`ow` (1.00x, 0.89x); the fission or a `subgraph-fission` alone on CPU (0.82x); the 7-move route, which continues to a second fission and gains nothing over the 3 moves (1.90x against 1.98x); on GPU, any CPU interchange after the fission (43x to 73x against 108x).

## What the reference is

- **Baseline:** stage 1, then `default_moves()` (tree above).
- **CPU, 72 threads:** 4 moves, then `finish_moves()`: `map co { map n { map oh { init; for ci { for kh { map ow { map kw (reduce) } } }; relu } } }`.
- **CPU, 1 thread:** the first 3 of those moves (no swap; the swap is neutral at 1 thread).
- **GPU:** one `map-fission` of the `ow` map, then `finish_moves()`.

## How it was created

From the output-stationary loop nest of the conv2d family's `_conv2d` (one group, stride 1), with bias and ReLU as in the `conv2d_*_relu` kernels. Probed on 2026-10-23 as a hard-tier (H2) candidate outside the set's other mechanisms; the timing showed medium-length answers (1 to 4 moves) on every device.

## What the solution is

CPU, 72 threads:

1. `map-fission` on `kernel4_0` (the `ow` map): init, sum and ReLU become three maps.
2. `interchange-map-loop` on (`kernel5_2`, `for6_0`): the `ow` sum map goes inside the `ci` loop.
3. `interchange-map-loop` on (`kernel6_0`, `for7_0`): and inside the `kh` loop.
4. `interchange-map-map` on (`kernel1_0`, `kernel2_0`): `co` outside `n`, so the parallel loop has 64 iterations instead of 8.
5. `finish_moves()`.

CPU, 1 thread: moves 1 to 3, `finish_moves()`. GPU: move 1, `finish_moves()`. Labels from jobs 5126748 and 5128277; they are not stable across versions.

## How to optimize it

Preset M, median of 5 calls after one warm-up, whole-program host time (GPU includes copies). Ratio is default time over variant time.

| Variant | CPU 1 thread | CPU 72 threads | GPU |
| --- | --- | --- | --- |
| baseline (default) | 797.77 ms | 100.57 ms | 264.92 ms |
| stage 1, `finish_moves()` | 797.47 ms (1.00x) | 100.53 ms (1.00x) | 266.38 ms (0.99x) |
| `map-fission` alone | 966.21 ms (0.83x) | 121.80 ms (0.83x) | **2.46 ms (107.5x)** |
| `subgraph-fission` alone (either cut) | 966.77 to 967.41 ms (0.82x) | 121.88 to 122.11 ms (0.82x) | 2.49 to 2.57 ms (103x to 107x) |
| moves 1 to 3 (`ow` inside the sums) | **403.24 ms (1.98x)** | 50.96 ms (1.97x) | 3.64 ms (72.7x) |
| swap `n`, `co` alone | 800.53 ms (1.00x) | 13.20 ms (7.62x) | 266.02 ms (1.00x) |
| moves 1 to 4 (CPU 72-thread reference) | 400.48 ms (1.98x) | **6.81 ms (14.9x)** | not run |
| 7-move route (second fission, `oh` inside the sums) | 420.58 ms (1.90x) | 52.87 ms (1.90x) | 6.11 ms (43.3x) |
| 7-move route plus the swap | 414.37 ms (1.91x) | 7.03 ms (14.4x) | not run |
| swap `co`, `oh` alone | 794.95 ms (1.00x) | 100.33 ms (1.00x) | 253.41 ms (1.05x) |
| swap `oh`, `ow` alone | 866.43 ms (0.92x) | 112.54 ms (0.89x) | 266.54 ms (0.99x) |
| walk sequence 1 (`subgraph-fission`, `map-fission`, swap `n`, `co`, `subgraph-fission`) | 994.93 ms (0.80x) | 18.43 ms (5.46x) | 244.16 ms (1.09x) |
| walk sequence 2 (three fissions, `map-fusion`) | 967.43 ms (0.82x) | 122.26 ms (0.82x) | 5.63 ms, **wrong at M** (see below) |
| walk sequence 3 (`subgraph-fission`, `map-fission`, `map-fusion`) | 967.80 ms (0.82x) | 121.82 ms (0.83x) | 2.49 ms (106.6x) |

The 1-thread rows for moves 1 to 4 and the 7-move route plus swap, and the 72-thread repeat of the baseline (101.25 ms) and of moves 1 to 3 (51.05 ms), are from jobs 5128278 and 5128277 on other nodes; both repeats agree within 1%.

## What the gains are

CPU 72 threads 100.57 ms against **6.81 ms (14.9x)** with 4 moves; CPU 1 thread 797.77 ms against **403.24 ms (1.98x)** with 3 moves; GPU 264.92 ms against **2.46 ms (107.5x)** with 1 move. Every timed variant matches the oracle at preset S (relative error at most 3.0e-12) and the baseline's output at preset M (CPU 0.0, GPU at most 1.6e-10), except walk sequence 2 on GPU.

**Wrong GPU result at preset M.** Walk sequence 2 (`subgraph-fission` of the `ow` body, `map-fission` of `oh`, `map-fission`, `map-fusion`; job 5127862) matches the oracle at S but differs from the baseline at M by a relative error of 4.3e29. It is a size-dependent GPU code generation bug, not a property of the schedule (the same schedule is correct on CPU at M). An agent that reaches this schedule on GPU produces a fast wrong program; the backtest must check the output at M, not only at S. Not yet reproduced or filed.

## CPU and GPU

Three different answers: GPU wants the fission only (the CPU interchanges cost it 1.5x to 2.5x of its gain); CPU one thread wants the fission and the two interchanges; CPU 72 threads also wants the parallel axis changed.

## What else matters

- The manifest declares no small symbols: with KH, KW declared small (`public.scales`), stage 1 builds a different nest and the route is not offered (jobs 5127775 to 5127781), as for H1, H3 and M8 (`a6c787e`).
- Preset M's `H`, `W` (57) are not multiples of a block size.
- The stage 5 build is the default one; the CPU 72-thread gap would change with a `collapse(2)` of the default's `n, co` loops, which no stage 2 move expresses.

## Reference solution

```yaml
status: entry
preset: M    # N 8, CI 64, CO 64, H 57, W 57, KH 3, KW 3
dace: 042413df2b96ed0f2ebb361c4879c7f68bad41d2
nestforge: interchange-benchmarks (7c6a69d) plus this kernel
hpcagent_bench: 2666c06b73eaa9d51f05568e6c584aaafa98ad67
host: Daint GH200 node, exclusive
references:
  - device: cpu
    threads: 72
    timing: job 5128277
    baseline:
      moves: [default_moves]
      time_ms: 101.25    # min 101.17, max 101.82
      repeats: 5
      oracle_rel_error: 3.0e-12   # preset S
      tree: |
        map (n, co, oh, ow) { out = b; for ci { for kh { map kw (reduce) } }; out = relu(out) }
        # kernel: one omp parallel for, over n (8 iterations)
    reference:
      moves:
        - kind: map-fission
          on: kernel4_0, the ow map (init, sum, relu)
        - kind: interchange-map-loop
          on: kernel5_2, for6_0 (ow sum map inside ci)
        - kind: interchange-map-loop
          on: kernel6_0, for7_0 (inside kh)
        - kind: interchange-map-map
          on: kernel1_0, kernel2_0 (co outside n)
        - finish_moves: true
      time_ms: 6.81      # min 6.70, max 7.14
      repeats: 5
      oracle_rel_error: 3.0e-12   # preset S
      tree: |
        map co { map n { map oh { init; for ci { for kh { map ow { map kw (reduce) } } }; relu } } }
        # kernel: omp parallel for over co (64), omp simd over ow
  - device: cpu
    threads: 1
    timing: job 5127860
    baseline: {moves: [default_moves], time_ms: 797.77, repeats: 5, oracle_rel_error: 3.0e-12}
    reference:
      moves: [map-fission kernel4_0, interchange-map-loop kernel5_2 for6_0, interchange-map-loop kernel6_0 for7_0, finish_moves]
      time_ms: 403.24    # min 402.89, max 403.83
      repeats: 5
      oracle_rel_error: 3.0e-12
      tree: |
        map n { map co { map oh { init; for ci { for kh { map ow { map kw (reduce) } } }; relu } } }
  - device: gpu
    timing: job 5127862
    baseline: {moves: [default_moves], time_ms: 264.92, repeats: 5, oracle_rel_error: 1.6e-12}
    reference:
      moves: [map-fission kernel4_0, finish_moves]
      time_ms: 2.46      # min 2.41, max 2.58
      repeats: 5
      oracle_rel_error: 3.0e-12
      tree: |
        map (n, co, oh) { map ow (init); map ow { for ci { for kh { for kw } } }; map ow (relu) }
        # kernel: grid (H, CO, N), one thread per ow
decoys:
  - moves: [map-fission]          # on CPU
    time_ms: {cpu_1: 966.21, cpu_72: 121.80}
  - moves: [the 7-move route]     # second fission and oh inside the sums
    time_ms: {cpu_1: 420.58, cpu_72: 52.87, gpu: 6.11}
  - moves: [interchange-map-map oh, ow]
    time_ms: {cpu_1: 866.43, cpu_72: 112.54}
backtest:
  default_moves: {cpu_1: 797.77, cpu_72: 100.57, gpu: 264.92}
  each_stage1_move_alone: see the table above (6 moves)
  walk_sequences: see the table above (3 sequences from job 5126537's walk)
```
