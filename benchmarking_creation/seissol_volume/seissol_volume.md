# seissol_volume (medium, CPU and GPU)

Explanation and solution record (`benchmarking_creation/AGENTS.md`) of a kernel derived from HPCAgent-Bench `scientific_computing/dense_linear_algebra/seissol_tensor_contraction` (`2666c06`). Written 2026-10-23 from Daint jobs 5127053 (first version, lifted to an `einsum` node), 5127112 (move walk from stage 1), 5127336 and 5127372 (route replay), 5127373, 5127388 and 5127389 (builds against the oracle), 5127637 (quickstart), 5127840 (replay of every variant through the manifest), 5127863, 5127864 and 5127865 (timing). Experiment note: `2026-10-23-h2-contenders` (thesis notes). How to optimize it, in short: [README.md](README.md).

**Status:** `entry`, timed on CPU (1 and 72 threads) and GPU.

Versions: DaCe `extended` `042413d`, HPCAgent-Bench `2666c06`, NestForge `interchange-benchmarks` (`7c6a69d` plus this kernel), Daint `venv-f9081bd`.

## What this benchmark is

The volume-integral update of SeisSol's ADER-DG scheme (seismic wave propagation): `upd[b, k, p] = sum_{d, l, q} kDivM[d, k, l] * I[b, l, q] * star[d, q, p]`, then `Q += dt * upd`. `b` runs over elements, `k, l` over basis functions (56 for order 6), `p, q` over the 9 elastic quantities, `d` over the 3 dimensions. The contraction is the corpus kernel's einsum; the time-step form (`upd`, then `Q += dt * upd`) is how the update is applied. The DaCe program is written output-stationary: one `(b, k, p)` entry, zeroed, accumulated over `d, l, q`. Preset M: `NB 4003, NBF 56, NQ 9, ND 3` (about 6 GFLOP per call).

## Why it is a good test

- **The answer depends on the device.** GPU: one `map-fission` (54x). CPU: the fission and two interchanges (1.62x on 1 thread, 1.45x on 72). Each device's answer is a trap on the other: the CPU route continued to 6 moves makes the GPU 6.8x slower than the default (0.15x), and the fission alone gains nothing on CPU (1.00x).
- **What the default does instead:** `default_moves()` keeps `map (b, k, p) { upd = 0; for d { for l { map q (reduce) } } }` plus the update map. On GPU each `(b, k, p)` entry is a 256-thread block whose threads share the 9-term `q` reduction (block reduction, `__syncthreads()`). On CPU every entry re-reads `kDivM[d, k, :]` and `star[d, :, p]`.
- **What makes it hard:** no interchange is offered from stage 1. `map-fission` of the `p` map (init from sum) unlocks the `p` pushes; it gains nothing alone on CPU.
- **The source form matters:** written as one perfect nest (`Q += ...` inside the loops), the frontend lifts it to a single `einsum` library node and stage 2 has no move at all (job 5127053).
- **Tempting wrong moves:** continuing past 3 moves (a second fission of the `k` map and two more pushes: no CPU gain, 0.15x on GPU); the swap of `b`, `k` (0.78x at 72 threads); walk sequence 1, three swaps that end with `k` outside `b` (0.13x at 72 threads).

## What the reference is

- **Baseline:** stage 1, then `default_moves()`.
- **CPU (1 and 72 threads):** 3 moves, then `finish_moves()`: `map b { map k { init; for d { for l { map p { map q (reduce) } } } } }`, then the update map.
- **GPU:** one `map-fission` of the `p` map, then `finish_moves()`.

## How it was created

From the corpus einsum, written output-stationary. The first version (perfect nest, job 5127053) had no moves; the second adds the zero init and the separate time-step update, as the ADER-DG step applies it. Probed as a hard-tier (H2) candidate; the timing showed that only the first 3 of the 6 moves found by the walk pay.

## What the solution is

1. `map-fission` on `kernel3_0` (the `p` map): init and sum become two maps.
2. `interchange-map-loop` on (`kernel4_1`, `for5_0`): the `p` sum map goes inside `l`.
3. `interchange-map-loop` on (`kernel5_0`, `for6_0`): and inside `d`.
4. `finish_moves()`.

GPU: move 1, `finish_moves()`. Labels from jobs 5127372 and 5127863; they are not stable across versions.

## How to optimize it

Preset M, median of 5 calls after one warm-up, whole-program host time (GPU includes copies). Ratio is default time over variant time.

| Variant | CPU 1 thread | CPU 72 threads | GPU |
| --- | --- | --- | --- |
| baseline (default) | 1952.69 ms | 27.99 ms | 287.73 ms |
| stage 1, `finish_moves()` | 1951.93 ms (1.00x) | 28.00 ms (1.00x) | 287.69 ms (1.00x) |
| `map-fission` alone | 1944.43 ms (1.00x) | 27.79 ms (1.01x) | **5.37 ms (53.6x)** |
| `subgraph-fission` alone | 1952.34 ms (1.00x) | 27.86 ms (1.00x) | 10.39 ms (27.7x) |
| moves 1 to 3 (`p` inside the sums) | **1203.10 ms (1.62x)** | **19.35 ms (1.45x)** | 6.93 ms (41.5x) |
| 6 moves (also fission of `k`, `k` inside the sums) | 1211.53 ms (1.61x) | 19.66 ms (1.42x) | 1952.18 ms (0.15x) |
| swap `b`, `k` | 1982.65 ms (0.98x) | 35.82 ms (0.78x) | 291.50 ms (0.99x) |
| swap `k`, `p` | 1951.31 ms (1.00x) | 28.02 ms (1.00x) | 289.71 ms (0.99x) |
| swap `b`, `k` of the update map | 1957.06 ms (1.00x) | 28.14 ms (0.99x) | 289.41 ms (0.99x) |
| swap `k`, `p` of the update map | 1955.62 ms (1.00x) | 27.97 ms (1.00x) | 287.68 ms (1.00x) |
| walk sequence 1 (swaps `b`/`k`, `k`/`p`, `b`/`k`) | 1963.28 ms (0.99x) | 217.44 ms (0.13x) | 290.20 ms (0.99x) |
| walk sequence 2 (two fissions, a swap) | 1949.92 ms (1.00x) | 27.92 ms (1.00x) | 464.67 ms (0.62x) |
| walk sequence 3 (fission, push, swap, fission, swap) | 1888.27 ms (1.03x) | 27.11 ms (1.03x) | 24.01 ms (12.0x) |

## What the gains are

CPU 1 thread 1952.69 ms against **1203.10 ms (1.62x)**; CPU 72 threads 27.99 ms against **19.35 ms (1.45x)**, both with 3 moves; GPU 287.73 ms against **5.37 ms (53.6x)** with 1 move. Every timed variant matches the oracle at preset S (relative error at most 3.9e-12) and the baseline's output at preset M (CPU 0.0, GPU at most 3.0e-8).

## CPU and GPU

GPU wants the fission only; the CPU pushes cost it part of the gain (41.5x at 3 moves) and the 6-move route loses to the default. CPU wants the fission plus the `p` pushes; the fission alone does nothing there.

## What else matters

- The manifest declares no small symbols: with NQ, ND declared small (`public.scales`), stage 1 builds a different nest and the route is not offered (jobs 5127775 to 5127781), as for H1, H3 and M8 (`a6c787e`).
- `NBF 56` is the order-6 basis size; preset M's `NB 4003` is not a multiple of a block size.

## Reference solution

```yaml
status: entry
preset: M    # NB 4003, NBF 56, NQ 9, ND 3
dace: 042413df2b96ed0f2ebb361c4879c7f68bad41d2
nestforge: interchange-benchmarks (7c6a69d) plus this kernel
hpcagent_bench: 2666c06b73eaa9d51f05568e6c584aaafa98ad67
host: Daint GH200 node, exclusive
references:
  - device: cpu
    threads: 1
    timing: job 5127863
    baseline:
      moves: [default_moves]
      time_ms: 1952.69   # min 1951.37, max 1953.60
      repeats: 5
      oracle_rel_error: 3.9e-12   # preset S
      tree: |
        map (b, k, p) { upd = 0; for d { for l { map q (reduce) } } }; map (b, k, p) { Q += dt * upd }
    reference:
      moves:
        - kind: map-fission
          on: kernel3_0, the p map (init, sum)
        - kind: interchange-map-loop
          on: kernel4_1, for5_0 (p sum map inside l)
        - kind: interchange-map-loop
          on: kernel5_0, for6_0 (inside d)
        - finish_moves: true
      time_ms: 1203.10   # min 1202.03, max 1204.16
      repeats: 5
      oracle_rel_error: 3.9e-12
      tree: |
        map b { map k { map p (init); for d { for l { map p { map q (reduce) } } } } }; map (b, k, p) { Q += dt * upd }
  - device: cpu
    threads: 72
    timing: job 5127864
    baseline: {moves: [default_moves], time_ms: 27.99, repeats: 5, oracle_rel_error: 3.9e-12}   # min 27.71, max 28.50
    reference: {moves: [the 3 CPU moves, finish_moves], time_ms: 19.35, repeats: 5, oracle_rel_error: 3.9e-12}   # min 18.33, max 19.97
  - device: gpu
    timing: job 5127865
    baseline:
      moves: [default_moves]
      time_ms: 287.73    # min 287.48, max 290.06
      repeats: 5
      oracle_rel_error: 1.4e-12
      tree: |
        map (b, k, p) { ... }   # kernel: grid (NQ, NBF, NB), 256 threads per entry sharing the 9-term q reduction
    reference:
      moves: [map-fission kernel3_0, finish_moves]
      time_ms: 5.37      # min 5.21, max 5.48
      repeats: 5
      oracle_rel_error: 3.9e-12
decoys:
  - moves: [the 6-move route]
    time_ms: {cpu_1: 1211.53, cpu_72: 19.66, gpu: 1952.18}
  - moves: [interchange-map-map b, k]
    time_ms: {cpu_1: 1982.65, cpu_72: 35.82}
  - moves: [map-fission]          # on CPU
    time_ms: {cpu_1: 1944.43, cpu_72: 27.79}
backtest:
  default_moves: {cpu_1: 1952.69, cpu_72: 27.99, gpu: 287.73}
  each_stage1_move_alone: see the table above (6 moves)
  walk_sequences: see the table above (3 sequences from job 5127112's walk)
```
