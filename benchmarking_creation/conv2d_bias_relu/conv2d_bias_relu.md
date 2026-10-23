# conv2d_bias_relu (H2 contender, hard, CPU)

Explanation and solution record (`benchmarking_creation/AGENTS.md`) of a kernel derived from the HPCAgent-Bench `machine_learning` conv2d family (`2666c06`). Written 2026-10-23 from Daint jobs 5126537 (move walk from stage 1, CPU and GPU), 5126691 and 5126748 (route replay), 5126804 and 5126805 (stage 3 to 5 builds against the oracle, CPU and GPU), 5127637 (quickstart of this folder). Experiment note: `2026-10-23-h2-contenders` (thesis notes). How to optimize it, in short: [README.md](README.md).

**Status:** `entry`, not timed. Route legal and oracle-checked on CPU and GPU at preset S; the gap to the default at preset M is not measured yet.

Versions: NestForge `h2/hard-contenders` (from `ib/on-main` `a823380`; the probes ran on `ib/on-main` `f9081bd`), DaCe `extended` `042413d`, HPCAgent-Bench `2666c06`.

## What this benchmark is

One convolution layer, `out[n, co, oh, ow] = relu(b[co] + sum_{ci, kh, kw} x[n, ci, oh + kh, ow + kw] * wt[co, ci, kh, kw])`, stride 1, input already padded. The DaCe program is written output-stationary, the way a direct or thread-per-output convolution is written: for each output element, the bias init, three summed loops accumulating into `out` in place, then ReLU. `n, co, oh, ow` are parallel; `ci, kh, kw` are reductions. The scheduling problem: on CPU, move the two summed loops `ci` and `kh` outside the output-plane maps `oh` and `ow`, so each weight is applied to a contiguous row of `x` and `out`.

Why this shape survives stage 1: only the innermost summed loop (`kw`) becomes a reduction map; `ci` and `kh` add into `out` in memory and stay loops. The init and the ReLU around the sums stop the frontend from lifting the nest to one library node (a perfect nest of the same kind became an `einsum` node, job 5127053).

## Why it is a good test

- **What the default does instead:** `default_moves()` collapses `n, co, oh, ow` into one 4-D map with `for ci { for kh { map kw (reduce) } }` inside: one thread per output, the sums strided through `x`. Moves left after it: one `map-fission`, two `subgraph-fission` (job 5126537).
- **What makes it hard:** none of the interchanges is offered from stage 1; only three `interchange-map-map` swaps, a `map-fission` and two `subgraph-fission` are. The route has two unlocks. `map-fission` of the `ow` map (splitting init, sum and ReLU) is what makes the first `interchange-map-loop` legal; then the fusion of the two bias-init maps is what makes `map-fission` of the `oh` map appear (job 5126691). Neither unlock gains anything by itself.
- **The answer depends on the device:** on GPU the output-stationary form, one thread per output, is the usual good schedule; the CPU route on GPU builds and is correct (job 5126805) but serializes `ci` and `kh` around the output-plane maps.
- **Tempting wrong moves:** the `interchange-map-map` swaps of `n`, `co`, `oh`, `ow` (always offered, change nothing that matters or move `ow` off the contiguous axis); stopping after the `ow` pushes (3 moves) with `oh` still outside the summed loops; `subgraph-fission` of the `ow` body (offered from stage 1, leads nowhere on the route).

## What the reference is

- **Baseline (CPU and GPU):** stage 1, then `default_moves()`: `map (n, co, oh, ow) { init; for ci { for kh { map kw (reduce) } }; relu }`.
- **Reference (CPU):** the 7 moves below, then `finish_moves()`: `map n { map co { init maps; for ci { for kh { map oh { map ow { map kw (reduce) } } } }; relu map } }`.
- **GPU:** the baseline is the expected reference. Not timed.

Both match the oracle at preset S on CPU (8 OpenMP threads; relative error 9.1e-12, job 5126804) and on GPU (8.1e-12 baseline, 9.1e-12 reference, job 5126805).

## How it was created

From the output-stationary loop nest of the conv2d family's `_conv2d` (one group, stride 1), with bias and ReLU as in `conv2d_*_relu` kernels. Chosen on 2026-10-23 as an H2 contender outside the set's existing mechanisms (CloudSC column carry, column physics, map pull-out, ADI sweeps); 6 other contenders probed the same day were dropped (see the experiment note).

## What the solution is

1. `map-fission` on `kernel4_0` (the `ow` map): init, sum and ReLU become three maps.
2. `interchange-map-loop` on (`kernel5_2`, `for6_0`): the `ow` sum map goes inside the `ci` loop.
3. `interchange-map-loop` on (`kernel6_0`, `for7_0`): and inside the `kh` loop.
4. `map-fusion` on (`kernel5_0`, `kernel5_1`): the two bias-init maps. Unlocks the next fission.
5. `map-fission` on `kernel3_0` (the `oh` map).
6. `interchange-map-loop` on (`kernel4_1`, `for5_0`): the `oh` map goes inside `ci`.
7. `interchange-map-loop` on (`kernel5_1`, `for6_0`): and inside `kh`.
8. `finish_moves()`.

Labels from jobs 5126748 and 5126804; they are not stable across versions.

## How to optimize it

| Decision | What happens | Evidence |
| --- | --- | --- |
| `default_moves()` | one 4-D output map, sums inside | 5126537, built 5126804 |
| move 1 only | init, sum and ReLU split; no reorder | built 5126804 |
| moves 1 to 3 | `ow` inside `ci`, `kh`; `oh` still outside | built 5126804, 5126805 |
| moves 1 to 7 | `oh` and `ow` inside `ci`, `kh` | built 5126804, 5126805 |
| `interchange-map-map` (`n`, `co`) | swap only | built 5126804 |

## What the gains are

Not measured. Preset M is `N 8, CI 64, CO 64, H 57, W 57, KH 3, KW 3`. Expected (a guess, not measured): on CPU the reference applies each `wt[co, ci, kh, kw]` to a contiguous row of `out` and `x`, which vectorizes over `ow` and keeps the `H x W` output plane in cache, while the baseline strides through `x` across `ci` for every output. To be timed in the batch session at 1 thread, 72 threads and on GPU.

## CPU and GPU

CPU: the 7-move route. GPU: the default (one thread per output); the CPU route is the trap. Both devices build every variant above.

## What else matters

- The kernel row and column counts `KH`, `KW` are declared small (`public.scales`).
- Preset M's `H`, `W` (57) are not multiples of a block size.

## Before calling it done

- [x] Oracle, manifest, DaCe program, `initialize()`, this record, README.
- [x] Route replayed and built at S on CPU and GPU, oracle checked.
- [ ] Timed at M (1 thread, 72 threads, GPU): baseline, reference, moves 1 to 3, a swap decoy.
