# conv3d_box (H2 contender, hard, CPU)

Explanation and solution record (`benchmarking_creation/AGENTS.md`) of a kernel derived from HPCAgent-Bench `scientific_computing/structured_grids/conv_3d` (`2666c06`). Written 2026-10-23 from Daint jobs 5127348 (move walk from stage 1, CPU and GPU), 5127530 and 5127531 (stage 3 to 5 builds against the oracle, CPU and GPU), 5127637 (quickstart of this folder). Experiment note: `2026-10-23-h2-contenders` (thesis notes). How to optimize it, in short: [README.md](README.md).

**Status:** `entry`, not timed. Route legal and oracle-checked on CPU and GPU at preset S; the gap to the default at preset M is not measured yet.

Versions: NestForge `h2/hard-contenders` (from `ib/on-main` `a823380`; the probes ran on `ib/on-main` `f9081bd`), DaCe `extended` `042413d`, HPCAgent-Bench `2666c06`.

## What this benchmark is

A non-separable 3-D box stencil, `out[i, j, k] = sum_{di, dj, dk} w[di, dj, dk] * padded[i + di, j + dj, k + dk]` with `K = 2R + 1 = 5` taps per axis (125 taps), on an edge-padded grid. The corpus kernel writes it tap-outer in NumPy; the DaCe program here is written output-stationary (one grid point, zeroed, three tap loops inside), as a direct C or GPU port writes it. The scheduling problem: on CPU, move the tap loops `di` and `dj` outside the `j` and `k` maps, so each weight is applied to whole contiguous rows of the grid.

## Why it is a good test

- **What the default does instead:** `default_moves()` collapses `i, j, k` into one 3-D map with `for di { for dj { map dk (reduce) } }` inside. Moves left: one `map-fission`, one `subgraph-fission` (job 5127348).
- **What makes it hard:** from stage 1 only `map-fission`, `subgraph-fission` and two `interchange-map-map` swaps are offered. `map-fission` of the `k` map unlocks its first push; `map-fission` of the `j` map unlocks the `j` pushes. The route interleaves the two maps (job 5127348, path at depth 6).
- **Tempting wrong moves:** the `i, j, k` swaps; pushing only `k` (2 moves); pushing `i` too (the parallel axis).

## What the reference is

- **Baseline:** stage 1, then `default_moves()`.
- **Reference (CPU):** the 6 moves below, then `finish_moves()`: `map i { init; for di { for dj { map j { map k { map dk (reduce) } } } } }`.

Both match the oracle at preset S on CPU (8 OpenMP threads; relative error 0, job 5127530) and on GPU (1.3e-15, job 5127531).

## How it was created

From `conv_3d`'s sum, written as the output-stationary loop nest with a zero init (without the init, the perfect nest is lifted to a library node, as `seissol_volume`'s first version was).

## What the solution is

1. `map-fission` on `kernel3_0` (the `k` map).
2. `interchange-map-loop` on (`kernel4_1`, `for5_0`): `k` inside `di`.
3. `map-fission` on `kernel2_0` (the `j` map).
4. `interchange-map-loop` on (`kernel3_1`, `for4_0`): `j` inside `di`.
5. `interchange-map-loop` on (`kernel6_0`, `for7_0`): `k` inside `dj`.
6. `interchange-map-loop` on (`kernel4_1`, `for5_0`): `j` inside `dj`.
7. `finish_moves()`.

Labels from jobs 5127348 and 5127530; they are not stable across versions.

## How to optimize it

| Decision | What happens | Evidence |
| --- | --- | --- |
| `default_moves()` | one 3-D map, taps inside | 5127348, built 5127530 |
| moves 1 and 2 | `k` inside `di` only | built 5127530 |
| moves 1 to 6 | `j`, `k` inside `di`, `dj` | built 5127530, 5127531 |
| `interchange-map-map` (`i`, `j`) | swap only | built 5127530 |

## What the gains are

Not measured. Preset M is `N 257, K 5`. Expected (a guess): on CPU the reference streams contiguous `k` rows per weight; the baseline does 125 scattered reads per point. To be timed in the batch session.

## CPU and GPU

CPU: the 6-move route. GPU: the default (one thread per point) is the expected reference.

## What else matters

- Same mechanism as `conv2d_bias_relu` (fission unlocks, then pushes of the output maps inside the summed loops), with no fusion unlock and a stencil instead of a channel sum. If only one of the two enters the set, `conv2d_bias_relu` has the longer chain.

## Before calling it done

- [x] Oracle, manifest, DaCe program, `initialize()`, this record, README.
- [x] Route replayed and built at S on CPU and GPU, oracle checked.
- [ ] Timed at M (1 thread, 72 threads, GPU).
