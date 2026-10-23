# graupel_sedim2b (slot M5.2)

Explanation and solution record (`benchmarking_creation/AGENTS.md`) of a synthetic kernel. Written 2026-10-23 from Daint jobs 5126471 (replay at S), 5126528 to 5126531 (timing). Earlier probes and timings: `2026-10-22-m52-graupel-sedim-timing` and `2026-10-22-h2-alternatives` (thesis notes).

**Status:** `entry`, timed at NestForge main's pins. CPU at 72 threads beats the default 15x to 20x; the route beats its one-move decoy 1.28x to 1.33x (two jobs on two nodes).

## What this benchmark is

Sedimentation of `NSPEC` precipitation species in a column model, written in the Fortran order. Each column `jl` takes `nsub[jl]` of `NSUB` substeps. In a substep each species falls down the levels: the flux leaving level `jk` enters level `jk + 1` (carried in `fl[jl]`), and species `js` also gains `w[js]` of what species `js - 1` lost at the same level (`G[jk, jl]`, overwritten in place). `q` and `c` are `(NSPEC, NLEV, KLON)`, `G` is `(NLEV, KLON)`, the rest `(KLON,)`; `KLON` is contiguous. The substep, species and level loops carry; the columns are independent. The column's precipitation `prec` sums the last species' outgoing flux of each substep.

The program is `for it { for js { for jk { map jl { sweep } } }; map jl { prec } }`. The scheduling problem: the column map sits under three sequential loops, so each level of each species of each substep opens a parallel region with `KLON` small iterations. Synthetic, in the style of the graupel/rain sedimentation of ICON's and CloudSC's microphysics.

## Why it is a good test

- **What the default does instead:** `default_moves()` leaves the stage 1 nest, `NSUB * NSPEC * NLEV` parallel regions per call (2160 at preset M), each with 1031 columns of a few flops.
- **What makes it hard:** two pulls of the same kind in a fixed order. The species pull is offered only after the level pull. The first pull already gives most of the gain (24 regions), so an agent that stops when a move pays off stops one move short.
- **What it covers:** `interchange-loop-map` twice, with the second unlocked by the first; a decision that depends on the thread count.
- **What a wrong decision looks like:** stopping after the level pull (the decoy, 1.28x to 1.33x slower than the route at 72 threads); trusting the 72-thread answer on a GPU, where nothing gains.

## What the reference is

- **Baseline:** stage 1, then `default_moves()`: the stage 1 tree (below).
- **Reference (CPU, 72 threads):** `interchange-loop-map` (level loop into the sweep's column map), `interchange-loop-map` (species loop into it), `finish_moves()`: `for it { map jl { for js { for jk } }; map jl }`, 16 parallel regions per call.
- **GPU:** no move. The default is within 10% of every variant.

## How it was created

From the 2026-10-22 H2 sketches. `graupel_sedim` (no species loop) had a one-move route. `graupel_sedim2` added the species loop but kept the flux in `F[js + 1, jk + 1, jl]`, an offset access, and `MoveLoopIntoMap` refused both pulls (job 5119404). `graupel_sedim2b` reads and writes every flux at the same element (`fl[jl]`, `G[jk, jl]`), and both pulls are offered. At DaCe `801cafd` the route lost its OpenMP loop after the second pull (widened memlets, spcl/NestForge#24) and needed memlets re-propagated (job 5119636). On main, PR #31 narrows them, and the route keeps its pragma with no workaround. `graupel_sedim3`, written to get a slower decoy, has no move that swaps a loop directly under a map (spcl/NestForge#29), so it was dropped.

## What the solution is

Labels from job 5126529 (CPU, main's pins).

1. `interchange-loop-map` on (`for2_0`, `kernel4_0`): the level loop into the sweep's column map.
2. `interchange-loop-map` on (`for1_0`, `kernel3_0`): the species loop into the column map. Not offered before move 1.
3. `finish_moves()`.

Tree after the moves:

```
|- state0_0
`- for0_0  _loop_it_0=0:NSUB
   `- state1_0  kernels=2
```

## How to optimize it

After stage 1, `list_moves` offers one `interchange-loop-map` (the level pull). After it, one more (the species pull). After `finish_moves()`, none. `default_moves()` takes neither. The trap is to stop after the first pull, which already gives 12x to 15x at 72 threads. The substep loop cannot be pulled: the precipitation map in the same loop body reads the last species' flux of each substep.

## What the gains are

Preset M (`NSPEC` 3, `NLEV` 90, `KLON` 1031, `NSUB` 8, every column all substeps), whole-program host time, median of 5 calls after one warm-up, default stage 5 build, `-fopenmp` on the CPU kernels, `OMP_PROC_BIND=close`, `OMP_PLACES=cores`. Every variant matches the oracle at S (relative error 0) and the baseline at M (0).

| Build | Baseline | Level pull (decoy) | Reference | Reference over baseline | Reference over decoy |
| --- | --- | --- | --- | --- | --- |
| CPU 72 threads (job 5126529) | 10.69 ms [10.60, 10.74] | 0.72 ms [0.66, 0.92] | **0.54 ms [0.48, 0.56]** | **19.7x** | 1.33x |
| CPU 72 threads, repeat (job 5126530) | 8.15 ms [8.08, 8.44] | 0.69 ms [0.66, 1.88] | **0.54 ms [0.48, 0.54]** | **15.1x** | 1.28x |
| CPU 1 thread (job 5126528) | 3.29 ms [3.18, 3.45] | **2.64 ms** [2.57, 2.67] | 2.86 ms [2.80, 3.09] | 1.15x | 0.92x |
| GPU (job 5126531) | 1.93 ms [1.89, 2.04] | 1.88 ms | 1.78 ms [1.73, 1.83] | 1.09x | 1.06x |

Stage 1 without `default_moves()` is the baseline's time on CPU and 25.3 ms on GPU. Re-propagating memlets (`reference_prop`) gives the same times as the reference.

## CPU and GPU

The gap exists on a threaded CPU. On one thread the first pull is the best schedule and the route is 8% slower than it. On GPU the default already lifts the column map, and all variants are within 10%.

## What else matters

- **Legality:** an offset access on a flux array (`F[js + 1, jk + 1, jl]`) blocks both pulls; the program keeps every flux at one element.
- **Contiguity and stride:** `KLON` is contiguous. After the pulls each thread walks its own column across levels and species, at stride `KLON`; the gain comes from the region count, not from stride.
- **Launches against stride:** 2160 parallel regions in the baseline, 24 after the first pull, 16 in the reference.
- **Decoys:** the level pull alone, measured (1.28x to 1.33x slower than the route at 72 threads, within one job each).
- **Oracle:** `q` uniform in `[0, 1e-3]`, `c` in `[0, 0.9]`, `ftop` in `[0, 1e-4]`, `w = [0, 0.3, 0.5]`, every column 8 substeps. Relative error 0.
- **Pins:** DaCe `042413d`, NestForge `ib/on-main` (origin/main `8545a4b` plus `interchange-benchmarks`), HPCAgent-Bench `2666c06`.

## Reference solution

```yaml
device: cpu
threads: 72
preset: M
dace: 042413df2b96ed0f2ebb361c4879c7f68bad41d2
nestforge: ib/on-main (origin/main 8545a4b + interchange-benchmarks 82bb3c3)
hpcagent_bench: 2666c06b73eaa9d51f05568e6c584aaafa98ad67
probe: Daint job 5126529, repeat 5126530
baseline:
  moves: [default_moves]
  time_ms: 10.69
  time_min_ms: 10.60
  time_max_ms: 10.74
  repeats: 5
  oracle_rel_error: 0.0
reference:
  moves:
    - kind: interchange-loop-map
      on: the level loop around the sweep's column map (for2_0, kernel4_0)
    - kind: interchange-loop-map
      on: the species loop around that map (for1_0, kernel3_0)
    - finish_moves: true
  time_ms: 0.54
  time_min_ms: 0.48
  time_max_ms: 0.56
  repeats: 5
  oracle_rel_error: 0.0
decoys:
  - moves: [interchange-loop-map (level loop), finish_moves]
    time_ms: 0.72
```
