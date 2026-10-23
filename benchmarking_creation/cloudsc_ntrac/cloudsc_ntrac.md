# cloudsc_ntrac (slot M8)

Explanation and solution record (`benchmarking_creation/AGENTS.md`) of a kernel derived from CloudSC. Written 2026-10-21 from Daint jobs 5112828, 5112859, 5112883, 5112902, 5112961 and 5113005. Set: `benchmark-set-15` (thesis notes). Entry note: `2026-10-21-w11c-cloudsc-ntrac` (thesis notes).

**Status:** `entry`, timed 2026-10-23 at NestForge main's pins. Target: GPU, 2.29x at preset M, which since 2026-10-23 has 1031 columns (it had 32771, where the route gained only 1.11x).

## What this benchmark is

`NTRAC` tracers sediment down the levels of each column, and a precipitation cover runs down the same levels, in one level loop `jk`. The tracer flux `pflux[jm, jk, jl]` is a recurrence over levels for each (tracer, column) pair: it reads `pflux[jm, jk - 1, jl]`, and the new tracer value is clamped at 0. The cover `pcov[jk, jl]` is a recurrence over levels for each column alone: it reads `pcov[jk - 1, jl]`, is floored at 0.1 and is gated to 0 where `prec` (last step's precipitation, an input) is 0. The two recurrences share nothing. `q` and `pflux` are (NTRAC, NLEV, KLON), the others (NLEV, KLON), with `KLON` contiguous; `jm` and `jl` are parallel, and `jk` is carried. The scheduling problem: on a GPU, the level loop should sit inside one kernel per recurrence, but the two recurrences share one level loop and must be split before either can be inverted. Source: CloudSC (`ecmwf-ifs/dwarf-p-cloudsc`, `src/cloudsc_fortran/cloudsc.F90`) 4.2 per-species fall (1715–1755) and cover (1757–1789), 5.2 diagonal (2594–2610), clamp (2681–2698), 5.3 flux (2700–2719), with the edits listed under "How it was created".

## Why it is a good test

- **What the default does instead:** stage 1 and `default_moves()` leave `for jk { map[jl] (cover); map[jm, jl] (tracers) }`, two GPU kernel launches per level, `2 × (NLEV − 1)` launches in all (272 at preset M), each moving a few MB. The reference has 2 launches.
- **What makes it hard:** no interchange is legal at stage 1, because the level loop holds two maps and `interchange-loop-map` needs exactly one. `loop-fission` gains nothing alone (it only splits one loop into two with the same launch count); it is the move that unlocks both interchanges. The tracer interchange is offered only after `finish_moves()` has collapsed the tracer maps `[jm] { [jl] }` into one `[jm, jl]` map (see "What else matters"). The answer depends on the device: on a CPU the same moves give a column walk at stride `KLON` and are expected to be slower.
- **What it covers:** `loop-fission` and `interchange-loop-map`. It is the only kernel in the set whose route starts with `loop-fission`.
- **What a wrong decision looks like:** `loop-fusion` right after the fission (undoes it), `interchange-map-loop` after an interchange (undoes it), stopping after one interchange (half the gain), or applying the route on CPU.

## What the reference is

- **Baseline:** stage 1, then `default_moves()`: `for jk { map[jl] cover; map[jm, jl] tracers }`.
- **Reference (GPU):** `loop-fission`, `interchange-loop-map` (cover), `finish_moves()`, `interchange-loop-map` (tracers), `finish_moves()`: `map[jm, jl] { for jk }; map[jl] { for jk }`.
- **Other device (CPU):** no move. The baseline keeps the level loop outside and the contiguous `jl` innermost; the GPU route is a decoy there (not timed).

## How it was created

From design R2 of the 2026-10-09 CloudSC-style benchmark notes (about 60 lines). Edits from CloudSC: `NTRAC` independent tracers (as for the IFS-AER aerosols) with fall speeds spread over CloudSC's `ZVQX` (ice 0.13, snow 1.0, rain 4.0 m/s), no off-diagonal solver terms, the flux stored as "out of level `jk`" instead of CloudSC's `ZPFPLSX(JK+1)`, and the cloud top at level 1. Inputs come from `initialize()` in `cloudsc_ntrac.py`: `dtgdp = dt g / dp` for a 3600 s step and a 450 to 1800 Pa layer, `rdtgdp` its inverse, a density profile, dry columns (a third) and slightly negative `q` (10%), so that both the cover gate and the clamp bind.

Four spellings failed on the way, each on a check that is recorded here because it shapes the kernel:

1. **v0, the design as sketched (job 5112828):** the cover gated by this level's rain plus snow flux, `pflux[2, jk, :] + pflux[3, jk, :]`. Stage 1 offers only 52 `map-fusion` and no `loop-fission`. DaCe's `LoopFission` cuts a same-level bridge only when every dimension of the access is a single point (`_is_per_iter_subset`, `loop_fission.py`), and this bridge is a row of columns, so both halves stay in one group. The input `prec` replaces the bridge.
2. **v1, slice spelling with `prec` (job 5112859):** `loop-fission` is offered, but each half is many small column maps, and the `(KLON,)` temporaries are shared across tracers, so no interchange follows. Stage 5 fails on them too (`KeyError('pflux_slice')` in `abi_ready_copy`). The kernel is written as scalar loops from v2 on.
3. **v2, scalar loops with `qn` stored (job 5112883):** builds and matches the oracle, but stage 1 splits the tracer body into a `pflux` map and a recomputing `qn` map, so the tracer half holds two maps again. `qn` is now a scalar.
4. **v3, `qn` a scalar, gate as an `if` (job 5112902):** the `if` becomes a conditional block inside the cover map, and the cover interchange is not offered. The gate is now arithmetic: `max(cov, 0.1) * min(prec * 1e14, 1.0)`, exactly 0 or 1 because `prec` is 0 or at least 1e-6 (variant b, jobs 5112902 and 5112961).

A further variant (c, job 5112961) carried the tracer loop with a column total over `jm`. It was pre-solved: stage 1 already offers both interchanges, and on GPU `default_moves()` inverts everything.

## What the solution is

Route 1:

1. `loop-fission` on `for0_0`: splits the level loop into `for0_0` (tracers, writes `pflux`) and `for0_1` (cover, writes `pcov`). Legal because the halves share no written array.
2. `interchange-loop-map` on `for0_1`, `kernel2_1`: moves the cover's level loop into its column map, `map[jl] { for jk }`. Not offered before step 1, because the loop held two maps.
3. `finish_moves()`: collapses the tracer maps `[jm] { [jl] }` into `[jm, jl]`.
4. `interchange-loop-map` on `for0_0`, `kernel2_0`: moves the tracers' level loop into the 2-D map, `map[jm, jl] { for jk }`. Not offered before step 3 (DaCe counts the nested `[jl]` map as a second map).
5. `finish_moves()`.

Route 2 (same tree, job 5113005, CPU and GPU): `loop-fission`, `finish_moves()`, then `interchange-loop-map` twice (both pairs are offered after the first `finish_moves()`), then `finish_moves()`.

Labels are from job 5113005, at the pins in the solution block; they are not stable across versions.

Tree after the last move and `finish_moves()` (loop nest only; the bodies are in the job's `R0-F-after-finish.tree.txt`):

```
|- state0_0
`- state0_1
   |- kernel1_0  [_loop_it_1=0:NTRAC, _loop_it_2=0:KLON]  writes=['pflux']
   |  `- for2_0  _loop_it_4=1:NLEV
   `- kernel1_1  [_loop_it_3=0:KLON]  writes=['pcov']
      `- for2_1  _loop_it_4=1:NLEV
```

## How to optimize it

Right after stage 1, `list_moves` offers `loop-fission` (1) and `interchange-map-map` (1, on the tracer maps `[jm]` and `[jl]`). `default_moves()` only collapses the tracer maps, and leaves `loop-fission` offered. The fission must come first; the tracer interchange needs a `finish_moves()` (or the collapse it does) before it is offered.

| Decision | What happens | Evidence |
| --- | --- | --- |
| `interchange-loop-map` at stage 1 | not offered: two maps in the level loop | 5113005 |
| `loop-fusion` after the fission | offered; fuses the halves back (undoes step 1) | 5113005 |
| `interchange-map-loop` after an interchange | offered; the inverse move, undoes it | 5113005 |
| stopping after the cover interchange | builds and matches the oracle; tracers still launch once per level; not timed | 5113005 (`R0-half`) |
| the full route on CPU | builds and matches the oracle; column walk at stride `KLON`; not timed | 5113005 |

## What the gains are

Whole-program host time (GPU times include host-device copies), median of 5 calls after one warm-up, default stage 5 build, `-fopenmp` on CPU kernels. DaCe `042413d`, HPCAgent-Bench `2666c06`, NestForge `ib/on-main`, 2026-10-23.

**Why preset M changed.** At the old M (`NLEV` 137, `KLON` 32771, `NTRAC` 16) the route gains 1.11x on GPU (38.6 ms against 34.8 ms, job 5126626): the host-device copies of the (`NTRAC`, `NLEV`, `KLON`) arrays dominate and hide the 272 launches the route removes. GPU sweep of preset M (jobs 5127148 to 5127151):

| `KLON` | `NLEV` | Baseline | Reference | Ratio | Half route |
| --- | --- | --- | --- | --- | --- |
| 32771 | 137 | 38.58 ms | 34.78 ms | 1.11x | 1.05x |
| 4099 | 137 | 10.57 ms | 6.80 ms | 1.55x | 1.21x |
| **1031** | **137** | 6.55 ms | 2.93 ms | **2.24x** | 1.38x |
| 1031 | 500 | 20.96 ms | 7.60 ms | 2.76x | 1.47x |
| 257 | 1000 | 32.64 ms | 6.36 ms | 5.13x | 1.67x |

Preset M is now `NLEV` 137, `KLON` 1031, `NTRAC` 16 (CloudSC's level count; 1031 columns, a GPU block of the size a blocked driver hands one kernel). L is 4099 columns and XL 32771.

**At the new preset M** (jobs 5127296, 5127297, 5127298):

| Variant | GPU | CPU 1 thread | CPU 72 threads |
| --- | --- | --- | --- |
| baseline (default) | 6.51 ms | 3.63 ms | 2.17 ms |
| reference (5 steps) | **2.85 ms (2.29x)** | 25.23 ms (0.14x) | 2.15 ms (1.01x) |
| half route (cover only) | 4.77 ms (1.37x) | 4.88 ms (0.74x) | **1.40 ms (1.55x)** |
| `loop-fission` + `loop-fusion` | n/a | 3.81 ms (0.95x) | 2.25 ms (0.96x) |

Oracle relative error at preset S: 0.0 for `pflux` and `pcov` on every variant, CPU and GPU.

Oracle relative error at preset S: 0.0 for `pflux` and `pcov`, for the baseline, both routes and the half route, on CPU and GPU.

## CPU and GPU

Stage 1 leaves the same tree on both devices, and the same moves are legal on both. The benchmark claims the GPU: the full route, 2.29x. On CPU the answer is different. On 1 thread no move helps: the route makes one column per iteration at stride `KLON` and is 7x slower, and the half route loses too. At 72 threads the half route (cover interchanged, tracers kept under the level loop) is the best schedule, 1.55x, and the full route ties the default. So the same five moves are right on GPU, a trap on one CPU thread, and one move too many at 72 threads.

Without the `NTRAC` small declaration: declaring `NTRAC` small (`public.scales`) turns the tracer map into a sequential loop at stage 1, and the route above is not offered (2026-10-23, walk job 5126614). The manifest does not declare it.

## What else matters

- **Legality:** `LoopFission` refuses a same-level bridge over a row (v0). `MoveLoopIntoMap` (DaCe `801cafd`, `move_loop_into_map.py`, "Check if body contains exactly one map") counts every map entry in the loop body, nested ones included, so the tracer half `[jm] { [jl] }` is refused until it is collapsed. A conditional block in the cover map blocks the cover interchange (v3).
- **Contiguity and stride:** `KLON` is contiguous. The GPU reference puts `jl` on threads (coalesced at each level); on CPU it walks `jk` innermost at stride `KLON`.
- **Launches against stride:** the route removes launches (good on GPU) and breaks the contiguous inner walk (bad on CPU).
- **Vectorization:** not checked. Stage 7 of the CPU quickstart reports "loop not vectorized: complicated access pattern" for the tracer kernel of the baseline.
- **Live values and carries:** none in the maps; the carries are the arrays `pflux` and `pcov`, read at `jk − 1`.
- **Oracle:** at preset S, `pcov` is 0 in 33% of cells (dry columns), at its floor 0.1 in 0.2%, and the clamp sets 11% of the `pflux` values to 0.
- **Backtest:** not run.
- **Pins:** DaCe `801cafd`, NestForge `w11c/cloudsc-ntrac` on `7c5480f`, HPCAgent-Bench `e63f9f1`.

## Reference solution

```yaml
status: entry
device: gpu
threads: 1
preset: M    # NLEV 137, KLON 1031, NTRAC 16
dace: 042413df2b96ed0f2ebb361c4879c7f68bad41d2
nestforge: ib/on-main (origin/main 8545a4b + interchange-benchmarks)
hpcagent_bench: 2666c06b73eaa9d51f05568e6c584aaafa98ad67
probe: Daint job 5113005 (route, preset S); timing 5127298 (GPU), 5127296 and 5127297 (CPU)
host: Daint GH200 node, exclusive
baseline:
  moves: [default_moves]
  time_ms: 6.51
  repeats: 5
  oracle_rel_error: 0.0   # preset S
  tree: |
    `- for0_0  _loop_it_4=1:NLEV
       `- state1_0
          |- kernel2_0  [_loop_it_3=0:KLON]  writes=['pcov']
          `- kernel2_1  [_loop_it_1=0:NTRAC, _loop_it_2=0:KLON]  writes=['pflux']
reference:
  moves:
    - kind: loop-fission
      on: for0_0, the level loop (splits tracers, writing pflux, from cover, writing pcov)
    - kind: interchange-loop-map
      on: for0_1, kernel2_1 (cover level loop into the column map)
    - finish_moves: true   # collapses the tracer maps [jm] { [jl] } into [jm, jl]
    - kind: interchange-loop-map
      on: for0_0, kernel2_0 (tracer level loop into the 2-D map)
    - finish_moves: true
  time_ms: 2.85
  repeats: 5
  oracle_rel_error: 0.0   # preset S
  tree: |
    `- state0_1
       |- kernel1_0  [_loop_it_1=0:NTRAC, _loop_it_2=0:KLON]  writes=['pflux']
       |  `- for2_0  _loop_it_4=1:NLEV
       `- kernel1_1  [_loop_it_3=0:KLON]  writes=['pcov']
          `- for2_1  _loop_it_4=1:NLEV
alternatives:
  - moves: [loop-fission, finish_moves, interchange-loop-map, interchange-loop-map, finish_moves]
    note: same tree (job 5113005)
decoys:
  - moves: [loop-fission, loop-fusion]
    effect: undoes the fission
    time_ms: null
  - moves: [loop-fission, interchange-loop-map, finish_moves]
    effect: half the gain (tracers still launch once per level)
    time_ms: 4.77
  - moves: [the reference moves, interchange-map-loop]
    effect: undoes one interchange
    time_ms: null
```

CPU: on 1 thread the reference is `default_moves()` (3.63 ms; the GPU route 25.23 ms); at 72 threads the half route is the best schedule (1.40 ms against 2.17 ms).
