# cloudsc_ntrac (slot M8)

Explanation and solution record (`benchmarking_creation/AGENTS.md`) of a kernel derived from CloudSC. Written 2026-10-21 from Daint jobs 5112828, 5112859, 5112883, 5112902, 5112961 and 5113005. Set: `benchmark-set-15` (thesis notes). Entry note: `2026-10-21-w11c-cloudsc-ntrac` (thesis notes).

**Status:** `entry`. The kernel, the route and its decoys are written and probed on Daint at preset S on CPU and GPU. Nothing is timed (W10).

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

Not timed yet (W10). Guess from the design: on GPU at preset M (`NLEV` 137, `KLON` 32771, `NTRAC` 16), 272 launches become 2, and each per-level tracer kernel moves about 13 MB, so the time is partly launch-bound: about 1.5 to 2.5× (a guess, nothing measured). The quickstart's stage 7 kernel times at preset S (job 5113005) are launch noise, not the gap.

Oracle relative error at preset S: 0.0 for `pflux` and `pcov`, for the baseline, both routes and the half route, on CPU and GPU.

## CPU and GPU

Stage 1 leaves the same tree on both devices, and the same moves are legal on both. The gap is claimed on GPU only. On CPU, the baseline already walks the contiguous `jl` innermost under the level loop. The route turns that into one column per iteration at stride `KLON`, which the 2026-10-02 CPU/GPU flip note found about 8.5× slower for a similar thread-per-column shape on one thread (a laptop smoke test, not a Daint measurement). W10 should time the CPU baseline and the route at 1 and 72 threads to confirm that CPU wants no move.

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
preset: M
dace: 801cafd
nestforge: 7c5480f + this kernel (branch w11c/cloudsc-ntrac)
hpcagent_bench: e63f9f1
probe: Daint job 5113005 (preset S, CPU and GPU, venv-e989491)
host: null
baseline:
  moves: [default_moves]
  time_ms: null
  time_min_ms: null
  time_max_ms: null
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
  time_ms: null
  time_min_ms: null
  time_max_ms: null
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
    time_ms: null
  - moves: [the reference moves, interchange-map-loop]
    effect: undoes one interchange
    time_ms: null
```

The same block for CPU: the reference is `default_moves()` (no move), and the GPU route is a decoy; to be filled in W10.
