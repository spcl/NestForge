# scattering_self_energies_m2 (slot M8, medium, CPU)

Explanation and solution record (`benchmarking_creation/AGENTS.md`) of a kernel derived from HPCAgent-Bench `scientific_computing/dense_linear_algebra/scattering_self_energies` (`e63f9f1`). Written 2026-10-21 from Daint jobs 5111782 (walk of the loop order, as variant v1), 5112069 (build check), 5112070 and 5112133 (walk and routes, as variant v1c) and 5112562 (validation of this folder). Set: `benchmark-set-15` (thesis notes). Entry note: `2026-10-21-w3-medium-refills` (thesis notes). How to optimize it, in short: [README.md](README.md). The 6-move original is the hard-tier candidate `scattering_self_energies`.

**Status:** `entry`

## What this benchmark is

The electron-phonon scattering self-energy of OMEN (via NPBench), `Sigma[k, E, a] += (G[k, E-w, neigh_idx[a, b]] @ dH[a, b, i]) @ (dH[a, b, j] * D[q, w, a, b, i, j])` behind the guard `E - w >= 0`, with the same oracle, arguments and presets as the corpus kernel. Two changes in the DaCe program, which is hand-written: the loop order is `k, E, q, w, a, b, i, j`, so the atom loop `a` sits under two reduction loops (`q`, `w`) instead of four, and the `10 x 10` products are explicit orbital loops `o1..o4` instead of `@`. `k`, `E`, `a` (and `o1`, `o2`) index the output; `q`, `w`, `b`, `i`, `j`, `o3`, `o4` are reductions. The scheduling problem: pull the atom map out of the two reduction loops so `(k, E, a)` is one parallel map.

## Why it is a good test

- **What the default does instead:** on CPU, `default_moves()` collapses `k, E` and `o1, o2` and leaves map `a` under `for q { for w }`, entered `Nqz * Nw` times per `(k, E)` point.
- **What makes it hard:** the second pull is offered only after the first; a move that looks similar (`interchange-map-loop` on map `a` and loop `b`, the unlock of the 6-move original) is offered at every step and goes the wrong way.
- **What it covers:** 2x `interchange-loop-map` in a real code, CPU only (the GPU default already makes the same schedule with lane maps), so the answer depends on the device.
- **What a wrong decision looks like:** `interchange-map-loop` on `a` and `b`; `interchange-map-map` on `k`, `E` or on the orbital maps; stopping after one pull.

## What the reference is

- **Baseline (CPU):** stage 1, then `default_moves()`: `map (k, E) { for q { for w { map a { for b { for i { for j { map (o1, o2) { guard { for o3 { for o4 } } } } } } } } } }`.
- **Reference (CPU):** 2x `interchange-loop-map`, then `finish_moves()`: `map (k, E, a) { for q { for w { ... } } }`.
- **GPU:** no move; `default_moves()` already gives `map (k, E, a_lane) { map (a, o1_lane) { for q { ... } } }`.

## How it was created

From `scattering_self_energies` (6 moves, and stages 3 to 5 refuse its body). Variant v1 reordered the nest to `k, E, q, w, a, b, i, j` (route of 2 pulls, job 5111782) but still failed to build (job 5112057, `KeyError`/`UnsupportedNest`); v1b (one-expression body) failed the same way (job 5112064), because the DaCe frontend makes `MatMult` library nodes for the `@` either way. v1c writes the products as explicit orbital loops and builds (job 5112069). v2 (`a` under one loop) was 1 move and the GPU default took it.

## What the solution is

1. `interchange-loop-map` on (`for4_0`, `kernel6_0`): loop `w` and map `a`; map `a` out past `w`.
2. `interchange-loop-map` on (`for3_0`, `kernel5_0`): loop `q` and map `a`; offered only after step 1.
3. `finish_moves()`: `k`, `E`, `a` collapse into one map.

Labels from job 5112133 (as v1c, same program); not stable across versions.

```
`- state0_1
   `- kernel1_0  [_loop_it_0=0:Nkz, _loop_it_1=0:NE, _loop_it_4=0:NA]  reads=['D', 'G', 'Sigma', 'dH', 'neigh_idx'] writes=['Sigma']
      `- for2_0  _loop_it_2=0:Nqz
         `- ... for w, for b, for i, for j, map (o1, o2), guard, for o3, for o4
```

## How to optimize it

After stage 1 (CPU and GPU), `list_moves` offers `interchange-loop-map` (`for4_0`, `kernel6_0`), `interchange-map-loop` (`kernel6_0`, `for7_0`), `interchange-map-map` (`kernel1_0`, `kernel2_0`) and (`kernel11_0`, `kernel12_0`). `default_moves()` on CPU takes none of the interchanges.

| Decision | What happens | Evidence |
| --- | --- | --- |
| `interchange-map-loop` on `a` and `b` | pushes loop `b` out of map `a`: map `a` entered `NB` times more often | job 5112133, route `decoy-map-loop`; not timed |
| `interchange-map-map` (`k`, `E`) | no structural gain on CPU | job 5112133 (offered at every step) |
| `interchange-map-map` on the orbital maps (`kernel11_0`, `kernel12_0`) | stage 1 already has `o1` outer and `o2` (the contiguous axis of `Sigma`) inner, even when the source writes `o2, o1`; the swap makes the inner map strided | job 5112135 (variant v1d); not timed |
| stopping after one pull | map `a` still under `q` | not timed |

## What the gains are

Not timed yet (W10). Preset M (`Nkz 4, NE 16, Nqz 4, Nw 2, N3D 3, NA 128, NB 4`), CPU, 1 and 72 threads.

| Build | Baseline (default) | Stage 1 only | Reference | Ratio |
| --- | --- | --- | --- | --- |
| CPU serial | | | | |
| CPU OpenMP, 72 threads | | | | |

Oracle relative error at preset S: job 5112562 (quickstart, CPU, default schedule).

## CPU and GPU

Stage 1 is the same on both; the GPU default already makes the reference shape, so the gap is CPU only. Expected on CPU: one parallel region over `Nkz * NE * NA` instead of `Nkz * NE` with an inner map re-entered `Nqz * Nw` times.

## What else matters

- **Legality:** both pulls offered and applied on CPU and GPU (job 5112133).
- **Build:** stages 3 to 5 pass on CPU for the route and the default (job 5112069 as v1c; job 5112562 for this folder).
- **Vectorization:** not checked.
- **Pins:** DaCe `801cafd`, NestForge `4df1830` (branch `bench/w3-medium-refills`), HPCAgent-Bench `e63f9f1`.

## Reference solution

```yaml
status: entry
device: cpu
threads: 72
preset: M
dace: 801cafd
nestforge: 4df1830
hpcagent_bench: e63f9f1
probe: Daint jobs 5112133 (route), 5112069 and 5112562 (build)
host: null
baseline:
  moves: [default_moves]
  time_ms: null
  repeats: 5
  oracle_rel_error: null
  tree: |
    map (k, E) { for q { for w { map a { for b { for i { for j { map (o1, o2) { guard { for o3 { for o4 } } } } } } } } } }
reference:
  moves:
    - kind: interchange-loop-map
      on: for4_0 (loop w), kernel6_0 (map a)
    - kind: interchange-loop-map
      on: for3_0 (loop q), kernel5_0 (map a)
    - finish_moves: true
  time_ms: null
  repeats: 5
  oracle_rel_error: null
  tree: |
    map (k, E, a) { for q { for w { for b { for i { for j { map (o1, o2) { guard { for o3 { for o4 } } } } } } } } }
alternatives: []
decoys:
  - moves: [interchange-map-loop]
    effect: map a entered NB times more often
    time_ms: null
  - moves: [interchange-loop-map, interchange-loop-map, interchange-map-loop]
    effect: undoes part of the pull
    time_ms: null
```
