# scattering_self_energies_m2 (slot M9, medium, CPU)

Explanation and solution record (`benchmarking_creation/AGENTS.md`) of a kernel derived from HPCAgent-Bench `scientific_computing/dense_linear_algebra/scattering_self_energies` (`e63f9f1`). Written 2026-10-21 from Daint jobs 5111782 (walk of the loop order, as variant v1), 5112069 (build check), 5112070 and 5112133 (walk and routes, as variant v1c) and 5112562 (validation of this folder). Set: `benchmark-set-15` (thesis notes). Entry note: `2026-10-21-w3-medium-refills` (thesis notes). How to optimize it, in short: [README.md](README.md). The 6-move original is the hard-tier candidate `scattering_self_energies`.

**Status:** `entry`, timed 2026-10-23 at NestForge main's pins with preset M changed to `Nkz` 2, `NE` 4: the route built with `collapse(3)` beats the default's best build (`collapse(2)`) **9.78x** on CPU at 72 threads (job 5127570), 9.80x and 9.76x in two repeat jobs (5127845, 5127847).

Slot: M9. Entered on 2026-10-21 as M8; `cloudsc_ntrac` holds M8, so this kernel moved to M9 (user decision, board package W12-M9).

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

DaCe `042413d`, HPCAgent-Bench `2666c06`, NestForge `ib/on-main`, CPU 72 threads (`OMP_PROC_BIND=close`, `OMP_PLACES=cores`), whole-program host time, median of 5 after one warm-up. Every variant matches the oracle at S (5.9e-15) and the baseline at M.

**At the old preset M** (`Nkz 4, NE 16`; job 5126572) the route gains little: 17.32 s against 15.68 s (1.10x) as built, the one-pull decoy 1.11x, and `interchange-map-loop` 1.17x, faster than the route. The default's `k, E` split already gives each of 64 threads a point; re-entering map `a` `Nqz * Nw = 8` times costs little. Raising `Nqz * Nw` (and shrinking `NA` to keep the run short) does not change that (1.14x, job 5127159).

**Preset M changed (2026-10-23)** to `Nkz 2, NE 4, Nqz 4, Nw 2, N3D 3, NA 128, NB 4`: 8 `(k, E)` points, fewer than the 72 threads, so the default's best build leaves threads idle and the route's `(k, E, a)` map fills them. This is the same lever as its sibling M5.1.

| Build (72 threads) | old M, 64 points (job 5127571) | **new M**, 8 points (job 5127570) |
| --- | --- | --- |
| default, `collapse(2)` (the baseline to beat) | 1157.4 ms | 1123.1 ms |
| reference, `collapse(2)` | 1040.5 ms (1.11x) | 1010.0 ms (1.11x) |
| **reference, `collapse(3)`** | 893.8 ms (1.30x) | **114.8 ms (9.78x)** |

Repeats at the new M (jobs 5127845, 5127847):

| Build (72 threads) | 5127845 | 5127847 |
| --- | --- | --- |
| default, as built | 3952.8 ms | 3951.1 ms |
| default, `collapse(2)` | 1134.0 ms | 1132.2 ms |
| reference, as built | 3560.3 ms (1.11x) | 3564.9 ms (1.11x) |
| **reference, `collapse(3)`** | **115.7 ms (9.80x over `collapse(2)`)** | **116.0 ms (9.76x)** |
| one pull, as built | 3574.3 ms | 3569.4 ms |
| `interchange-map-loop`, as built | 3372.6 ms | 3381.0 ms |
| partial undo, as built | 3573.5 ms | 3571.5 ms | `interchange-map-loop` built with `collapse(3)` does not compile (the collapsed nest is not perfectly nested), so that decoy is measured as built only.

The answer has two parts, as for M5.1: the two pulls (stage 2) and a kernel built with `collapse(3)` (stage 5, through `bind`). Its route is shorter than M5.1's (two pulls, no round trip), so the pair grades the same lever at two depths.

Oracle check at preset S: quickstart passes all seven stages on CPU with the default schedule (job 5112722, after the C_PTR fix `27080f2`; stage 6 picked `g++:contract-fma` among 9 variant cells). Relative error not recorded separately.

## CPU and GPU

Stage 1 is the same on both; the GPU default already makes the reference shape, so the gap is CPU only. Expected on CPU: one parallel region over `Nkz * NE * NA` instead of `Nkz * NE` with an inner map re-entered `Nqz * Nw` times.

## What else matters

- **Legality:** both pulls offered and applied on CPU and GPU (job 5112133).
- **Build:** stages 3 to 5 pass on CPU for the route and the default (job 5112069 as v1c; job 5112562 for this folder); stage 6 needed the C_PTR fix `27080f2` (`std::complex<double>` entry buffers), then quickstart passes (job 5112722).
- **Vectorization:** not checked.
- **Pins:** DaCe `801cafd`, NestForge `4df1830` (branch `bench/w3-medium-refills`), HPCAgent-Bench `e63f9f1`.

## Reference solution

```yaml
status: entry
device: cpu
threads: 72
preset: M    # Nkz 2, NE 4, Nqz 4, Nw 2, N3D 3, NA 128, NB 4 (since 2026-10-23)
dace: 042413df2b96ed0f2ebb361c4879c7f68bad41d2
nestforge: ib/on-main (origin/main 8545a4b + interchange-benchmarks)
hpcagent_bench: 2666c06b73eaa9d51f05568e6c584aaafa98ad67
probe: Daint jobs 5112133 (route), 5112069 and 5112562 (build); timing 5127570
host: Daint GH200 node, exclusive
baseline:
  moves: [default_moves]
  kernel: "#pragma omp parallel for collapse(2)"
  time_ms: 1123.1
  repeats: 5
  oracle_rel_error: 5.9e-15   # preset S
  tree: |
    map (k, E) { for q { for w { map a { for b { for i { for j { map (o1, o2) { guard { for o3 { for o4 } } } } } } } } } }
reference:
  moves:
    - kind: interchange-loop-map
      on: for4_0 (loop w), kernel6_0 (map a)
    - kind: interchange-loop-map
      on: for3_0 (loop q), kernel5_0 (map a)
    - finish_moves: true
  kernel: "#pragma omp parallel for collapse(3)"
  time_ms: 114.8
  repeats: 5
  oracle_rel_error: 5.9e-15   # preset S
  tree: |
    map (k, E, a) { for q { for w { for b { for i { for j { map (o1, o2) { guard { for o3 { for o4 } } } } } } } } }
alternatives: []
decoys:
  - moves: [interchange-map-loop]
    effect: map a entered NB times more often
    time_ms: 14826.1   # old preset M, kernel as built (job 5126572); does not compile with collapse(3)
  - moves: [interchange-loop-map, interchange-loop-map, interchange-map-loop]
    effect: undoes part of the pull
    time_ms: 15662.2   # old preset M, kernel as built (job 5126572)
```
