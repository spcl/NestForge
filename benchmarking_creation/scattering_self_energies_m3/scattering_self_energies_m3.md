# scattering_self_energies_m3 (slot M5.1, medium, CPU)

Explanation and solution record (`benchmarking_creation/AGENTS.md`) of a kernel derived from HPCAgent-Bench `scientific_computing/dense_linear_algebra/scattering_self_energies` (`e63f9f1`). Written 2026-10-22 from Daint jobs 5113053 (routes from stage 1), 5113054 and 5113055 (default build), 5113159 (the 5-move route on CPU and GPU), 5114046, 5114047 and 5115143 (timing and the memlet problem) and 5116121 (timing with collapsed kernels). Experiment notes: `2026-10-21-w3-medium-refills` and `2026-10-21-h2-m3-timing` (thesis notes). How to optimize it, in short: [README.md](README.md). Sibling kernels: `scattering_self_energies_m2` (M9, 2 pulls) and the 6-move original `scattering_self_energies`.

**Status:** `entry`, re-timed 2026-10-23 at NestForge main's pins with preset M changed to `Nkz` 2, `NE` 8: the route built with `collapse(3)` beats the default's best build (`collapse(2)`) **5.08x and 4.97x** (two jobs) on CPU at 72 threads.

Slot: M5.1. Created by W3 as a hard-tier (H2) candidate; logged as medium on 2026-10-22 (user decision, board package W12-M5.1).

## What this benchmark is

The electron-phonon scattering self-energy of OMEN (via NPBench), `Sigma[k, E, a] += (G[k, E-w, neigh_idx[a, b]] @ dH[a, b, i]) @ (dH[a, b, j] * D[q, w, a, b, i, j])` behind the guard `E - w >= 0`, with the same oracle, arguments and presets as the corpus kernel. Two changes in the DaCe program, which is hand-written: the loop order is `k, E, q, w, i, a, b, j`, so the atom loop `a` sits under three reduction loops (`q`, `w`, `i`), and the `10 x 10` products are explicit orbital loops `o1..o4` instead of `@`. `k`, `E`, `a` (and `o1`, `o2`) index the output; `q`, `w`, `i`, `b`, `j`, `o3`, `o4` are reductions. The scheduling problem: pull the atom map out of the three reduction loops so `(k, E, a)` is one parallel map, and build that map with all three loops split.

## Why it is a good test

- **What the default does instead:** on CPU, `default_moves()` collapses `k, E` and leaves map `a` under `for q { for w { for i } }`. The kernel parallelizes only `k` (no `collapse`), 4 iterations at preset M.
- **What makes it hard:** from stage 1 none of the pulls is offered (job 5113053). The unlock is a round trip: `interchange-map-loop` on `a` and `b`, which looks backwards, then `interchange-loop-map` back. Only then are the three pulls offered, one after the other.
- **It needs the context of a later stage:** the moves pay off only if stage 5 builds the kernel with `collapse(3)`. With `collapse(2)`, as the default can already have, they gain nothing.
- **What a wrong decision looks like:** stopping after the first move; a partial pull; collapsing the default's `k, E` and stopping there. The last one is a large gain (16x) with no move, so it is the baseline the moves have to beat.

## What the reference is

- **Baseline (CPU):** stage 1, then `default_moves()`: `map (k, E) { for q { for w { for i { map a { for b { for j { map (o1, o2) { guard { for o3 { for o4 } } } } } } } } } }`. Compared both as built (`#pragma omp parallel for`) and with `collapse(2)`, its best build.
- **Reference (CPU):** the 5 moves, then `finish_moves()`: `map (k, E, a) { for q { for w { for i { for b { for j { ... } } } } } }`, memlets re-propagated (see Legality), kernel built with `collapse(3)`.
- **GPU:** the route applies (job 5113159); not built or timed.

## How it was created

From `scattering_self_energies` (6 moves: `interchange-map-loop` on `a`/`b`, then 5 pulls; its stage 3 to 5 build failed until spcl/NestForge#19 and #21). W3 reordered the nest so that `a` sits under three reduction loops instead of four, and wrote the products as explicit orbital loops as in `scattering_self_energies_m2` (the `@` products become `MatMult` library nodes, which stages 3 to 5 refused). The direct 3-pull route is not offered from stage 1 (job 5113053, `pull-2`, `pull-3`: the pulls are missing); the round trip of moves 1 and 2 makes it legal (job 5113159).

## What the solution is

1. `interchange-map-loop` on (`kernel7_0`, `for8_0`): map `a`, loop `b`.
2. `interchange-loop-map` on (`for6_0`, `kernel8_0`): loop `b`, map `a`; back to stage 1's order.
3. `interchange-loop-map` on (`for5_0`, `kernel7_0`): loop `i`, map `a`.
4. `interchange-loop-map` on (`for4_0`, `kernel6_0`): loop `w`, map `a`.
5. `interchange-loop-map` on (`for3_0`, `kernel5_0`): loop `q`, map `a`.
6. `finish_moves()`: `k`, `E`, `a` collapse into one map.

Labels from jobs 5113159 and 5116121; not stable across versions. Kernel loop nest of the timed reference (job 5116121):

```
#pragma omp parallel for collapse(3)
for (int64_t _loop_it_0 = 0; _loop_it_0 < Nkz; _loop_it_0 += 1) {
    for (int64_t _loop_it_1 = 0; _loop_it_1 < NE; _loop_it_1 += 1) {
        for (int64_t _loop_it_5 = 0; _loop_it_5 < NA; _loop_it_5 += 1) {
            nested_state4_0_0_0_0(...);   // for q { for w { for i { for b { for j { map (o1, o2) { ... } } } } } }
```

## How to optimize it

After stage 1 on CPU, `list_moves` offers `interchange-map-loop` (`kernel7_0`, `for8_0`) but no `interchange-loop-map` that pulls `a` (job 5113053). `default_moves()` takes none of the interchanges.

| Decision | What happens | Evidence (preset M, 72 threads) |
| --- | --- | --- |
| `default_moves()` | `a` under three reduction loops; `k` parallel only | 16.05 s (5116121) |
| `default_moves()`, kernel with `collapse(2)` | `k, E` split: 64 iterations | 1.01 s (5116121) |
| move 1 only | `b` pushed out of `a` | 16.01 s (5114047) |
| moves 1 and 2 | stage 1's order, but the memlet is widened (Legality) | 62.02 s; 15.99 s re-propagated (5114047, 5115143) |
| the 5 moves, no `collapse` | only `k` split, as in the default | 15.94 s re-propagated (5115143); 59.13 s without (5114047) |
| the 5 moves, `collapse(2)` | `k, E` split; `a` not used | 1.05 s (5116121) |
| the 5 moves, `collapse(3)` | `k, E, a` split: 8192 iterations | **0.89 s** (5116121) |

## What the gains are

**2026-10-23, preset M changed.** At the old M (`Nkz` 4, `NE` 16) the default's best build already splits 64 `(k, E)` points over 72 threads, and the route gains 1.30x over it (job 5126560: 1137 ms against 874 ms). With fewer `(k, E)` points than threads, the default's best build leaves threads idle, and only the route's `(k, E, a)` map fills them. Sweep at 72 threads (jobs 5127167 to 5127169; the default's best build takes the same ~1.13 s at every size because each thread holds one `(k, E)` point):

| `Nkz` x `NE` | default, `collapse(2)` | reference, `collapse(3)` | Ratio |
| --- | --- | --- | --- |
| 4 x 16 = 64 (old M) | 1137 ms | 874 ms | 1.30x |
| 2 x 8 = 16 (**new M**) | 1142 ms | 224 ms | **5.10x** |
| 1 x 16 = 16 | 1125 ms | 224 ms | 5.02x |
| 2 x 4 = 8 | 1133 ms | 115 ms | 9.84x |

At the new preset M (`Nkz 2, NE 8, Nqz 4, Nw 2, N3D 3, NA 128, NB 4`), DaCe `042413d`, HPCAgent-Bench `2666c06`, NestForge `ib/on-main`, CPU 72 threads (`OMP_PROC_BIND=close`, `OMP_PLACES=cores`), median of 5 after one warm-up, jobs 5127627 and 5127628 (repeat), 5127629 (1 thread). No memlet workaround: on `main` the kernel keeps its OpenMP loop (spcl/NestForge#31). Every variant matches the oracle at S (5.8e-15) and the baseline at M.

| Build | 72 threads | repeat | 1 thread |
| --- | --- | --- | --- |
| default, as built | 8562.7 ms | 8430.9 ms | 17126.0 ms |
| default, `collapse(2)` (the baseline to beat) | 1139.5 ms | 1132.2 ms | 17022.9 ms |
| first move only, as built | 7370.6 ms | 7295.2 ms | 14831.5 ms |
| moves 1 and 2 (round trip), as built | 7648.6 ms | 7557.8 ms | n/a |
| reference, as built | 7627.4 ms | 7537.8 ms | 15242.1 ms |
| reference, `collapse(2)` | 1024.5 ms | 1017.5 ms | 15315.1 ms |
| **reference, `collapse(3)`** | **224.3 ms (5.08x)** | **227.9 ms (4.97x)** | 14721.4 ms (1.16x) |

The benchmark is the CPU at 72 threads, and its answer has two parts: the five moves (stage 2) and a kernel built with `collapse(3)` (stage 5, through `bind` of a rebuilt kernel). Neither alone is enough: the moves with the default build are 1.12x, and the best collapse of the default is the baseline above. On 1 thread nothing matters (1.16x at most).

**2026-10-22, old preset M** (DaCe `801cafd`, the record as first written):

Preset M (`Nkz 4, NE 16, Nqz 4, Nw 2, N3D 3, NA 128, NB 4`), CPU, Daint node `nid005539`, 72 threads (`OMP_PROC_BIND=close`, `OMP_PLACES=cores`), `g++` `strict-ieee` with `-fopenmp`, whole-program host time, mean of 10 calls after one warm-up (min and max not recorded). Every variant matches the oracle (rtol 1e-8, atol 1e-10) at S and M.

| Build | Preset S | Preset M |
| --- | --- | --- |
| default | 19.15 ms | 16.05 s |
| default, `collapse(2)` | 5.53 ms | 1.01 s |
| reference, no `collapse` | 19.00 ms | 16.04 s |
| reference, `collapse(2)` | 5.54 ms | 1.05 s |
| reference, `collapse(3)` | 0.97 ms | 0.89 s |

Ratio of the reference to the best default (`collapse(2)`): 1.14x at M, 5.7x at S; to the default as built: 18x at M. The gain is larger at S because `k, E` (8 iterations at S) cannot fill 72 threads and `a` can; at M, `k, E` already gives 64. Serial (job 5114046): default 62.80 s, stage 1 without moves 62.51 s; the reference was not timed serially. The ratio against the best default is below the 2x of the checklist.

## CPU and GPU

CPU is the device measured. On GPU the route is offered and applies (job 5113159), but neither the default nor the route was built or timed, so whether the GPU default already reaches `(k, E, a)` (as it does for `scattering_self_energies_m2`) is not known.

## What else matters

- **Legality:** the pulls are absent from stage 1 and offered after moves 1 and 2 (jobs 5113053, 5113159).
- **Memlets:** DaCe `MoveLoopIntoMap` (move 2 onwards) widens the write `map a -> Sigma` to `0:Nkz, 0:NE, 0:NA, 0:10, 0:10`; `map_scope_carries_dependency` then makes the outer map `Sequential` and the kernel has no `#pragma omp parallel for` (59.13 s). `propagate_memlets_sdfg` after `finish_moves()` and on each kernel copy restores `_loop_it_0, _loop_it_1, 0:NA, …` and the pragma. Tracked as spcl/NestForge#24; the times above use the re-propagation.
- **Collapse:** NestForge's CPU kernel build emits `#pragma omp parallel for` without `collapse`; the `collapse(N)` builds above rewrote the pragma before the OpenMP rebuild (both nests are perfectly nested). Reaching the reference through NestForge needs a kernel source with `collapse(3)` (`set_kernel_source`) or a stage 5 option for it.
- **Contiguity:** stage 1 puts `o2`, the contiguous axis of `Sigma`, innermost.
- **Vectorization:** not checked.
- **Oracle:** NumPy oracle of the corpus kernel, inputs from `initialize()` in `scattering_self_energies_m3.py`, `Norb = 10` pinned.
- **Pins:** DaCe `801cafd`, HPCAgent-Bench `e63f9f1`, NestForge `ib/h2` `8c87f76` (with #19) plus the complex-buffer patch `27080f2`. Not yet re-checked on `interchange-benchmarks`.

## Reference solution

```yaml
status: entry
device: cpu
threads: 72
preset: M    # Nkz 2, NE 8, Nqz 4, Nw 2, N3D 3, NA 128, NB 4 (since 2026-10-23)
dace: 042413df2b96ed0f2ebb361c4879c7f68bad41d2
nestforge: ib/on-main (origin/main 8545a4b + interchange-benchmarks)
hpcagent_bench: 2666c06b73eaa9d51f05568e6c584aaafa98ad67
probe: Daint jobs 5113053, 5113159 (routes); timing 5127627, repeat 5127628
host: Daint GH200 node, exclusive
baseline:
  moves: [default_moves]
  kernel: "#pragma omp parallel for collapse(2)"
  time_ms: 1139.5
  repeats: 5
  oracle_rel_error: "matches (rtol 1e-8); max error not recorded"
  tree: |
    map (k, E) { for q { for w { for i { map a { for b { for j { map (o1, o2) { guard { for o3 { for o4 } } } } } } } } } }
reference:
  moves:
    - kind: interchange-map-loop
      on: kernel7_0 (map a), for8_0 (loop b)
    - kind: interchange-loop-map
      on: for6_0 (loop b), kernel8_0 (map a)
    - kind: interchange-loop-map
      on: for5_0 (loop i), kernel7_0 (map a)
    - kind: interchange-loop-map
      on: for4_0 (loop w), kernel6_0 (map a)
    - kind: interchange-loop-map
      on: for3_0 (loop q), kernel5_0 (map a)
    - finish_moves: true
  kernel: "#pragma omp parallel for collapse(3)"
  time_ms: 224.3
  repeats: 5
  oracle_rel_error: "matches (rtol 1e-8); max error not recorded"
  tree: |
    map (k, E, a) { for q { for w { for i { for b { for j { map (o1, o2) { guard { for o3 { for o4 } } } } } } } } }
alternatives:
  - moves: [default_moves]
    kernel: "#pragma omp parallel for"
    time_ms: 8562.7
  - moves: [the 5 moves, finish_moves]
    kernel: "#pragma omp parallel for collapse(2)"
    time_ms: 1024.5
decoys:
  - moves: [interchange-map-loop]
    effect: pushes loop b out of map a; the first move alone
    time_ms: 7370.6   # kernel as built
```
