# How to optimize scattering_self_energies_m3

Contains the solution. Do not give this file to an agent whose job is to find the schedule. Full record: [scattering_self_energies_m3.md](scattering_self_energies_m3.md).

## The problem

`Sigma[k, E, a]` (a `10 x 10` complex block per point) accumulates over the reduction loops `q`, `w`, `i`, `b`, `j` and the orbital sums, behind a guard `E - w >= 0`. The output axes `k`, `E` and `a` are parallel. On CPU, stage 1 and `default_moves()` leave the atom map `a` under three reduction loops:

```
map (k, E) { for q { for w { for i { map a { for b { for j { map (o1, o2) { if E - w >= 0 { for o3 { for o4 } } } } } } } } } }
```

so the parallel map over atoms is entered `Nqz * Nw * N3D` times per `(k, E)` point, and the kernel's only parallel loop is `k, E` (`Nkz * NE` = 64 iterations at preset M).

## The schedule to reach (CPU)

```
map (k, E, a) { for q { for w { for i { for b { for j { ... } } } } } }
```

One parallel region over `Nkz * NE * NA` points (8192 at preset M), each owning its `Sigma[k, E, a]` block, built with the three outer loops collapsed (`#pragma omp parallel for collapse(3)`).

## The moves, in order (5, then finish)

1. `interchange-map-loop` on map `a` and loop `b`: pushes loop `b` out of map `a`. It looks backwards and gains nothing by itself.
2. `interchange-loop-map` on loop `b` and map `a`: puts map `a` back outside loop `b`. The loop order is stage 1's again, but now the three pulls below are offered; from stage 1 they are not.
3. `interchange-loop-map` on loop `i` and map `a`.
4. `interchange-loop-map` on loop `w` and map `a`.
5. `interchange-loop-map` on loop `q` and map `a`: map `a` is now directly under `k` and `E`.
6. `finish_moves()`: collapses `k`, `E`, `a` into one map.

Then, in stage 5, the kernel must collapse all three loops of that map. With the default kernel build (no `collapse`) the route is no faster than the default.

## Traps

- **Stopping after move 1.** It only pushes `b` out; no gain (16.01 s against 16.05 s for the default at M).
- **Collapse without the moves.** `collapse(2)` on the default's `k, E` is a 16x gain by itself (16.05 s to 1.01 s). It is the baseline to beat, not the answer.
- **The moves with `collapse(2)`.** 1.05 s, no better than the default with `collapse(2)`: the moves only pay off when the `a` dimension they expose is split too.
- **Stopping after one or two pulls.** Map `a` still sits under a reduction loop.

## Known issue

DaCe `MoveLoopIntoMap` (moves 2 to 5) widens the `Sigma` memlet out of map `a` to the whole array, so the kernel build makes the outer map sequential and the route runs at serial speed (59 s). Until that is fixed (spcl/NestForge#24), re-propagate memlets (`propagate_memlets_sdfg`) after `finish_moves()` and on each kernel copy.

## GPU

The 5-move route is offered and applies on GPU too (job 5113159); the GPU program has not been built or timed.

## Status

Timed at DaCe `801cafd`, HPCAgent-Bench `e63f9f1`, NestForge `8c87f76` (+ `27080f2`), Daint job 5116121, CPU, 72 threads: 0.89 s for the route with `collapse(3)` against 1.01 s for the default with `collapse(2)` at preset M (1.14x).
