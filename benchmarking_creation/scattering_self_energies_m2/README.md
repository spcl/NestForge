# How to optimize scattering_self_energies_m2

Contains the solution. Do not give this file to an agent whose job is to find the schedule. Full record: [scattering_self_energies_m2.md](scattering_self_energies_m2.md).

## The problem

`Sigma[k, E, a]` (a `10 x 10` complex block per point) accumulates over the reduction loops `q`, `w`, `b`, `i`, `j` and the orbital sums, behind a guard `E - w >= 0`. The output axes `k`, `E` and `a` are parallel. On CPU, stage 1 and `default_moves()` leave the atom map `a` under two reduction loops:

```
map (k, E) { for q { for w { map a { for b { for i { for j { map (o1, o2) { if E - w >= 0 { for o3 { for o4 } } } } } } } } } }
```

so the parallel map over atoms is entered `Nqz * Nw` times per `(k, E)` point.

## The schedule to reach (CPU)

```
map (k, E, a) { for q { for w { for b { for i { for j { ... } } } } } }
```

One parallel region over `Nkz * NE * NA` points, each owning its `Sigma[k, E, a]` block.

## The moves, in order (2, then finish)

1. `interchange-loop-map` on loop `w` and map `a`: map `a` goes out past `w`.
2. `interchange-loop-map` on loop `q` and map `a`: map `a` goes out past `q`, directly under `k` and `E`.
3. `finish_moves()`: collapses `k`, `E`, `a` into one map.

The second pull is offered only after the first.

## Traps

- **`interchange-map-loop` on map `a` and loop `b`.** Offered at every step. It pushes loop `b` out of map `a`, which makes the nest worse here (in `scattering_self_energies`, the 6-move version, this is the move that unlocks the route; here it is not needed).
- **`interchange-map-map` on `k` and `E`.** Legal, but no gain on CPU.
- **`interchange-map-map` on the orbital maps.** Stage 1 already put `o2`, the contiguous axis of `Sigma`, innermost; swapping makes the inner map strided.
- **Stopping after one pull.** Map `a` is still under `q`.

## GPU

On GPU, `default_moves()` already pulls `a` out (lane maps over `a` and `o1`), so there the default is the answer and this is not a test.

## Status

Route checked at DaCe `801cafd`, NestForge `4df1830`, HPCAgent-Bench `e63f9f1` (Daint job 5112133, as variant v1c). Not timed yet.
