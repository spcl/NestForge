# How to optimize seissol_volume

Contains the solution. Do not give this file to an agent whose job is to find the schedule. Full record: [seissol_volume.md](seissol_volume.md).

## The problem

The SeisSol ADER-DG volume update, written one output entry at a time:

```
map b { map k { map p { upd = 0; for d { for l { map q (reduce) } } } } }
map (b, k, p) { Q += dt * upd }
```

`default_moves()` keeps this order: for every `(b, k, p)` the whole `d, l, q` sum, re-reading `kDivM` and `star` per entry.

## The schedule to reach (CPU)

```
map b { init; for d { for l { map k { map p { map q (reduce) } } } } }
```

Each element `b` is one parallel task; inside it, each `kDivM[d, k, l]` is applied to the element's whole `(k, p)` block.

## The moves, in order (6, then finish)

1. `map-fission` of the `p` map (init from sum). Gains nothing alone; makes move 2 legal.
2. `interchange-map-loop`: `p` inside `l`.
3. `interchange-map-loop`: `p` inside `d`.
4. `map-fission` of the `k` map. Gains nothing alone; makes move 5 legal.
5. `interchange-map-loop`: `k` inside `d`.
6. `interchange-map-loop`: `k` inside `l`.
7. `finish_moves()`.

## Traps

- **The map swaps** of `b`, `k`, `p`, offered at every step.
- **Stopping after move 3**, with `k` still outside the sums.
- **Pushing `b` inward**: it is the axis to parallelize.
