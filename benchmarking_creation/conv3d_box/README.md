# How to optimize conv3d_box

Contains the solution. Do not give this file to an agent whose job is to find the schedule. Full record: [conv3d_box.md](conv3d_box.md).

## The problem

A 125-tap 3-D box stencil written one grid point at a time:

```
map i { map j { map k { out = 0; for di { for dj { map dk (reduce) } } } } }
```

`default_moves()` collapses `i, j, k` and keeps the taps inside: 125 scattered reads per point.

## The schedule to reach (CPU)

```
map i { init; for di { for dj { map j { map k { map dk (reduce) } } } } }
```

## The moves, in order (6, then finish)

1. `map-fission` of the `k` map. Makes move 2 legal.
2. `interchange-map-loop`: `k` inside `di`.
3. `map-fission` of the `j` map. Makes moves 4 and 6 legal.
4. `interchange-map-loop`: `j` inside `di`.
5. `interchange-map-loop`: `k` inside `dj`.
6. `interchange-map-loop`: `j` inside `dj`.
7. `finish_moves()`.

## Traps

- The `i, j, k` swaps; stopping after `k` alone; pushing `i` inward.
