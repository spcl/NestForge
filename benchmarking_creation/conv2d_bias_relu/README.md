# How to optimize conv2d_bias_relu

Contains the solution. Do not give this file to an agent whose job is to find the schedule. Full record: [conv2d_bias_relu.md](conv2d_bias_relu.md).

## The problem

A convolution layer with bias and ReLU, written one output element at a time:

```
map n { map co { map oh { map ow { out = b; for ci { for kh { map kw (reduce) } }; out = relu(out) } } } }
```

`default_moves()` collapses the four output maps into one and leaves the sums inside: one thread per output, reading `x` with a stride across `ci` for every output.

## The schedule to reach (CPU)

```
map n { map co { init; for ci { for kh { map oh { map ow { map kw (reduce) } } } }; relu } }
```

Each weight `wt[co, ci, kh, kw]` is applied to whole contiguous rows of `x` and `out`; the `H x W` output plane of one `(n, co)` stays in cache.

## The moves, in order (7, then finish)

1. `map-fission` of the `ow` map: init, sum and ReLU become separate maps. Gains nothing alone; makes move 2 legal.
2. `interchange-map-loop`: `ow` inside `ci`.
3. `interchange-map-loop`: `ow` inside `kh`.
4. `map-fusion` of the two bias-init maps. Gains nothing alone; makes move 5 appear.
5. `map-fission` of the `oh` map.
6. `interchange-map-loop`: `oh` inside `ci`.
7. `interchange-map-loop`: `oh` inside `kh`.
8. `finish_moves()`.

## Traps

- **The map swaps.** `interchange-map-map` on `n, co, oh, ow` is always offered; none of it helps, and moving `ow` off the innermost position breaks the contiguous rows.
- **Stopping after move 3.** `ow` is inside the sums but `oh` is not: the output plane is re-entered per row.
- **Missing move 4.** Without the init fusion the `oh` fission is never offered, so the route looks finished after move 3.
- **On GPU.** The default is the right schedule; the CPU route serializes `ci` and `kh` around the output maps.
