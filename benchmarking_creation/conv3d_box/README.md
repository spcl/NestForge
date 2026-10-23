# How to optimize conv3d_box

Contains the solution. Do not give this file to an agent whose job is to find the schedule. Full record: [conv3d_box.md](conv3d_box.md).

## The problem

A 125-tap 3-D box stencil written one grid point at a time:

```
map i { map j { map k { out = 0; for di { for dj { map dk (reduce) } } } } }
```

`default_moves()` collapses `i, j, k` and keeps the taps inside. On CPU every point does 125 scattered reads; on GPU each point is a 256-thread block sharing a 5-term reduction.

## The schedule to reach

- **CPU (1 and 72 threads), 5 moves:** `map-fission` of the `k` map; `interchange-map-loop`: `k` inside `di`; `map-fission` of the `j` map; `interchange-map-loop`: `j` inside `di`; `interchange-map-loop`: `k` inside `dj`; `finish_moves()`. 1.89x on 1 thread, 1.72x on 72.
- **GPU, 1 move:** `subgraph-fission` of the `k` map's body, `finish_moves()`. 34x (`map-fission` instead: 17x).

## Traps

- **Each fission alone on CPU** (0.99x): required, gains nothing by itself.
- **Stopping after `k` inside `di`** (2 moves): 1.53x against 1.89x.
- **A sixth move** (`j` inside `dj`) is offered; it gains nothing.
- **CPU moves on GPU:** every prefix of the CPU route is slower on GPU than the single fission; the 6-move route is slower than the default, and the 5-move route gives a wrong GPU result at preset M (code generation bug; correct at S).
