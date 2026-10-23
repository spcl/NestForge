# How to optimize seissol_volume

Contains the solution. Do not give this file to an agent whose job is to find the schedule. Full record: [seissol_volume.md](seissol_volume.md).

## The problem

The SeisSol ADER-DG volume update, written one output entry at a time:

```
map b { map k { map p { upd = 0; for d { for l { map q (reduce) } } } } }
map (b, k, p) { Q += dt * upd }
```

`default_moves()` keeps this order. On GPU each entry is a 256-thread block sharing a 9-term reduction; on CPU each entry re-reads `kDivM` and `star`.

## The schedule to reach

- **GPU:** `map-fission` of the `p` map (init from sum), then `finish_moves()`. 54x.
- **CPU (1 and 72 threads):** that fission, then `interchange-map-loop` twice: `p` inside `l`, then inside `d`. 1.62x on 1 thread, 1.45x on 72.

## Traps

- **The fission alone on CPU** (1.00x): required first, gains nothing by itself.
- **Continuing the route.** A second fission (of `k`) and two more pushes are offered; on CPU they gain nothing, on GPU they make the program 6.8x slower than the default.
- **The swaps** of `b`, `k` (0.78x at 72 threads), and sequences that move `k` outside `b` (0.13x).
