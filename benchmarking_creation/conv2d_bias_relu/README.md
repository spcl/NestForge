# How to optimize conv2d_bias_relu

Contains the solution. Do not give this file to an agent whose job is to find the schedule. Full record: [conv2d_bias_relu.md](conv2d_bias_relu.md).

## The problem

A convolution layer with bias and ReLU, written one output element at a time:

```
map n { map co { map oh { map ow { out = b; for ci { for kh { map kw (reduce) } }; out = relu(out) } } } }
```

`default_moves()` collapses the four output maps and keeps the sums inside. On CPU the kernel parallelizes only `n` (8 iterations); on GPU each output is a 256-thread block sharing a 3-tap reduction.

## The schedule to reach

- **GPU:** `map-fission` of the `ow` map, then `finish_moves()`. One thread per output, sums sequential in the thread. 108x.
- **CPU, 1 thread:** that fission, then `interchange-map-loop` twice: `ow` inside `ci`, then inside `kh`. Each weight is applied to a contiguous `ow` row (vectorized). 1.98x.
- **CPU, 72 threads:** the three CPU moves, then `interchange-map-map` on `n`, `co` so the parallel loop is `co` (64). 14.9x; the interchanges alone 2.0x, the swap alone 7.6x.

## Traps

- **The fission alone on CPU.** Required first, but 0.83x by itself.
- **Continuing the route.** After the three moves a fusion of the init maps unlocks a second fission and two more pushes (`oh` inside the sums). They gain nothing (1.90x against 1.98x).
- **The other swaps.** `co`, `oh` changes nothing; `oh`, `ow` moves the contiguous axis out (0.89x).
- **CPU moves on GPU.** After the fission, each interchange costs GPU time (73x, then 43x for the full route, against 108x).
- **A wrong GPU program.** Three fissions plus a fusion give a GPU program that is correct at preset S and wrong at M (code generation bug). Check the output at the timed size.
