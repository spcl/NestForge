# How to optimize adi_column_sweep

Contains the solution. Do not give this file to an agent whose job is to find the schedule. Full record: [adi_column_sweep.md](adi_column_sweep.md).

## The problem

Polybench ADI: each time step runs a row sweep (Thomas recurrences on `pt`, `qt`, then `v`, along `j` with `i` parallel, rows contiguous in `i`) and a column sweep (recurrences on `p`, `q`, then `u`, along `j` with `i` parallel, but `p[i, j]`, `q[i, j]`, `u[i, j]` are strided by `N` in `i`). After stage 1 every sweep is `for j { map i }`: one map launch or parallel region per `j`, `N - 2` per sweep per step.

## The schedule to reach (GPU)

- Every sweep as one map over `i` with its `j` loop inside (`map i { for j }`): one launch per sweep instead of `N - 2`.
- The three column sweeps (`p`, `q`, `u`) in one map: `map i { for j (p); for j (q); for j (u) }`, each row computed in one kernel while it is in cache. **This fusion is refused at the current pins** (see Known issues), so the reachable schedule stops at the interchanges.

## The moves, in order

1. `interchange-loop-map` on the column sweep that writes `p`.
2. `interchange-loop-map` on the column sweep that writes `q`.
3. `interchange-loop-map` on the column sweep that writes `u`.
4. `interchange-loop-map` on the row sweep that writes `v`.
5. (blocked) `map-fusion` of the `p` and `q` row maps, then of that and the `u` map, after `finish_moves()` puts them in one state.

Labels move after every move; name each sweep by the array its map writes. The forward row sweep (`pt`, `qt`) is two maps under one loop and cannot be interchanged.

## Traps

- **The column interchanges alone** (moves 1 to 3): no gain on GPU (1.694 s against 1.702 s at M), because the row sweeps are then left as `N - 2` launches each, which the GPU default had removed.
- **The `v` row interchange alone** (move 4): worse than the default on GPU (2.716 s).
- **The row interchange on CPU**: the row sweeps are contiguous in `i`; interchanging `v` on CPU makes it strided (0.770 s against 1.003 s for stage 1, but the CPU default is 0.519 s).
- **Following the CPU default on GPU, or the GPU default on CPU**: the defaults split by device (CPU takes the column interchanges, GPU the row ones).

## CPU

The CPU default already makes the three column interchanges and keeps the row sweeps; no move beats it (0.519 s at M, 72 threads), and the hand-fused column map does not either (0.525 s). No gap on CPU.

## Known issues

- `map-fusion` of the column row maps is refused by DaCe `MapFusionVertical` (also with `permissive=True`, DaCe alone): the producer is a recurrence that reads its own intermediate. With the fusion made by hand in the source, the GPU default reaches 0.160 s at M, 10.6x the default.
- `describe()` runs DaCe work-depth on every call; on this kernel it fails (SymPy `RecursionError`) or runs for many minutes after `default_moves()` and after the interchanges, so a session needs a guard that skips the costs.

## Status

Blocked (not finished): the reachable route is under the 2x bar; the fusion that clears it is spcl/NestForge#27. Measured at DaCe `801cafd`, HPCAgent-Bench `e63f9f1`, NestForge `bench/m6` (`interchange-benchmarks` `4a0c043`), Daint, with a harness guard around work-depth: GPU 1.180 s (4 interchanges) against 1.702 s (default) at M, 1.44x.
