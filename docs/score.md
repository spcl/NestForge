# Static move score

prev: [2 Moves](stages/2-moves.md)

`Session.score_moves()` ranks every legal stage 2 move without building anything, and `Session.guided_moves()` applies
the best one while it lowers the score. Code: `nestforge/stages/score.py`.

```
S = lines / p + c * E        lower is better
```

| term | meaning |
|---|---|
| `E` | parallel-region entries: every top-level map times the trips of the loops around it (one fork or launch each) |
| `lines` | cache lines the tasklets touch along their innermost iterator: `volume * min(1, |stride| * bytes / 64)` per memlet, times the executions; an array of at most 1 MiB counts its lines once |
| `p` | workers sharing the traffic |
| `c` | the price of one region entry, in cache lines |

Only `k = c * p` decides the sign of a change. The weights are frozen per build (`WEIGHTS`): CPU serial `k = 1e4`,
CPU OpenMP `k = 3.2e5`, GPU `k = 2.05e7`. With these, `E` decides whenever a move changes it and `lines` breaks ties:
moving a sequential loop into a map, or fusing two maps, wins; on one thread a stride-one inner access wins.

The weights were fitted on measured runtimes of seven small kernels (TSVC `s231`, `s233`, `s235`, `s2233`, `s256` and
two `column_scan` layouts) and checked on new two-move sequences of the same kernels, predicted before timing. It is
a ranking for the moves stage 2 offers, not a runtime model. Known misses: some fusions that are slower at 32 OpenMP
threads, where neither term changes in the right direction.

How the score was built, the tests, the results and where it fails: [score-design.md](score-design.md).

`examples/score_demo.py` compares stage 2 by the score with the default stage 2 on HPCAgent-Bench kernels. One run
(CPU OpenMP, 32 threads, preset M; program time, ratio to the default stage 2):

| kernel | fitted on | score's move | score-guided | default, then score |
|---|---|---|---|---|
| `tsvc_2_s231` | yes | interchange | 0.20x | 0.20x |
| `tsvc_2_s1119` | no | interchange | 0.22x | 0.22x |
| `scan_levels_k1` | no | fusion; interchange after the default | 0.93x | 1.86x |
| `scan_levels_k32` | no | fusion; interchange after the default | 1.02x | 2.00x |
| `column_scan_nproma_nlev` | yes | 5 fusions | 1.01x | 0.99x |

The score adds what the default never does (interchange) and wins where the predicted gain is large; the two losses
are interchanges with a score drop of 3% to 4% that multiply `lines` by 8.

## Possible improvements

From the demo and from hardware counters on the slow 32-thread OpenMP cases. The fused kernels move the same data
through L1 and L2 as the kernels they replace but miss the L3 4x to 22x more. Their slowdown at 32 threads shows when
kernels are timed back to back and reverses when calls are spaced apart, so the first two rows are untested. None is
in the code.

| change | why |
|---|---|
| apply a move only when the score drops by more than a margin | the demo's two losses had a 3% to 4% drop, its wins about 90% |
| measured `c(p)` instead of a fitted `c` | the cost per region entry grows with threads (empty region: 0.24, 2.4, 7.2 us at 1, 8, 32 threads); the fit only bounds `c` |
| per region `max(c(p), lines / bandwidth(p))` instead of the sum | small kernels on many threads are bound by synchronization, not memory; `lines / p` assumes traffic splits over threads |
| reuse across kernels in `lines` (reuse distance) | some fusions are slower already on one thread: one fused sweep loses the reuse between consecutive iterations, which per-tasklet `lines` cannot see |
| validate on whole-program time at the target thread count | per-kernel timing in isolation changes how threads sleep and wake, and with it the sign of some fusions |
