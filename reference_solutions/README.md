# Reference solutions

One file per benchmark, `<kernel>.py`. Its `solve(s, ctx)` applies the reference moves; replay times the result against `default_moves()` and checks it against the bench oracle.

```
python -m nestforge.replay reference_solutions/<kernel>.py
```

**Adding a solution:** keep the script short, like `tsvc_2_s231.py`: the kernel, the config and the moves, picked by kind and by what they read and write. Add a kernel here once its moves beat the default by more than the timing noise (10-17% between Daint nodes). CI replays every file here at preset S on one CPU thread (`tests/test_replay.py`), so a solution that stops applying or no longer matches the oracle fails the build. A script that cannot run on the pins yet sets `CI_SKIP` to the reason, and CI skips it.

| Kernel | Moves | Config | Default | Reference | Gain |
| --- | --- | --- | --- | --- | --- |
| `aes_graupel` | `map-fission` of the map computing `sink` and `sx2x`, then the default | CPU, 72 threads | 7705 ms | 1022 ms | 7.54x (previous pins; re-timing on main pending) |
| `tsvc_2_s233` | `loop-fission`, then `interchange-loop-map` on the `aa` half and the `bb` half | CPU, 72 threads | 30.7 ms | 8.8 ms | 3.47x |
| `tsvc_2_s231` | `interchange-loop-map` (the `j` loop into the `i` map) | CPU, 72 threads | 39.5 ms | 16.5 ms | 2.40x |
| `cloudsc_nproma` | `map-fission` of the column map, then `interchange-map-loop` of the body map and the level loop | CPU, 1 thread | 110.2 ms | 46.9 ms | 2.35x (DaCe `042413d`; not timed on main, `CI_SKIP`) |
| `fuse_stencil_through_transient` | the default's fusions, then the `map-fusion` they leave (the stencil into the map that writes its transient) | CPU, 72 threads | 12.49 ms | 3.72 ms | 3.35x (main `fb130f1`) |
| `bellman_ford` | the default, then the one `map-fission` it leaves | GPU | 4347 ms | 2828 ms | 1.54x (main `fb130f1`) |

Preset M on a Daint GH200 node, mean of 5 calls, every variant matches the oracle; NestForge main `0e1a475`, DaCe `f77695949`, HPCAgent-Bench `1fa634af`. `aes_graupel` was timed at DaCe `3469167e5`, HPCAgent-Bench `0cc69ba`.

`cloudsc_nproma` is CI-skipped: the kernel is only on HPCAgent-Bench branch `benchmarking-nestforge`, and on DaCe `7683e143` its `map-fission` reads and writes `pfplsx` through one access node, a cyclic state that the stages after fusion reject (`NetworkXUnfeasible` in `route_ordering_through_data`). On main the default still leaves the route on the menu. At 72 threads the route was 1.19x at preset M and 2.04x at `NPROMA` 1024 (older pins).
