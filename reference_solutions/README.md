# Reference solutions

One file per benchmark, `<kernel>.py`. Its `solve(s, ctx)` applies the reference moves; replay times the result against `default_moves()` and checks it against the bench oracle.

```
python -m nestforge.replay reference_solutions/<kernel>.py
```

**Adding a solution:** keep the script short, like `tsvc_2_s231.py`: the kernel, the config and the moves, picked by kind and by what they read and write. Add a kernel here once its moves beat the default by more than the timing noise (10-17% between Daint nodes). CI replays every file here at preset S on one CPU thread (`tests/test_replay.py`), so a solution that stops applying or no longer matches the oracle fails the build.

| Kernel | Moves | Config | Default | Reference | Gain |
| --- | --- | --- | --- | --- | --- |
| `aes_graupel` | `map-fission` of the map computing `sink` and `sx2x`, then the default | CPU, 72 threads | 7705 ms | 1022 ms | 7.54x (previous pins; re-timing on main pending) |
| `tsvc_2_s233` | `loop-fission`, then `interchange-loop-map` on the `aa` half and the `bb` half | CPU, 72 threads | 30.7 ms | 8.8 ms | 3.47x |
| `tsvc_2_s231` | `interchange-loop-map` (the `j` loop into the `i` map) | CPU, 72 threads | 39.5 ms | 16.5 ms | 2.40x |

Preset M on a Daint GH200 node, mean of 5 calls, every variant matches the oracle; NestForge main `0e1a475`, DaCe `f77695949`, HPCAgent-Bench `1fa634af`. `aes_graupel` was timed at DaCe `3469167e5`, HPCAgent-Bench `0cc69ba`.
