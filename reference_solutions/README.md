# Reference solutions

One file per benchmark, `<kernel>.py`. Its `solve(s, ctx)` applies the reference moves; replay times the result against `default_moves()` and checks it against the bench oracle.

```
python -m nestforge.replay reference_solutions/<kernel>.py
```

**Adding a solution:** keep the script and its section short and easy to scan, like `tsvc_2_s231` below. The script holds only the moves, picked by kind and by what they read and write. The section lists only the moves, a few words on when they apply, and a small table of each config's default time, reference time and gain. Leave full replay output in `experiments/replay/`. CI replays every file here at preset S on one CPU thread (`tests/test_replay.py`), so a solution that stops applying or no longer matches the oracle fails the build.

## tsvc_2_s231

**Move:** `interchange-for-doall`, which moves the `j` loop inside the `i` map. It applies on a threaded CPU and on the GPU; on one thread, no move.

| Config | Default | Reference | Gain |
| --- | --- | --- | --- |
| CPU, 72 threads | 39.8 ms | 16.6 ms | 2.39x |
| CPU, 1 thread | 55.4 ms | 55.4 ms | 1.00x (the interchange would be 17x slower) |
| GPU | 38.4 ms | 37.7 ms | 1.02x (the default already interchanges) |

Preset M on a Daint GH200 node; DaCe `3469167e5`, HPCAgent-Bench `0cc69ba`.
