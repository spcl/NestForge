# Reference solutions

One file per benchmark, `<kernel>.py`, named after its HPCAgent-Bench corpus kernel. A file defines:

- `KERNEL`: the corpus kernel.
- `CONFIGS`: the configurations it is measured on, `Context(device, threads, preset)`.
- `solve(s, ctx)`: the reference schedule, as stage 2 calls on a `nestforge.replay.Script`.
- `DECOYS` (optional): named alternatives that are timed next to the reference.

Moves are picked by kind and by the containers their rows read and write, never by label: `s.apply(s.pick("interchange-loop-map", reads=["bb"], writes=["aa"]))`, then `s.finish()`. If a pick matches no move, or several, it fails and prints the offered moves and the tree. That is how a solution that no longer applies on a new pin shows up.

```
python -m nestforge.replay reference_solutions/<kernel>.py [--config cpu:72:M] [--md reference_solutions/README.md]
```

For each configuration, replay runs the baseline (`default_moves()`), the reference and the decoys on copies of the canonicalized session. Each copy then goes through the default stages 3 to 5. Replay times the whole program (the mean of `--reps` calls) and checks it against the bench oracle. Each configuration runs in a process launched with its OpenMP thread count and the bench's OpenMP launch environment. Records go to `experiments/replay/<kernel>/<device>-<threads>-<preset>.json`. `--md` rewrites that kernel's table below.

## tsvc_2_s231

`aa[j, i] = aa[j-1, i] + bb[j, i]`. After stage 1 the program is `for j { map i }`. The reference takes the one offered move, `interchange-loop-map`, on a threaded CPU and on the GPU. On one thread it takes nothing, because the interchanged column walk is strided. Measured on Daint (GH200, job 5161429) at DaCe `3469167e5` and HPCAgent-Bench `0cc69ba`:

<!-- replay:tsvc_2_s231:begin -->
| Config | Variant | Moves | Time (ms) | Speedup | Oracle |
| --- | --- | --- | --- | --- | --- |
| cpu, 1 thread(s), preset M | baseline | default_moves | 55.43 | 1.00x | ok |
| cpu, 1 thread(s), preset M | reference | none | 55.35 | 1.00x | ok |
| cpu, 1 thread(s), preset M | interchange | interchange-loop-map | 958.54 | 0.06x | ok |
| cpu, 1 thread(s), preset M | no-move | none | 55.43 | 1.00x | ok |
| cpu, 72 thread(s), preset M | baseline | default_moves | 39.76 | 1.00x | ok |
| cpu, 72 thread(s), preset M | reference | interchange-loop-map | 16.61 | 2.39x | ok |
| cpu, 72 thread(s), preset M | interchange | interchange-loop-map | 16.48 | 2.41x | ok |
| cpu, 72 thread(s), preset M | no-move | none | 52.88 | 0.75x | ok |
| gpu, 1 thread(s), preset M | baseline | default_moves | 38.42 | 1.00x | ok |
| gpu, 1 thread(s), preset M | reference | interchange-loop-map | 37.66 | 1.02x | ok |
| gpu, 1 thread(s), preset M | interchange | interchange-loop-map | 37.42 | 1.03x | ok |
| gpu, 1 thread(s), preset M | no-move | none | 171.06 | 0.22x | ok |

- cpu-1-M: 2026-10-27T06:52:50+01:00, mean of 5 calls; nestforge `0aa3ec0+replay-uncommitted`, dace `3469167e5`, hpcagent_bench `0cc69ba10`.
- cpu-72-M: 2026-10-27T06:51:27+01:00, mean of 5 calls; nestforge `0aa3ec0+replay-uncommitted`, dace `3469167e5`, hpcagent_bench `0cc69ba10`.
- gpu-1-M: 2026-10-27T06:59:13+01:00, mean of 5 calls; nestforge `0aa3ec0+replay-uncommitted`, dace `3469167e5`, hpcagent_bench `0cc69ba10`.
<!-- replay:tsvc_2_s231:end -->
