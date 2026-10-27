# Replay of tsvc_2_s231

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
