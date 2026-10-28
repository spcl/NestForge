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

Preset M on a Daint GH200 node, mean of 5 calls, every variant matches the oracle; NestForge main `0e1a475`, DaCe `f77695949`, HPCAgent-Bench `1fa634af`. `aes_graupel` was timed at DaCe `3469167e5`, HPCAgent-Bench `0cc69ba`.

`cloudsc_nproma` is CI-skipped: the kernel is only on HPCAgent-Bench branch `benchmarking-nestforge`, and on DaCe `7683e143` its `map-fission` reads and writes `pfplsx` through one access node, a cyclic state that the stages after fusion reject (`NetworkXUnfeasible` in `route_ordering_through_data`). On main the default still leaves the route on the menu. At 72 threads the route was 1.19x at preset M and 2.04x at `NPROMA` 1024 (older pins).

## Rewritten programs and changed parameters

A solution may start from a rewritten loop nest of a corpus kernel, or run a corpus kernel at other parameters, while the bench's own inputs and NumPy reference still grade it:

- `PROGRAM = "programs/<name>_dace.py"`: a `@dace.program` with the corpus kernel's arguments, optimized instead of the kernel's own `_dace.py`. `KERNEL` names the corpus kernel whose manifest, inputs and oracle it uses.
- `PARAMS = {...}`: preset parameters merged over the preset (the bench's `params_override`). Integer entries also become the program's sizes.

| Solution | Corpus kernel | Program or parameters | Route | Status |
| --- | --- | --- | --- | --- |
| `scattering_self_energies_m2` | `scattering_self_energies` | explicit loop nest `k, E, q, w, a, b, i, j`; `Nkz` 2, `NE` 4 | 2 `interchange-loop-map` (map `a` out past `w`, `q`) | applies, oracle ok at S |
| `scattering_self_energies_m3` | `scattering_self_energies` | the same nest with `i` before `a, b`; `Nkz` 2, `NE` 8 | `interchange-map-loop`, then 4 `interchange-loop-map` | applies, oracle ok at S |
| `conv3d_box` | `conv_3d` | output-stationary box stencil, padding inside | GPU: one `subgraph-fission`; CPU: 2 `map-fission` and 3 `interchange-map-loop` | GPU 6.55x at S, oracle ok |
| `seissol_volume` | `seissol_tensor_contraction` | output-stationary contraction, `Q += upd` | GPU: one `map-fission`; CPU: the fission and 2 `interchange-map-loop` | `CI_SKIP`: replay at S does not finish |
| `dbcsr_small_blocks` | `dbcsr` | `block_size` 4, `density` 0.05 | `interchange-loop-if` | `CI_SKIP`: the default fails to build |

The programs and parameters come from the kernels of HPCAgent-Bench branch `benchmarking-nestforge` (`d22922c5c`, spcl/HPCAgent-Bench#124, reverted in #128 as duplicates of corpus kernels). None is timed on these pins yet.

