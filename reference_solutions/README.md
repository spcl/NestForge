# Reference solutions

One file per benchmark, `<kernel>.py`. Its `solve(s, ctx)` applies the reference moves; replay times the result against `default_moves()` and checks it against the bench oracle.

```
python -m nestforge.replay reference_solutions/<kernel>.py
```

**Adding a solution:** keep the script short, like `tsvc_2_s231.py`: the kernel, the config and the moves, picked by kind and by what they read and write. Add a kernel here once its moves beat the default by more than the timing noise (10-17% between Daint nodes). CI replays every file here at preset S on one CPU thread (`tests/test_replay.py`), so a solution that stops applying or no longer matches the oracle fails the build. A script that cannot run on the pins yet sets `CI_SKIP` to the reason, and CI skips it.

All 13 benchmarks, the latest measurement of each. Preset M on a Daint GH200 node; every variant matches the bench oracle. **Pins** is the NestForge commit the numbers were taken on: `1309709` is this branch (DaCe `98482ac25`, HPCAgent-Bench `ed1f78857`); a row on older pins has not been re-timed here yet. **CI** says whether `tests/test_replay.py` replays the file at `cpu:1:S`.

| Kernel | Config | Moves | Default | Reference | Gain | Pins | CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `cholesky2` | CPU, 72 threads | `expand-library` of the `np.linalg.cholesky` call (NestForge's own loops instead of the library), then the default | 22.7 s | 121 ms | 187x (3 alternated rounds) | `1309709` | skipped (`CI_SKIP`) |
| `hmm_forward` | CPU, 72 threads | `expand-library` of the product `alpha @ trans` and of `sum(alpha)` (no vendor call left in the time loop), then the default | 1750.0 ms | 184.7 ms | 9.47x (3 alternated rounds) | `1309709` | replayed |
| `bfs` | GPU | the default, then the `interchange-doall-for` it leaves on `reach` (the source-vertex loop back outside the target-vertex map) | 1427.2 ms | 364.1 ms | 3.92x (see the note below) | `1309709` | replayed |
| `tsvc_2_s233` | CPU, 72 threads | `for-fission`, then `interchange-for-doall` on the `aa` half and the `bb` half | 30.7 ms | 8.8 ms | 3.47x | `0e1a475` (DaCe `f77695949`) | replayed |
| `fuse_stencil_through_transient` | CPU, 72 threads | the default's fusions, then the `doall-fusion` they leave (the stencil into the map that writes its transient) | 12.49 ms | 3.72 ms | 3.35x | `fb130f1` (DaCe `0f8ac0227`) | replayed |
| `pagerank` | CPU, 72 threads | `expand-library` of the product `trans @ rank` and of `sum(rank)` (no vendor call left in the loop), then the default | 493.2 ms | 150.0 ms | 3.29x (3 alternated rounds) | `1309709` | replayed |
| `tsvc_2_s231` | CPU, 72 threads | `interchange-for-doall` (the `j` loop into the `i` map) | 39.5 ms | 16.5 ms | 2.40x | `0e1a475` (DaCe `f77695949`) | replayed |
| `cloudsc_nproma` | CPU, 1 thread | `doall-fission` of the column map, then `interchange-doall-for` of the body map and the level loop | 110.2 ms | 46.9 ms | 2.35x | DaCe `042413d` | skipped (`CI_SKIP`) |
| `bellman_ford` | GPU | the default, then the one `doall-fission` it leaves | 4319.3 ms | 2821.4 ms | 1.53x | `1309709` | replayed |
| `fuse_physics_into_scan` | GPU | the default, then the `doall-fission` of the column map that writes `s` (the physics out of the scan kernel) | 16.03 ms | 11.34 ms | 1.41x | `1309709` | replayed |
| `velocity_tendencies` | CPU, 72 threads | the default, then the edge phase as one kernel rewritten to one 64-column tile (no menu move; see the note below) | 171.26 ms | 131.81 ms | 1.30x | `1309709` | replayed |
| `harris_corner` | GPU | every producer fused by hand into the box-sum doall that reads it (the products, then `Ix`, `Iy` and their chains; recomputed per tap), then the default: one kernel `img -> R` | 27.53 ms | 22.14 ms | 1.24x (6 alternated rounds, 1.15-1.39x) | `1309709` | replayed |
| `aes_graupel` | CPU, 72 threads | `doall-fission` of the map computing `sink` and `sx2x`, then the default | 459.5 ms | 519.1 ms | **0.89x: no gain on this branch** (7.54x, 7705 to 1022 ms, at DaCe `3469167e5`, bench `0cc69ba`) | `1309709` | replayed |

`aes_graupel` no longer beats the default on this branch's pins (jobs 5197631, 5197819): its reference needs a new route or removal, and it should not be counted until then.

`bfs` is close to "take no move", so treat it with care. On the GPU the default itself interchanges the source-vertex loop of `frontier @ graph` into the target-vertex map, which leaves the N = 1200 threads of preset M each summing a whole column alone; the reference undoes that interchange. Taking no move and skipping the default's fusions is almost as fast: 410.1 ms, 3.45x, so the reference beats it by only 5%, within the node noise. The kernel is kept because a real move beats the default (as for `bellman_ford`), but its gain mostly measures what the default gets wrong on the GPU, not what the move adds. On the CPU the default keeps the loop outside and leaves the opposite move (`interchange-for-doall`), which is slower there (0.84x in the screen), so the reference is the default; the script declares the no-move variant as a decoy (`stage1`).

`fuse_physics_into_scan` also declares a decoy: the reference, then the physics map interchanged so the levels are outermost, which leaves only the 90 levels parallel on the GPU (300 ms, 0.05x of the default).

`harris_corner` was timed with the per-kernel check against the emitted Python skipped (moves, scopes, placement and kernel builds once, then the whole program against the bench oracle): that check runs the fused kernel as a Python loop over every point, hours at preset M, so a plain `python -m nestforge.replay` of this file at M is very slow. On the CPU the same route is slower than the default (0.75x at 2048 x 4096, DaCe `0f8ac0227`), so the reference there is the default.

`cholesky2` is CI-skipped: the per-kernel check against the emitted Python fails on the expanded Cholesky kernel (`NameError: name 'conj' is not defined`, a NestForge emitter gap, also seen on `quatrex_rgf`); the timings above check the whole program against the bench oracle. It depends on the device: on the GPU the same `expand-library` takes 9.5 s against the default's 8.5 ms (the vendor solver), so the GPU reference is the default and the expansion is its decoy; on the CPU the decoy is `set-library-implementation` to OpenBLAS, which times like the default (22.3 s). The CPU default's 22 s (about 1 Gop/s for the factorization) is far below what OpenBLAS reaches on one core, so part of this gain is the library setup on Daint, not only the schedule; the default still takes that path.

`pagerank` needs both expansions: expanding only the product fails to build (`KeyError: '_out_gemv'`, the same error as `atax` after expanding its products), so that half-route cannot be listed as a decoy. On the GPU the default does not build (`CompilationError`), while the two expansions build and run (148.5 ms, oracle ok); the GPU becomes a configuration once the default builds there.

`hmm_forward` takes the same route as `pagerank` on another program (forward probabilities of a hidden Markov model): one family, two kernels. Its GPU default does not build (`CompilationError`), so the GPU is not a configuration yet.

`cloudsc_nproma` is CI-skipped: the kernel is not in HPCAgent-Bench `main` (it came with spcl/HPCAgent-Bench#124, which was reverted; its files are at bench commit `d22922c5c`), and on DaCe `7683e143` its `doall-fission` reads and writes `pfplsx` through one access node, a cyclic state that the stages after fusion reject (`NetworkXUnfeasible` in `route_ordering_through_data`). On main the default still leaves the route on the menu. At 72 threads the route was 1.19x at preset M and 2.04x at `NPROMA` 1024 (older pins).

`velocity_tendencies` is not a menu move. After `default_moves` the edge phase (the nests that write `vt`, `vn_ie`, `z_kin_hor_e`, `z_vt_ie`, `z_w_concorr_me` and `z_v_grad_w`) is one kernel, and that kernel's source is rewritten so one OpenMP loop over tiles of 64 columns holds every level loop of the phase. The default strip-mines each nest alone, so the next nest reads `vn` and `vt` back from memory. Preset M, 72 threads, mean of 5, oracle ok, job 5194998 on one GH200: 159.52 ms to 118.26 ms, 1.35x, on main `fb130f1` (DaCe `0f8ac0227`, HPCAgent-Bench `e7d161d6f`). A 16-thread login-node run of the same route was 206.89 ms against 448.36 ms in that process. The gather-loop interchanges and the subgraph cuts were slower than the default.
