# cloudsc_lateral_drift

GPU scheduling benchmark. The gap is the two lateral sweeps. First measured 2026-10-09 on Daint (GH200), job 5012446, DaCe `73f0a120e`, NestForge `6e1ff61`. Rerun on this checkout on 2026-10-15, job 5066154, DaCe `11d27d22e`, NestForge `d14a393`: baseline 770.85 ms, reference 30.51 ms, both relative error 0. Rerun at the `interchange-benchmarks` pins on 2026-10-16, job 5069251, DaCe `73f0a120e`, NestForge `40f140d`: baseline 770.79 ms, reference 30.48 ms (25.3x), both relative error 0; this is the first block under "Reference solution". The sections below are the October run. The reruns are at the end.

## What this benchmark is

Rain sedimentation down the levels, then an implicit-upwind drift of the flux along the columns, eastward into `outE` and westward into `outW`, then a pointwise pick into `out` from a per-level `int32` mask `east`. Arrays are row-major `(level, column)` with the column index contiguous. `pfplsx` is carried down the levels (columns independent). `outE` is carried eastward and `outW` westward (levels independent). `flx`, `outE` and `outW` are work arrays. Only `out` is a result.

The scheduling problem is the two sweeps. Stage 1 on the GPU leaves each as a host loop over columns around a map over levels, so the program launches once per column. The reference pulls each column loop inside its map. The vertical carry is already a map over columns. Synthetic lateral drift on CloudSC section 4.2 / 5.3 flux code (`ecmwf-ifs/dwarf-p-cloudsc` `cloudsc.F90` at `f7ba9f8`); the drift itself is not in CloudSC.

## What the reference is

Both programs were checked against an independent vectorized NumPy oracle. Device: one NVIDIA GH200. Time is whole-program host wall time (`perf_counter` around the linked call), so host-device copies sit inside the number. Default kernel configs, no stage 6 search. Median of 5 calls after one correctness call.

- **Baseline.** Stage 1, then `default_moves()`. The sweeps stay host loops over columns. Tree under the reference block.
- **Reference.** Two `interchange-loop-map` moves, then `finish_moves()`. Each sweep is one map over levels; the column loop sits inside the kernel and does not appear as a tree row.

CPU is a different pair and has no measured gap. Stage 1 there already gives a map over levels for each sweep, and `default_moves()` fuses those sweeps. This benchmark claims the GPU pair only.

## How it was created

Source lines: `cloudsc.F90` rain sedimentation 1721–1756, the 5.2 diagonal at 2603, the 5.3 flux at 2705–2709, the section 8 flux sum at 2795–2800. The edit adds the two lateral sweeps and the pick.

The first sketch wrote both directions in place into one `out` array with `where`. `interchange-loop-map` was illegal on both sweeps (jobs 5011531, 5011602). The rewrite that was timed gives each direction its own array, so neither sweep reads the element it writes. `min` is the limiter and also stops `LoopToScan` from lifting the sweep.

Probes that failed on the way, all in [[2026-10-09-cloudsc-style-benchmarks]]:

- `map-fusion` of the two sweeps is refused: "nests are in different states". The reference is the two interchanges alone (job 5011820).
- `loop-fusion` of the two sweep loops is refused: different ranges, or a loop-carried dependency.
- The baseline host loops do not build until `_loop_it_5` and `_loop_it_7` are declared as symbols on the parent SDFG. That declaration is a harness workaround, not a stage 2 move (job 5012446).
- A reference without `finish_moves()` dies in stage 3: `UnsupportedNest: nested SDFG nested_single_state_body rebinds outer names ['NCLDTOP']` (job 5012671).
- Two harder variants (`b1_coupled`, `b1_tmp`) never get a legal interchange. They are not this benchmark.

The DaCe program in this folder is that timed body, renamed from `b1` to `cloudsc_lateral_drift`. The trees below were printed for SDFG `b1_kernel_b1`.

## What the solution is

On the GPU, after stage 1. Labels are from job 5012446 and are not stable across versions. Each move is the only `interchange-loop-map` offered at that moment. The second is not offered as a second row beside the first; it appears after the first applies, on the sweep that is still a host loop.

1. `interchange-loop-map` on the host loop and the map that writes `outE` (the run's labels `for0_0`, `kernel2_0`).
2. `interchange-loop-map` on the host loop and the map that writes `outW` (labels regenerated; the run recorded `for0_0`, `kernel2_0` again).
3. `finish_moves()`. Required. Without it the program does not build.

Tree after those moves and `finish_moves()`:

```
SDFG 'b1_kernel_b1'  epoch=4
|- state0_0
`- state0_1
   |- kernel1_0  [_loop_it_1=0:KLON]  reads=['pap', 'paph', 'pfplsx', 'zqr', 'ztp1'] writes=['pfplsx']
   |- kernel1_1  [_loop_it_2=0:NLEV + 1, _loop_it_3=0:KLON]  reads=['pfplsx', 'pfsnow'] writes=['flx']
   |- kernel1_2  [_loop_it_4=0:NLEV + 1]  reads=['flx'] writes=['outE', 'outW']
   |- kernel1_3  [_loop_it_6=0:NLEV + 1]  reads=['cap', 'flx', 'outE'] writes=['outE']
   |- kernel1_4  [_loop_it_8=0:NLEV + 1]  reads=['cap', 'flx', 'outW'] writes=['outW']
   `- kernel1_5  [_loop_it_9=0:NLEV + 1, _loop_it_10=0:KLON]  reads=['east', 'outE', 'outW'] writes=['out']
```

`kernel1_3` is the eastward sweep and `kernel1_4` the westward sweep. The column loop is inside each kernel.

## How to optimize it

Right after stage 1 on the GPU, `list_moves` offers `interchange-loop-map` on each sweep, `interchange-map-loop` on the vertical carry, and `map-fission` on the init kernel (job 5012193, rechecked). `default_moves()` does not take the sweep interchange. The baseline tree is the stage 1 shape.

Order: eastward sweep, then westward sweep, then `finish_moves()`. Either sweep alone is legal; the second stays offered after the first. Fusion of the two resulting maps is not offered.

Traps:

- `interchange-map-loop` on the carry (`kernel1_0` and its inner loop). Legal. `finish_moves()` puts the loop back, so a decoy that then calls `finish_moves()` is the reference again (30.60 ms in job 5012446, not a separate program).
- Undoing one sweep after the reference (`interchange-map-loop` on that sweep) leaves one host loop and about half the gap. Undoing both returns the baseline.
- Greedy "first interchange, up to 8 steps" lands on the reference only because it ping-pongs and the cap stops on an even count (job 5012951). That is not evidence the policy found the schedule.
- `map-fission` and `map-fusion` of the init pieces apply in random sequences and do not change the time.

## What the gains are

Preset M, one GH200, whole linked program, host wall time, 5 repeats, job 5012446. DaCe `73f0a120e`, NestForge `6e1ff61`, HPCAgent-Bench `26a4f0cfc`. Relative error against the independent NumPy oracle is 0 for every row.

| Program | Median | Min | Max |
| --- | --- | --- | --- |
| Baseline | 771.12 ms | 770.38 | 772.20 |
| Reference | 30.65 ms | 30.54 | 31.02 |

Ratio 25.2× (771.12 / 30.65). One run, one node.

Preset S whole-program baseline and reference were not measured. The reference at preset S passed the oracle on all six kernels (job 5011927); those kernel times are 10 to 70 µs and are not the gap.

The carry-inversion decoy without `finish_moves()` is 27.54 ms (min 27.01, max 27.70, job 5012671), 0.90× the reference. 30.65 ms is the designed schedule, not a minimum.

## CPU and GPU

GPU stage 1 leaves `for j { map k }` on both sweeps and offers `interchange-loop-map`. The 25.2× gap is on that device.

CPU stage 1 already gives a single map over levels for each sweep, and `default_moves()` fuses them into one kernel (jobs 5011752, 5012193). There is no sweep interchange left to apply. No CPU pair was timed. The benchmark is GPU-only.

The gap is launch count, not a thread-count effect. It is reported as whole-program host time.

## What else matters

- **Legality.** `map-fusion` across the two sweeps: "nests are in different states, a control-flow dependency map fusion never crosses". `loop-fusion` of the sweep loops: "different iteration ranges, or a loop-carried dependency". In-place `where` into one `out` (the first sketch): `interchange-loop-map` illegal on both sweeps; the reason string does not say which dependence.
- **Contiguity.** The column axis is contiguous. The reference walks it inside the level map. The baseline launches once per column.
- **Launches against stride.** At preset M the baseline host loops trip 32,767 and 32,766 times. The reference replaces each with one kernel. How much of the 30.65 ms is copies was not separated, so the kernel-only ratio is at least this and was not measured on its own.
- **Vectorization.** No GPU vectorization report was collected. The gain is not a SIMD claim.
- **Live values.** The sweeps carry `outE` and `outW` along columns. Splitting each sweep so the interchange needs a prior fusion was tried (`b1_tmp`) and `map-fusion` refused. This benchmark does not require that promotion.
- **Decoys.** One sweep undone: 404.09 ms (min 403.63, max 404.67). Both undone: 774.48 ms (min 773.80, max 778.01). Both job 5012951, through `finish_moves()`, error 0. The carry flip without `finish_moves()` is faster than the reference, so it is not a penalty at this size.
- **Oracle.** `initialize` in `cloudsc_lateral_drift.py`, seed 0: monotone pressure, temperature in `[220, 300]` K, `cap` small enough that the limiter binds (about 28% of updates at preset S, job 5012193). Relative error of every timed variant at preset M is 0.
- **Pins.** DaCe `extended` `73f0a120e` (2.0.0a10), NestForge `6e1ff61` (PR #3; the Daint tree `NestForge-b1` has no `.git`, the SHA is the one that session recorded), HPCAgent-Bench `26a4f0cfc`. A run on `d14a393` / DaCe `11d27d22e` is not this result. The baseline on the measured pin built only after the two host-loop symbols were declared by hand.

## Backtest

Score a later run by the oracle and by preset M host time near 30.7 ms. Matching the move list only checks that the sequence still replays.

Same setup as the reference (jobs 5012951 and 5012952), every schedule through `finish_moves()`:

| Schedule | Median |
| --- | --- |
| `default_moves()` | 771.12 ms |
| reference, then undo the eastward sweep | 404.09 ms |
| reference, then undo both sweeps | 774.48 ms |
| greedy, first offered interchange, 8 steps | 30.74 ms |
| random seed 0, up to 4 legal moves | 400.21 ms |
| random seed 1 | 399.36 ms |
| random seed 2 | 400.21 ms |
| random seed 3 | 770.06 ms |

None of the four random sequences reached the reference. Three interchanged one sweep. Seed 3 applied both interchanges and then undid both. Greedy matched the reference time because the 8-step cap stopped on the reference, after three undo-and-redo pairs.

## Rerun on the bench checkout (2026-10-15)

Job 5066154, 2026-10-15, debug partition, one GH200 (`nid006559`), preset M, 5 repeats. NestForge from `bench/cloudsc-lateral-drift-solution` at `d14a393` (the files in this folder), DaCe `11d27d22e`, HPCAgent-Bench `26a4f0cfc`. The linked program used `libomp` from `$SCRATCH/thesis/omp-env` because this uenv only has `libgomp`. The baseline again needed `_loop_it_5` and `_loop_it_7` declared on the parent SDFG. The reference declared none. Both trees match the October run. Relative error against the vectorized NumPy is 0.

| Program | Median | Min | Max |
| --- | --- | --- | --- |
| Baseline | 770.85 ms | 769.43 | 771.23 |
| Reference | 30.51 ms | 30.43 | 30.81 |

Ratio 25.3× (770.85 / 30.51).

```yaml
device: gpu
threads: 1
preset: M
dace: 11d27d22e143ce45c2d647a6255f5f047185ed2a
nestforge: d14a39399803eced745d7b73536065126dadc9eb
hpcagent_bench: 26a4f0cfc1540cead109dfc8d736f1c45263e390
baseline:
  moves: [default_moves]
  time_ms: 770.85
  time_min_ms: 769.43
  time_max_ms: 771.23
  repeats: 5
  oracle_rel_error: 0.0
reference:
  moves:
    - kind: interchange-loop-map
      on: the row that writes outE
    - kind: interchange-loop-map
      on: the row that writes outW
    - finish_moves: true
  time_ms: 30.51
  time_min_ms: 30.43
  time_max_ms: 30.81
  repeats: 5
  oracle_rel_error: 0.0
```

## Rerun at the interchange-benchmarks pins

Job 5069251, 2026-10-16, debug partition, one GH200 (`nid006548`), exclusive, preset M, 5 repeats, this folder's `time_both.py`. NestForge `interchange-benchmarks` at `40f140d` (this folder copied from `bench/cloudsc-lateral-drift-solution` unchanged), DaCe `73f0a120e` with HPCAgent-Bench `26a4f0cfc` (venv `venv-b1`), `libomp` from `$SCRATCH/thesis/omp-env`. The baseline again needed `_loop_it_5` and `_loop_it_7` declared; the reference declared none. Stage 1, baseline and reference trees match the October run. Relative error against the vectorized NumPy is 0.

| Program | Median | Min | Max |
| --- | --- | --- | --- |
| Baseline | 770.79 ms | 770.40 | 771.71 |
| Reference | 30.48 ms | 30.41 | 31.23 |

Ratio 25.3x (770.79 / 30.48).

## Reference solution

The first block is the current one, at the pins of `interchange-benchmarks`; the second is the October run, kept for its trees.

```yaml
device: gpu
threads: 1
preset: M
dace: 73f0a120e4f3ec38dd1fadeb27792e7b972ec324
nestforge: 40f140d
hpcagent_bench: 26a4f0cfc1540cead109dfc8d736f1c45263e390
host: nid006548, Daint job 5069251
baseline:
  moves: [default_moves]
  time_ms: 770.79
  time_min_ms: 770.40
  time_max_ms: 771.71
  repeats: 5
  oracle_rel_error: 0.0
  tree: |
    for0_0  _loop_it_5=1:KLON
      kernel2_0  [_loop_it_6=0:NLEV + 1]  reads=['cap', 'flx', 'outE'] writes=['outE']
    for0_1  _loop_it_7=0:KLON - 1
      kernel2_1  [_loop_it_8=0:NLEV + 1]  reads=['cap', 'flx', 'outW'] writes=['outW']
    (plus the carry, flux, init and pick kernels, as in the October tree below)
reference:
  moves:
    - kind: interchange-loop-map
      on: for0_0, kernel2_0 (the row that writes outE)
    - kind: interchange-loop-map
      on: for0_0, kernel2_0 (the row that writes outW, relabeled after the first move)
    - finish_moves: true
  time_ms: 30.48
  time_min_ms: 30.41
  time_max_ms: 31.23
  repeats: 5
  oracle_rel_error: 0.0
  tree: |
    kernel1_0  [_loop_it_1=0:KLON]  reads=['pap', 'paph', 'pfplsx', 'zqr', 'ztp1'] writes=['pfplsx']
    kernel1_1  [_loop_it_2=0:NLEV + 1, _loop_it_3=0:KLON]  reads=['pfplsx', 'pfsnow'] writes=['flx']
    kernel1_2  [_loop_it_4=0:NLEV + 1]  reads=['flx'] writes=['outE', 'outW']
    kernel1_3  [_loop_it_6=0:NLEV + 1]  reads=['cap', 'flx', 'outE'] writes=['outE']
    kernel1_4  [_loop_it_8=0:NLEV + 1]  reads=['cap', 'flx', 'outW'] writes=['outW']
    kernel1_5  [_loop_it_9=0:NLEV + 1, _loop_it_10=0:KLON]  reads=['east', 'outE', 'outW'] writes=['out']
```

```yaml
device: gpu
threads: 1
preset: M
dace: 73f0a120e4f3ec38dd1fadeb27792e7b972ec324
nestforge: 6e1ff61
hpcagent_bench: 26a4f0cfc1540cead109dfc8d736f1c45263e390
baseline:
  moves: [default_moves]
  time_ms: 771.12
  time_min_ms: 770.38
  time_max_ms: 772.20
  repeats: 5
  oracle_rel_error: 0.0
  tree: |
    SDFG 'b1_kernel_b1'  epoch=2
    |- state0_0
    |- state0_1
    |  |- kernel1_0  [_loop_it_1=0:KLON]  reads=['pap', 'paph', 'pfplsx', 'zqr', 'ztp1'] writes=['pfplsx']
    |  |- kernel1_1  [_loop_it_2=0:NLEV + 1, _loop_it_3=0:KLON]  reads=['pfplsx', 'pfsnow'] writes=['flx']
    |  `- kernel1_2  [_loop_it_4=0:NLEV + 1]  reads=['flx'] writes=['outE', 'outW']
    |- for0_0  _loop_it_5=1:KLON
    |  `- state1_0
    |     `- kernel2_0  [_loop_it_6=0:NLEV + 1]  reads=['cap', 'flx', 'outE'] writes=['outE']
    |- for0_1  _loop_it_7=0:KLON - 1
    |  `- state1_1
    |     `- kernel2_1  [_loop_it_8=0:NLEV + 1]  reads=['cap', 'flx', 'outW'] writes=['outW']
    `- state0_2
       `- kernel1_3  [_loop_it_9=0:NLEV + 1, _loop_it_10=0:KLON]  reads=['east', 'outE', 'outW'] writes=['out']
reference:
  moves:
    - kind: interchange-loop-map
      on: the row that writes outE
    - kind: interchange-loop-map
      on: the row that writes outW
    - finish_moves: true
  time_ms: 30.65
  time_min_ms: 30.54
  time_max_ms: 31.02
  repeats: 5
  oracle_rel_error: 0.0
  tree: |
    SDFG 'b1_kernel_b1'  epoch=4
    |- state0_0
    `- state0_1
       |- kernel1_0  [_loop_it_1=0:KLON]  reads=['pap', 'paph', 'pfplsx', 'zqr', 'ztp1'] writes=['pfplsx']
       |- kernel1_1  [_loop_it_2=0:NLEV + 1, _loop_it_3=0:KLON]  reads=['pfplsx', 'pfsnow'] writes=['flx']
       |- kernel1_2  [_loop_it_4=0:NLEV + 1]  reads=['flx'] writes=['outE', 'outW']
       |- kernel1_3  [_loop_it_6=0:NLEV + 1]  reads=['cap', 'flx', 'outE'] writes=['outE']
       |- kernel1_4  [_loop_it_8=0:NLEV + 1]  reads=['cap', 'flx', 'outW'] writes=['outW']
       `- kernel1_5  [_loop_it_9=0:NLEV + 1, _loop_it_10=0:KLON]  reads=['east', 'outE', 'outW'] writes=['out']
```
