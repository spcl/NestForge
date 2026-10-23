# cloudsc_kcache_cpu

CPU scheduling benchmark (slot M3 of the benchmark set, 2 moves). The gap is the order of the column map and the level loop. First measured 2026-10-09 on Daint (GH200) as `kc_arr2`, DaCe `73f0a120e`, an older NestForge checkout (`NestForge-b1`, SHA not recorded). Route re-probed on 2026-10-21 at DaCe `73f0a120e`, NestForge `40f140d` (job 5110945). Added to this folder on 2026-10-21 at DaCe `801cafd`, NestForge `interchange-benchmarks` `4df1830`, HPCAgent-Bench `e63f9f1` (job 5111095): the DaCe program matches the NumPy oracle there, but stage 1 does not run on it at these pins (see "Blocker at the current pins"). **Not finished:** no time has been measured at these pins, and the reference block below holds the 2026-10-09 numbers.

## What this benchmark is

Rain and snow precipitation physics from CloudSC (cloud cover, rain evaporation, melting, sedimentation flux, cumulative fluxes) in the loop structure of the dwarf's GPU `k-caching` port: one loop over columns `jl` whose body is a per-column prologue, then the sequential level loop `jk`. Arrays are row-major `(level, column)`, so `klon` is the contiguous axis; 19 arrays of shape `(nlev, klon)` or `(nlev + 1, klon)`. Columns are independent and levels are sequential. Six values are carried from one level to the next within a column (`zcov`, `zcovmax`, `zpfr`, `zpfs`, `zsqrf`, `zsqsf`); they live in `(klon,)` transient arrays, set in the prologue and read once and written once per level.

The scheduling problem is to recover the CPU order from GPU code: `map jl { prologue; for jk { body } }` should become `map jl { prologue }; for jk { map jl { body } }`, so that at each level the inner map walks the columns at stride 1 and is parallel. Source: `ecmwf-ifs/dwarf-p-cloudsc` at `f7ba9f8`, loop structure from `src/cloudsc_gpu/cloudsc_gpu_scc_k_caching_mod.F90` (prologue 635–658, level loop 665–2636), physics from `cloudsc.F90` (4.2 sedimentation 1715–1793, 4.5 rain evaporation 2128–2196, 5.3 fluxes 2700–2725, section 8 fluxes 2795–2879). The line references are in `cloudsc_kcache_cpu_numpy.py`.

## What the reference is

- **Baseline.** Stage 1, then `default_moves()`: one map over columns with the level loop inside (the GPU order). `default_moves()` does not change the stage 1 tree.
- **Reference (CPU).** `subgraph-fission` of the column map after the prologue, then `interchange-map-loop` on the second map and its level loop, then `finish_moves()`: a prologue map over columns, then `for jk { map jl { level body } }`.

The pair is for the CPU, at 1 and at 72 threads. On GPU the reference is no move (see "CPU and GPU").

## How it was created

Built on 2026-10-09 as B2 of [[2026-10-09-cloudsc-style-benchmarks]]. The first program (`kc_kernel`) kept the carries as body-local scalars. There `subgraph-fission` applies and `interchange-map-loop` is then not offered, and the fission dies in stage 3 with `UnsupportedNest: nested connector 'zcov' is (1,), but 'zcov' is (KLON,)` (job 5014410). The rewrite that was timed, `kc_arr2`, stores each carry as a `(KLON,)` array, read once into a scalar at the start of a level and written once at its end (no statement reads the element it writes). On it both moves apply (job 5014476).

`cloudsc_kcache_cpu_dace.py` is `kc_arr2` with two edits: the size symbols are renamed from `NLEV`, `KLON`, `NCLDTOP` to the manifest's `nlev`, `klon`, `ncldtop`, and the constants are inlined instead of imported. It is hand-written, not emitted. `cloudsc_kcache_cpu.py` holds the input generator (`initialize`); `cloudsc_kcache_cpu_numpy.py` is the 2026-10-09 oracle unchanged. The 2026-10-09 timings used `kc_kernel.make_inputs`, not this `initialize`.

## What the solution is

On the CPU, after stage 1. Labels are from job 5110945 (DaCe `73f0a120e`) and are not stable across versions.

1. `subgraph-fission` on the column map `kernel1_0`, cut after `state2_0` (the prologue). The map becomes `kernel1_0` (reads `paph`, writes the six carries and `zpaphd`) and `kernel1_1` (the level loop). `interchange-map-loop` is not offered before this move, because the map body is the prologue plus the loop, not the loop alone.
2. `interchange-map-loop` on `kernel1_1` and its level loop `for2_0`.
3. `finish_moves()`. On CPU it keeps the shape.

Tree after the moves (job 5110945, CPU; memset rows and bodies left out):

```
SDFG 'kc_arr2_kc'  epoch=4
|- state0_0
|- state0_1
|  `- kernel1_0  [_loop_it_0=0:KLON]  reads=['paph'] writes=['zcov', 'zcovmax', 'zpaphd', 'zpfr', 'zpfs', 'zsqrf', 'zsqsf']
`- for0_0  _loop_it_1=NCLDTOP:NLEV
   `- state1_0
      `- kernel2_0  [_loop_it_0=0:KLON]  reads=['pa', 'pap', 'paph', 'pq', 'pt', 'qr', 'qs', 'tt', 'zcov', 'zcovmax', 'zpaphd', 'zpfr', 'zpfs', 'zsqrf', 'zsqsf'] writes=['pcovptot', 'pfhpsl', 'pfhpsn', 'pfplsl', 'pfplsn', 'pfsqrf', 'pfsqsf', 'tnd_q', 'tnd_qr', 'tnd_qs', 'tnd_t', 'zcov', 'zcovmax', 'zpfr', 'zpfs', 'zsqrf', 'zsqsf']
```

## How to optimize it

After stage 1, `list_moves` offers `map-fission` (1) and `subgraph-fission` (1), on CPU and GPU alike (job 5110945). `default_moves()` takes neither. Moves offered on the way (job 5110945):

| Step | Offered |
| --- | --- |
| stage 1 | `map-fission` 1, `subgraph-fission` 1 |
| after `subgraph-fission` | `map-fission` 1, `subgraph-fission` 1, `interchange-map-loop` 1 |
| after `interchange-map-loop` | `map-fission` 2, `subgraph-fission` 1, `interchange-loop-map` 1 (the undo) |
| `map-fission` from stage 1 instead | `map-fusion` 21, `interchange-map-loop` 1 |

The fission is worth almost nothing by itself (359 ms against 364 ms for the baseline, one thread, 2026-10-09); it is worth making only for the interchange it unlocks. `map-fission` from stage 1 is a second route: it splits the column map into 8 maps (7 one-statement prologue maps and the map with the level loop) and makes `interchange-map-loop` legal on the last one. That route is untimed. The 21 `map-fusion` pairs it offers re-fuse prologue maps and change nothing that matters. Traps: undoing the interchange (`interchange-loop-map` after the reference, 398 ms on one thread), and the CPU reference on GPU.

## What the gains are

Not measured at the current pins. The 2026-10-09 numbers (DaCe `73f0a120e`, NestForge `NestForge-b1`, preset M = 137 × 32768, whole program, median of 5, error 0 against NumPy on every row, GH200 node):

| Setting | Baseline (default) | Reference | Ratio | Job |
| --- | --- | --- | --- | --- |
| CPU, 1 thread | 364.09 ms | 321.23 ms (rerun 302.85 ms) | 1.13× (1.20×) | 5014533 (5014618) |
| CPU, 72 threads | 6.67 ms | 6.03 ms | 1.11× | 5014533 |
| CPU, 1 thread, fast-math | 327.42 ms | 242.24 ms | 1.35× | 5014656 |
| CPU, 72 threads, fast-math | 15.86 ms | 6.08 ms | 2.61× (the baseline slows down; the reference is unchanged from 6.03 ms) | 5014656 |
| GPU | 8.24 ms | 8.70 ms | 0.95× | 5014677 |

Min and max were not recorded for these rows. At preset S the reference is slower than the baseline (6.06 to 16.45 ms, job 5014476), which is launch noise, not the gap.

**A searched schedule beat the reference.** Of four random legal sequences of up to 4 moves at one thread, one ran in 244 ms, against 321 ms for the reference (job 5014677). Its moves were not logged. The reference is therefore a designed schedule, not a minimum. The `map-fission` route is a plausible candidate for the faster one.

## CPU and GPU

Stage 1 leaves the same program on both: one column map with the level loop inside, which is already the GPU order. On CPU the reference applies and `finish_moves()` keeps it (job 5110945). On GPU both moves are offered and apply, but `finish_moves()` puts the level loop back inside the column map (job 5110945, DaCe `73f0a120e`), so the program equals the baseline; on 2026-10-09, with the older NestForge, the reference stayed and was slower (8.24 to 8.70 ms). The gap is CPU only, small at default flags on 1 and 72 threads.

## What else matters

- **Legality.** `interchange-map-loop` is not offered at stage 1. On the scalar-carry program it is not offered after the fission either (job 5014410).
- **Contiguity and stride.** `klon` is contiguous. The reference walks it innermost at every level; the baseline walks `jk` innermost inside each column (stride `klon`).
- **Launches against stride.** One parallel region per level in the reference (122 at preset M) against one for the baseline; the reference still wins at 72 threads, by 1.11×.
- **Vectorization.** It does not vectorize: the compiler reports `vector .2d fp ops 0` on both the baseline and the reference body (2026-10-09). The level body calls `exp`, `sqrt` and `pow`; at one thread fast-math speeds the reference up more than the baseline, which suggests strict IEEE libm calls (`-fmath-errno`) block vectorization. Unchecked: `-fopt-info-vec-missed`, and one build with only `-fno-math-errno`.
- **Live values and carries.** The six carries must be `(klon,)` arrays for the interchange to be legal; body-local scalars block it (see "How it was created").
- **Decoys.** Undoing the interchange: 398 ms on one thread (job 5014677). The CPU reference on GPU: `finish_moves()` silently undoes it at `73f0a120e`, so it is not a measured decoy at that pin.
- **Oracle.** Inputs from `initialize()` in `cloudsc_kcache_cpu.py`: monotone `paph` (the kernel divides by `paph[jk+1] - paph[jk]`), `pap` the mid-levels, `pq` below saturation, `pt` rising through freezing so the melt limiter is partly active, `qr`, `qs`, `pa` with about 30% exact zeros. At the current pins the DaCe program matches the oracle at preset S with relative error 4.8e-13 (job 5111095); on 2026-10-09 the timed builds matched NumPy with error 0 at preset M.
- **Pins.** The timings are from DaCe `73f0a120e` and an older NestForge; the route is confirmed at `73f0a120e` only. Neither is this benchmark's result at the current pins.

## Blocker at the current pins

At DaCe `801cafd`, NestForge `4df1830`, HPCAgent-Bench `e63f9f1` (job 5111095), `canonicalize()` fails on this kernel, on CPU and GPU, with `ValueError: The argument 'nan' is not comparable.` The original `kc_arr2` fails the same way, and the other repo-local kernels and three corpus kernels pass stage 1, so it is this kernel's body, not the symbol rename. The cause is in DaCe's work-depth analysis, which `Session.describe()` runs on every call (`nestforge/ir/introspect.py`, `program_work_depth`): `get_static_symbols` in `dace/sdfg/performance_evaluation/helpers.py` maps each tasklet input connector to the name of the array it reads and drops the index, so `zdp = paph[jk + 1, jl] - paph[jk, jl]` becomes `paph - paph = 0`. Then `zdtgdp = PTSPHY * RG / zdp` becomes complex infinity, and `Min(zdpr * zdtgdp, zqrf)` raises on the resulting `nan`. Until DaCe or NestForge handles this, the route cannot be re-checked and nothing can be built or timed at these pins.

## Backtest

From job 5014677 (one thread, preset M, 2026-10-09): `default_moves()` 364 ms; fission alone 359 ms; the reference 321 ms; greedy 342 ms; four random sequences 244, 339, 290 and 285 ms (moves not logged); the reference plus the undo 398 ms. To redo at the current pins once the blocker is fixed, with every random sequence logged.

## Reference solution

Filled from the 2026-10-09 run, not from the current pins; the trees are from job 5110945 at `73f0a120e`. To be replaced by a run at the current pins.

```yaml
device: cpu
threads: 1
preset: M
dace: 73f0a120e4f3ec38dd1fadeb27792e7b972ec324
nestforge: NestForge-b1 (2026-10-09 checkout, SHA not recorded)
hpcagent_bench: 26a4f0cfc1540cead109dfc8d736f1c45263e390
host: GH200 node, Daint job 5014533
note: not measured at the current pins (DaCe 801cafd, NestForge 4df1830, HPCAgent-Bench e63f9f1); min and max not recorded
baseline:
  moves: [default_moves]
  time_ms: 364.09
  repeats: 5
  oracle_rel_error: 0.0
  tree: |
    SDFG 'kc_arr2_kc'
    `- state0_0
       `- kernel1_0  [_loop_it_0=0:KLON]  (prologue, then the level loop)
reference:
  moves:
    - kind: subgraph-fission
      on: kernel1_0 (the column map), cut after state2_0 (the prologue)
    - kind: interchange-map-loop
      on: kernel1_1 (the column map holding the level loop), for2_0 (the level loop)
    - finish_moves: true
  time_ms: 321.23
  repeats: 5
  oracle_rel_error: 0.0
  tree: |
    SDFG 'kc_arr2_kc'
    |- state0_1
    |  `- kernel1_0  [_loop_it_0=0:KLON]  reads=['paph'] writes=['zcov', 'zcovmax', 'zpaphd', 'zpfr', 'zpfs', 'zsqrf', 'zsqsf']
    `- for0_0  _loop_it_1=NCLDTOP:NLEV
       `- state1_0
          `- kernel2_0  [_loop_it_0=0:KLON]  (the level body)
```

```yaml
device: cpu
threads: 72
preset: M
dace: 73f0a120e4f3ec38dd1fadeb27792e7b972ec324
nestforge: NestForge-b1 (2026-10-09 checkout, SHA not recorded)
hpcagent_bench: 26a4f0cfc1540cead109dfc8d736f1c45263e390
host: GH200 node, Daint job 5014533
note: not measured at the current pins; same moves as the 1-thread block
baseline: {moves: [default_moves], time_ms: 6.67, repeats: 5, oracle_rel_error: 0.0}
reference: {moves: [subgraph-fission, interchange-map-loop, finish_moves], time_ms: 6.03, repeats: 5, oracle_rel_error: 0.0}
```

```yaml
device: gpu
preset: M
note: no move is the reference; at 73f0a120e finish_moves() undoes the CPU reference on GPU (job 5110945); 2026-10-09 with the CPU reference applied, job 5014677
baseline: {moves: [default_moves], time_ms: 8.24, repeats: 5}
trap: {moves: [subgraph-fission, interchange-map-loop, finish_moves], time_ms: 8.70, repeats: 5}
```
