# cloudsc_nproma

CPU scheduling benchmark (slot M4 of the benchmark set, 2 moves). The gap is the order of the ragged column map and the level loop inside each block. First measured 2026-10-09 on Daint (GH200), DaCe `73f0a120e`, an older NestForge checkout (`NestForge-b1`). Route re-probed 2026-10-21 at DaCe `801cafd`, NestForge `4df1830`, HPCAgent-Bench `e63f9f1` (jobs 5112741, 5112794). Entry note: `2026-10-21-w11a-cloudsc-nproma` (thesis notes).

**Status:** `entry` (route, decoys and preset-S oracle at this branch's pins; not timed here). The reference build needs the `passes_through` fix (spcl/NestForge#19); see "Blocker at the current pins".

## What this benchmark is

CloudSC precipitation fall and cover in the driver's blocked NPROMA layout, written in the GPU loop order. Arrays are `(nblk, [5,] nlev[+1], nproma)` in C order, so `nproma` is contiguous. The nest is a block map, a ragged column map (`min(nproma, ngptot - ibl * nproma)`), then the level loop. The level loop carries the species flux `pfplsx` (`jk + 1`) and the cover `pcov` (`jk` reads `jk - 1`). Columns are independent. `nblk = ceil(ngptot / nproma)` and `ngptot` is not a multiple of `nproma`, so the last block is short. Source: `ecmwf-ifs/dwarf-p-cloudsc` at `f7ba9f8`, driver block loop `cloudsc_driver_mod.F90` 126–135, GPU loop order `cloudsc_driver_gpu_scc_mod.F90` 138 and `cloudsc_gpu_scc_mod.F90` 492–495, body `cloudsc.F90` 4.2 (1715–1755), cover (1757–1789) and the 5.3 reset (2700–2719) written as `min(qpre * RQPRE, 1)`.

The scheduling problem is to recover the CPU order from that GPU nest: `map ibl { map jl { prologue; for jk } }` should become `map ibl { prologue; for jk { map jl } }`, so each level walks a contiguous `nproma` run.

## What the reference is

- **Baseline.** Stage 1, then `default_moves()`: `map ibl { map jl { for jk } }`. `default_moves()` does not take the fission or the interchange.
- **Reference (CPU).** `map-fission` of the column map, then `interchange-map-loop` on the body map and the level loop, then `finish_moves()`: `map ibl { prologue memsets; for jk { map jl } }`.

The pair is for the CPU. The same two moves are legal on GPU; that schedule was not timed.

## How it was created

Designed 2026-10-09 as B3 of the CloudSC-style benchmarks and timed from `$SCRATCH/thesis/cloudsc_nproma/cloudsc_nproma_kernel.py`. `cloudsc_nproma_dace.py` is that program with the size symbols lowercased to the manifest names. `cloudsc_nproma_numpy.py` is the same body. `cloudsc_nproma.py` holds `initialize()`. The species count stays a Python int (`5`) so the species loop unrolls, which is the nest the route was read on.

## What the solution is

On the CPU, after stage 1. Labels are from jobs 5112741 and 5112794 (the uppercase-symbol program) and are not stable across versions.

1. `map-fission` of `kernel2_0`, the ragged column map. It splits into six prologue maps and one body map, `kernel3_6`, which still holds the level loop. The stride does not change. `interchange-map-loop` is not offered before this move.
2. `interchange-map-loop` on (`kernel3_6`, `for4_0`), `MapLoopInterchange`. `for4_0` is `ncldtop:nlev`.
3. `finish_moves()`. It turns the prologue maps into memset library nodes and keeps `for jk { map jl }`.

Tree after `finish_moves()` (CPU; body statements left out):

```
kernel1_0  [_loop_it_0=0:NBLK]
   |- memsetLib_pcov_0, memsetLib_pfplsx_1 .. _5
   `- for2_0  _loop_it_3=NCLDTOP:NLEV
      `- kernel4_0  [_loop_it_1=0:Min(NPROMA, NGPTOT - NPROMA*_loop_it_0)]
```

## How to optimize it

Offered at each step, CPU and GPU the same (job 5112741):

| Step | Offered |
| --- | --- |
| stage 1 | `map-fission` 1, `subgraph-fission` 1 |
| after `default_moves()` | `map-fission` 1, `subgraph-fission` 1 |
| after `map-fission` | `map-fusion` 15, `map-fission` 1, `subgraph-fission` 1, `interchange-map-loop` 1 |
| after `interchange-map-loop` | `map-fusion` 15, `map-fission` 2, `subgraph-fission` 1, `interchange-loop-map` 1 |
| after `finish_moves()` | `map-fission` 1, `subgraph-fission` 1, `interchange-loop-map` 1 |

The fifteen `map-fusion` pairs glue the fission back together. `interchange-loop-map` on (`for2_0`, `kernel4_0`) puts the level loop back inside the column map; on 2026-10-09 that undo cost 107.45 ms against the reference's 45.71 ms at one thread. A second `interchange-map-loop` on the block map is illegal: `the body of kernel1_0 is not exactly the loop for2_0`, because the prologue shares that map (job 5112794). `subgraph-fission` is offered at stage 1 and `can_be_applied` fails.

## What the gains are

Not measured at the current pins. The 2026-10-09 numbers (DaCe `73f0a120e`, NestForge `NestForge-b1`, GH200, median of 21 interleaved repeats, relative error 0):

| Setting | Baseline | Reference | Ratio |
| --- | --- | --- | --- |
| preset M, 1 thread | 109.30 ms | 45.71 ms | 2.39× |
| preset M, 1 thread, undo (`interchange-loop-map`) | | 107.45 ms | |
| preset M, 72 threads | 6.07 ms | 4.89 ms | 1.24× |
| preset L (`nproma` 1024), 72 threads | 9.67 ms | 4.73 ms | 2.04× |

## CPU and GPU

Stage 1 leaves the same nest on both devices, and both moves apply on both (job 5112741). The baseline builds on CPU at these pins. The reference build is the blocker below, on either device, because the emitter runs before codegen. The gap was measured on CPU only.

## What else matters

- **Legality.** `interchange-map-loop` is absent until the column map is split. The fission does not itself put the level loop outside the column map.
- **Contiguity and stride.** `nproma` is contiguous. The reference walks it innermost at every level. The baseline walks `jk` innermost inside each column, at stride `nproma`.
- **Decoys.** `interchange-loop-map` undoes the reference. `interchange-map-loop` on the block map is refused. The `map-fusion` pairs undo the fission. `subgraph-fission` does not apply.
- **Oracle.** `initialize()` with seed 0. Baseline relative error 0 at preset S on CPU (job 5112741, NestForge `4df1830`). Reference relative error 0 at preset S on CPU once `passes_through` is present (job 5112794, NestForge-h2).
- **Pins.** DaCe `801cafd`, NestForge `interchange-benchmarks` at `7c5480f` for this folder, HPCAgent-Bench `e63f9f1`. The route probe ran on the snapshot of `4df1830`; the nest matches the program in this folder.

## Blocker at the current pins

On NestForge `4df1830` the reference (the two moves, then `finish_moves()`) fails in `emit_python` with `UnsupportedNest: nested SDFG loop_body rebinds outer names ['NGPTOT']`. The ragged range uses `ngptot`, and DaCe stores the identity symbol mapping as a string, so the identity test fails. This is the same false refusal as M1's `NCLDTOP` ([[2026-10-21-unsupportednest-rebinds-outer-names]] in the thesis notes). With `passes_through` (spcl/NestForge#19, also on `ib/h2`) the reference builds and matches the oracle. The baseline already matches on `4df1830`. This branch does not contain that fix, so a timing run of the reference has to wait for it.

## Reference solution

```yaml
status: entry
device: cpu
threads: null
preset: M
dace: 801cafd179790a844eaf460084232a0c6452571d
nestforge: interchange-benchmarks (code at 7c5480f) + benchmarking_creation/cloudsc_nproma
hpcagent_bench: e63f9f1ce04141ae766bcb2cc5164277b57fa59d
probe: Daint jobs 5112741 (route and baseline build at S), 5112794 (reference build with the passes_through fix)
host: null
baseline:
  moves: [default_moves]
  time_ms: null
  time_min_ms: null
  time_max_ms: null
  repeats: 5
  oracle_rel_error: 0.0   # preset S, CPU, NestForge 4df1830
  tree: |
    kernel1_0  [_loop_it_0=0:NBLK]
      kernel2_0  [_loop_it_1=0:Min(NPROMA, NGPTOT - NPROMA*_loop_it_0)]
        for3_0  _loop_it_3=NCLDTOP:NLEV
reference:
  moves:
    - kind: map-fission
      on: kernel2_0, the ragged column map
    - kind: interchange-map-loop
      on: kernel3_6 (column body map), for4_0 (level loop)
    - finish_moves: true
  time_ms: null
  time_min_ms: null
  time_max_ms: null
  repeats: 5
  oracle_rel_error: 0.0   # preset S, CPU, with passes_through; fails on this branch without it
  tree: |
    kernel1_0  [_loop_it_0=0:NBLK]
      memsetLib rows for the prologue
      for2_0  _loop_it_3=NCLDTOP:NLEV
        kernel4_0  [_loop_it_1=0:Min(NPROMA, NGPTOT - NPROMA*_loop_it_0)]
```
