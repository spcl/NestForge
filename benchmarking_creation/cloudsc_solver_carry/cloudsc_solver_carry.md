# cloudsc_solver_carry (slot H1)

<!-- The format and the rules are in benchmarking_creation/AGENTS.md. This file is the record of one benchmark, and it holds the solution. Do not give it to an agent whose job is to find the schedule. -->

Explanation and solution record of a kernel derived from CloudSC. Written 2026-10-21 from a local probe (DaCe `801cafd`, this branch `7c5480f`), not a Daint job. Set: `benchmark-set-15` (thesis notes). Entry note: `2026-10-21-h1-solver-carry`.

**Status:** `entry`

## What this benchmark is

The section 5.2 cloud solver and the vertical precipitation-flux carry, cut from dwarf-p-cloudsc `cloudsc.F90` at `f7ba9f8` (sedimentation 1715–1752, matrix and LU 2596–2675, flux write 2700–2708, tendencies 2726–2767) and the same index order as HPCAgent-Bench `e63f9f1` `cloudsc_numpy.py`. Species order is 0 liquid, 1 ice, 2 rain, 3 snow, 4 vapour. Arrays are `(nclv, nlev, klon)` with `klon` contiguous; `flux` has `nlev + 1` levels. One column loop holds a species prologue, a level loop that writes `flux[s, jk + 1]` from the sedimented first guess, then the matrix, the non-pivoting LU, the substitution, the clamp and the tendencies as their own level loops. `solqa` and `solqb` are inputs. `zdtgdp` and `zrdtgdp` are inputs. The scheduling problem is to split that column body so the 5×5 does not stay live across every statement, in an order `default_moves()` does not take.

## Why it is a good test

- **What the default does instead:** `default_moves()` leaves three column maps. The third holds a single level map (`kernel2_0`, range `ncldtop:nlev`). The solver statements stay in that one map.
- **What makes it hard:** seven moves, and `interchange-loop-map` is absent from the stage-1 menu. It is offered after `loop-fission` and the fissions below. Stage 1 also offers 16 `subgraph-fission` cuts and, after the route, hundreds of `map-fusion` pairs.
- **What it covers:** `loop-fission`, `subgraph-fission`, `map-fission` (twice), `interchange-map-map` (twice), `interchange-loop-map`.
- **What a wrong decision looks like:** the two stage-1 `map-fusion`s, a repeated `subgraph-fission` on the flux map (it stays offered and, on the smaller skeleton, fills the graph with empty column maps), and the undo interchanges offered after the route.

## What the reference is

- **Baseline:** stage 1, then `default_moves()`: one state whose content is four memsets and three column maps, the third containing one level map. CPU and GPU summaries match; the SDFG hashes do not.
- **Reference (CPU, threads not measured):** the seven moves below, then `finish_moves()`. The third column map then holds level maps `kernel2_4` through `kernel2_42`. Hash differs from the CPU default.
- **GPU:** the same seven labels apply. `finish_moves()` leaves the same summary shape and a hash different from the GPU default.

A search for a sequence of five or fewer moves that reaches this nest was not run. Whether the nest is faster is the timing session.

## How it was created

Arithmetic from `cloudsc.F90` at the lines above and from `cloudsc_numpy.py` at the pin. The flux written to the next level uses the sedimented first guess, so the carry closes in the flux loop; CloudSC writes that flux from the solved mixing ratios after the LU. Process rates, the precipitation-cover branches and the phase `if` are not in the kernel: `latent[s]` is data (`RALVDCP`, `RALSDCP`, or 0). The DaCe program is a hand override (no autogen marker) with the same loops. `scripts/emit_local_dace.py` was not run; hpcagent_bench is not installed in the probe environment.

Probes that did not become the route, all at these pins, are in `2026-10-21-h1-solver-carry`:

- One level loop and six straight-line regions: `subgraph-fission` does not chain, and `interchange-loop-map` leaves a single loop in the map so the cuts disappear.
- Serial species recurrences inside the column map: `map-fission` plus `interchange-map-loop` is six applies, and `finish_moves()` does not return (timeout, then a sympy recursion).
- Six separate level loops: stage 1 already puts `klon` innermost. Five `interchange-map-map` plus the carry inversion apply and finish, and they put the level axis innermost.
- The same loops written column-outer: stage 1 rewrites them to the level-outer, column-inner nest.

The program that is committed puts every region in one column loop, which is the nest stage 1 does not fully split.

## What the solution is

Labels from the local replay on CPU and on GPU (same labels on both). Each step changed the SDFG hash.

1. `loop-fission` on `for2_0`.
2. `subgraph-fission` on `kernel1_0`, `state2_0`.
3. `map-fission` on `kernel1_2`.
4. `map-fission` on `kernel2_4`. Not the first `map-fission` in the stage-1 menu (`kernel1_0`); that label is the one offered after steps 1 and 2.
5. `interchange-map-map` on `kernel2_0`, `kernel3_1`.
6. `interchange-map-map` on `kernel2_1`, `kernel3_2`.
7. `interchange-loop-map` on `for2_0`, `kernel4_0`. Absent from the stage-1 menu. Offered once steps 1–4 have been applied in this order; steps 5 and 6 were also applied before it in the replay.
8. `finish_moves()`.

Tree after `finish_moves()` (CPU; GPU summary matches), tasklet text omitted:

```
state0_1
  memset fallsrce, fallsink
  kernel1_0  [0:klon]
  kernel1_1  [0:klon]
  memset lhs, qn
  kernel1_2  [0:klon]
    kernel2_4 .. kernel2_42  [ncldtop:nlev]
```

## How to optimize it

After stage 1 on either device, `list_moves` offers `loop-fission` 1, `map-fusion` 2 (`kernel4_0` with `kernel4_1` and with `kernel4_2`), `map-fission` 7, `subgraph-fission` 16. No interchange. `default_moves()` takes none of the seven steps above: afterwards the offers are `map-fission` and `subgraph-fission` only (8 and 16 on CPU, 8 and 15 on GPU).

| Decision | What happens | Evidence |
| --- | --- | --- |
| `map-fusion` `kernel4_0` `kernel4_1` or `kernel4_2` at stage 1 | joins two of the maps the route splits | stage-1 menu, local replay |
| `subgraph-fission` `kernel1_0` `state2_0` repeated | on the two-species skeleton, each apply stays legal and adds an empty column map | local skeleton probe |
| Stopping after the two stage-1 `map-fission`s `kernel1_0` and `kernel1_1` | `finish_moves()` hash differs from both the default and the seven-move finish | local replay, one process |
| Undo interchanges after the route | offered (`interchange-loop-map` 1 and `interchange-map-map` 4 after an eight-move extension that also applied one `interchange-map-loop`) | local replay |

## What the gains are

Not timed yet (W10). Preset M is `nlev` 137, `klon` 32771, `ncldtop` 15, `nclv` 5. No device build has been run, so there is no oracle error from generated code. The NumPy function at preset S is finite, the flux above `ncldtop` stays 0, and the clamp sees a negative species 5635 times (four condensate species over levels `ncldtop:nlev` and 263 columns).

## CPU and GPU

Stage 1 offers the same kinds and the same labels on both devices. The seven-move replay uses the same labels on both. `finish_moves()` leaves a different SDFG from `default_moves()` on each device (hashes compared in one process per device). The gap, if the split body is faster, is expected on both. Not timed at 1 thread or at 72.

## What else matters

- **Legality:** the route written down before this kernel (one `interchange-loop-map`, then five `subgraph-fission` cuts of the six regions) does not survive at this pin. After that inversion the map body is one loop, and a `subgraph-fission` that does apply does not leave a further `subgraph-fission` of the same kind. `interchange-map-loop` after `map-fission` of a serial inner loop does not finish.
- **Contiguity and stride:** `klon` is contiguous. Default and the reference both keep a column map outside the solver's level maps. The difference that survives `finish_moves()` is how many level maps sit inside that column map.
- **Launches against stride:** not measured. The reference has many more level maps than the default.
- **Vectorization:** not checked.
- **Live values and carries:** the 5×5 is unrolled in the column body. The flux carry is `flux[s, jk + 1]` written from the first guess inside the column loop. Fission is what separates the solver statements.
- **Oracle:** `initialize()` in `cloudsc_solver_carry.py`. `nclv` is 5. `zdtgdp` is positive, `zrdtgdp` is its reciprocal, `solqb` is diagonally dominant, `vfall` is 0 for vapour. The device oracle check has not been run.
- **Backtest:** not run.
- **Pins:** DaCe `801cafd179790a844eaf460084232a0c6452571d` (local checkout equals the pin), NestForge `bench/h1-solver-carry` `7c5480f`, HPCAgent-Bench `e63f9f1ce04141ae766bcb2cc5164277b57fa59d`. Probe machine: local, not Daint. `venv-e989491` was not used.

## Reference solution

```yaml
status: entry
device: cpu
threads: null
preset: M
dace: 801cafd179790a844eaf460084232a0c6452571d
nestforge: 7c5480f (bench/h1-solver-carry)
hpcagent_bench: e63f9f1ce04141ae766bcb2cc5164277b57fa59d
probe: local replay 2026-10-21, preset S; no Daint job
host: null
baseline:
  moves: [default_moves]
  time_ms: null
  time_min_ms: null
  time_max_ms: null
  repeats: 5
  oracle_rel_error: null
  tree: |
    state0_1
      memset fallsrce, fallsink, lhs, qn
      kernel1_0  [0:klon]
      kernel1_1  [0:klon]
      kernel1_2  [0:klon]
        kernel2_0  [ncldtop:nlev]
reference:
  moves:
    - kind: loop-fission
      on: for2_0
    - kind: subgraph-fission
      on: kernel1_0, state2_0
    - kind: map-fission
      on: kernel1_2
    - kind: map-fission
      on: kernel2_4
    - kind: interchange-map-map
      on: kernel2_0, kernel3_1
    - kind: interchange-map-map
      on: kernel2_1, kernel3_2
    - kind: interchange-loop-map
      on: for2_0, kernel4_0
    - finish_moves: true
  time_ms: null
  time_min_ms: null
  time_max_ms: null
  repeats: 5
  oracle_rel_error: null
  tree: |
    state0_1
      memset fallsrce, fallsink
      kernel1_0  [0:klon]
      kernel1_1  [0:klon]
      memset lhs, qn
      kernel1_2  [0:klon]
        kernel2_4 .. kernel2_42  [ncldtop:nlev]
alternatives: []
decoys:
  - moves: [map-fusion kernel4_0 kernel4_1]
    effect: no change measured; joins maps the route splits
    time_ms: null
  - moves: [subgraph-fission kernel1_0 state2_0, repeated]
    effect: on the skeleton, empty column maps; not replayed on the full kernel
    time_ms: null
  - moves: [map-fission kernel1_0, map-fission kernel1_1, finish_moves]
    effect: a different SDFG from the reference and from the default
    time_ms: null
```

```yaml
status: entry
device: gpu
threads: 1
preset: M
dace: 801cafd179790a844eaf460084232a0c6452571d
nestforge: 7c5480f (bench/h1-solver-carry)
hpcagent_bench: e63f9f1ce04141ae766bcb2cc5164277b57fa59d
probe: local replay 2026-10-21, preset S; no Daint job
host: null
baseline:
  moves: [default_moves]
  time_ms: null
  time_min_ms: null
  time_max_ms: null
  repeats: 5
  oracle_rel_error: null
  tree: |
    state0_1 kernels=7
    offers after default: map-fission 8, subgraph-fission 15
reference:
  moves:
    - kind: loop-fission
      on: for2_0
    - kind: subgraph-fission
      on: kernel1_0, state2_0
    - kind: map-fission
      on: kernel1_2
    - kind: map-fission
      on: kernel2_4
    - kind: interchange-map-map
      on: kernel2_0, kernel3_1
    - kind: interchange-map-map
      on: kernel2_1, kernel3_2
    - kind: interchange-loop-map
      on: for2_0, kernel4_0
    - finish_moves: true
  time_ms: null
  time_min_ms: null
  time_max_ms: null
  repeats: 5
  oracle_rel_error: null
  tree: |
    same summary as the CPU reference; hash differs from the GPU default
alternatives: []
decoys: []
```
