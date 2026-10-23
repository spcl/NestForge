# column_physics_cpu (slot H3)

Explanation and solution record (`benchmarking_creation/AGENTS.md`) of a synthetic kernel. Written 2026-10-21 from Daint jobs 5111100 to 5111248 (this branch's pins) and 5110958 to 5111055 (the 2026-10-16 pins). Set: `benchmark-set-15` (thesis notes). Entry note: `2026-10-21-h3-column-physics` (thesis notes).

**Status:** `entry` (route, decoys and probe at this branch's pins; not timed).

## What this benchmark is

A column model written the way a GPU port writes it: the column loop `jl` is outermost. Per column, a cloud-cover recurrence runs down the levels (`covc` carries a running maximum, `pcov` records it), and a tracer recurrence runs down the levels for `NTRAC` tracers (`tr[jk] = min(tr[jk-1] * fall + src[jk], cap)`). A diagnostic pass then combines cover and the first tracer, level by level, and is itself carried over levels (`out[jk]` reads `out[jk-1]`). `tr` is `(NLEV, NTRAC, KLON)`, the others `(NLEV, KLON)`; `KLON` is contiguous. Levels are sequential in all three parts; columns and tracers are parallel. The scheduling problem: bring every level loop outside its column map and make the contiguous column index innermost, which takes six moves of five kinds, most of which gain nothing until a later one applies. Synthetic, modeled on the CloudSC column structure (thesis slot H3, variant v1 of 2026-10-21).

## Why it is a good test

- **What the default does instead:** `default_moves()` fuses the cover and tracer maps back into one column map with both level loops inside (the GPU order), and fuses the diagnostic's four maps into one. Every column then walks the levels at stride `KLON`.
- **What makes it hard:** six moves in an order with dependences. The prologue fission unlocks the cover interchange; the second `map-fission` unlocks `interchange-map-map`; both interchanges unlock `loop-fusion`. Four of the six are expected to gain nothing alone (not timed): they only make a later move legal. An agent that judges each move by its immediate effect stops early or never starts.
- **What it covers:** `subgraph-fission`, `interchange-map-loop` (x2), `map-fission`, `loop-fusion`, `interchange-map-map`, plus three `map-fusion`. It is the set's only answer with `loop-fusion` and `interchange-map-map`, and it uses no `interchange-loop-map`.
- **What a wrong decision looks like:** re-fusing the cover and tracer maps (`map-fusion kernel1_0 kernel1_1`, offered at stage 1, which is what the default does), the tracer `interchange-loop-map` (the GPU order), any of the dozens of diagnostic map-fusion orders taken before the structural moves, and the undo moves offered after the route.

## What the reference is

- **Baseline:** stage 1, then `default_moves()`: `map jl { covc = 0; for jk { cover }; for jk { map jm { tracer } } }; for jk { map jl { diagnostic } }`.
- **Reference (CPU, threads to be measured):** the nine moves below, then `finish_moves()`: `map jl { covc = 0 }; for jk { map jl { cover }; map [jm, jl] { tracer } }; for jk { map jl { diagnostic } }`.
- **GPU:** stage 1 already gives the GPU order (column maps outermost, level loops inside); no move is expected to help. Not probed beyond stage 1 and the default.

## How it was created

From the 2026-10-16 draft (`column_physics_cpu` in `benchmark-set-15`). As drafted (v0: the diagnostic pass a plain 2-D map), a walk of every sequence up to 6 moves at the 2026-10-16 pins never offered `loop-fusion` or the tracer interchange (job 5110958). Making the diagnostic carried over levels (v1, this oracle) keeps it a level loop through stage 1 and makes both appear.

At the 2026-10-16 pins (DaCe `73f0a120e`, HPCAgent-Bench `26a4f0cfc`) the route was seven moves: a first `map-fission` splitting cover from tracer, then the six below (jobs 5111051, 5111055). At this branch's pins (DaCe `801cafd`, HPCAgent-Bench `e63f9f1`), stage 1 makes that split itself and leaves the diagnostic as four one-statement maps, so the route starts one move later and the old 7-move replay is refused (job 5111111). The six moves were found by a walk at these pins (job 5111139, 298 programs, stopped by its 15-minute budget with 1269 sequences queued, most of them diagnostic map-fusion orders) and replayed (job 5111245).

The cap is fixed at 0.6 in `initialize()`; an equal-ended `domain` is refused by the manifest schema. It makes the tracer clamp bind in 21% of the updates at preset S (job 5111111).

## What the solution is

Labels from job 5111245, at the pins in the solution block.

1. `subgraph-fission` on `kernel1_0` after `state2_0`: cut `covc = 0` off the cover map, so the map holds only its level loop.
2. `interchange-map-loop` on `kernel1_2`, `for2_1`: the tracer's level loop moves outside its column map: `for jk { map jl { map jm } }`.
3. `map-fission` on `kernel2_0`: split the tracer's column-map body so the tracer map sits directly inside it. `interchange-map-map` is offered only after this.
4. `interchange-map-loop` on `kernel1_1`, `for2_0`: the cover's level loop moves outside: `for jk { map jl }`. Not offered before step 1.
5. `loop-fusion` on `for0_0`, `for0_1`: the cover and tracer level loops become one. Offered only once both interchanges are made.
6. `interchange-map-map` on `kernel3_0`, `kernel4_0`: tracer map outside, column map inside, so `KLON` is innermost.
7. to 9. `map-fusion` on `kernel2_1`, `kernel2_2`, three times: the diagnostic's four maps become one (horizontal, then vertical twice). The default makes these fusions too; they are in the route so that the reference is not worse than the default on the diagnostic.
10. `finish_moves()`.

Tree after the last move and `finish_moves()` (rows only):

```
|- state0_1
|  `- kernel1_0  [_loop_it_8=0:KLON]  reads=[] writes=['covc']
|- for0_0  _loop_it_9=1:NLEV
|  `- state1_0
|     |- kernel2_0  [_loop_it_8=0:KLON]  reads=['covc', 'za'] writes=['covc', 'pcov']
|     `- kernel2_1  [_loop_it_2=0:NTRAC, _loop_it_10=0:KLON]  reads=['cap', 'fall', 'src', 'tr'] writes=['tr']
`- for0_1  _loop_it_3=1:NLEV
   `- state1_1
      `- kernel2_2  [_loop_it_4=0:KLON]  reads=['cap', 'out', 'pcov', 'tr'] writes=['out']
```

## How to optimize it

After stage 1 on CPU, `list_moves` offers 5 `map-fusion` (four among the diagnostic maps, one re-fusing cover and tracer), `map-fission` on the cover map, the prologue `subgraph-fission`, and the tracer `interchange-loop-map`. `default_moves()` takes the fusions and none of the structural moves. The order constraint: step 1 before step 4, step 3 before step 6, steps 2 and 4 before step 5. Steps 7 to 9 should be free to come at any point; that order was not replayed.

| Decision | What happens | Evidence |
| --- | --- | --- |
| `map-fusion kernel1_0 kernel1_1` at stage 1 | cover and tracer back in one column map, as the default; the route is then longer | job 5111245 (stage 1 offers) |
| `interchange-loop-map for2_1 kernel4_0` at stage 1 | tracer becomes `map jl { map jm { for jk } }`, the GPU order; not timed | job 5111139 |
| Diagnostic map-fusions first, in any order | legal and harmless, but there are dozens of orders; an agent that explores them first spends its budget there | job 5111139 (724 map-fusion offers in 298 programs) |
| Stop after step 4 | both nests level-outer, tracer still `map jl { map jm }`; expected slower than the full route; not timed | job 5111245 |
| Undo after the route (`interchange-map-map` back, `loop-fission`, `map-fission`) | all offered at the end; undo the gain | job 5111245 |
| `interchange-loop-map for0_1 kernel2_2` after `finish_moves()` | diagnostic becomes `map jl { for jk }`: one parallel region instead of `NLEV`; may win at many threads (as in `tsvc_2_s231`); not timed | job 5111245 |

## What the gains are

Not timed yet (W10 of the thesis notes' 2026-10-21 board). Preset M is `NLEV` 137, `KLON` 32771, `NTRAC` 16. To time: the baseline, the reference, the reference without steps 5 and 6, and the reference plus the diagnostic `interchange-loop-map`, at 1 and 72 threads.

## CPU and GPU

Stage 1 gives the same tree on both devices: column maps outermost, level loops inside, the diagnostic a level loop over four maps. That is the GPU order, so the gap is expected on CPU only. The GPU side has not been walked; only its stage 1 tree and its `default_moves()` result were recorded (job 5111139).

## What else matters

- **Legality:** at this branch's pins every route move is offered right after its prerequisite (job 5111245 lists the offers after each step). At the 2026-10-16 pins the walk of v0 never offered `loop-fusion` or the tracer interchange.
- **Contiguity and stride:** `KLON` is contiguous. The baseline walks levels inside each column (stride `KLON`); the reference walks columns innermost in all three parts.
- **Launches against stride:** the reference has `NLEV` parallel regions per level loop (two loops) where the baseline has one region over columns. On a serial CPU the stride gain is the question; at many threads the region count may matter (see the last decoy).
- **Vectorization:** not checked.
- **Live values and carries:** the cover carry is a `(KLON,)` array (`covc`) read and written once per level, so no fission has to promote a scalar.
- **Oracle:** inputs from `initialize()` (`cap` = 0.6). The committed DaCe program matches the NumPy oracle at preset S on CPU, max relative error 6.4e-16 (`tr`), 4.3e-16 (`out`), 0 (`pcov`, `covc`) (job 5111248). The shared check script of the manifest migration errors on the scalar `cap` (it passes a 0-d array); job 5111248 passes it as a float.
- **Tool:** `describe()` shows the tracer state empty between steps 2 and 6 even after `expand()`, although `list_moves()` names its maps.
- **Backtest:** not run.
- **Pins:** DaCe `801cafd`, NestForge `bench/template` `2cc3828` plus this kernel, HPCAgent-Bench `e63f9f1`.

## Reference solution

```yaml
status: entry
device: cpu
threads: null            # to be measured at 1 and 72
preset: M
dace: 801cafd179790a844eaf460084232a0c6452571d
nestforge: 2cc3828 (bench/template) + benchmarking_creation/column_physics_cpu
hpcagent_bench: e63f9f1ce04141ae766bcb2cc5164277b57fa59d
probe: Daint jobs 5111139 (walk), 5111245 (route replay), 5111248 (oracle at S)
host: null
baseline:
  moves: [default_moves]
  time_ms: null
  time_min_ms: null
  time_max_ms: null
  repeats: 5
  oracle_rel_error: null
  tree: |
    kernel1_0  [_loop_it_8=0:KLON]  writes=['covc', 'pcov', 'tr']   (for2_0 cover, for2_1 { kernel4_0 [NTRAC] } inside)
    for0_0  _loop_it_3=1:NLEV
      kernel2_0  [_loop_it_5=0:KLON]  writes=['out']
reference:
  moves:
    - kind: subgraph-fission
      on: kernel1_0 after state2_0 (covc = 0 off the cover map)
    - kind: interchange-map-loop
      on: kernel1_2, for2_1 (tracer level loop out)
    - kind: map-fission
      on: kernel2_0 (tracer column-map body)
    - kind: interchange-map-loop
      on: kernel1_1, for2_0 (cover level loop out)
    - kind: loop-fusion
      on: for0_0, for0_1 (cover and tracer level loops)
    - kind: interchange-map-map
      on: kernel3_0, kernel4_0 (KLON innermost in the tracer nest)
    - kind: map-fusion
      on: kernel2_1, kernel2_2 (diagnostic, three times)
    - finish_moves: true
  time_ms: null
  time_min_ms: null
  time_max_ms: null
  repeats: 5
  oracle_rel_error: null
  tree: |
    kernel1_0  [_loop_it_8=0:KLON]  writes=['covc']
    for0_0  _loop_it_9=1:NLEV
      kernel2_0  [_loop_it_8=0:KLON]  writes=['covc', 'pcov']
      kernel2_1  [_loop_it_2=0:NTRAC, _loop_it_10=0:KLON]  writes=['tr']
    for0_1  _loop_it_3=1:NLEV
      kernel2_2  [_loop_it_4=0:KLON]  writes=['out']
alternatives:
  - moves: [diagnostic map-fusions first, then steps 1 to 6, finish_moves]
    note: the fusions are independent of the structural moves; same final program expected (not replayed)
decoys:
  - moves: [map-fusion kernel1_0 kernel1_1]
    effect: back to the default's single column map
    time_ms: null
  - moves: [interchange-loop-map for2_1 kernel4_0]
    effect: tracer in the GPU order
    time_ms: null
  - moves: [<reference moves>, interchange-map-map kernel3_0 kernel4_0]
    effect: undoes step 6
    time_ms: null
  - moves: [<reference moves>, interchange-loop-map for0_1 kernel2_2]
    effect: diagnostic as map jl { for jk }; may be faster at many threads
    time_ms: null
```
