# How the static move score was built

prev: [Static move score](score.md)

This page explains where `S = lines / p + c * E` comes from: the question, the experiments that chose each term, the
tests of the frozen score, the demo on kernels it never saw, what works, what does not, and what to try next. Every
number was measured on one Daint GH200 node (CPU: Grace, OpenMP at 32 threads; GPU: one H100) with NestForge `598fe59`
and DaCe `a161a3e62`, except the demo (NestForge `4a0862c`, DaCe `0270fcec6`).

## 1. The question

Stage 2 offers moves (`map-fusion`, `loop-fission`, `map-fission`, `interchange-loop-map`, ...). Building and timing
every candidate costs about 25 s on the CPU and up to 5 minutes on the GPU per program. We look for a number `S`,
computed from the SDFG without building it, with one property:

> a move that lowers the measured runtime lowers `S` (and one that raises it raises `S`).

Only the sign has to be right, per move and per device. `S` is a ranking for a search or an agent, not a runtime
model; absolute times and compiler flags are out of scope (every variant uses the default kernel configuration, so
two variants differ only in the program).

## 2. Ground truth

| item | choice |
|---|---|
| kernels | 7 small kernels: TSVC `s231`, `s233`, `s235`, `s2233`, `s256`, and `column_scan` in two layouts (`nlev_nproma`, `nproma_nlev`); `cloudsc_tidy` stage 1 against default only; `aes_graupel` scored but not timed (oracle bug) |
| variants | per kernel: stage 1, NestForge's default stage 2, every legal single move, two `fission -> interchange` sequences; 46 variants, presets S and M |
| builds | CPU serial, CPU OpenMP (32 threads), GPU |
| time | `sum over kernels of kernel time x calls`, kernel time from NestForge's `validate_kernel` (standalone calls, oracle-checked), calls from the trip counts of the loops around the call |
| class | 5 independent measurements per cell; faster (slower) only when the min-max intervals do not overlap, with a 2% margin; otherwise a tie, which is not scored |
| error | a wrong sign: faster with `dS >= 0`, or slower with `dS <= 0` |

## 3. How the terms were chosen

### Step 0: untimed prototypes

Work `W`, depth `D`, bytes moved `Q` (bytes crossing map boundaries, an array below the cache size fetched once) and
parallel-region entries `E` were computed on every move of three large stage 1 SDFGs. `W` did not change under any of
36 moves and `D` only ever rose. `Q` and `E` had the expected sign per move kind, but nothing was timed yet.

### E0: first time next to a clock

Every term on every variant of section 2, against measured runtimes on the three builds. Findings:

1. **`E` is right wherever threads exist.** OpenMP: 26 faster variants with `E` down, 0 with `E` up. GPU: 41 and 0.
2. **`E` breaks on serial interchanges.** Five interchanges are 60% to 459% slower on one thread while `E` falls
   (`s231`: `E` 10,240 to 1, 5.5x slower serial). On one thread the interchange buys nothing and walks memory with a
   large stride.
3. **`W` is flat, `D` never falls on a faster variant.** Dropped.
4. **`Q` rises on the fastest interchanges** (the moved map is entered once per loop trip), and it was identical at
   cache sizes 1, 4 and 32 MiB. Dropped as the memory term.
5. **Stride decides on serial.** Two `s233` interchanges with equal `E` and `Q` run +60% and -82%; the two
   `column_scan` layouts run 186 and 931 ms. A stride-aware term was needed.

### Session A: scheduled depth, rejected

`D_sched` adds the depth of independent maps in one state run one after another (a CPU backend does not overlap
them). It equalled `D` in 154 of 156 cells: no information.

### Session B: the stride term `lines`

```
lines = sum over memlets next to a tasklet of  volume * min(1, |stride| * itemsize / 64)  * executions
```

The stride is taken along the tasklet's innermost iterator (the innermost map parameter, else the nearest loop),
so a unit-stride double access costs 1/8 of a cache line and a strided one a whole line. Cache rule: an array of at
most 1 MiB counts its lines once, whatever its stride. It separates the `s233` pair (3.08e8 against 3.85e7 lines)
and the `column_scan` layouts (5.78e8 against 1.39e8).

### E5: combining the terms

```
dS = d_lines / p + c * d_E
```

`p` = workers that share the traffic, `c` = price of one region entry in cache lines. `(p, c)` was fitted per build
on a grid (`p` in {1} serial, {1..32} OpenMP, {1..2048} GPU; `c` from 0 to 1e5) to minimise wrong signs.

| build | scored | `E` alone | `lines` alone | `lines / p + c * E` | leave one kernel out |
|---|---|---|---|---|---|
| CPU serial | 28 | 9 | 13 | **2** | 2 |
| CPU OpenMP | 30 | 2 | 24 | **2** | 2 |
| GPU | 48 | 5 | 32 | **0** | 0 |

Fitted on TSVC only and scored on `column_scan` and `cloudsc_tidy`: serial 2, OpenMP 1, GPU 0 wrong, the same cells
as in the full fit, and fewer than `E` alone (serial 4, GPU 5).

**Simpler stand-ins for `lines` lose a build** (E5b): the strided share alone gets 7 / 2 / 5 wrong (serial / OpenMP
/ GPU), the element count alone 7 / 2 / 0, share times elements 4 / 1 / 5. `lines` is the only one right on all
three: it is a volume whose unit cost depends on the stride, and dropping either part loses a build.

### Freezing

Many grid points tie, and only `k = c * p` decides the sign of a change, so `p` and `c` are not identified, only an
interval of `k`. Before the out-of-sample test the tie set was saved and a rule picks one point: first the smallest
`c` (pre-registered), now the largest `k` (changed after E6, see 4.2). Frozen weights:

| build | `p` | `c` | `k` |
|---|---|---|---|
| CPU serial | 1 | 1e4 | 1e4 |
| CPU OpenMP | 32 | 1e4 | 3.2e5 |
| GPU | 2048 | 1e4 | 2.05e7 |

With these `k`, `E` decides whenever a move changes it and `lines` breaks ties. The score is close to lexicographic.

## 4. Tests of the frozen score

### 4.1 E6: new move sequences, predicted before timing

74 new two-move variants (a second legal move on each timed parent), 888 sign predictions written and hashed before
any timing, then timed on the three builds.

| wrong signs | CPU serial | CPU OpenMP | GPU |
|---|---|---|---|
| E0 single moves (used for the fit) | 2/28 | 2/30 | 0/48 |
| E6 vs parent, **pre-registered** | 3/26 | 1/51 | **0/120** |
| E6 vs parent, after the fix and the rule change below | 0/26 | 0/51 | 0/120 |
| E6 vs stage 1 | 0/45 | 2/76 | 0/112 |

The honest out-of-sample number is the pre-registered row: **4 wrong of 197 (about 2%)**.

### 4.2 Two decisions taken after seeing E6

- **Bug fix in `lines`.** The three serial errors came from tasklets whose iterator sits outside their nested SDFG
  (after an inverse interchange), which `lines` skipped. Fixed; parameters unchanged; E0 unaffected.
- **Tie-break rule.** The one OpenMP error was an interchange 18x faster that the score said to skip; it came from
  the smallest-`c` rule. The largest-`k` rule fixes it and changes nothing on GPU.

Both are post hoc. Pre-registered numbers are reported first everywhere.

### 4.3 Regret: what a wrong choice costs

A wrong sign on a 3% change and on an 18x change counted the same, so the main metric became regret: among the
sibling moves of a parent, with "do not move" as an option, take the one the score ranks best; regret is how much
slower it is than the best option. E6, 50 sibling sets per build, mean / max:

| ranking by | CPU serial | CPU OpenMP | GPU |
|---|---|---|---|
| **score** | **1.7% / 9%** | **0.6% / 6%** | **0.7% / 4%** |
| `E` alone | 57% / 517% | 0.6% / 4% | 0.7% / 4% |
| `lines` alone | 61% / 592% | 606% / 3376% | 967% / 12032% |
| a random move | 91% | 512% | 716% (pre-registered rule) |

On serial the score beats both terms alone; on OpenMP and GPU it equals `E` alone. This is the bar a new score has to
beat.

### 4.4 The misses, and what they are not

The remaining errors are OpenMP fusions on `column_scan_nproma_nlev` (`m0`/`m1` + `map-fusion-5`, +12% and +13%) and
the default stage 2 of `cloudsc_tidy` (+27%) and `column_scan_nproma_nlev` (+40%). In all of them `E`, `lines` and
`Q` fall, so no weighting of these terms can get them right.

| test | result |
|---|---|
| E8, thread sweep (1, 8, 32 threads; three placements) | the slowdowns appear only at 32 threads; not placement; fused kernels scale 1.9-2.1x from 8 to 32 threads against 3.8-4.3x for the kernels they replace |
| per-kernel loads | inside a program, kernel time ranks with per-kernel `lines` (Spearman +0.81 to +0.85) and with strided streams (+0.93): the score finds the hotspot kernel |
| liveness across kernel boundaries | flat in most comparisons, fixes no miss |
| memory-capacity rule (time = footprint / bandwidth of the level it fits, bandwidths microbenchmarked) | **rejected**: serial regret 99%, OpenMP 35%; the fused kernels' 35 MiB fits the L3 at full bandwidth |
| E8b, hardware counters (8, 16, 32 threads) | fused kernels: same L1/L2 traffic, 3.6x to 22x more last-level misses, more TLB walks; and the slowdown depends on the timing loop (below) |

**The timing loop matters.** In the same process, NestForge's timing loop (one kernel called back to back) has the
fused variants 1.10x to 1.30x slower than stage 1 at 32 threads; a loop that pauses between calls has them 0.43x to
0.95x, faster (0.77x to 0.95x with threads kept spinning). The one-thread restore of in-place outputs before each call
is not the cause. So on these cells the ground truth's timing method is in question as much as the score.

## 5. In NestForge

`nestforge/stages/score.py` computes `E` and `lines` on the SDFG at concrete sizes and scores every legal move on a
copy (`rank_moves`). `Session.score_moves()` ranks; `Session.guided_moves()` applies the best-scored move while the
score falls. Choosing costs 0.03 to 7 s per small kernel (372 s for `vadv`, 51 moves), against 25 s to 5 min to build
one candidate. `examples/score_demo.py` runs four stage 2 pipelines per kernel: stage 1 only, the default, the
score-guided one, and the default followed by the score.

## 6. Kernels it never saw

**Static probe** (nothing built): on fusion-heavy kernels the greedy one-move-at-a-time search ends with more region
entries than NestForge's default fusion stage (`vadv` 625 against 448, `kmeans` 123 against 103), so the score does
not replace the default. Over 32 unseen kernels with a loop around a map, it finds an interchange on 3; 26 get no
move from any pipeline.

**Timed demo** (CPU OpenMP 32 threads, preset M, 20 of 20 programs oracle-checked; ratio to the default stage 2):

| kernel | fitted on | default | score-guided | default, then score | score's move |
|---|---|---|---|---|---|
| `tsvc_2_s1119` | no | 130.2 ms | **0.22x** | 0.22x | interchange |
| `scan_levels_k1` | no | 8.0 ms | 0.93x | **1.86x** | fusion; interchange after the default |
| `scan_levels_k32` | no | 6.8 ms | 1.02x | **2.00x** | same |
| `tsvc_2_s231` | yes | 96.5 ms | **0.20x** | 0.20x | interchange |
| `column_scan_nproma_nlev` | yes | 35.8 ms | 1.01x | 0.99x | 5 fusions (stage 1: 0.83x) |

One unseen kernel won 4.5x with a move the default never makes; two lost 2x. In the losses the score dropped only
3% to 4% (`E` 765 to 3, `lines` up 8x); in the wins it dropped about 90%.

## 7. What works and what does not

**Works**
- Sign of a move on the 7 kernels, out of sample on new move sequences: 4 of 197 wrong, pre-registered.
- Choosing among moves: regret under 2% mean on every build.
- Interchange, which the default stage 2 never does: 5x on `s231` and 4.5x on `s1119` at 32 threads.
- Cheap: seconds per kernel, no build.
- Pointing at the hotspot kernel inside a program.

**Does not work, or is not shown**
- New kernels: one demo, 1 win and 2 losses out of 3; no frozen held-out test yet.
- Fusion-heavy kernels: the greedy search is worse than the default fusion stage; the score adds moves, it does not
  replace the default.
- Small predicted gains: a 3% score drop is not a reliable move.
- The OpenMP 32-thread fusion slowdowns: no term sees them, and they depend on how kernels are timed.
- Thread count: validated at 32 threads only (and 1 for serial); the ground truth changes between 8 and 32.
- Size: small, loop-sized kernels; `aes_graupel` untimed, `cloudsc_tidy` default only.
- Identification: `p` and `c` are not identified, only `k`; two choices (stride fix, tie-break) were made after E6.
- Ground truth: per-kernel time in isolation, not the whole program; copies and launches between kernels excluded.

## 8. Possible improvements

| change | why | status |
|---|---|---|
| move only when the score drops by more than a margin | the demo's losses dropped 3-4%, its wins about 90% | post hoc, needs a frozen test |
| reuse across kernels and iterations in `lines` (reuse distance) | fused kernels miss L3 4x to 22x more while L1/L2 traffic is equal; per-tasklet `lines` cannot see lost reuse | not built |
| measured `c(p)` instead of a fitted `c` | an empty parallel region costs 0.24, 2.4, 7.2 us at 1, 8, 32 threads; the fit only bounds `c` | untested |
| per region `max(c(p), lines / bandwidth(p))` instead of the sum | small regions on many threads are bound by synchronization, not memory | untested |
| search beyond one move at a time, or start from the default | greedy search misses what the default's broad fusion reaches | `default, then score` exists |
| ground truth = whole program at the target thread count | per-kernel timing in isolation changes the sign of some fusions | needs a harness change |
| a frozen held-out test on new kernel families | the only clean generalization test | next |

## 9. Questions for the meeting

1. Is a sign-only ranking the right target, or is a calibrated runtime model needed?
2. Which kernels would make a credible held-out test (stencils, `cloudsc_monolith`, others)?
3. Is per-kernel time acceptable as ground truth, or should the whole program be timed?
4. Is `lines / p + c * E` defensible, or should it be an analytical roofline-style model with measured parameters?
5. The 32-thread fusion slowdown that depends on the timing loop: prefetcher, TLB, or thread wake-up first?
6. Is "largest `k`" acceptable given it was chosen after E6, or should the next test use an explicit lexicographic
   rule (`E` first, then `lines`)?
