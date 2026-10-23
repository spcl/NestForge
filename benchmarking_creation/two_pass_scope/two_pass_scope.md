# two_pass_scope (slot M7)

Explanation and solution record (`benchmarking_creation/AGENTS.md`) of a synthetic kernel. Written 2026-10-22 from Daint job 5119028. Entry note: `2026-10-22-m7-two-pass-scope` (thesis notes).

**Status:** `entry` (route and oracle at preset S; not timed).

## What this benchmark is

Two passes over a grid `(N, M)`, with `M` contiguous. The producer walks columns. A scalar starts at 0; column `j` of `t` is `a * s + b`, then `s` becomes `s + t[0, j] + w[j]` and that value is stored in `carry[j]`. The consumer walks rows and writes `out = t * t - a * carry[M - 1]`. The column loop cannot be a map, because the next scalar reads the column just written. The row loop can. They are separate blocks, and no move fuses across that boundary. The scheduling problem is to put both passes in one kernel. Synthetic (thesis slot M7).

## What the reference is

CPU. The GPU reference builds; the GPU baseline does not (below).

- **Baseline.** Stage 1, then `default_moves()`, then one kernel per parallel top-level map. Two kernels: the column map, launched once per column from the host loop that updates the scalar, and the collapsed consumer map.
- **Reference.** `define_scope` on the init state, the column loop and the consumer state. One kernel. No `finish_moves()`.

## How it was created

`layout_flip_fuse` was dropped the same morning: at DaCe `801cafd` the default already fuses the two maps ([[2026-10-22-m7-layout-flip-fuse]]). A prefix `acc[j] = previous + w[j]` with the map outside the carry is lifted to a scan in the same state, and the default fuses the maps (job 5118878). The carry has to read the map output, so the map stays inside the loop (this source). Job 5119028.

## What the solution is

Labels from job 5119028, epoch 1.

`define_scope` on `state0_0`, `for0_0`, `state0_1`. `state0_0` is the scalar init. `for0_0` is `0:M` with one map `0:N` writing `t` and reading the scalar. `state0_1` is the consumer nest writing `out`. The call returns applied, kernel `extcall_0`.

Stage 1 tree:

```
state0_0
for0_0  0:M
  kernel2_0  [0:N]  writes t
state0_1
  kernel1_0  [0:N]
    kernel2_1  [0:M]  writes out
```

After the scope the tree is one state holding `extcall_0` (reads `a`, `b`, `carry`, `t`, `w`; writes `carry`, `out`, `t`).

## How to optimize it

`list_moves` after stage 1 offers one move: `interchange-map-map` on `kernel1_0`, `kernel2_1`. That swaps the consumer so `M` is outer and `N` is inner. It leaves the column loop where it is, so the program is still two kernels. That is the decoy. `default_moves()` does not fuse the passes and leaves no further move. `map-fusion` is not offered: the nests are not in one state. Naming the two maps to `define_scope` is refused, because they are in different states; the blocks have to be named.

## What the gains are

Not timed. Preset M is the size for that measurement. Preset S checked correctness only.

## CPU and GPU

CPU and GPU stage 1 trees match, and the same scope applies on both. CPU baseline, CPU reference and CPU decoy match the oracle at S (relative error 0 on `t`, `out` and `carry`). GPU reference matches (0). GPU baseline and GPU decoy do not compile: the parent passes the device pointer `s_gpu` into `extcall_0`'s by-value `double s`. The reference does not split that scalar across a kernel call, so it compiles.

## What else matters

- **Legality.** `define_scope` on the two top-level maps is refused: they are in different states. The consumer interchange is legal and is not the answer.
- **Contiguity and stride.** `M` is contiguous. The consumer nest after stage 1 has `N` outer and `M` inner. The decoy reverses that.
- **Launches against stride.** The baseline launches the column kernel once per column and keeps the scalar update on the host. The reference is one kernel. Not timed, so which one wins is not known.
- **Live values and carries.** The scalar is loop-carried through the column map. `carry` records it so the consumer reads an array. `chain_length` of `carry` is `M`.
- **Decoys.** `interchange-map-map` on the consumer. It builds on CPU and matches the oracle, and it leaves two kernels.
- **Oracle.** Inputs `a`, `b`, `w` uniform on `[-1, 1]`, seed 0. `t`, `out` and `carry` start at 0. Preset S, relative error 0 on CPU for all three schedules, and on GPU for the scope.
- **Pins.** DaCe `801cafd179790a844eaf460084232a0c6452571d`, NestForge `4a0c0434ae260140cb35a1f89ec9d10d70bb2e74`, HPCAgent-Bench `e63f9f1`.

## Reference solution

```yaml
status: entry
device: cpu
threads: null
preset: M
dace: 801cafd179790a844eaf460084232a0c6452571d
nestforge: 4a0c0434ae260140cb35a1f89ec9d10d70bb2e74
hpcagent_bench: e63f9f1
probe: Daint job 5119028 (oracle at S; no timing)
host: null
baseline:
  moves: [default_moves]
  time_ms: null
  time_min_ms: null
  time_max_ms: null
  repeats: null
  oracle_rel_error: 0.0
  tree: |
    for0_0  _loop_it_0=0:M
      kernel2_0  [0:N]  writes t
    state0_2
      kernel1_0  [0:N, 0:M]  writes out
reference:
  moves:
    - kind: define_scope
      on: state0_0, for0_0, state0_1
  time_ms: null
  time_min_ms: null
  time_max_ms: null
  repeats: null
  oracle_rel_error: 0.0
  tree: |
    state0_0
      extcall_0  reads a, b, carry, t, w  writes carry, out, t
```
