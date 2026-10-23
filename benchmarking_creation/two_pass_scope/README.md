# How to optimize two_pass_scope

Contains the solution. Do not give this file to an agent whose job is to find the schedule. Full record: [two_pass_scope.md](two_pass_scope.md).

## The problem

Two passes over `(N, M)`, with `M` contiguous. The column loop carries a scalar: column `j` of `t` is scaled by it, then the scalar is updated from `t[0, j] + w[j]` and stored in `carry[j]`. The row loop writes `out` from `t` and `carry[M - 1]`. Stage 1 leaves them as separate blocks:

```
state0_0                          # scalar starts at 0
for0_0  0:M
  kernel2_0  [0:N]  writes t
state0_1
  kernel1_0  [0:N]
    kernel2_1  [0:M]  writes out
```

The column loop cannot become a map, because the next scalar reads the column just written. `map-fusion` does not cross that block. `default_moves()` leaves two kernels: the column map, launched once per column, and the collapsed consumer.

## The schedule to reach

One kernel over the init state, the column loop, and the consumer state. The kernel written in stage 5 then holds both passes.

## The move

`define_scope` on `state0_0`, `for0_0`, `state0_1`. It applies and returns `extcall_0` (reads `a`, `b`, `carry`, `t`, `w`; writes `carry`, `out`, `t`). No `finish_moves()`.

Naming the two maps instead is refused: they are in different states. Name the blocks.

## Traps

- **`interchange-map-map` on `kernel1_0`, `kernel2_1`.** The only move `list_moves` offers after stage 1. It swaps the consumer so `M` is outer. The column loop stays put, so the program is still two kernels. On CPU it matches the oracle.
- **Doing nothing.** `default_moves()` does not fuse the passes.

## GPU

The same scope applies, and that one kernel matches the oracle at S. The two-kernel default does not compile: the parent passes the device pointer `s_gpu` into `extcall_0`'s by-value `double s`.

## Status

Route checked at DaCe `801cafd`, NestForge `4a0c043`, HPCAgent-Bench `e63f9f1` (Daint job 5119028). CPU baseline, reference, and the interchange match the oracle at preset S (relative error 0). Timed on CPU at preset M (job 5119199): the scope is 1.89× at 1 thread (1577 ms against 2972 ms) and 0.34× at 72 threads (521 ms against 175 ms). The interchange ties the default at 72 threads. The scoped kernel's scalar update is three length-1 parallel loops per column, so the barriers cost more than the baseline's host launches once the team is large.
