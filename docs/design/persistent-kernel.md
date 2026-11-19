# Persistent kernel (megakernel) move

[Designs](README.md) · [Stages](../stages.md) · [Move families](../move-families.md)

Status: built (2026-11-18) on DaCe extended `8681a8388a`, as `persistent-kernel` `[for|state|doall ...]` with
`scope` (owner's decision: a list in one move). The rows are consecutive blocks of one region of the kernel's body
(for loops and states; a doall stands for its state; a block between them may hold no doall), and they run as one
persistent kernel with a barrier where a launch or an implicit barrier was. The kernel's rendering runs
`MegakernelLoops(blocks=...)` for its device (`nestforge.stages.kernel.persistent_loops`): on the GPU behind a runtime
test that the largest doall fits the resident threads, with the per-launch blocks as the fallback (owner's decision);
on the CPU in one OpenMP team. A kernel holds one persistent run; to change it, roll back and apply a longer list. The
move is refused when the rows are not consecutive in one region, or a trial render fuses nothing. A straight run of
kernels can be slower as one persistent kernel (DaCe measured ICON velocity tendencies 2x slower), so the 3% guard of
`get_performance` decides by measurement. The tree marks the kernel row `persistent: <blocks>`. Tested on the CPU
(`tests/test_persistent_kernel.py`); the GPU path needs a GPU run.

## Goal

A time loop whose body is only kernels launches every kernel once per iteration. A persistent kernel launches once
and runs the whole loop, with a grid barrier where a launch boundary was. It saves the launches and keeps data in
registers and caches across iterations, at the price of grid barriers and of a cap: every kernel's work per iteration
must fit the threads resident on the device at once. That is a trade-off the agent decides per loop, so it is a move.

## What DaCe gives (branch `mk-1117`)

- `MegakernelLoops` (`loops.py`): fires on the outermost sequential for loop whose body is all kernels. On the GPU it
  keeps the loop twice behind a runtime test: the persistent kernel when every kernel's work per iteration fits the
  resident threads (work that grows linearly with the loop index bounded at the ends of the range), else the
  per-launch loop. A size it cannot bound leaves the loop alone. On the CPU one OpenMP team runs the whole loop.
- The megakernel passes (chains, gather kernels, grid-stride loops, grid barriers, thread ownership), the grid `Sync`
  node, cooperative launches.
- Not in canonicalization; it stays out of NestForge's defaults too (the move is the agent's choice).

## The move

| Move | Row | Family | Arguments |
|---|---|---|---|
| `persistent-kernel` | a for loop at program level whose body is only kernels | `schedule` | none |

- Phase: after placement, as a device move (`DEVICE_KINDS`), since it needs the kernels' devices and their placed
  bodies; it rewrites the program around the kernels, not a kernel's body.
- Applies `MegakernelLoops` to the one loop the row names (not as a pass over the program).
- Refused, with DaCe's reason: a body with a statement outside a kernel, kernels on two devices, a size DaCe cannot
  bound, a loop inside another loop of kernels (only the outermost).
- The tree shows the loop as `for7 [t=0:T]  persistent (gpu, size test)`.
- `rollback` takes it back like any move (the rollback equivalence test covers it once it exists).

## Tests

1. A GPU time loop of two kernels becomes one cooperative launch behind the size test, both versions present
   (structure; runs on a login node).
2. Kernels in a straight line, outside a loop, are not offered.
3. A loop with a host statement between its kernels is refused with the reason.
4. CPU `jacobi_2d`: one `omp parallel` around the loop, exact results (local).
5. GPU `jacobi_2d` at N=64 and N=4096: exact on both sides of the size test (beverin, ask first).

## Build order

1. DaCe: `mk-1117` on extended, then the bench's DaCe pin, then NestForge's.
2. The move, its tree text, its refusals, tests 1 to 4.
3. Test 5 on beverin.
