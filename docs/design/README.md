# Designs

[Overview](../../README.md) · [Code map](../README.md) · [Rules for changing Tactics](../../AGENTS.md)

One page per feature that is designed and not yet built. A page states the goal, what exists, the design, its tests,
and what "done" means, so that someone who did not write it can build it. A page moves into the stage documentation
when its feature merges, and leaves this folder.

The steps marked "Built" below are implemented; nothing else in this folder is. The pages were written and reviewed on 2026-11-16 against Tactics `bf068ae`,
DaCe extended `561defb2a6`, and HPCAgent-Bench `cc1124254`.

## Pages

| Page | Feature | Needs from the bench | Can start |
|---|---|---|---|
| [comm-nodes.md](comm-nodes.md) | programs that communicate between MPI ranks, and the overlap of communication with compute | B1 to B5, B11, B12 | packages 1 to 7, 11, 12, 14, 17, 18 built; 8 to 10, 15, 16 wait for the bench |
| [schedule-moves.md](schedule-moves.md) | `for-to-doall` (with `force`), `set-schedule-cpu`, `set-schedule-gpu`, schedules in the tree | none | built; `guided` and a chunk on the GPU need dace-extended 1.0.1 |
| [bench-requests.md](bench-requests.md) | the sixteen requests to HPCAgent-Bench, B1 to B16 | | |
| [persistent-kernel.md](persistent-kernel.md) | `persistent-kernel`: a time loop of kernels as one launch (megakernel) | none | built on the CPU; GPU run pending |
| [transformation-requests.md](transformation-requests.md) | `request_transformation`: the agent asks for a move no family offers | none | request, clustering, the authored catalogue and the gate built; the authoring agent is the owner's step |
| [mlir-llvm-agent.md](mlir-llvm-agent.md) | an agent that applies MLIR then LLVM passes one move at a time, with the loops shown after each | B16 for GPU kernels | CPU part yes; GPU part needs B16 |

## Where to start

Build in this order. Each step is one pull request with its tests, and each page lists its tests and its "done".

1. **The plugin record.** Built: [plugin-record.md](../plugin-record.md).
2. **The `tactics_default` column.** Built: `tactics_bench.DefaultOptimizer`.
3. **Move families.** Built: [move-families.md](../move-families.md).
4. **`comm-nodes`, package 1.** Built: `tests/comm_programs.py` and `tests/test_comm_canonical.py`, green on DaCe
   extended `6ed7dc19e6`, which fixes the skeleton ([comm-nodes.md](comm-nodes.md), section 9).
5. **`comm-nodes`, packages 2 to 7,** in the page's order. Package 12 (`split-doall`) is independent and fits anywhere.
6. **`comm-nodes`, packages 8 to 10,** once the bench has B1 to B4.
7. **`comm-nodes`, packages 11 to 16,** the overlap.
8. **`skew-wavefront`.** Built: DaCe extended `ef08de68e9` (D3), Tactics [stages.md](../stages.md). The bench's
   DaCe pin follows with HPCAgent-Bench #196.
9. **Hoist and sink moves.** Built: `hoist-invariant-code`, `sink-invariant-code`, `hoist-invariant-guard` and
   `sink-invariant-guard` replace five kinds (`tactics/stages/sinking.py`, [stages.md](../stages.md)).

## What is verified and what is not

Every claim about today's code names its file and function, and was read in the code on 2026-11-16. Four things were
not run and are the first to check:

1. Whether DaCe's canonicalization keeps the order and the loop of every communication node (`comm-nodes.md`,
   section 9). `LoopToMap` has no check for side effects.
2. Whether a packet's `env` reaches the process that runs the agent-side tool wrapper
   ([move-families.md](../move-families.md), the switch). The wrapper's test sets the variable itself; a stub episode
   under `tactics-full-no-moves` is the check that remains. The packet `tactics` already sets `JUDGE_INPUT_MODE`
   this way.
3. Whether DaCe's timer instrumentation writes one report per MPI rank (`comm-nodes.md`, section 12.4).
4. Whether a DaCe program that takes a communicator argument puts it on the `_comm` connector of every node
   (`comm-nodes.md`, section 7).

## The study these designs serve

The experiment design lives outside this repository. In short: 120 kernels (`llr80` and `scicomp40`), one episode per kernel, the bench's protocol
with its three verdicts, six agent configurations on the CPU, one second model on `scicomp40`, three configurations
on the GPU, and an A/A study first. The MPI experiment, with GPUs over NCCL, RCCL, or MPI, is not designed yet; the
owner wants it, and [comm-nodes.md](comm-nodes.md) so far excludes GPU targets for programs that communicate.

## Decisions

Taken by the owner on 2026-11-16, each recorded in its page:

- no move rewrites a communication operation;
- the order of operations that start communication is strict;
- the overlap is done by moving rows and statements, and the default pipeline never does it;
- the application targets are the ICON dynamical core and OMEN, with independent agents;
- `tactics_reference` keeps its fallback to the default moves, and its record says which it was;
- the budget covers no ablation run per family.
- the study runs the all-or-none move ablation; `skew-wavefront` is a move and takes no skew factors; the Tactics agent and the LLVM-pass agent (B13) both start from the canonical form
  without wavefronts, rendered with its OpenMP pragmas; Polly counts as a pass; the canonical form's own gain is measured by columns without an LLM.

Open, each listed at the end of its page: which ablation the study runs, the grouping of the families, GPU targets
for communicating programs, and how ICON and OMEN become bench tasks.

## Rules for a design page

- The first lines give the status and the commits the page was written against.
- A claim about today's code names the file and the function.
- A decision that belongs to the owner is listed at the end with a recommendation, and is never taken in the text.
- A need from another project is a numbered request in `bench-requests.md`, never a patch on this side.
- Work in this repository follows [AGENTS.md](../../AGENTS.md): tests run serially with the MPI environment
  variables, `pre-commit run --all-files` before every push, and `uv.lock` is never committed.
