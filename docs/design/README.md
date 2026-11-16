# Designs

[Overview](../../README.md) · [Code map](../README.md) · [Rules for changing NestForge](../../AGENTS.md)

One page per feature that is designed and not yet built. A page states the goal, what exists, the design, its tests,
and what "done" means, so that someone who did not write it can build it. A page moves into the stage documentation
when its feature merges, and leaves this folder.

Nothing in this folder is implemented. The pages were written and reviewed on 2026-11-16 against NestForge `bf068ae`,
DaCe extended `561defb2a6`, and HPCAgent-Bench `cc1124254`.

## Pages

| Page | Feature | Needs from the bench | Can start |
|---|---|---|---|
| [default-flavor.md](default-flavor.md) | the `nestforge_default` column | nothing | now |
| [run-extractions.md](run-extractions.md) | the plugin record of an episode, refusal rates, and the token split | B7 for the exact token split | now |
| [move-families.md](move-families.md) | switching move families off, and attributing a run's speedup to its moves | nothing | now |
| [comm-nodes.md](comm-nodes.md) | programs that communicate between MPI ranks, and the overlap of communication with compute | B1 to B5, B11, B12 | packages 1 to 7 and 12 now |
| [skew-wavefront.md](skew-wavefront.md) | the `skew-wavefront` move, and a canonical form without wavefronts for both agents of the study | B13 starts from it | after DaCe's D3 |
| [bench-requests.md](bench-requests.md) | the thirteen requests to HPCAgent-Bench, B1 to B13 | | |

## Where to start

Build in this order. Each step is one pull request with its tests, and each page lists its tests and its "done".

1. **`run-extractions`, the plugin record.** Every later run should carry the record, so it lands before the first
   study. It also adds the `refusal` field that the other pages use.
2. **`default-flavor`.** Small, and every comparison needs the column. It writes the record's `reference` field.
3. **`move-families`.** The family table and the switch first. The replay script follows once a run with a record
   exists.
4. **`comm-nodes`, package 1.** The fixtures and the canonicalization test. Its result decides whether DaCe needs a
   fix before anything else in that page is built.
5. **`comm-nodes`, packages 2 to 7,** in the page's order. Package 12 (`split-doall`) is independent and fits anywhere.
6. **`comm-nodes`, packages 8 to 10,** once the bench has B1 to B4.
7. **`comm-nodes`, packages 11 to 16,** the overlap.
8. **`skew-wavefront`,** package 1 on DaCe extended first. It changes the starting point of every run, so it lands
   before the first study run, together with `default-flavor`.

## What is verified and what is not

Every claim about today's code names its file and function, and was read in the code on 2026-11-16. Four things were
not run and are the first to check:

1. Whether DaCe's canonicalization keeps the order and the loop of every communication node (`comm-nodes.md`,
   section 9). `LoopToMap` has no check for side effects.
2. Whether a packet's `env` reaches the process that runs the agent-side tool wrapper (`move-families.md`, the
   switch). The packet `nestforge` already sets `JUDGE_INPUT_MODE` this way.
3. Whether DaCe's timer instrumentation writes one report per MPI rank (`comm-nodes.md`, section 12.4).
4. Whether a DaCe program that takes a communicator argument puts it on the `_comm` connector of every node
   (`comm-nodes.md`, section 7).

## Decisions

Taken by the owner on 2026-11-16, each recorded in its page:

- no move rewrites a communication operation;
- the order of operations that start communication is strict;
- the overlap is done by moving rows and statements, and the default pipeline never does it;
- the application targets are the ICON dynamical core and OMEN, with independent agents;
- `nestforge_reference` keeps its fallback to the default moves, and its record says which it was;
- the budget covers no ablation run per family.
- `skew-wavefront` is a move; the NestForge agent and the LLVM-pass agent (B13) both start from the canonical form
  without wavefronts; Polly counts as a pass; the canonical form's own gain is measured by columns without an LLM.

Open, each listed at the end of its page: which ablation the study runs, the grouping of the families, GPU targets
for communicating programs, and how ICON and OMEN become bench tasks.

## Rules for a design page

- The first lines give the status and the commits the page was written against.
- A claim about today's code names the file and the function.
- A decision that belongs to the owner is listed at the end with a recommendation, and is never taken in the text.
- A need from another project is a numbered request in `bench-requests.md`, never a patch on this side.
- Work in this repository follows [AGENTS.md](../../AGENTS.md): tests run serially with the MPI environment
  variables, `pre-commit run --all-files` before every push, and `uv.lock` is never committed.
