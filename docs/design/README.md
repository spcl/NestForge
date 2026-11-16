# Designs

[Overview](../../README.md) · [Code map](../README.md)

One page per feature that is designed and not yet built. A page states the goal, what exists, the design, its tests,
and what "done" means, so that someone who did not write it can build it. A page moves into the stage documentation
when its feature merges, and leaves this folder.

| Page | Feature | Needs from the bench | Can start |
|---|---|---|---|
| [comm-nodes.md](comm-nodes.md) | programs that communicate between MPI ranks | B1 to B5 | packages 1 to 6 now |
| [default-flavor.md](default-flavor.md) | the `nestforge_default` column | B8 for one point | now |
| [move-families.md](move-families.md) | switching a move family off, and attributing a run's speedup to its moves | B9 | now |
| [run-extractions.md](run-extractions.md) | refusal rates and the token split of finished runs | B7 | now |
| [bench-requests.md](bench-requests.md) | the ten requests to HPCAgent-Bench | | |

## Order

1. `default-flavor`: small, and every comparison needs the column.
2. `run-extractions` and the plugin record: every later run should carry the record, so it lands before the first
   study.
3. `move-families`: the switch first, the replay script after the first runs exist.
4. `comm-nodes`, packages 1 to 6, in parallel with the three above. Packages 7 to 9 follow the bench's B1 to B4.

## Rules for a design page

- The first lines give the status and the commits the page was written against.
- A claim about today's code names the file and the function.
- A decision that belongs to the owner is listed at the end with a recommendation, and is never taken in the text.
- A need from another project is a numbered request in `bench-requests.md`, never a patch on this side.
