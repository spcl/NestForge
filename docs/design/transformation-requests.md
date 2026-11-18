# Transformation requests

[Designs](README.md) · [Stages](../stages.md) · [Move families](../move-families.md)

Status: step 1 built (the tool, its checks, the record); clustering, authoring and the gate are not built.

An agent that finds no fitting move may name one with `request_transformation` (packet `nestforge-requests`; E1 runs
without it, so requests never count in E1's cost). A request names the rows, describes the transformation, and gives at
least five pairs of small `@dace.program` modules, before and after. It never changes the program.

Checks, all without an LLM (`nestforge/stages/requests.py`), each refusing with its reason:

| Check | Refusal class |
|---|---|
| every row is in the current tree | `unknown-label` |
| at least `MIN_EXAMPLES` (5) pairs | `argument` |
| no example holds `LEAK_LINES` (3) consecutive lines of the task's NumPy reference | `argument` |
| each `before` and `after` runs and defines one `@dace.program` | `argument` |
| no pair is the same program after canonicalization (it would ask for nothing) | `argument` |
| each pair leaves the same values in every argument on random inputs (symbols at 7) | `argument` |

A recorded request is kept whole, examples included, in `requests-<episode>@<kernel>.jsonl` beside the journal; the
plugin record counts the call as `recorded` or `refused`. The journal never repeats a request.

Next steps (owner's design, `notes/design-transformation-requests.md` in the workspace): cluster requests by
canonical pattern, author one move per cluster in the family `authored`, gate it on its examples, the full suite and
a differential test over the corpus, and review it before it enters the catalogue.
