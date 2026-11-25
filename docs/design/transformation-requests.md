# Transformation requests

[Designs](README.md) · [Stages](../stages.md) · [Move families](../move-families.md)

Status: steps 1 and 2 built (the tool, its checks, the record; clustering), and the gate (2026-11-23,
`scripts/gate_authored_move.py`, `tactics/stages/gate.py`, `tests/test_gate.py`): a candidate passes on its cluster's
examples, is a fixed point of the cleanup after every application, matches the bench's reference at every site of the
corpus kernels, and passes the unit suite with the move loaded; the report then waits for the owner's review. Writing
the candidate (the authoring agent) is not built.

Clustering (`scripts/cluster_requests.py <run folders> --out clusters.json`, `tactics.stages.requests.cluster_requests`):
two requests are one candidate when one of their example pairs is the same after canonicalization, up to the names of
arrays and symbols (and the identifiers code generation derives from them, such as `a_idx`); candidates come out most
asked first, by kernels, then episodes. Floating-point constants are ignored (x / 2 and x / 8 are one pattern); integer
literals count, since they are offsets and bounds. Requests whose examples differ in shape stay apart, and the author
merges those by reading. An example that does not build is listed with
its cluster and kept out of every key.

An agent that finds no fitting move may name one with `request_transformation` (packet `tactics-requests`; E1 runs
without it, so requests never count in E1's cost). A request names the rows, describes the transformation, and gives at
least five pairs of small programs, before and after: Python modules with one `@dace.program` each, or C
(`language: c`). It never changes the program.

Checks, all without an LLM (`tactics/stages/requests.py`), each refusing with its reason:

| Check | Refusal class |
|---|---|
| every row is in the current tree | `unknown-label` |
| at least `MIN_EXAMPLES` (5) pairs | `argument` |
| no example holds `LEAK_LINES` (3) consecutive lines of the task's NumPy reference | `argument` |
| each Python `before` and `after` runs and defines one `@dace.program` | `argument` |
| no pair is the same program after canonicalization (it would ask for nothing) | `argument` |
| each pair leaves the same values in every argument on random inputs (symbols at 7) | `argument` |
| a C pair: each side compiles (`cc -std=c11 -fsyntax-only`) and the two differ beyond whitespace; the author checks that they compute the same | `argument` |

A recorded request is kept whole, examples included, in `requests-<episode>@<kernel>.jsonl` beside the journal; the
plugin record counts the call as `recorded` or `refused`. The journal never repeats a request.

Next steps (owner's design, `notes/design-transformation-requests.md` in the workspace): cluster requests by
canonical pattern, author one move per cluster in the family `authored`, gate it on its examples, the full suite and
a differential test over the corpus, and review it before it enters the catalogue.
