# AGENTS.md

Guide for LLM agents that drive NestForge and for coding agents that change it. Stage details live in
[docs/stages](docs/stages/); agents reach NestForge through the bench plugin in [plugin/nestforge](plugin/nestforge).

## Driving NestForge

The bench plugin's tools are listed in the [README](README.md#agents-the-hpcagent-bench-plugin); the
[method prompt](plugin/nestforge/methods/nestforge/method.md) is what the agent reads.

- Never edit SDFG nodes or memlets; request moves. Each move is checked for legality before it applies.
- Name rows by the labels the tree prints, unique across the program, with the epoch from its first line. Labels
  regenerate after every change: an old epoch returns `stale`. Only `applied` changed the program.
- When no move fuses two regions that should run as one, `set_scope` makes them one kernel; `unset_scope` undoes it.
- Keep a change when the whole-program time drops and the program stays correct: wrong and fast loses.

## Changing NestForge

- Setup, tests and pins: see the [README](README.md#install-and-test). Bump the pins whenever DaCe extended gains a
  transformation or fix NestForge needs.
- `pre-commit run --all-files` runs ruff check and ruff format (120 columns) and the header check, as CI does.
- Python is written as if statically typed: annotate everything, one name keeps one type, absolute imports, no
  leading-underscore names.
- No `getattr`/`hasattr`: declare every attribute with its type, `X | None` when it may be absent.
- Answers are frozen pydantic records (`nestforge/records.py`), outcomes are enums compared with `is`; every other
  class is slotted. DaCe library nodes, whose properties need an instance `__dict__`, are the exception.
- Small, well-named helpers; cyclomatic complexity at most 15.
- Nothing hardcoded but the repository root, derived from the package. No magic numbers: every numeric constant is
  a named module-level constant with a one-line comment.
- Iterate `dict` or `OrderedSet`, never a plain `set`, where order can reach generated code.
- Reuse DaCe (canonicalize, transformations, CPF codegen, analyses) before writing anything new.
- Comments short; docstrings with `:param:` only on public functions. Docs stay brief, one page per concept.
- Every bug fix gets a test. Tests assert structure as well as values; a failing test is fixed, never weakened.

## Experiments that need more than NestForge gives

A research run may extend NestForge and DaCe to read more out of a program (analyses, counters, per-scope timers,
estimates of a move's effect). Do not hold back for lack of an existing hook.

- Keep it additive and off by default: a new module or an opt-in flag; no change to a default stage, label or result.
- Work on an experimental branch (NestForge) or on the fork `nicolasribeiro2028/dace` (DaCe), one commit per idea, so
  each can be dropped or upstreamed alone.
- While exploratory, that code may skip the typing, complexity and docs rules above; rewrite it to them before it goes
  upstream.
- Record the DaCe and NestForge SHAs and the branch with every result that depends on the change.
- Pushing a branch, opening a PR or an issue follows the normal contribution route and needs the owner's say-so each
  time.

## Writing a benchmark

Kernels under `benchmarking_creation/` follow [benchmarking_creation/AGENTS.md](benchmarking_creation/AGENTS.md). Read it before adding or changing one.

A kernel is not done without the HPCAgent-Bench files (`<name>_numpy.py`, `<name>.yaml`, generated `<name>_dace.py`), the explanation `<name>.md`, and a measured reference solution in that explanation. The explanation states what the benchmark is, what the reference is, how it was created, what the solution is, how to optimize it, what the gains are, and how CPU and GPU differ. Score a run by the oracle and by preset M time. Do not give `<name>.md` to an agent whose job is to find the schedule.
