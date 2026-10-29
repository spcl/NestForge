# AGENTS.md

Guide for LLM agents that drive NestForge and for coding agents that change it. Stage details live in
[docs/stages.md](docs/stages.md); agents reach NestForge through the bench plugin in [plugin/nestforge](plugin/nestforge).

## Driving NestForge

The bench plugin's tools are listed in the [README](README.md#agents-the-hpcagent-bench-plugin); the
[skill page](plugin/nestforge/skills/nestforge/SKILL.md) and [prompt texts](plugin/nestforge/prompt/) are what the agent reads.

- Never edit SDFG nodes or memlets; request moves. Each move is checked for legality before it applies.
- Name rows by the labels the tree prints, unique across the program. A row keeps its label while it lives; a move
  retires the labels it took, which then return `not-found`. Only `applied` changed the program.
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
