# AGENTS.md

Guide for LLM agents that drive Tactics and for coding agents that change it. Stage details live in
[docs/stages.md](docs/stages.md); agents reach Tactics through the bench plugin in [plugin/tactics](plugin/tactics).

## Driving Tactics

The bench plugin's tools are listed in the [README](README.md#agents-the-hpcagent-bench-plugin); the
[skill page](plugin/tactics/skills/tactics/SKILL.md) and [prompt texts](plugin/tactics/prompt/) are what the agent reads.

- Never edit SDFG nodes or memlets; request moves. Each move is checked for legality before it applies.
- Name rows by the labels the tree prints, unique across the program. A row keeps its label while it lives; a move
  retires the labels it took, which then return `not-found`. Only `applied` changed the program; a call that fails
  leaves it as it was.
- When no move fuses two regions that should run as one, `set_scope` makes them one kernel; `unset_scope` undoes it.
- Keep a change when the whole-program time drops and the program stays correct: wrong and fast loses.

## Changing Tactics

- Setup, tests and pins: see the [README](README.md#install-and-test). Bump the pins whenever DaCe extended gains a
  transformation or fix Tactics needs.
- `pre-commit run --all-files` runs ruff check and ruff format (120 columns) and the header check, as CI does.
- Python is written as if statically typed: annotate everything, one name keeps one type, absolute imports, no
  leading-underscore names.
- `mypy && mypy tests` (strict, `[tool.mypy]` in pyproject.toml) passes with no errors, as CI checks. DaCe is
  untyped except `dace.program` (`typings/dace`).
- No `getattr`/`hasattr`: declare every attribute with its type, `X | None` when it may be absent.
- Answers are frozen pydantic records (`tactics/records.py`), outcomes are enums compared with `is`; every other
  class is slotted. DaCe library nodes, whose properties need an instance `__dict__`, are the exception.
- Small, well-named helpers; cyclomatic complexity at most 15.
- Nothing hardcoded but the repository root, derived from the package. No magic numbers: every numeric constant is
  a named module-level constant with a one-line comment.
- Iterate `dict` or `OrderedSet`, never a plain `set`, where order can reach generated code.
- Reuse DaCe (canonicalize, transformations, CPF codegen, analyses) before writing anything new.
- Comments short; docstrings with `:param:` only on public functions. Docs stay brief, one page per concept.
- Every bug fix gets a test. Tests assert structure as well as values; a failing test is fixed, never weakened.
- A bug's test carries `# Reproducer: <what broke, where> (<issue, kernel, job or commit>)` on the line above its
  `def` (or its decorators). Test pruning never deletes a tagged test.
- Tests run through pytest only: no `if __name__ == "__main__":` runners in test files.
