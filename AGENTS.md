# AGENTS.md

Guide for LLM agents that drive NestForge and for coding agents that change it. Stage details live in
[docs/stages](docs/stages/); the built-in loop is [docs/agent.md](docs/agent.md).

## Driving NestForge

| Stage | Agent calls | Reads |
|---|---|---|
| [2 Moves](docs/stages/2-moves.md) | `list_moves`, `apply_move`, `metrics` | structure tree |
| [3 Scopes](docs/stages/3-scopes.md) | `define_scope` | structure tree |
| [4 Placement](docs/stages/4-placement.md) | `place` | kernels, devices, kernel DAG |
| [5 Kernels](docs/stages/5-kernels.md) | `set_kernel_source`, `metrics` | kernel sources, [feedback](docs/stages/7-feedback.md) |

Stages 1 and 6 are deterministic only. Every stage has a deterministic default, so a run works with any subset of
agents.

- Never edit SDFG nodes or memlets; request moves. Each move is checked for legality before it applies.
- Name rows by the labels `describe()` prints, unique across the whole program, with the epoch from its first line.
  Labels regenerate after every change: an old epoch returns `stale`. Only `applied` changed the program.
- When no move fuses two regions that should run as one, `define_scope` makes them one kernel; write the fusion in
  that kernel's stage 5 source.
- A kernel source keeps its `extern "C"` entry and argument order. It counts only after it matches the oracle:
  wrong and fast loses.

## Changing NestForge

- Setup: `sudo bash scripts/setup_apt.sh`, then `pip install -e ".[dev,agent]"` (or `uv pip install`) and
  `pre-commit install`. Unit tests: `pytest -m "not integration and not gpu and not vendor"`; integration:
  `pytest -m integration`.
- DaCe and HPCAgent-Bench are pinned to commit SHAs in `pyproject.toml`. `python scripts/bump_deps.py` moves both
  to their branch tips; bump whenever DaCe extended gains a transformation or fix NestForge needs.
- `pre-commit run --all-files` runs ruff check and ruff format (120 columns) and the header check, as CI does.
- Python is written as if statically typed: annotate everything, one name keeps one type, absolute imports, no
  leading-underscore names.
- No `getattr`/`hasattr`: declare every attribute with its type, `X | None` when it may be absent.
- Every class is slotted (`@dataclass(slots=True)` or `__slots__`); DaCe library nodes, whose properties need an
  instance `__dict__`, are the exception.
- Small, well-named helpers; cyclomatic complexity at most 15.
- Nothing hardcoded but the repository root, derived from the package. No magic numbers: every numeric constant is
  a named module-level constant with a one-line comment.
- Iterate `dict` or `OrderedSet`, never a plain `set`, where order can reach generated code.
- Reuse DaCe (canonicalize, transformations, CPF codegen, analyses) before writing anything new.
- Comments short; docstrings with `:param:` only on public functions. Docs stay brief, one page per concept.
- Every bug fix gets a test. Tests assert structure as well as values; a failing test is fixed, never weakened.
