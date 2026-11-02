# How NestForge is built and how to extend it

[Overview](../README.md). What each stage decides is in [stages.md](stages.md). This page is the map of the code: the
layers, the phases an agent walks through, and where to add a move, a tool, a phase or a packet.

## The layers

Calls go down this list. `nestforge` never imports the bench at module level; only `nestforge_bench` and the plugin's
judge route do.

| Layer | Where | Holds |
|---|---|---|
| Agent tools | `plugin/nestforge/tools/nestforge.py` | One thin `POST /nestforge/<tool>` per tool. The agent image has no DaCe. |
| Tool table | `plugin/nestforge/wire.py` | Each tool's phase, arguments, description and effect. Pure pydantic, read by both sides. |
| Judge route | `plugin/nestforge/routes/nestforge.py` | One session process per episode and kernel, one call at a time. |
| Session process | `nestforge/server.py` | A `Session` in its own process; `CALLS` maps each tool to a session call and closes tools by state. |
| Session | `nestforge/session.py` | The one API over the stages, used by agents, humans and the reference driver. Owns the SDFG. |
| Stages | `nestforge/stages/` | One module per stage: `canonicalize`, `moves`, `expand`, `scopes`, `placement`, `kernel`, `variants`, `feedback`, `performance`, `finalize`. |
| IR helpers | `nestforge/ir/` | Tree rows and labels (`names`, `introspect`), outlining (`extract`), kernel dependencies (`depends`). |
| Build | `nestforge/build/` | Toolchains, flags, the timing arena, SDFG build settings. |
| Corpus | `nestforge/corpus/` | Bench kernels as DaCe programs, and the Python oracle a kernel is checked against. |
| Bench columns | `nestforge_bench/` | `nestforge_reference` (no LLM) and `nestforge_stub` (CI stand-in for an agent). |

A session starts from `server.start()`, which canonicalizes the program. Every stage has a default, so a run needs no
agent. `Session.trial()` copies a session for work that must not stick: `performance` and `finalize` both run on one,
so reading a time or taking a score never changes the program.

## The phases

`plugin.toml` lists the phases in order. A packet's `agent` key names the phases the agent works; the rest run their
`default` when the agent calls `next_phase`. The bench owns the phase gate (a tool of a later phase is refused) and
walks the order; NestForge owns what each step does to the session.

| Half | Phase | Runs | Session state afterwards |
|---|---|---|---|
| structured | `moves` | agent, tools `apply_move`, `run_defaults`, ... | the program, reshaped |
| structured | `scopes` | agent in `nestforge-full`, else default | some kernels; moves close at the first one |
| fixed point | `placement` | always default | what no scope covers gets its default scope; kernels are fixed |
| unstructured | `kernels` | agent in `nestforge-full`, else nothing | each kernel's source, one at a time |
| unstructured | `codegen` | agent in `nestforge-full`, else nothing | each kernel's compiler configuration |
| | `round` | always default | back to the program as the moves left it |

Three rules follow, and `nestforge/server.py` enforces the first two:

- **Moves close at the first kernel.** A move takes rows of the program before its scopes. `apply_move` and
  `run_defaults` with `moves` or `finish` answer 400 once the program has a kernel.
- **Scopes are fixed from placement on.** `set_scope` and `unset_scope` answer 400 once the kernels are placed, so the
  kernel work has a stable set of kernels.
- **A new round reopens both.** `Session.rewind` restores the program kept before the first kernel, with its row labels.
  Moves stay; scopes, placement and kernel code go. `next_phase` from the last agent phase reaches it, so an agent that
  wants to keep its kernels calls `submit` instead.

The packets:

| Packet | Agent works | For |
|---|---|---|
| `nestforge` | `moves` | Structured only. The agent writes no code. |
| `nestforge-full` | `moves`, `scopes`, `kernels`, `codegen` | Structured, then unstructured: one agent takes the kernels one at a time. |
| `nestforge-subagents` | the same | As `nestforge-full`, with one `kernel-optimizer` subagent per kernel. Each has its own context. |

The subagent file lists no `tools` on purpose: the harnesses name tools differently, and the closing rules hold for
every caller of the route. A session serves one call at a time, so subagents running at once wait for each other.

## How to extend it

**A move.** In `nestforge/stages/moves.py`: add the kind to `MOVE_SHAPES` with the row types it takes, a planner to
`PLANNERS` that returns a `Rewrite` or the reason it is illegal, candidate rows to `CANDIDATES`, and the transformation
and its effect to the table `move_kinds` reads. A planner mutates nothing; only `Rewrite.commit` does. Use an existing
DaCe transformation for the rewrite. Then add the entry to the Moves section of the skill page.

**A tool.** Add a `ToolDef` to `TOOLS` in `wire.py` with its phase, arguments and `Effect`, then the session call to
`CALLS` in `server.py`. Add it to `TIMED` if its answer should carry performance. List it under its phase in
`plugin.toml`, or under a packet's `tools` for `Phase.ANY`. `tests/test_bench_plugin.py` checks that the three agree and
that the call takes the tool's arguments.

**A closing rule.** Wrap the call in `until(closed, why, call)` in `server.py`, so the session API stays free for humans
and reference solutions. Put state that outlives a call on `Session`, and add it to `trial()` if a copy needs it.

**A phase.** Add `[phases.<id>]` to `plugin.toml` with `title`, `summary` and either `text` (the agent can run it) or a
`default` call; a phase with neither is only named in the prompt. Add the id to `Phase` in `wire.py` if it has tools.
The bench writes the workflow text from the phases, so there is no prompt to edit beyond `prompt/phases/<id>.md`.

**A default stage.** Add the literal to `Stage` in `session.py` and to `RunDefaults` in `wire.py`, and the case to
`Session.run_defaults`. A default that changes the program ends in `self.bump()`, which retires labels and drops
what depends on the old program.

**A packet.** Add `[packets.<key>]` to `plugin.toml`, usually with `extends`: tables merge key by key, lists replace.
Prompt changes go in an overlay YAML listed under `overlays`; check the result with
`python -m hpcagent_bench.prompt_build --packet <key> --show`, run from a bench checkout.

**A subagent.** Add `plugin/nestforge/agents/<name>.md` with `name` and `description` frontmatter and name it in a
packet's `agents`. Its body is its whole system prompt, so it says what the tools do and where it stops.

**A kernel hint.** `nestforge/stages/feedback.py`: `hints` turns one kernel's evidence into ranked lines, failures
first. `report` keeps the top lines, which `performance` returns.

**A compiler configuration.** `nestforge/stages/variants.py`: `VARIANTS_BY_DEVICE` lists the sweep per device. The
sweep verifies every variant against the oracle, so a wrong one loses by itself.

**A reference solution.** Add `reference_solutions/<kernel>.py` with `solve(s, ctx)`; see that folder's
[README](../reference_solutions/README.md).

## Pins

The bench owns the DaCe pin. `python scripts/bump_deps.py [bench-branch]` moves the bench, DaCe and relayout pins
together. A change to the plugin's tools or phases needs the bench's `.github/user-plugins.toml` bumped afterwards, or
the bench's CI keeps testing the old commit.
