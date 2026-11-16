# Move families: a switch, and attribution by replay

[Designs](README.md)

Status: design, not implemented. Written 2026-11-16 against NestForge `bf068ae`.

## Goal

Two things a study of NestForge's moves needs from the code.

1. A switch that takes one family of moves, or all of them, away for a whole run.
2. An attribution of a finished run's speedup to its moves and their families. This needs no agent run: a replay of
   the run's journal, one prefix at a time, costs builds only.

The owner has said that the budget does not cover one set of agent runs per family (2026-11-16). The attribution by
replay is therefore the main instrument, and the switch serves at most one ablation run. Which ablation the NestForge
study runs is open (see the end).

## Families

A family is a fixed set of move kinds. The table is the single source: `nestforge/stages/families.py` holds it as
`FAMILIES: dict[str, frozenset[str]]`, and a test checks that every move kind of `moves.py`, `lib_motion.py`,
`layout.py`, and `vectorize.py` is in exactly one family.

| Family | Move kinds |
|---|---|
| `fusion` | `for-fusion`, `doall-fusion`, `collapse-doalls` |
| `fission` | `for-fission`, `doall-fission`, `subgraph-fission`, and `split-doall` once built |
| `interchange` | `interchange-doall-doall`, `interchange-doall-for`, `interchange-for-doall`, `interchange-doall-if`, `interchange-for-if`, `interchange-if-for` |
| `tiling` | `tile-doall`, `tile-for`, `untile-doall`, `untile-for`, `unroll-doall`, `unroll-for`, `expand-doall` |
| `hoisting` | `hoist-invariant-doall`, `hoist-invariant-for`, `snapshot-reads`, `stage-loads-shared`, `propagate-config` |
| `library` | `lib-before`, `lib-after`, `lib-into`, `expand-library`, `set-library-implementation` |
| `motion` | `block-before`, `block-after`, `overlap-communication`, once built ([comm-nodes.md](comm-nodes.md), section 12) |
| `layout` | `relayout` |
| `vectorize` | `auto-vectorize-scope` |
| `schedule` | `doall-to-for` |
| `wavefront` | `skew-wavefront`, once built ([skew-wavefront.md](skew-wavefront.md)) |

`rollback` belongs to no family and is never switched off. The table holds every move kind the code had on
2026-11-16, and the test keeps the two in step.

## The switch

1. `Setup` gains `disabled_families: frozenset[str] = frozenset()`. An unknown name is refused when the session
   starts, with the list of known families. The name `all` stands for every family.
2. `Session.list_moves` and `list_move_kinds` leave out the kinds of a disabled family. `apply_move` on such a kind
   answers `illegal: <kind> belongs to the family <family>, which this run has switched off.`
3. `run_defaults` skips the moves of a disabled family, so the default pipeline and the reference replay obey the
   same switch. A reference solution that needs a disabled move stops at that move and reports it.
4. An ablation is a packet, so it is a name in the launch line and a column in the results. A packet names the
   families it switches off in its `env`:

   ```toml
   [packets.nestforge-full-no-moves]
   extends = "nestforge-full"
   name = "NestForge (all phases, no moves)"
   agent = ["scopes", "kernels", "codegen"]
   env = { NESTFORGE_DISABLED_FAMILIES = "all" }
   ```

   The session lives in the judge's process, and a packet's `env` is set where the agent's tools run. The agent-side
   tool wrapper (`plugin/nestforge/tools/nestforge.py:poster`) therefore reads the variable and sends it as the field
   `disabled_families` with every call. The route fixes the value at the episode's first call and refuses a later
   call with another value. The agent cannot change it: the packet gives it no shell, and the field is not a tool
   argument. No bench change is needed.
5. The prompt's move list is built from `list_move_kinds`, so the agent never reads about a move it cannot call.

## Attribution by replay

The plugin already writes every program-changing call of an episode to a journal (`routes/nestforge.py:record`,
`journal-<episode>@<kernel>.jsonl`). A new script, `scripts/attribute_moves.py`, reads one journal.

```
attribute_moves.py <journal> --kernel K --preset P --datatype D [--gpu] --out attribution.json
```

1. It starts a session with the run's setup and measures the program (`get_performance`), the baseline.
2. For every prefix of the journal (calls 1..k, k = 1..n), it replays the prefix on a fresh session, runs the
   remaining phases on their defaults, as `finalize` does, and measures the program.
3. It writes one row per call: the tool, the move kind, the family, the time after the prefix, and the ratio to the
   previous prefix. A second table sums the logarithms of the ratios per family, which is the family's share of the
   run's speedup in the order the agent worked.

The ratio of prefix k to prefix k-1 is the move's contribution in the order the agent applied it. The script reports
this order-dependent number and nothing else. It makes no claim about a move alone.

The replay covers the journal up to its first `write_kernel` entry. The record holds a hash in place of a kernel's
source, so the calls after that entry are attributed together as "kernel code".

Cost: n builds for a journal of n calls. The script takes `--stride s` to measure every s-th prefix, and it reuses
NestForge's build cache, which keys on the program.

The journal lives in the episode's folder, which does not outlast the run. The plugin record of
[run-extractions.md](run-extractions.md) carries it under the key `journal`, so the results database holds it, and
the script reads it from there (`hpcagent_bench.stats.plugin_records.latest`).

Each measurement is one `get_performance` call, which repeats the program within its time budget. A ratio inside the
spread of the two measurements is reported as 1.

## Tests

1. Every move kind is in exactly one family.
2. With `fusion` off, `list_moves` offers no fusion kind, `apply_move` refuses one with the family's name, and
   `run_defaults` applies none.
3. An unknown family name refuses the session.
4. On a three-move journal, the script's last prefix time equals the time of the full replay, and each row names the
   move's family.
5. The plugin record of a stub episode holds the journal.

## Done means

A packet with a switched-off family passes the plugin check and a stub episode. The script reproduces the final time
of a stub episode from its plugin record and reports the share per family.

## Decisions

Taken by the owner on 2026-11-16: the study runs one ablation, the same agent with every move family off (the
all-or-none packet above), on the frontier model with three repetitions.

Open for the owner:

1. The grouping of the table, above all whether `tiling` and `unroll` are one family and where `doall-to-for` goes. It
   decides only how the replay's rows are summed per family, and it can change after the runs.
