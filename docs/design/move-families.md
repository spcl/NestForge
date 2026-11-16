# Move families: a switch, and attribution by replay

[Designs](README.md)

Status: design, not implemented. Written 2026-11-16 against NestForge `bf068ae`.

## Goal

Two questions of the ablation experiment need support in the code.

1. What does an agent reach without one family of moves? This needs a switch that removes a family for a whole run.
2. Which moves of a finished run produced its speedup? This needs a replay of the run's journal, one prefix at a time.

## Families

A family is a fixed set of move kinds. The table is the single source: `nestforge/stages/families.py` holds it as
`FAMILIES: dict[str, frozenset[str]]`, and a test checks that every move kind of `moves.py`, `lib_motion.py`,
`layout.py`, and `vectorize.py` is in exactly one family.

| Family | Move kinds |
|---|---|
| `fusion` | `for-fusion`, `doall-fusion`, `collapse-doalls` |
| `fission` | `for-fission`, `doall-fission`, `subgraph-fission` |
| `interchange` | `interchange-doall-doall`, `interchange-doall-for`, `interchange-for-doall`, `interchange-doall-if`, `interchange-for-if`, `interchange-if-for` |
| `tiling` | `tile-doall`, `tile-for`, `untile-doall`, `untile-for`, `unroll-doall`, `unroll-for`, `expand-doall` |
| `hoisting` | `hoist-invariant-doall`, `hoist-invariant-for`, `snapshot-reads`, `stage-loads-shared`, `propagate-config` |
| `library` | `lib-before`, `lib-after`, `lib-into`, `expand-library`, `set-library-implementation` |
| `layout` | `relayout` |
| `vectorize` | `auto-vectorize-scope` |
| `schedule` | `doall-to-for` |

`rollback` belongs to no family and is never switched off. The implementer confirms the rows against the code before
the test freezes them. The owner decides the grouping (below).

## The switch

1. `Setup` gains `disabled_families: frozenset[str] = frozenset()`. An unknown name is refused when the session
   starts, with the list of known families.
2. `Session.list_moves` and `list_move_kinds` leave out the kinds of a disabled family. `apply_move` on such a kind
   answers `illegal: <kind> belongs to the family <family>, which this run has switched off.`
3. `run_defaults` skips the moves of a disabled family, so the default pipeline and the reference replay obey the
   same switch. A reference solution that needs a disabled move stops at that move and reports it.
4. An ablation is a packet, so it is a name in the launch line and a column in the results. One packet per ablation
   extends `nestforge` and names the families it switches off:

   ```toml
   [packets.nestforge-no-fusion]
   extends = "nestforge"
   name = "NestForge without fusion"
   env = { NESTFORGE_DISABLED_FAMILIES = "fusion" }
   ```

   The session lives in the judge's process, and a packet's `env` is set for the agent. The implementer confirms how
   the route learns the episode's packet. If it cannot, the route request must carry the packet's name
   ([bench request B9](bench-requests.md)), and the plugin reads the families from its own packet table.
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
   previous prefix.

The ratio of prefix k to prefix k-1 is the move's contribution in the order the agent applied it. The script reports
this order-dependent number and nothing else. It makes no claim about a move alone.

Cost: n builds for a journal of n calls. The script takes `--stride s` to measure every s-th prefix, and it reuses
NestForge's build cache, which keys on the program.

The journal lives in the episode's folder. For an analysis after the run, the plugin stores the journal in the
bench's plugin record at finalize (`plugin_api.keep_record`, capped at 256 KB), so the results database holds it and
the script can read it from there (`hpcagent_bench.stats.plugin_records.latest`). A journal over the cap is stored
without the arguments of its view calls. Program-changing calls are never dropped.

## Tests

1. Every move kind is in exactly one family.
2. With `fusion` off, `list_moves` offers no fusion kind, `apply_move` refuses one with the family's name, and
   `run_defaults` applies none.
3. An unknown family name refuses the session.
4. On a three-move journal, the script's last prefix time equals the time of the full replay, and each row names the
   move's family.
5. The plugin record of a stub episode holds the journal.

## Done means

One ablation packet per family passes the plugin check, and the script reproduces the final time of a stub episode
from its plugin record.

## Decisions for the owner

1. The grouping of the table, above all whether `tiling` and `unroll` are one family and where `doall-to-for` goes.
2. Which families the paper ablates. Each one costs a full set of agent runs.
