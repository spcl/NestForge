# Move families: a switch, and attribution by replay

[Overview](../README.md) · [Code map](README.md) · [Plugin record](plugin-record.md)

A family is a fixed set of move kinds. A run can switch families off for its whole length, and a finished run's
speedup is attributed to its moves and their families by replaying its journal. Code:
`tactics/stages/families.py` holds the table, and a test keeps every move kind in exactly one family.

| Family | Move kinds |
|---|---|
| `fusion` | `for-fusion`, `doall-fusion`, `snapshot-reads` |
| `fission` | `for-fission`, `doall-fission`, `subgraph-fission`, `split-doall` |
| `interchange` | `interchange-doall-doall`, `interchange-doall-for`, `interchange-for-doall` |
| `shape` | `flatten-doalls`, `unflatten-doall`, `doall-to-for`, `for-to-doall` |
| `tiling` | `tile-doall`, `tile-for`, `untile-doall`, `untile-for` |
| `hoisting` | `hoist-invariant-code`, `sink-invariant-code`, `hoist-invariant-guard`, `sink-invariant-guard` |
| `specialize` | `propagate-config`, `unroll-for`, `unroll-doall` |
| `library` | `lib-before`, `lib-after`, `lib-into`, `split-blocking`, `inline-library`, `set-library-implementation` |
| `layout` | `relayout` |
| `precision` | `set-fp-mode`, `change-precision` |
| `vectorize` | `vectorize-doall` |
| `schedule` | `set-schedule-cpu`, `set-schedule-gpu`, `stage-loads-shared`, `copy-to-tma`, `set-pipeline-depth`, `persistent-kernel` |
| `wavefront` | `skew-wavefront` |

`rollback` belongs to no family and is never switched off. A new move kind joins a family in the same change, or
the test fails. The grouping decides only how the replay sums its rows, so it can change after the runs.

## The switch

`Session(..., disabled_families=...)` and `Setup.disabled_families` take family names, and `all` for every family.
An unknown name refuses the session.

- `list_moves` and `list_move_kinds` leave out the kinds of a disabled family.
- `apply_move` of such a kind answers `illegal: <kind> belongs to the family <family>, which this run has switched
  off.`, with the refusal class `family-off`.
- `run_defaults("moves")` skews the wavefronts (family `wavefront`), then runs canonicalization's fusion stage
  (family `fusion`). A family switched off skips its part; the stages after fusion, which every program needs, always
  run.

A packet switches families off through its `env`, as the study's ablation does:

```toml
[packets.nestforge-full-no-moves]
extends = "nestforge-full"
agent = ["scopes", "kernels", "codegen"]
env = { ..., NESTFORGE_DISABLED_FAMILIES = "all" }
```

The session lives in the judge's process, and a packet's `env` reaches only the agent's tools. The agent-side tool
wrapper (`plugin/tactics/tools/nestforge.py`) therefore sends the variable as `disabled_families` with every call.
The route fixes the value at the episode's first call, refuses a later call that names other families, and writes the
families into the plugin record. The agent has no shell under the packet, and the field is not a tool argument.

## Attribution by replay

```text
python scripts/attribute_moves.py --journal journal.jsonl --kernel K --preset S --out attribution.json
python scripts/attribute_moves.py --db results.db --setup S --episode E --kernel K --preset S --out attribution.json
```

The script starts a session of the run's kernel and measures the program, then replays the journal one call at a
time and measures after each. A measurement runs the remaining phases on their defaults, as `get_performance` does.
Each row holds the call, its move kind and family, the time after it, and its ratio to the time before it. The second
table sums the logarithms of the ratios per family: the family's share of the run's speedup, in the order the agent
worked. The script reports this order-dependent number and makes no claim about a move alone.

Each prefix is measured `--repeats` times (2 by default). A ratio inside the larger spread of the two prefixes counts
as 1.

A journal from before a rename replays under today's names (`moves.RENAMED_KINDS`, `moves.journaled_arguments`):
`collapse-doalls` is `flatten-doalls`, `expand-doall` `unflatten-doall`, `expand-library` `inline-library`,
`auto-vectorize-scope` `vectorize-doall` (its `tile_sizes` become `lanes`), and the sizes of `stage-loads-shared` and
`set-schedule-gpu` become `block`.

The replay stops at the first `write_kernel`. The plugin record keeps a kernel's source as its hash, so the calls
from there on are counted together as `kernel code`, unattributed. The journal comes from the route's journal file
or from the episode's plugin record in the results database.
