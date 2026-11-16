# What a finished run must let us count

[Designs](README.md)

Status: design, not implemented. Written 2026-11-16 against NestForge `bf068ae` and HPCAgent-Bench `cc1124254`.

## Goal

Two experiments read finished runs: how often a tool call was refused and why, and how many tokens an agent spent
before and after it first wrote kernel code. Both numbers must come from data the run already stores, through one
script, with no reading of transcripts by hand.

## The plugin record

The bench archives every file `plugin_records/<plugin>.json` of an episode's folder with each graded submission
(`hpcagent_bench.harness.plugin_records.archive`), and a plugin writes the file with `plugin_api.keep_record`.
NestForge writes none today.

The route writes the record at every `finalize`, since a `/submit` grade always follows one:
`keep_record(agent_root(episode_id), "nestforge", 1, data)`. A tuner flavor writes the same record through
`FlavorOptimizer.keep_record`. A later write replaces the earlier one, so the record is always the whole episode so
far. Schema version 1:

```json
{
  "kernel": "jacobi_2d",
  "packet_phases": ["moves"],
  "disabled_families": [],
  "reference": null,
  "calls": [
    {"index": 0, "tool": "apply_move", "kind": "doall-fusion", "family": "fusion",
     "status": "applied", "refusal": "", "phase": "moves", "message": null}
  ],
  "journal": [["apply_move", {"kind": "doall-fusion", "labels": ["doall2", "doall4"]}]],
  "first_write_kernel": null,
  "final": {"time_us": 1830.0, "correct": true, "kernels": 3}
}
```

- `calls` holds every tool call of the episode that reached the plugin, view calls included, without arguments.
- `status` is the call's outcome: `applied`, `illegal`, or `not-found` for a labeled call (`MoveStatus` in
  `nestforge/records.py`), `ok` for a call that changes nothing, and `failed` for an error.
- `refusal` is a class, never the full text. `MoveResult` (`nestforge/session.py`) gains a field
  `refusal: Refusal | None`, an enum set where the refusal is made: `UNKNOWN_LABEL`, `WRONG_ROW_KIND`, `FAMILY_OFF`,
  `DEPENDENCE` (DaCe's legality check said no), `EFFECTS` (a crossed block touches what the row reads or writes),
  `COMMUNICATION` (a rule or an invariant of [comm-nodes.md](comm-nodes.md)), `ARGUMENT` (a bad tile size, split
  point, or operation), and `OTHER`. A test fails when a refusal of the test suite carries `OTHER`.
- `journal` is the list the plugin already keeps (`routes/nestforge.py:record`): the calls that changed the program,
  in order, with their arguments. A `write_kernel` entry holds the SHA-256 of its source in place of the source.
- `message` is the index of the assistant message that made the call, `null` until the bench sends it
  ([bench request B7](bench-requests.md)).
- `first_write_kernel` is the `index` of the first `write_kernel` call, or `null`.
- `reference` is `true` or `false` for the reference flavor ([default-flavor.md](default-flavor.md)) and `null`
  otherwise.

The bench caps a record at 256 KB. When the record is larger, the plugin drops the `ok` entries of `calls` first and
sets `truncated: true`. The journal is never shortened.

## The script

`scripts/run_statistics.py <results.db> --experiment <prefix> --out stats.parquet` reads the plugin records and the
bench's usage table through `hpcagent_bench.stats`, and writes one row per episode.

| Column | From |
|---|---|
| `calls`, `applied`, `refused`, `refused_by_class` | the record |
| `refusal_rate` | `refused / (applied + refused)` |
| `tokens_before_write`, `tokens_after_write` | usage per message, split at `first_write_kernel`'s message |
| `build_failures`, `wrong_results` | the bench's verdicts |

Figures are drawn through `hpcagent_bench.stats` from this file. The script draws none.

## What the bench must provide

The token split needs to know which assistant message made a tool call ([bench request B7](bench-requests.md)).
Until the bench sends it, the script joins on order: the k-th tool call to a NestForge tool in the transcript is the
entry with `index` k. This join is exact as long as every tool call reaches the route, which the tool wrapper
guarantees.

## Tests

1. A stub episode's record validates against the schema and holds every call the stub made.
2. Every refusal text the test suite produces has a class.
3. On a two-episode fixture database, the script's token split adds up to the episode's total.

## Done means

The script runs on the database of a stub study and yields one row per episode with no missing column.
