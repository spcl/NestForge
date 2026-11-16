# What a finished run must let us count

[Designs](README.md)

Status: design, not implemented. Written 2026-11-16 against NestForge `bf068ae` and HPCAgent-Bench `cc1124254`.

## Goal

Two experiments read finished runs: how often a tool call was refused and why, and how many tokens an agent spent
before and after it first wrote kernel code. Both numbers must come from data the run already stores, through one
script, with no reading of transcripts by hand.

## The plugin record

The bench archives one record per plugin with every graded submission (`plugin_api.keep_record`,
`hpcagent_bench.stats.plugin_records`). NestForge writes none today. The plugin writes one at `finalize`, schema
version 1:

```json
{
  "kernel": "jacobi_2d",
  "packet_phases": ["moves"],
  "disabled_families": [],
  "calls": [
    {"index": 0, "tool": "apply_move", "kind": "doall-fusion", "family": "fusion",
     "status": "applied", "refusal": "", "phase": "moves", "message": 12}
  ],
  "first_write_kernel": null,
  "final": {"time_us": 1830.0, "correct": true, "kernels": 3}
}
```

- `calls` holds every tool call of the episode that reached the plugin, view calls included, without arguments.
- `status` is the tool's outcome (`applied`, `illegal`, `not-found`, `failed`, `view`).
- `refusal` is a class, never the full text: the text up to the first colon after `illegal:`, mapped through a table
  in `plugin/nestforge/refusals.py` (`blocked-by-dependence`, `unknown-label`, `family-off`, `not-a-doall`, ...). A
  text without a class maps to `other`, and a test fails when a refusal of the test suite maps to `other`.
- `message` is the index of the assistant message that made the call, which joins the record to the bench's token
  usage per message.
- `first_write_kernel` is the `index` of the first `write_kernel` call, or `null`.

The journal of [move-families.md](move-families.md) is the subset of `calls` that changed the program, with
arguments. Both live in the same record under two keys.

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

The join needs the index of the assistant message that made a tool call. If the plugin route does not receive it,
that is [bench request B7](bench-requests.md). Until then, the script joins on the order of the tool calls in the
transcript, which the record's `index` preserves.

## Tests

1. A stub episode's record validates against the schema and holds every call the stub made.
2. Every refusal text the test suite produces has a class.
3. On a two-episode fixture database, the script's token split adds up to the episode's total.

## Done means

The script runs on the database of a stub study and yields one row per episode with no missing column.
