# The plugin record of an episode

[Overview](../README.md) · [Code map](README.md) · [Stages](stages.md)

Every finished Tactics episode leaves one record that says what the agent called, what each call did, and why a
refused call changed nothing. Two numbers of the study come from it through one script: how often calls were refused
and why, and how many tokens an agent spent before and after it first wrote kernel code.

## Where it comes from

The judge route (`plugin/tactics/routes/tactics.py`) logs every call that reaches it, the refused and the failed
included, in `calls-<episode>@<kernel>.jsonl` beside the journal. At every `finalize` it writes the record with the
bench's `plugin_api.keep_record` into the episode's folder, as `plugin_records/tactics.json`. A graded submission
always follows a `finalize`, and the bench archives the file with each `/submit` grade, so the record of a grade holds
the episode up to that grade.

Schema version 2 (2026-11-23), the shape of version 1 below. Only the labels a journal names differ: from
version 2 a label the kernel phase handed out is never handed out again after `rewind`, while a version-1
journal names the labels `rewind` reset the namer to hand out again. `scripts/attribute_moves.py` replays a version-1 journal with
`Session.reissue_dropped_labels` (from a results DB the version comes with the record; a journal file takes
`--schema 1`). `scripts/run_statistics.py` reads both.

The shape:

```json
{
  "kernel": "jacobi_2d",
  "disabled_families": [],
  "reference": null,
  "calls": [
    {"index": 0, "tool": "apply_move", "kind": "doall-fusion", "family": null, "status": "illegal",
     "refusal": "dependence", "forced": false, "phase": "moves", "tokens": 41210}
  ],
  "journal": [["apply_move", {"kind": "doall-fusion", "labels": ["doall2", "doall4"]}]],
  "first_write_kernel": null,
  "final": {"kernels": 3}
}
```

| Field | Holds |
|---|---|
| `calls` | every call in the order it ran, without its arguments |
| `status` | the call's outcome: `applied`, `illegal` or `not-found` for a labeled call, a kernel outcome (`ok`, `unchecked`, `refused`, `build-failed`, `wrong`) for `write_kernel`, `ok` for a call that changes nothing, and `failed` for an error or arguments that do not validate |
| `refusal` | the class of a refusal, empty otherwise (next section) |
| `forced` | `true` for a move the agent forced past DaCe's legality check (`for-to-doall` with `force`) |
| `tokens` | the agent's cumulative tokens when it made the call; the bench sends them with every judge call |
| `family` | the move's [family](move-families.md), `null` for a call that is not a move |
| `disabled_families` | the families the run switched off ([move-families.md](move-families.md)) |
| `reference` | whether the `tactics_reference` column replayed a reference solution (`false`: the kernel has none, so it ran the default moves); `null` for every other column |
| `journal` | the changes that made the program, with their arguments; a written kernel's source as `sha256:<digest>` |
| `first_write_kernel` | the `index` of the first `write_kernel` call, or `null` |
| `final` | the number of kernels the finalized program has |

The bench caps a record at 256 KB. A larger record drops the calls with status `ok` and says `"truncated": true`.
The journal is never shortened. A record that cannot be written costs the record, never the agent's call; the judge
log says why.

The packet the episode ran is not in the record: the bench's results database stores it with the episode. The two
columns without an LLM (`tactics_bench`) write the same record through `FlavorOptimizer.keep_record`, with no
calls and the default stages as `run_defaults` entries of the journal.

## Refusal classes

`MoveResult.refusal` (`tactics/session.py`) holds the class of every refused labeled call. The code that refuses
sets it: a planner returns a `Refused` (`tactics/records.py`), a `str` that carries its `Refusal`, so every caller
that reads the text reads it as before.

| Class | The call was refused because |
|---|---|
| `unknown-label` | no row holds the label, or a change retired it |
| `wrong-row-kind` | the rows are not of the kinds, or not in the places, the move takes |
| `family-off` | the packet switched the move's family off |
| `dependence` | DaCe's legality check, or its dependence analysis, said no |
| `effects` | a crossed block touches what the row reads or writes |
| `communication` | a rule of communicating programs |
| `argument` | a tile size, cap, implementation, value, layout, or scope is wrong |
| `other` | nothing classed it |

A refused result of class `other` warns with `UnclassedRefusal`, and the test suite turns that warning into an
error, so every refusal a test meets is classed. `tests/test_refusal_classes.py` also plans every candidate of every
move kind on four corpus programs and requires a class for each refusal.

## The statistics

`scripts/run_statistics.py <results.db> --setup <prefix> --out stats.csv` reads the records through
`hpcagent_bench.stats.plugin_records` and writes one row per episode and kernel, from its newest record.

| Column | From |
|---|---|
| `calls`, `applied`, `refused`, `refused_<class>` | the calls' statuses and refusal classes |
| `refusal_rate` | `refused / (applied + refused)` |
| `build_failures`, `wrong_results` | `write_kernel` calls whose kernel did not build or was wrong |
| `forced`, `wrong_after_forced` | the forced moves, and the wrong kernels after the first of them |
| `tokens_before_write`, `tokens_after_write` | the tokens at the first `write_kernel`, and those spent after it up to the last call |
| `fresh_input_tokens`, `cached_input_tokens`, `output_tokens` | the bench's token columns of the episode |

The script draws no figure. Figures go through `hpcagent_bench.stats` from its file.
