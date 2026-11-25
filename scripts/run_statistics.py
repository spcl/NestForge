# Copyright 2021 ETH Zurich and the Tactics authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""One row per Tactics episode of a results DB, from the plugin record its last graded submission kept
(docs/plugin-record.md) and the episode's token columns: calls, refusals by class, attempts that failed
to build or were wrong, and the tokens spent before and after the first ``write_kernel``.

    python scripts/run_statistics.py results.db --setup tactics- --out stats.csv
"""

import argparse
import csv
import pathlib
from collections import Counter
from typing import Any

from hpcagent_bench.harness import results_db
from hpcagent_bench.stats import plugin_records

#: The plugin whose records this script reads, and the record schema it understands.
PLUGIN = "tactics"
#: The record versions this script reads; version 2 changed what labels a journal names, not the shape.
SCHEMAS = frozenset({1, 2})
#: Every refusal class (``tactics.records.Refusal``), spelled as the record stores it.
REFUSALS = (
    "unknown-label",
    "wrong-row-kind",
    "family-off",
    "dependence",
    "effects",
    "communication",
    "argument",
    "other",
)
#: Call outcomes that refused a change, and kernel outcomes that wasted an attempt.
REFUSED = frozenset({"illegal", "not-found", "refused"})
BUILD_FAILED = "build-failed"
WRONG = "wrong"
#: The episodes table's token columns, reported as the bench splits them.
TOKEN_COLUMNS = ("fresh_input_tokens", "cached_input_tokens", "output_tokens")


def episode_tokens(db: pathlib.Path) -> dict[tuple[str, str], dict[str, int]]:
    """``(setup, label)`` -> the episode's token columns."""
    query = f"SELECT setup, label, {', '.join(TOKEN_COLUMNS)} FROM episodes"
    with results_db.reading(db, finished=results_db.finished_db(db)) as conn:
        return {(str(row[0]), str(row[1])): dict(zip(TOKEN_COLUMNS, map(int, row[2:]))) for row in conn.execute(query)}


def forced_moves_kept(calls: list[dict[str, Any]]) -> list[int]:
    """The indices of the forced moves no later ``rollback`` took back; each rollback takes back the last applied change."""
    kept: list[tuple[int, bool]] = []
    for call in calls:
        if call["status"] != "applied":
            continue
        if call.get("kind") == "rollback":
            if kept:
                kept.pop()
        else:
            kept.append((call["index"], bool(call.get("forced", False))))  # records before the field have none
    return [index for index, forced in kept if forced]


def row_of(data: dict[str, Any]) -> dict[str, Any]:
    """The counts of one record."""
    calls: list[dict[str, Any]] = data["calls"]
    statuses = Counter(str(call["status"]) for call in calls)
    refused = sum(statuses[status] for status in REFUSED)
    applied = statuses["applied"]
    classes = Counter(str(call["refusal"]) for call in calls if call["status"] in REFUSED)
    tokens = [call["tokens"] for call in calls if isinstance(call.get("tokens"), int)]
    first = data["first_write_kernel"]
    before = next((call["tokens"] for call in calls if call["index"] == first), None) if first is not None else None
    last = tokens[-1] if tokens else None
    forced = forced_moves_kept(calls)
    return {
        "calls": len(calls),
        "applied": applied,
        "refused": refused,
        **{f"refused_{name.replace('-', '_')}": classes[name] for name in REFUSALS},
        "refusal_rate": refused / (applied + refused) if applied + refused else 0.0,
        "build_failures": statuses[BUILD_FAILED],
        "wrong_results": statuses[WRONG],
        "forced": len(forced),
        "wrong_after_forced": sum(
            1 for call in calls if forced and call["index"] > forced[0] and call["status"] == WRONG
        ),
        "tokens_at_last_call": last,
        "tokens_before_write": before,
        "tokens_after_write": last - before if last is not None and isinstance(before, int) else None,
        "truncated": bool(data.get("truncated", False)),
    }


def rows(db: pathlib.Path, setup_prefix: str) -> list[dict[str, Any]]:
    """One row per episode and kernel whose setup starts with ``setup_prefix``: its newest record."""
    newest: dict[tuple[str, str, str], plugin_records.Kept] = {}
    for kept in plugin_records.load(db, plugin=PLUGIN):
        if kept.setup.startswith(setup_prefix):
            newest[(kept.setup, kept.episode, kept.kernel)] = kept  # oldest first, so the last one stays
    tokens = episode_tokens(db)
    out = []
    for (setup, episode, kernel), kept in newest.items():
        if kept.record.schema_version not in SCHEMAS:
            raise ValueError(
                f"{episode}: record schema {kept.record.schema_version}, this script reads {sorted(SCHEMAS)}"
            )
        data = kept.record.data
        assert isinstance(data, dict), "a Tactics record is a JSON object"
        spent = tokens.get((setup, episode), dict.fromkeys(TOKEN_COLUMNS, 0))
        out.append({"setup": setup, "episode": episode, "kernel": kernel, **row_of(data), **spent})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("db", type=pathlib.Path, help="the bench's results DB")
    parser.add_argument("--setup", default="", help="read only setups whose name starts with this")
    parser.add_argument("--out", type=pathlib.Path, required=True, help="the CSV to write")
    args = parser.parse_args()
    table = rows(args.db, args.setup)
    if not table:
        raise SystemExit(f"no {PLUGIN} record in {args.db} under setup prefix {args.setup!r}")
    with args.out.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"{len(table)} episodes -> {args.out}")


if __name__ == "__main__":
    main()
