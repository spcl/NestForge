# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The plugin record of an episode (``docs/plugin-record.md``): its shape, written by the judge route for an agent and
by the tuner columns for themselves. No bench import: each writer passes the bench's own ``keep``."""

import hashlib
from collections.abc import Callable, Sequence
from typing import Any

#: The record's file name in the episode's ``plugin_records`` folder, and the schema version of its shape.
PLUGIN = "nestforge"
SCHEMA = 1


def journal_entry(name: str, arguments: dict[str, Any]) -> list[object]:
    """One change of the journal as the record holds it: a written kernel's source as its SHA-256."""
    if "source" in arguments:
        digest = hashlib.sha256(str(arguments["source"]).encode("utf-8")).hexdigest()
        arguments = {**arguments, "source": f"sha256:{digest}"}
    return [name, arguments]


def episode_record(
    kernel: str,
    calls: Sequence[dict[str, Any]],
    journal: Sequence[list[object]],
    kernels: int,
    reference: bool | None = None,
) -> dict[str, Any]:
    """The record of an episode so far. ``reference`` says whether the reference column replayed reference moves;
    ``None`` for every other column."""
    return {
        "kernel": kernel,
        "disabled_families": [],
        "reference": reference,
        "calls": list(calls),
        "journal": list(journal),
        "first_write_kernel": next((call["index"] for call in calls if call["tool"] == "write_kernel"), None),
        "final": {"kernels": kernels},
    }


def keep_capped(keep: Callable[[dict[str, Any]], object], data: dict[str, Any]) -> None:
    """``keep(data)``; over the bench's cap, again without the calls that changed nothing, marked ``truncated``. The
    journal is never shortened.

    :raises ValueError: The record is over the cap even so."""
    try:
        keep(data)
    except ValueError:
        keep({**data, "calls": [call for call in data["calls"] if call["status"] != "ok"], "truncated": True})
