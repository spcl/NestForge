# Copyright 2021 ETH Zurich and the Tactics authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Move families: fixed sets of move kinds a run can switch off as a whole, and the unit the replay attribution sums
a run's speedup by. The table is the single source; a test keeps it in step with the move kinds."""

from collections.abc import Iterable

#: Every family and its move kinds. ``rollback`` belongs to none and is never switched off.
FAMILIES: dict[str, frozenset[str]] = {
    "fusion": frozenset({"for-fusion", "doall-fusion", "snapshot-reads"}),
    "fission": frozenset({"for-fission", "doall-fission", "subgraph-fission", "split-doall"}),
    "interchange": frozenset({"interchange-doall-doall", "interchange-doall-for", "interchange-for-doall"}),
    "shape": frozenset({"flatten-doalls", "unflatten-doall", "doall-to-for", "for-to-doall"}),
    "tiling": frozenset({"tile-doall", "tile-for", "untile-doall", "untile-for"}),
    "hoisting": frozenset(
        {
            "hoist-invariant-code",
            "sink-invariant-code",
            "hoist-invariant-guard",
            "sink-invariant-guard",
            "drop-implied-guard",
        }
    ),
    "specialize": frozenset({"propagate-config", "unroll-for", "unroll-doall"}),
    "library": frozenset(
        {"lib-before", "lib-after", "lib-into", "split-blocking", "inline-library", "set-library-implementation"}
    ),
    "layout": frozenset({"relayout"}),
    "precision": frozenset({"set-fp-mode", "change-precision"}),
    "vectorize": frozenset({"vectorize-doall"}),
    "schedule": frozenset(
        {
            "set-schedule-cpu",
            "set-schedule-gpu",
            "stage-loads-shared",
            "copy-to-tma",
            "set-pipeline-depth",
            "persistent-kernel",
        }
    ),
    "wavefront": frozenset({"skew-wavefront"}),
    # the moves of the catalogue (``tactics.stages.authored``): their kinds are known only when a session loads it
    "authored": frozenset(),
}

#: The move kinds no family holds, which every run keeps: ``rollback``, and the placement of kernels (the ``run-on-*``
#: moves of the scopes phase), which a run without moves still makes.
FAMILY_FREE: frozenset[str] = frozenset(
    {"rollback", "run-on-cpu", "run-on-gpu", "run-on-cpu-persistent", "run-on-gpu-persistent"}
)

#: The name that stands for every family.
ALL = "all"

#: The family of canonicalization's fusion stage, which the default moves run.
DEFAULT_MOVES_FAMILY = "fusion"
#: The family of the wavefront skew, which the default moves run before the fusion stage.
WAVEFRONT_FAMILY = "wavefront"


def family_of(kind: str) -> str | None:
    """The family of move ``kind``; ``None`` for one in no family (``rollback``)."""
    return next((name for name, kinds in FAMILIES.items() if kind in kinds), None)


def checked_families(names: Iterable[str]) -> frozenset[str]:
    """``names`` as families, :data:`ALL` expanded.

    :raises ValueError: A name is no family."""
    wanted = frozenset(names)
    if ALL in wanted:
        return frozenset(FAMILIES)
    unknown = sorted(wanted - FAMILIES.keys())
    if unknown:
        raise ValueError(f"no move family {unknown}; families are {[*FAMILIES, ALL]}")
    return wanted


def switched_off(kind: str, disabled: frozenset[str]) -> str | None:
    """The disabled family ``kind`` belongs to, or ``None`` when the kind may run."""
    family = family_of(kind)
    return family if family in disabled else None
