# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Move families: fixed sets of move kinds a run can switch off as a whole, and the unit the replay attribution sums
a run's speedup by. The table is the single source; a test keeps it in step with the move kinds."""

from collections.abc import Iterable

#: Every family and its move kinds. ``rollback`` belongs to none and is never switched off.
FAMILIES: dict[str, frozenset[str]] = {
    "fusion": frozenset({"for-fusion", "doall-fusion", "snapshot-reads"}),
    "fission": frozenset({"for-fission", "doall-fission", "subgraph-fission"}),
    "interchange": frozenset({"interchange-doall-doall", "interchange-doall-for", "interchange-for-doall"}),
    "shape": frozenset({"flatten-doalls", "unflatten-doall", "doall-to-for", "for-to-doall"}),
    "tiling": frozenset({"tile-doall", "tile-for", "untile-doall", "untile-for"}),
    "hoisting": frozenset(
        {"hoist-invariant-code", "sink-invariant-code", "hoist-invariant-guard", "sink-invariant-guard"}
    ),
    "specialize": frozenset({"propagate-config", "unroll-for", "unroll-doall"}),
    "library": frozenset({"lib-before", "lib-after", "lib-into", "inline-library", "set-library-implementation"}),
    "layout": frozenset({"relayout"}),
    "vectorize": frozenset({"vectorize-doall"}),
    "schedule": frozenset({"set-schedule-cpu", "set-schedule-gpu", "stage-loads-shared"}),
    "wavefront": frozenset({"skew-wavefront"}),
}

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
