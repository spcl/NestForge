# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Render arena results: for each nest and FP mode, the winning compiler x flag, plus the grid."""
from __future__ import annotations

from nestforge.arena import ArenaResult
from nestforge.perf import flags


def render_markdown(result: ArenaResult) -> str:
    lines = [f"## nest `{result.name}`", ""]

    lines.append("### winner per FP mode")
    lines.append("| FP mode | compiler | max-diff vs numpy | time (us) |")
    lines.append("|---|---|---|---|")
    for mode in flags.FP_LEVELS:
        w = result.winners.get(mode)
        if w is None:
            lines.append(f"| {mode} | — (no correct build) | — | — |")
        else:
            lines.append(f"| {mode} | {w.compiler} | {w.maxdiff:g} | {w.time_us:.2f} |")
    lines.append("")

    lines.append("### full grid")
    lines.append("| compiler | FP mode | correct | max-diff | time (us) | measured |")
    lines.append("|---|---|---|---|---|---|")
    for c in sorted(result.cells, key=lambda c: (c.compiler, c.fp_mode)):
        t = "inf" if c.time_us == float("inf") else f"{c.time_us:.2f}"
        d = "inf" if c.maxdiff == float("inf") else f"{c.maxdiff:g}"
        # a collapsed cell's timing is real but borrowed; unmarked it reads as an independent sample
        measured = "yes" if c.same_as is None else f"= {c.same_as}"
        lines.append(f"| {c.compiler} | {c.fp_mode} | {'yes' if c.ok else 'no'} | {d} | {t} | {measured} |")
    if result.collapsed:
        lines += ["", "### collapsed (identical artifact, measured once)", ""]
        lines += [f"- `{note}`" for note in result.collapsed]
    return "\n".join(lines) + "\n"
