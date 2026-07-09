"""Render arena results: for each nest and FP mode, the winning compiler x flag, plus the grid."""
from __future__ import annotations

from typing import Iterable

from nestforge.arena import ArenaResult, FP_MODES


def render_markdown(result: ArenaResult) -> str:
    lines = [f"## nest `{result.name}`", ""]

    lines.append("### winner per FP mode")
    lines.append("| FP mode | compiler | max-diff vs numpy | time (us) |")
    lines.append("|---|---|---|---|")
    for mode in FP_MODES:
        w = result.winners.get(mode)
        if w is None:
            lines.append(f"| {mode} | — (no correct build) | — | — |")
        else:
            lines.append(f"| {mode} | {w.compiler} | {w.maxdiff:g} | {w.time_us:.2f} |")
    lines.append("")

    lines.append("### full grid")
    lines.append("| compiler | FP mode | correct | max-diff | time (us) |")
    lines.append("|---|---|---|---|---|")
    for c in sorted(result.cells, key=lambda c: (c.compiler, c.fp_mode)):
        t = "inf" if c.time_us == float("inf") else f"{c.time_us:.2f}"
        d = "inf" if c.maxdiff == float("inf") else f"{c.maxdiff:g}"
        lines.append(f"| {c.compiler} | {c.fp_mode} | {'yes' if c.ok else 'no'} | {d} | {t} |")
    return "\n".join(lines) + "\n"


def render_many(results: Iterable[ArenaResult]) -> str:
    return "\n".join(render_markdown(r) for r in results)
