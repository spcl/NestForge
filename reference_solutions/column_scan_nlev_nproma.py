# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Column scan, (NLEV, NPROMA) layout: the default stage 2, then one kernel around the whole step loop (stage 3,
``set_scope``), so each of the ``nsteps`` passes no longer launches a kernel of its own.

The default scopes leave the step loop on the host: its body is the level loop, which holds the ``k == 0`` branch,
and a loop body with a branch is not opened, so the level loop becomes the kernel and runs once per step. A
``peel-for`` of the first level (1.20x at preset M, 72 threads, job 5305247; the scope 1.20-1.39x in job 5306973) and a ``tile-for`` of the step loop reach
the same layout by accident: the peel wraps the body in a guard and the tile adds a loop level, and either makes the
default scopes take the whole step loop as one kernel. The scope is the lever; the peel and the tile are near-equals
(see T46-79)."""

from collections.abc import Sequence

from dace.sdfg.state import LoopRegion

from nestforge.ir.introspect import Row
from nestforge.replay import Context, Script

KERNEL = "column_scan_nlev_nproma"
CONFIGS = (Context("cpu", 72, "M"),)


def is_loop(rows: Sequence[Row]) -> bool:
    return isinstance(rows[0][0], LoopRegion)


def solve(s: Script, ctx: Context) -> None:
    s.defaults()
    s.scope(where=is_loop)


def peel_first(s: Script, ctx: Context) -> None:
    """The peel of the first level before the default: the default then leaves the level loop as it was (1.05x at M)."""
    s.apply(s.pick("peel-for"))
    s.defaults()


def level_map_sequential(s: Script, ctx: Context) -> None:
    """The level-row map made a sequential loop before the default (0.95x at M)."""
    s.apply(s.pick("doall-to-for"))
    s.defaults()


DECOYS = {"peel-first": peel_first, "level-map-sequential": level_map_sequential}
