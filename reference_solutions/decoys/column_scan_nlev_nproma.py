# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Decoys of ``column_scan_nlev_nproma`` (branch ``decoys``): schedules an agent could reasonably take that are not the reference solution."""

from nestforge.replay import Context, Decoy, Script


def peel_first(s: Script, ctx: Context) -> None:
    """The peel of the first level before the default: the default then leaves the level loop as it was (1.05x at M)."""
    s.apply(s.pick("peel-for"))
    s.defaults()


def level_map_sequential(s: Script, ctx: Context) -> None:
    """The level-row map made a sequential loop before the default (0.95x at M)."""
    s.apply(s.pick("doall-to-for"))
    s.defaults()


DECOYS = {
    "peel-first": Decoy(
        peel_first,
        "wrong-order",
        "a peel of the first level is one of three moves that reach the reference's layout, but taken before the"
        " default it changes nothing",
    ),
    "level-map-sequential": Decoy(
        level_map_sequential,
        "looks-right-loses",
        "a sequential level loop matches the scan's dependence, but it gives up the parallel level map",
    ),
}
