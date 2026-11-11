# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Decoys of ``floyd_warshall`` (branch ``decoys``): schedules an agent could reasonably take that are not the reference solution."""

from nestforge.replay import Context, Decoy, Script


def stage1(s: Script, ctx: Context) -> None:
    """No move and no default fusions: the same time as the default (0.98-1.01x at M)."""
    s.finish()


DECOYS = {
    "stage1": Decoy(
        stage1,
        "stopped-early",
        "the menu offers only tiling and expansion, so taking no move looks like the only safe choice",
    )
}
