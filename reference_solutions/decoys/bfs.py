# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Decoys of ``bfs`` (branch ``decoys``): schedules an agent could reasonably take that are not the reference solution."""

from nestforge.replay import Context, Decoy, Script


def stage1(s: Script, ctx: Context) -> None:
    """No move and no default fusions: close to the reference on the GPU at M."""
    s.finish()


DECOYS = {
    "stage1": Decoy(
        stage1,
        "stopped-early",
        "skipping the default's fusions and taking no move is nearly as fast as the interchange on the GPU",
    )
}
