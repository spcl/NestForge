# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Decoys of ``warpx_boris_push`` (branch ``decoys``): schedules an agent could reasonably take that are not the reference solution."""

from nestforge.replay import Context, Decoy, Script


def stage1(s: Script, ctx: Context) -> None:
    """No move and no default fusions: one parallel region per elementwise map and step (0.45-0.46x at L)."""
    s.finish()


DECOYS = {"stage1": Decoy(stage1, "stopped-early", "without the lane hoist the default's fusions look complete")}
