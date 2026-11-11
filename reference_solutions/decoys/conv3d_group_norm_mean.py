# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Decoys of ``conv3d_group_norm_mean`` (branch ``decoys``): schedules an agent could reasonably take that are not the reference solution."""

from nestforge.replay import Context, Decoy, Script


def stage1(s: Script, ctx: Context) -> None:
    """No move and no default fusions (0.74-0.75x at M)."""
    s.finish()


DECOYS = {"stage1": Decoy(stage1, "stopped-early", "no move needs no reasoning about the convolution's loop order")}
