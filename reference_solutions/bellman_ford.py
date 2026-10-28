# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Bellman-Ford on the GPU: the default's fusions, then the one ``map-fission`` it leaves, then the stages after
fusion. On the CPU the default leaves no move, so the reference is the default there."""

from nestforge.replay import Context, Script

KERNEL = "bellman_ford"
CONFIGS = (Context("gpu", 1, "M"),)


def solve(s: Script, ctx: Context) -> None:
    s.session.default_moves()
    if ctx.device == "gpu":
        s.apply(s.pick("map-fission"))
    s.finish()
