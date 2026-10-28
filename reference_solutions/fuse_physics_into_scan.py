# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Physics fused into a level scan, on the GPU: the default moves the level loop into one column map that computes
``p1`` and ``p2`` for every level and then scans; the ``map-fission`` of that column map (the one writing ``s``)
puts the physics in its own kernel again, then the stages after fusion. On the CPU the default stays the
reference."""

from nestforge.replay import Context, Script

KERNEL = "fuse_physics_into_scan"
CONFIGS = (Context("gpu", 1, "M"),)


def solve(s: Script, ctx: Context) -> None:
    s.session.default_moves()
    if ctx.device == "gpu":
        s.apply(s.pick("map-fission", writes=["s"]))
    s.finish()


def physics_levels_outside(s: Script, ctx: Context) -> None:
    """The reference, then the physics map interchanged so the levels are outermost: on the GPU only the 90
    levels stay parallel (0.04x of the default at M)."""
    s.session.default_moves()
    if ctx.device == "gpu":
        s.apply(s.pick("map-fission", writes=["s"]))
        s.apply(s.pick("interchange-map-map", reads=["x", "t"]))
    s.finish()


DECOYS = {"physics-levels-outside": physics_levels_outside}
