# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Decoys of ``fuse_physics_into_scan`` (branch ``decoys``): schedules an agent could reasonably take that are not the reference solution."""

from nestforge.replay import Context, Decoy, Script


def physics_levels_outside(s: Script, ctx: Context) -> None:
    """The reference, then the physics map interchanged so the levels are outermost: on the GPU only the 90
    levels stay parallel (0.04x of the default at M)."""
    s.session.default_moves()
    if ctx.device == "gpu":
        s.apply(s.pick("doall-fission", writes=["s"]))
        s.apply(s.pick("interchange-doall-doall", reads=["x", "t"]))
    s.finish()


DECOYS = {
    "physics-levels-outside": Decoy(
        physics_levels_outside,
        "looks-right-loses",
        "moving the levels outermost looks like the usual way to parallelize a column scan, but on the GPU it leaves"
        " only the levels parallel",
    )
}
