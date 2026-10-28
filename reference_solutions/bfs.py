# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Breadth-first search on the GPU: the default's fusions, then the ``interchange-map-loop`` it leaves on ``reach``,
then the stages after fusion. On the GPU the default moves the source-vertex loop of ``frontier @ graph`` inside the
target-vertex map, so each of the N threads sums a whole column alone; the move puts the loop back outside, one
row update of ``reach`` per source vertex. On the CPU the default keeps that order and leaves the opposite move
(``interchange-loop-map``, slower there), so the reference is the default.

Caveat: taking no move and skipping the default's fusions (``stage1``) is within 5% of the reference at preset M on
the GPU (410.1 against 390.5 ms; default 1415.5 ms, job 5191988), so the gain mostly measures what the default gets
wrong, not what the move adds."""

from nestforge.replay import Context, Script

KERNEL = "bfs"
CONFIGS = (Context("gpu", 1, "M"),)


def solve(s: Script, ctx: Context) -> None:
    s.session.default_moves()
    if ctx.device == "gpu":
        s.apply(s.pick("interchange-map-loop", writes=["reach"]))
    s.finish()


def stage1(s: Script, ctx: Context) -> None:
    """No move and no default fusions: close to the reference on the GPU at M."""
    s.finish()


DECOYS = {"stage1": stage1}
