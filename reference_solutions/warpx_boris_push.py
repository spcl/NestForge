# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""WarpX Boris momentum pusher: fuse the maps of the step loop (``doall-fusion``), move the loop inside the one particle map (``interchange-for-doall``), then the
default stage 2 on top.

The kernel pushes every particle ``nsteps`` times; each step is a chain of elementwise maps over the particles. The
default keeps the step loop outside, so every step streams ``ux``, ``uy``, ``uz`` and the six field arrays through
memory again. With the step loop inside one particle map, each particle runs all its steps before the next and the
arrays are streamed once (132.1 ms to 82.9 ms, 1.59x at preset L, 72 threads, 3 alternated rounds 1.59-1.60x, on
690de2f, job 5303150). The move is on this branch only (not on main bc07961)."""

from nestforge.replay import Context, Script

KERNEL = "warpx_boris_push"
CONFIGS = (Context("cpu", 72, "L"),)


def solve(s: Script, ctx: Context) -> None:
    while fusions := s.session.list_moves("doall-fusion"):
        s.apply(fusions[0])
    s.apply(s.pick("interchange-for-doall", writes=["ux", "uy", "uz"]))
    s.defaults()


def stage1(s: Script, ctx: Context) -> None:
    """No move and no default fusions: one parallel region per elementwise map and step (0.45-0.46x at L)."""
    s.finish()


DECOYS = {"stage1": stage1}
