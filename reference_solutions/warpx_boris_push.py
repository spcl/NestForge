# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""WarpX Boris momentum pusher: fuse the maps of the step loop (``doall-fusion``), move the loop inside one particle
map (``interchange-for-doall``), then the default stage 2 on top.

The kernel pushes every particle ``nsteps`` times; each step is a chain of elementwise maps over the particles, behind
ifs on ``momentum_push_type``. The default keeps the step loop outside, so every step streams ``ux``, ``uy``, ``uz``
and the six field arrays through memory again. With the step loop inside one particle map, each particle runs all its
steps before the next and the arrays are streamed once (132.1 ms to 82.9 ms, 1.59x at preset L, 72 threads, 3
alternated rounds 1.59-1.60x, measured on 690de2f with the single move that is now these two; to be retaken)."""

from nestforge.replay import Context, Script

KERNEL = "warpx_boris_push"
CONFIGS = (Context("cpu", 72, "L"),)


def solve(s: Script, ctx: Context) -> None:
    for _ in range(200):  # the maps behind one branch fuse; the ifs between the branches stop it
        if not (fusions := s.session.list_moves("doall-fusion")):
            break
        s.apply(fusions[0])
    s.apply(s.pick("interchange-for-doall", writes=["ux", "uy", "uz"]))
    s.defaults()
