# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""ICON AES graupel: split the map computing the sedimentation sink and its ratio (``sink``, ``sx2x``), then the
default stage 2 on top.

That map is the loop over the 6 species of each level, with the column maps inside it: left whole, only its 6
iterations run in parallel. Split, each statement becomes a species map the default collapses with its column map,
so the parallel space is species by columns (1.83x at preset M, 72 threads, on c2daa40, job 5256868)."""

from nestforge.replay import Context, Script

KERNEL = "icon_aes_graupel"
CONFIGS = (Context("cpu", 72, "M"),)


def solve(s: Script, ctx: Context) -> None:
    s.apply(s.pick("doall-fission", writes=["__inl1_sink", "__inl1_sx2x"]))
    s.defaults()
