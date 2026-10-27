# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""ICON AES graupel: split the map computing the sedimentation sink and its ratio (``sink``, ``sx2x``), then the
default stage 2 on top."""

from nestforge.replay import Context, Script

KERNEL = "aes_graupel"
CONFIGS = (Context("cpu", 72, "M"),)


def solve(s: Script, ctx: Context) -> None:
    s.apply(s.pick("map-fission", writes=["__inl1_sink", "__inl1_sx2x"]))
    s.defaults()
