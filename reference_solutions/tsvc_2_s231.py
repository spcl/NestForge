# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""TSVC s231: move the j loop inside the i map, one parallel region instead of one per j."""

from nestforge.replay import Context, Script

KERNEL = "tsvc_2_s231"
CONFIGS = (Context("cpu", 72, "M"),)


def solve(s: Script, ctx: Context) -> None:
    s.apply(s.pick("interchange-loop-map", reads=["bb"], writes=["aa"]))
