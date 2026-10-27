# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""TSVC s233: split the i loop into its aa and bb halves, then move each half's loop inside its map."""

from nestforge.replay import Context, Script

KERNEL = "tsvc_2_s233"
CONFIGS = (Context("cpu", 72, "M"),)


def solve(s: Script, ctx: Context) -> None:
    s.apply(s.pick("loop-fission"))
    s.apply(s.pick("interchange-loop-map", writes=["aa"]))
    s.apply(s.pick("interchange-loop-map", writes=["bb"]))
