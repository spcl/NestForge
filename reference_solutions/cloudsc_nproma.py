# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""CloudSC blocked layout, written in GPU order: split the column map, then take the level loop out of its body map,
so each level is one contiguous ``NPROMA`` map (the CPU order)."""

from nestforge.replay import Context, Script

KERNEL = "cloudsc_nproma"
CONFIGS = (Context("cpu", 1, "M"), Context("cpu", 72, "L"))
CI_SKIP = (
    "cloudsc_nproma is only on HPCAgent-Bench branch benchmarking-nestforge, and on DaCe 7683e143 its map-fission "
    "(MapFission) reads and writes pfplsx through one access node, a cyclic state the stages after fusion reject"
)


def solve(s: Script, ctx: Context) -> None:
    s.apply(s.pick("map-fission", writes=["pfplsx", "pcov"]))
    s.apply(s.pick("interchange-map-loop", writes=["pfplsx"]))
    s.finish()
