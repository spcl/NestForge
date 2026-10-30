# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""PageRank power iteration on the CPU: each sweep is ``trans @ rank`` and ``sum(rank)``, two library calls the default
keeps. ``expand-library`` of both (the product into a row-parallel doall with an inner reduction, the sum into a doall
reduction) leaves no vendor call in the loop, then the default: 486-502 ms to 150 ms at preset M on 72 threads.
Expanding only the product does not build (``KeyError: '_out_gemv'``). On the GPU the default does not build at all,
so the GPU is not a configuration yet."""

from nestforge.replay import Context, Script

KERNEL = "pagerank"
CONFIGS = (Context("cpu", 72, "M"),)


def solve(s: Script, ctx: Context) -> None:
    if ctx.device == "cpu":
        s.apply(s.pick("expand-library", writes=["trans_matmul_rank"]))
        s.apply(s.pick("expand-library", writes=["sum_rank"]))
    s.defaults()
