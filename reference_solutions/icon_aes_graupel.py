# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""ICON AES graupel: the default stage 2.

The canonical form unrolls the loop over the 6 species of each level (DaCe d3b47fd05c), so every species already has
its own column map: the parallel space is species by columns without a move. Before that, the species loop was one
map holding the column maps, and this solution split it with ``doall-fission`` on ``sink`` and ``sx2x`` (1.83x at
preset M, 72 threads, on c2daa40, job 5256868)."""

from tactics.replay import Context, Script

KERNEL = "icon_aes_graupel"
CONFIGS = (Context("cpu", 72, "M"),)


def solve(s: Script, ctx: Context) -> None:
    s.defaults()
