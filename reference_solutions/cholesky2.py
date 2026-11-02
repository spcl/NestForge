# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cholesky factor plus the strict upper triangle, on the CPU: the default keeps ``np.linalg.cholesky`` as a library
call, OpenBLAS's, and on the CPU that call is the slow part of the program. ``expand-library`` turns it into NestForge's
own loops, which the default then parallelizes: 22.3-23.3 s to 121 ms at preset M on 72 threads. The default never
expands a library call. On the GPU the answer flips: the default's vendor solver takes 8.5 ms and the expansion 9.5 s,
so the default stays the reference there."""

from nestforge.replay import Context, Script

KERNEL = "cholesky2"
CONFIGS = (Context("cpu", 72, "M"),)


def solve(s: Script, ctx: Context) -> None:
    if ctx.device == "cpu":
        s.apply(s.pick("expand-library"))
    s.defaults()
