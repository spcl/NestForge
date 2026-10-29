# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""HMM forward algorithm on the CPU: each time step is ``alpha @ trans`` and ``sum(alpha)``, two library calls the
default keeps inside the time loop. ``expand-library`` of both (the product into a doall over states with an inner
reduction, the sum into a doall reduction) leaves no vendor call in the loop, then the default: 1737-1771 ms to
183-187 ms at preset M on 72 threads. The same route as ``pagerank``, on a different program. On the GPU the default
does not build, so the GPU is not a configuration yet."""

from nestforge.replay import Context, Script

KERNEL = "hmm_forward"
CONFIGS = (Context("cpu", 72, "M"),)


def solve(s: Script, ctx: Context) -> None:
    if ctx.device == "cpu":
        s.apply(s.pick("expand-library", writes=["alpha_matmul_trans"]))
        s.apply(s.pick("expand-library", writes=["scale__v2"]))
    s.defaults()
