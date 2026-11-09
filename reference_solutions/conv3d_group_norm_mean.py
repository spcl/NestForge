# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Conv3D + GroupNorm + mean: the default stage 2, then the one ``interchange-doall-for`` it leaves, which takes the
kernel-position loop out of the output-point maps (1.32-1.33x at preset M, 72 threads, job 5303945, graded against the
bench's NumPy reference).

The convolution is a reduction over the input channel and the kernel positions. After the default, the innermost
kernel-position loop sits inside the output-point maps, and each output point calls it as a function that adds one
product to ``acc`` in memory. The interchange puts that loop outside the five output maps, so the innermost loop is the
output width, a contiguous ``omp simd`` loop that adds one product per element. Decoys: no move and no default
fusions (``stage1``, 0.74-0.75x), and the same interchange before the default (``interchange-first``)."""

from nestforge.replay import Context, Script

KERNEL = "conv3d_group_norm_mean"
CONFIGS = (Context("cpu", 72, "M"),)
CI_SKIP = (
    "the bench grades this kernel against KernelBench's torch model, which the pip install of HPCAgent-Bench lacks "
    "(T46-74): every variant would run unchecked. Timed against the bench's NumPy reference at rtol 1e-8 instead."
)


def solve(s: Script, ctx: Context) -> None:
    s.session.default_moves()
    s.apply(s.pick("interchange-doall-for"))
    s.finish()


def stage1(s: Script, ctx: Context) -> None:
    """No move and no default fusions (0.74-0.75x at M)."""
    s.finish()


def interchange_first(s: Script, ctx: Context) -> None:
    """The interchange before the default: the default then redoes the fusions on the interchanged tree."""
    s.apply(s.pick("interchange-doall-for"))
    s.defaults()


DECOYS = {"stage1": stage1, "interchange-first": interchange_first}
