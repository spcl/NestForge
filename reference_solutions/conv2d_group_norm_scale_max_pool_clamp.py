# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Conv2D + GroupNorm + scale + max-pool + clamp: the default stage 2, then ``expand-library`` of the ``TensorDot`` that
computes each convolution tap (NestForge's own loops instead of the TTGT call), then the stages after fusion.

The default keeps the nine per-tap contractions as ``TensorDot`` library calls, which DaCe lowers to a transpose, a GEMM
and a transpose. Expanded, the tap is one parallel loop nest that reads the padded window in place. Decoys: no move and
no default fusions (``stage1``), and the ``pure`` implementation of the same node, which keeps it a library call but
loops naively (0.15x at preset M)."""

from dace.libraries.linalg.nodes.tensordot import TensorDot

from nestforge.ir.introspect import Row
from nestforge.replay import Context, Script

KERNEL = "conv2d_group_norm_scale_max_pool_clamp"
CONFIGS = (Context("cpu", 72, "M"),)
CI_SKIP = (
    "needs two DaCe fixes that the pin 7c06b5194 lacks: the edge validation of a strided slice with a symbolic step "
    "(spcl/NestForge#86, fork commit 845c931cf) and the TTGT temporary of a sliced operand (#74, fork commit cc9642937); "
    "and the bench's torch oracle is missing (T46-74), so the timings grade against the bench's NumPy function in float64 "
    "at rtol 1e-8."
)


def is_tensordot(rows: list[Row]) -> bool:
    return isinstance(rows[0][0], TensorDot)


def solve(s: Script, ctx: Context) -> None:
    s.session.default_moves()
    s.apply(s.pick("expand-library", where=is_tensordot))
    s.finish()


def stage1(s: Script, ctx: Context) -> None:
    """No move and no default fusions (65.04 ms against the default's 63.75 ms at preset M)."""
    s.finish()


def pure_tensordot(s: Script, ctx: Context) -> None:
    """The ``pure`` implementation of the contraction instead of its expansion: 437.78 ms at preset M, 0.15x."""
    s.session.default_moves()
    s.apply(s.pick("set-library-implementation", where=is_tensordot))
    s.finish()


DECOYS = {"stage1": stage1, "pure-tensordot": pure_tensordot}
