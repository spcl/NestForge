# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""3-D box stencil, written output-stationary (``programs/conv3d_box_dace.py``): on the GPU, one fission of the
``k`` map's body; on the CPU, two fissions that unlock pushing the ``k`` and ``j`` maps inside the tap loops."""

from nestforge.replay import Context, Script, move_matches

KERNEL = "conv_3d"
PROGRAM = "programs/conv3d_box_dace.py"
CONFIGS = (Context("cpu", 72, "M"), Context("gpu", 1, "M"))
#: Pushes offered together after the second fission.
PUSHES = 2


def both_pushes(s: Script) -> None:
    """``j`` inside ``di`` and ``k`` inside ``dj``: both are offered together after the fissions, and the route takes
    both, in the order they are offered (``j`` first)."""
    for _ in range(PUSHES):
        rows = s.session.row_index()
        offered = s.session.list_moves("interchange-doall-for")
        s.apply(next(m for m in offered if move_matches([rows[label] for label in m.labels], (), ["out_grid"], None)))


def solve(s: Script, ctx: Context) -> None:
    if ctx.device == "gpu":
        s.apply(s.pick("subgraph-fission", writes=["out_grid"]))
    else:
        s.apply(s.pick("doall-fission", writes=["out_grid"]))
        s.apply(s.pick("interchange-doall-for", writes=["out_grid"]))
        s.apply(s.pick("doall-fission", writes=["out_grid"]))
        both_pushes(s)
    s.finish()
