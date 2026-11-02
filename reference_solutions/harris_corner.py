# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Harris corner response on the GPU: the default keeps the gradient stencil and the 3x3 box sums in two kernels,
with ``Ixx``, ``Iyy`` and ``Ixy`` written to memory and read back at 9 offsets. Fusing every producer into the
box-sum doall that reads it (the products, then ``Ix`` and ``Iy`` and their chains), which recomputes them per
tap, leaves one kernel reading ``img`` and writing ``R``. The default never takes these fusions. On the CPU the
same route is slower (0.75x at 2048 x 4096), so the default stays the reference there."""

from collections.abc import Callable

from dace.sdfg import nodes

from nestforge.replay import Context, Script, row_containers
from nestforge.session import Move, Session

KERNEL = "harris_corner"
CONFIGS = (Context("gpu", 1, "M"),)

FUSION = "doall-fusion"


def feeds(session: Session, move: Move, array: str) -> bool:
    """One doall of the fusion writes ``array`` and the other reads it."""
    rows = session.row_index()
    (reads0, writes0), (reads1, writes1) = (row_containers(rows[label]) for label in move.labels)
    return (array in writes0 and array in reads1) or (array in writes1 and array in reads0)


def into_box_sums(session: Session, move: Move) -> bool:
    """One doall of the fusion runs over the box-sum range ``(H - 4) x (W - 4)``."""
    rows = session.row_index()
    return any(
        isinstance(rows[label][0], nodes.MapEntry) and "H - 4" in str(rows[label][0].map.range) for label in move.labels
    )


def fuse_while(s: Script, wanted: Callable[[Session, Move], bool]) -> None:
    while move := next((m for m in s.session.list_moves(FUSION) if wanted(s.session, m)), None):
        s.apply(move)


def solve(s: Script, ctx: Context) -> None:
    if ctx.device == "gpu":
        for array in ("Ixx", "Iyy", "Ixy", "Ix", "Iy"):
            fuse_while(s, lambda session, move, a=array: feeds(session, move, a))
        fuse_while(s, into_box_sums)
    s.defaults()
