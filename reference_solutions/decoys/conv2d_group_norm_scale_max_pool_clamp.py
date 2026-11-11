# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Decoys of ``conv2d_group_norm_scale_max_pool_clamp`` (branch ``decoys``): schedules an agent could reasonably take that are not the reference solution."""

from dace.libraries.linalg.nodes.tensordot import TensorDot

from nestforge.ir.introspect import Row
from nestforge.replay import Context, Decoy, Script


def is_tensordot(rows: list[Row]) -> bool:
    return isinstance(rows[0][0], TensorDot)


def stage1(s: Script, ctx: Context) -> None:
    """No move and no default fusions (65.04 ms against the default's 63.75 ms at preset M)."""
    s.finish()


def pure_tensordot(s: Script, ctx: Context) -> None:
    """The ``pure`` implementation of the contraction instead of its expansion: 437.78 ms at preset M, 0.15x."""
    s.session.default_moves()
    s.apply(s.pick("set-library-implementation", where=is_tensordot))
    s.finish()


DECOYS = {
    "stage1": Decoy(
        stage1, "stopped-early", "the default is already correct and no move is offered that obviously pays"
    ),
    "pure-tensordot": Decoy(
        pure_tensordot,
        "looks-right-loses",
        "the pure implementation is the other way to lower the contraction, but it is not the expansion into loops",
    ),
}
