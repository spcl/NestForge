# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The loops a kernel sits inside: their iterators are symbols the kernel takes, defined by the loop and by no
declaration, and a standalone check of the kernel needs a value for each."""

from __future__ import annotations

import dace
from dace.libraries.standard.nodes.external_call import ExternalCall
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation.passes.analysis import loop_analysis

from nestforge.ir.extract import INT64

#: A loop's iteration count is halved to pick its middle iteration, an interior value that boundary reads miss.
HALF = 2


def enclosing_loops(state: SDFGState) -> list[LoopRegion]:
    """The counted loops around ``state``, outermost first."""
    loops: list[LoopRegion] = []
    region: ControlFlowRegion | dace.SDFG | None = state.parent_graph
    while isinstance(region, ControlFlowRegion) and not isinstance(region, dace.SDFG):
        if isinstance(region, LoopRegion) and region.loop_variable:
            loops.append(region)
        region = region.parent_graph
    return loops[::-1]


def kernel_state(sdfg: dace.SDFG, ext: ExternalCall) -> SDFGState:
    """The state holding the kernel node ``ext``."""
    for block in sdfg.all_control_flow_blocks():
        if isinstance(block, SDFGState) and any(node is ext for node in block.nodes()):
            return block
    raise KeyError(f"kernel {ext.name!r} is not in the program")


def declare_scope_symbols(parent: dace.SDFG, nsdfg: nodes.NestedSDFG) -> None:
    """Declare on ``parent`` each symbol ``nsdfg`` takes that only a loop around it defines. An ``ExternalCall`` is
    lowered against the declared symbols and gets no symbol mapping, so an undeclared iterator has no type there."""
    for name in nsdfg.symbol_mapping:
        if str(name) not in parent.symbols:
            parent.add_symbol(str(name), nsdfg.sdfg.symbols.get(str(name), INT64))


def middle_value(loop: LoopRegion, env: dict[dace.symbolic.symbol, int]) -> int | None:
    """The iterator value halfway through ``loop`` under ``env``, or ``None`` for a loop that is not a counted range."""
    start = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if start is None or end is None or stride is None:
        return None
    first, last, step = (int(dace.symbolic.evaluate(x, env)) for x in (start, end, stride))
    count = (last - first) // step + 1
    return first + (max(count, 1) - 1) // HALF * step


def loop_symbol_values(state: SDFGState, sizes: dict[str, int]) -> dict[str, int]:
    """A value for the iterator of each loop around ``state``: the middle of its range under ``sizes`` and the
    values already chosen for the loops outside it."""
    env: dict[dace.symbolic.symbol, int] = {dace.symbolic.symbol(k): v for k, v in sizes.items()}
    values: dict[str, int] = {}
    for loop in enclosing_loops(state):
        value = middle_value(loop, env)
        if value is not None:
            values[loop.loop_variable] = value
            env[dace.symbolic.symbol(loop.loop_variable)] = value
    return values
