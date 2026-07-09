"""Extract any loop-nest (CFG ``LoopRegion``) or map-nest (``MapEntry``) into a standalone SDFG.

One primitive unifies DaCe's two outliners:
  * ``MapEntry``   -> ``nest_state_subgraph(sdfg, state, state.scope_subgraph(entry))``
  * ``LoopRegion`` -> ``nest_sdfg_subgraph(sdfg, subgraph)``
Both produce a ``NestedSDFG`` whose ``.sdfg`` is the standalone SDFG. The returned
:class:`Boundary` records the in/out data and symbols and keeps a handle on the placed
``NestedSDFG`` so the lowering pass can later swap it for an ``ExternalCall`` node.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import List, Optional, Union

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion, SDFGState
from dace.transformation import helpers

NestNode = Union[nodes.MapEntry, LoopRegion]


@dataclass
class Boundary:
    """The interface of an extracted nest, in the order the arena/libnode will use."""
    inputs: List[str]  # data read by the nest (NestedSDFG in-connectors)
    outputs: List[str]  # data written by the nest (NestedSDFG out-connectors)
    symbols: List[str]  # free symbols (sizes) the nest depends on
    nsdfg_node: nodes.NestedSDFG  # the NestedSDFG placed in the parent (replacement anchor)
    state: SDFGState  # parent state holding ``nsdfg_node``
    standalone_sdfg: dace.SDFG  # detached, independently compilable copy of the nest
    parent_sdfg: dace.SDFG = field(repr=False, default=None)


def _detach(sdfg: dace.SDFG) -> dace.SDFG:
    """Deep-copy an outlined nested SDFG and cut its parent links so it stands alone."""
    det = copy.deepcopy(sdfg)
    det.parent = None
    det.parent_sdfg = None
    det.parent_nsdfg_node = None
    det.reset_cfg_list()
    return det


def _find_state_of_node(sdfg: dace.SDFG, node: nodes.Node) -> SDFGState:
    """Return the ``SDFGState`` in ``sdfg`` that contains ``node``."""
    for state in sdfg.states():
        if node in state.nodes():
            return state
    raise ValueError(f"node {node} not found in any state of SDFG {sdfg.label}")


def _boundary_from_nsdfg(nsdfg_node: nodes.NestedSDFG, state: SDFGState, parent_sdfg: dace.SDFG) -> Boundary:
    inputs = sorted(nsdfg_node.in_connectors.keys())
    outputs = sorted(nsdfg_node.out_connectors.keys())
    symbols = sorted(str(s) for s in nsdfg_node.symbol_mapping.keys())
    return Boundary(inputs=inputs,
                    outputs=outputs,
                    symbols=symbols,
                    nsdfg_node=nsdfg_node,
                    state=state,
                    standalone_sdfg=_detach(nsdfg_node.sdfg),
                    parent_sdfg=parent_sdfg)


def extract_map_nest(parent_sdfg: dace.SDFG, map_entry: nodes.MapEntry, name: Optional[str] = None) -> Boundary:
    """Outline a whole map scope (entry..exit + body) into a standalone SDFG.

    ``full_data=True`` nests the *entire* boundary arrays (not just the accessed sub-range), so the
    external call receives full arrays and the kernel keeps the original global indices -- otherwise
    DaCe shrinks + rebases each connector to its accessed slice (e.g. a ``[1:N-1]`` stencil would
    pass a size-``N-2`` buffer and index ``B[i-1]``), which would break the generated C signature.
    """
    state = _find_state_of_node(parent_sdfg, map_entry)
    subgraph = state.scope_subgraph(map_entry, include_entry=True, include_exit=True)
    nsdfg_node = helpers.nest_state_subgraph(parent_sdfg, state, subgraph, name=name or "nest", full_data=True)
    return _boundary_from_nsdfg(nsdfg_node, state, parent_sdfg)


def extract_loop_nest(parent_sdfg: dace.SDFG, loop: LoopRegion, name: Optional[str] = None) -> Boundary:
    """Outline a CFG loop region into a standalone SDFG (M1)."""
    from dace.sdfg.graph import SubgraphView
    subgraph = SubgraphView(parent_sdfg, [loop])
    inner_state = helpers.nest_sdfg_subgraph(parent_sdfg, subgraph)
    # nest_sdfg_subgraph returns the state that now holds the NestedSDFG; find that node.
    nsdfg_node = next(n for n in inner_state.nodes() if isinstance(n, nodes.NestedSDFG))
    return _boundary_from_nsdfg(nsdfg_node, inner_state, parent_sdfg)


def extract_nest_to_sdfg(parent_sdfg: dace.SDFG, node: NestNode, name: Optional[str] = None) -> Boundary:
    """Extract any map-nest or loop-nest into a standalone SDFG.

    :returns: a :class:`Boundary`; the standalone SDFG is ``boundary.nsdfg_node.sdfg``.
    """
    if isinstance(node, nodes.MapEntry):
        return extract_map_nest(parent_sdfg, node, name=name)
    if isinstance(node, LoopRegion):
        return extract_loop_nest(parent_sdfg, node, name=name)
    raise TypeError(f"cannot extract node of type {type(node).__name__}; expected MapEntry or LoopRegion")
