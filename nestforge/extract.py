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


def detach(sdfg: dace.SDFG) -> dace.SDFG:
    """Deep-copy an outlined nested SDFG and cut its parent links so it stands alone."""
    det = copy.deepcopy(sdfg)
    det.parent = None
    det.parent_sdfg = None
    det.parent_nsdfg_node = None
    det.reset_cfg_list()
    return det


def find_state_of_node(sdfg: dace.SDFG, node: nodes.Node) -> SDFGState:
    """Return the ``SDFGState`` in ``sdfg`` that contains ``node``."""
    for state in sdfg.states():
        if node in state.nodes():
            return state
    raise ValueError(f"node {node} not found in any state of SDFG {sdfg.label}")


def boundary_from_nsdfg(nsdfg_node: nodes.NestedSDFG, state: SDFGState, parent_sdfg: dace.SDFG) -> Boundary:
    inputs = sorted(nsdfg_node.in_connectors.keys())
    outputs = sorted(nsdfg_node.out_connectors.keys())
    symbols = sorted(str(s) for s in nsdfg_node.symbol_mapping.keys())
    return Boundary(inputs=inputs,
                    outputs=outputs,
                    symbols=symbols,
                    nsdfg_node=nsdfg_node,
                    state=state,
                    standalone_sdfg=detach(nsdfg_node.sdfg),
                    parent_sdfg=parent_sdfg)


def extract_map_nest(parent_sdfg: dace.SDFG, map_entry: nodes.MapEntry, name: Optional[str] = None) -> Boundary:
    """Outline a whole map scope (entry..exit + body) into a standalone SDFG.

    ``full_data=True`` nests the *entire* boundary arrays (not just the accessed sub-range), so the
    external call receives full arrays and the kernel keeps the original global indices -- otherwise
    DaCe shrinks + rebases each connector to its accessed slice (e.g. a ``[1:N-1]`` stencil would
    pass a size-``N-2`` buffer and index ``B[i-1]``), which would break the generated C signature.
    """
    state = find_state_of_node(parent_sdfg, map_entry)
    subgraph = state.scope_subgraph(map_entry, include_entry=True, include_exit=True)
    nsdfg_node = helpers.nest_state_subgraph(parent_sdfg, state, subgraph, name=name or "nest", full_data=True)
    return boundary_from_nsdfg(nsdfg_node, state, parent_sdfg)


def nest_defined_symbols(loop: LoopRegion) -> set:
    """Symbols DEFINED inside the loop nest: every loop variable (the region itself and any nested
    LoopRegion) plus every inter-state-edge assignment target. Mirrors the ``ndefined_symbols`` set
    ``helpers.nest_sdfg_subgraph`` builds for its symbolic-output plumbing."""
    syms = set()
    for b in [loop, *loop.all_control_flow_blocks()]:
        if isinstance(b, LoopRegion) and b.loop_variable and b.init_statement:
            syms.add(b.loop_variable)
    for e in loop.all_interstate_edges():
        syms.update(e.data.assignments.keys())
    return syms


def extract_loop_nest(parent_sdfg: dace.SDFG, loop: LoopRegion, name: Optional[str] = None) -> Boundary:
    """Outline a CFG loop region into a standalone SDFG (M1)."""
    from dace.sdfg.graph import SubgraphView
    # nest_sdfg_subgraph emits a "symbolic output" for each symbol defined in the nest and looks up its
    # dtype in the nested-or-PARENT sdfg.symbols. A loop index DaCe never registered as a symbol (it need
    # not be) would KeyError there; pre-declare each nest-defined symbol on the parent as int64.
    for s in nest_defined_symbols(loop):
        if s not in parent_sdfg.symbols:
            parent_sdfg.add_symbol(s, dace.int64)
    subgraph = SubgraphView(parent_sdfg, [loop])
    inner_state = helpers.nest_sdfg_subgraph(parent_sdfg, subgraph)
    # nest_sdfg_subgraph returns the state that now holds the NestedSDFG; find that node.
    nsdfg_node = next(n for n in inner_state.nodes() if isinstance(n, nodes.NestedSDFG))
    return boundary_from_nsdfg(nsdfg_node, inner_state, parent_sdfg)


def extract_nest_to_sdfg(parent_sdfg: dace.SDFG, node: NestNode, name: Optional[str] = None) -> Boundary:
    """Extract any map-nest or loop-nest into a standalone SDFG.

    :returns: a :class:`Boundary`; the standalone SDFG is ``boundary.nsdfg_node.sdfg``.
    """
    if isinstance(node, nodes.MapEntry):
        return extract_map_nest(parent_sdfg, node, name=name)
    if isinstance(node, LoopRegion):
        return extract_loop_nest(parent_sdfg, node, name=name)
    raise TypeError(f"cannot extract node of type {type(node).__name__}; expected MapEntry or LoopRegion")


def whole_program_boundary(sdfg: dace.SDFG) -> Boundary:
    """A :class:`Boundary` wrapping the WHOLE (un-split) kernel SDFG -- the whole-program-scope lane, where
    each external tool receives the entire program and auto-optimizes across nests, instead of one extracted
    nest.

    There is no extraction and no parent: ``standalone_sdfg`` is a detached copy of the whole SDFG, and the
    replacement handles (``nsdfg_node``/``state``/``parent_sdfg``) are ``None`` -- the whole-program lane
    emits + compiles + times, it never swaps a libnode. ``inputs``/``outputs`` come from the SDFG's
    read/write sets (an in-place array is in BOTH, as for a nest boundary); ``symbols`` are the non-array
    arguments (the size symbols). Restricted to NON-transient arrays: transients are the kernel's own
    scratch, allocated by the emitted code, not part of the caller interface -- exactly the split
    :func:`boundary_from_nsdfg` gets for free from the nest's connectors."""
    detached = detach(sdfg)
    read, write = detached.read_and_write_sets()
    arrays = {n for n, desc in detached.arrays.items() if not desc.transient}
    inputs = sorted(a for a in arrays if a in read)
    outputs = sorted(a for a in arrays if a in write)
    symbols = [a for a in detached.arglist() if a not in detached.arrays]
    return Boundary(inputs=inputs,
                    outputs=outputs,
                    symbols=symbols,
                    nsdfg_node=None,
                    state=None,
                    standalone_sdfg=detached,
                    parent_sdfg=None)
