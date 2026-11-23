# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The labels the structure tree prints, ``<kind><n>`` and stable while their row lives (:class:`Namer`), and the inlining
of top-level nested SDFGs that canonicalization runs first."""

from __future__ import annotations

import heapq
import re
from dataclasses import dataclass, field
from typing import Any

import dace
from dace.libraries.comm.nodes.node import CommNode
from dace.libraries.standard.nodes.copy import CopyLibraryNode
from dace.libraries.standard.nodes.copy.common import INPUT_CONNECTOR_NAME, OUTPUT_CONNECTOR_NAME
from dace.libraries.standard.nodes.external_call import ExternalCall
from dace.sdfg import nodes
from dace.sdfg.state import (
    AbstractControlFlowRegion,
    BreakBlock,
    ConditionalBlock,
    ContinueBlock,
    ControlFlowBlock,
    LoopRegion,
    ReturnBlock,
    SDFGState,
)
from dace.transformation.interstate.expand_nested_sdfg_inputs import ExpandNestedSDFGInputs
from dace.transformation.interstate.multistate_inline import InlineMultistateSDFG
from dace.transformation.passes.analysis import loop_analysis
from dace.utils import find_new_name


def in_order(graph: AbstractControlFlowRegion | SDFGState) -> list[Any]:
    """A graph's nodes in topological order, ties broken by insertion order, so labels are deterministic."""
    all_nodes: list[Any] = list(graph.nodes())  # blocks of a region or nodes of a state
    rank = {id(n): i for i, n in enumerate(all_nodes)}
    indegree = {id(n): 0 for n in all_nodes}
    for edge in graph.edges():
        indegree[id(edge.dst)] += 1
    ready = [rank[id(n)] for n in all_nodes if indegree[id(n)] == 0]
    heapq.heapify(ready)
    ordered: list[Any] = []
    while ready:
        node = all_nodes[heapq.heappop(ready)]
        ordered.append(node)
        for edge in graph.out_edges(node):
            indegree[id(edge.dst)] -= 1
            if indegree[id(edge.dst)] == 0:
                heapq.heappush(ready, rank[id(edge.dst)])
    seen = {id(n) for n in ordered}
    return ordered + [n for n in all_nodes if id(n) not in seen]


# no top-level nested SDFG


def top_level_map_entries(state: SDFGState) -> list[nodes.MapEntry]:
    """The maps of ``state`` not nested in another map, in ``scope_children`` order (breadth first from the
    sources), which kernel names follow."""
    return [n for n in state.scope_children()[None] if isinstance(n, nodes.MapEntry)]


def is_copy_tasklet(node: nodes.Node) -> bool:
    """Whether ``node`` is the tasklet DaCe's ``copy_node`` makes of a one-element copy (``_cpy_out = _cpy_in``, a cast
    when the types differ): the same copy as a ``CopyLibraryNode``, which its automatic expansion would become."""
    return (
        isinstance(node, nodes.Tasklet)
        and set(node.in_connectors) == {INPUT_CONNECTOR_NAME}
        and set(node.out_connectors) == {OUTPUT_CONNECTOR_NAME}
    )


def is_copy(node: nodes.Node) -> bool:
    """Whether ``node`` is a copy, as a ``CopyLibraryNode`` or as its one-element tasklet."""
    return isinstance(node, CopyLibraryNode) or is_copy_tasklet(node)


def is_call(state: SDFGState, node: nodes.Node) -> bool:
    """Whether ``node`` is a library call row of ``state``: a library node (a kernel NestForge built included), or a
    one-element copy's tasklet outside every map (inside one it is a statement of the body)."""
    return isinstance(node, nodes.LibraryNode) or (is_copy_tasklet(node) and state.entry_node(node) is None)


def top_level_doalls(state: SDFGState) -> list[nodes.MapEntry | nodes.CodeNode]:
    """The doalls of ``state`` not nested in a map: its maps, and its library calls (each a black-box doall; a
    kernel NestForge built is not one, nor is a communication row, which never joins a kernel), in
    ``scope_children`` order."""
    return [
        n
        for n in state.scope_children()[None]
        if isinstance(n, nodes.MapEntry) or (is_call(state, n) and not isinstance(n, (ExternalCall, CommNode)))
    ]


def top_level_nsdfgs(sdfg: dace.SDFG) -> list[tuple[SDFGState, nodes.NestedSDFG]]:
    """Every ``NestedSDFG`` outside all map scopes; one inside a map is a kernel body."""
    return [
        (state, node)
        for state in sdfg.states()
        for node in state.scope_children()[None]
        if isinstance(node, nodes.NestedSDFG)
    ]


def inline_top_level_nsdfgs(sdfg: dace.SDFG) -> int:
    """Widen and inline every top-level nested SDFG; returns how many transformations applied."""
    if not top_level_nsdfgs(sdfg):
        return 0
    applied: int = sdfg.apply_transformations_repeated(
        ExpandNestedSDFGInputs, options={"top_level_only": True}, validate=False
    )
    return applied + int(sdfg.apply_transformations_repeated(InlineMultistateSDFG, validate=False))


# canonical labels


def counted(loop: LoopRegion) -> bool:
    """Whether ``loop`` runs over a range it knows up front (a ``for``), not until a condition fails (a ``while``)."""
    return bool(loop.loop_variable) and all(
        part is not None for part in (loop_analysis.get_init_assignment(loop), loop_analysis.get_loop_end(loop))
    )


def block_kind(block: ControlFlowBlock) -> str:
    """The tree keyword of a block: a sequential loop over a known range is a ``for``, one that runs until its
    condition fails a ``while``; a parallel map is a ``doall`` (:func:`relabel_state`), a loop proven to have
    independent iterations. A kernel is a scope, not a row."""
    if isinstance(block, LoopRegion):
        return "for" if counted(block) else "while"
    if isinstance(block, ConditionalBlock):
        return "if"
    if isinstance(block, ContinueBlock):
        return "continue"
    if isinstance(block, BreakBlock):
        return "break"
    if isinstance(block, ReturnBlock):
        return "return"
    if isinstance(block, SDFGState):
        return "state"
    return "block"


#: A row name :class:`Namer` hands out: its kind, then the kind's counter.
ROW_NAME = re.compile(r"^(?P<kind>[a-z_]+?)(?P<index>\d+)$")


@dataclass(slots=True)
class Namer:
    """The row names of one program: ``<kind><n>``, one counter per kind. A name stays on its row while the row
    lives and is never handed out again once :meth:`retire` d, so a name an agent read either still means that row
    or is refused as unknown."""

    counters: dict[str, int] = field(default_factory=dict)
    retired: set[str] = field(default_factory=set)

    def retire(self, names: tuple[str, ...]) -> None:
        self.retired.update(names)

    def name(self, kind: str, label: str, taken: set[str]) -> str:
        """``label`` when this namer gave it to a row of ``kind`` and no other row holds it yet, else a fresh name."""
        match = ROW_NAME.match(label)
        if (
            match is not None
            and match["kind"] == kind
            and int(match["index"]) < self.counters.get(kind, 0)
            and label not in self.retired
            and label not in taken
        ):
            taken.add(label)
            return label
        index = self.counters.get(kind, 0)
        self.counters[kind] = index + 1
        fresh = f"{kind}{index}"
        taken.add(fresh)
        return fresh


def normalize_labels(sdfg: dace.SDFG, namer: Namer | None = None) -> None:
    """Name every block ``<kind><n>``, every map ``doall<n>`` and every library call ``lib_<family><n>`` (``namer``,
    a fresh one by default), keeping the names ``namer`` gave before. A kernel NestForge built (``ExternalCall``)
    keeps its label unless an earlier one holds it."""
    namer = namer if namer is not None else Namer()
    relabel_cfg(sdfg, namer, set())
    unique_library_labels(sdfg)
    shorten_temporaries(sdfg, namer)


#: A container name DaCe made up (the frontend's ``__tmp3``, ``A_slice``, ``A_slice_plus_A_slice``, a fusion's
#: ``__map_fusion_T``, a reduction's ``_wcr_priv__Add____out``), as opposed to one the program itself chose.
GENERATED = re.compile(r"^_|__|_slice")
#: The kinds of the names :func:`shorten_temporaries` gives: ``tmp<n>`` for a scalar, ``Tmp<n>`` for an array.
SCALAR_TEMPORARY = "tmp"
ARRAY_TEMPORARY = "Tmp"


def scalar_like(desc: dace.data.Data) -> bool:
    """Whether ``desc`` holds one element: a scalar, or an array of one element."""
    return isinstance(desc, dace.data.Scalar) or all(str(extent) == "1" for extent in desc.shape)


def shorten_temporaries(sdfg: dace.SDFG, namer: Namer) -> None:
    """Rename every generated temporary in the program itself, so the tree, the kernel source and a move's
    ``container`` all say one short name: a scalar ``tmp<n>``, an array ``Tmp<n>``, and the frontend's read of an
    index array (``<program>_A_index``) ``A_index`` while no other container holds that name. A name is renamed in every program it is declared in, a nested
    program's connector of that name with it; an argument of the program keeps its name, and no name is handed out
    twice."""
    renames = temporary_renames(sdfg, namer)
    for node, state in list(sdfg.all_nodes_recursive()):
        if isinstance(node, nodes.NestedSDFG):
            rename_connectors(state, node, renames)
    for sd in list(sdfg.all_sdfgs_recursive()):
        own = {name: new for name, new in renames.items() if name in sd.arrays}
        if own:
            sd.replace_dict(own)
    unshadow_connectors(sdfg)


def index_array_of(name: str, arguments: set[str]) -> str | None:
    """The argument whose index array the frontend read into ``name`` (``<program>_A_index``), or ``None``."""
    return next((a for a in sorted(arguments) if name.endswith(f"_{a}_index") and name != f"{a}_index"), None)


def temporary_renames(sdfg: dace.SDFG, namer: Namer) -> dict[str, str]:
    """The short name :func:`shorten_temporaries` gives each generated temporary of ``sdfg``, by its name now."""
    declared: set[str] = set()
    shapes: dict[str, bool] = {}
    arguments = {name for name, desc in sdfg.arrays.items() if not desc.transient}
    for sd in sdfg.all_sdfgs_recursive():
        declared |= sd.arrays.keys() | sd.symbols.keys()
        for name, desc in sd.arrays.items():
            shapes[name] = shapes.get(name, True) and scalar_like(desc)
    renames: dict[str, str] = {}
    for name in sorted(declared - arguments - sdfg.symbols.keys()):
        if name not in shapes:
            continue
        index_of = index_array_of(name, arguments)
        if index_of is not None and f"{index_of}_index" not in declared:
            renames[name] = f"{index_of}_index"
            declared.add(renames[name])
            continue
        if index_of is None and not GENERATED.search(name):
            continue
        kind = SCALAR_TEMPORARY if shapes[name] else ARRAY_TEMPORARY
        fresh = namer.name(kind, "", set())
        while fresh in declared:
            fresh = namer.name(kind, "", set())
        renames[name] = fresh
    return renames


def unshadow_connectors(sdfg: dace.SDFG) -> None:
    """Give a nested program's connector the name of the container it is bound to when its own name is a different
    container of an enclosing program (after a fusion, a nest's ``T`` holds the temporary while the program's ``T``
    is an argument): one name, one container, in every view."""
    for node, state in list(sdfg.all_nodes_recursive()):
        if not isinstance(node, nodes.NestedSDFG):
            continue
        enclosing: set[str] = set()
        outer: dace.SDFG | None = state.sdfg
        while outer is not None:
            enclosing |= outer.arrays.keys()
            outer = outer.parent_sdfg
        bound: dict[str, set[str]] = {}
        for edge in state.in_edges(node):
            if edge.dst_conn and edge.data.data:
                bound.setdefault(edge.dst_conn, set()).add(edge.data.data)
        for edge in state.out_edges(node):
            if edge.src_conn and edge.data.data:
                bound.setdefault(edge.src_conn, set()).add(edge.data.data)
        inner = node.sdfg
        taken = inner.arrays.keys() | inner.symbols.keys()
        renames = {
            conn: next(iter(data))
            for conn, data in bound.items()
            if len(data) == 1 and conn in enclosing and conn not in data and next(iter(data)) not in taken
        }
        if renames:
            rename_connectors(state, node, renames)
            inner.replace_dict(renames)


def rename_connectors(state: SDFGState, node: nodes.NestedSDFG, renames: dict[str, str]) -> None:
    """The connectors of ``node`` named in ``renames``, and the edges that end at them, under their new names."""
    for old, new in renames.items():
        if old in node.in_connectors:
            node.add_in_connector(new, node.in_connectors[old], force=True)
            node.remove_in_connector(old)
            for edge in state.in_edges(node):
                if edge.dst_conn == old:
                    edge.dst_conn = new
        if old in node.out_connectors:
            node.add_out_connector(new, node.out_connectors[old], force=True)
            node.remove_out_connector(old)
            for edge in state.out_edges(node):
                if edge.src_conn == old:
                    edge.src_conn = new


def unique_library_labels(sdfg: dace.SDFG) -> None:
    """Rename each library node whose label an earlier one holds, nested SDFGs included."""
    taken: dict[str, None] = {}
    for node, _ in sdfg.all_nodes_recursive():
        if not isinstance(node, nodes.LibraryNode):
            continue
        if node.label in taken:
            # LibraryNode sets name and label alike; keep them equal. DaCe's Node stub declares label read-only,
            # though LibraryNode's label is a writable Property
            node.name = node.label = find_new_name(node.label, taken)  # pyright: ignore[reportAttributeAccessIssue]
        taken[node.label] = None


def library_family(node: nodes.CodeNode) -> str:
    """The DaCe library ``node`` comes from (``blas``, ``fft``, ``linalg``, ...); a standard node is named by its own
    kind (``reduce``, ``scan``, ``fill``, ``copy``), a one-element copy's tasklet ``copy`` too."""
    if is_copy_tasklet(node):
        return "copy"
    parts = type(node).__module__.split(".")
    package = parts[2] if parts[:2] == ["dace", "libraries"] and len(parts) > 2 else "lib"
    return type(node).__name__.lower().removesuffix("librarynode") if package == "standard" else package


def relabel_cfg(cfg: AbstractControlFlowRegion, namer: Namer, taken: set[str]) -> None:
    """Name one CFG's blocks, recursing into the regions and states among them."""
    for block in in_order(cfg):
        block.label = namer.name(block_kind(block), block.label, taken)
        if isinstance(block, SDFGState):
            relabel_state(block, namer, taken)
        elif isinstance(block, AbstractControlFlowRegion):  # a ConditionalBlock's nodes are its branches
            relabel_cfg(block, namer, taken)


#: Each scope's child nodes, keyed by its entry node, the top level by ``None``.
ScopeChildren = dict[nodes.Node | SDFGState | None, list[nodes.Node]]


def ordered_scope_children(state: SDFGState) -> ScopeChildren:
    """Each scope's children in execution order (:func:`in_order`), the order labels are given and printed in."""
    rank = {id(n): i for i, n in enumerate(in_order(state))}
    return {scope: sorted(kids, key=lambda n: rank[id(n)]) for scope, kids in state.scope_children().items()}


def relabel_state(state: SDFGState, namer: Namer, taken: set[str]) -> None:
    """Name every map of ``state`` ``doall<n>`` and every library call ``lib_<family><n>``, nested SDFGs included."""
    children = ordered_scope_children(state)

    def descend(scope: nodes.MapEntry | None) -> None:
        for node in children[scope]:
            if isinstance(node, nodes.MapEntry):
                node.map.label = namer.name("doall", node.map.label, taken)
                descend(node)
            elif isinstance(node, nodes.NestedSDFG):
                relabel_cfg(node.sdfg, namer, taken)
            elif isinstance(node, nodes.LibraryNode) and not isinstance(node, ExternalCall):
                # LibraryNode sets name and label alike; keep them equal (see unique_library_labels)
                node.name = node.label = namer.name(f"lib_{library_family(node)}", node.label, taken)  # pyright: ignore[reportAttributeAccessIssue]
            elif scope is None and is_copy_tasklet(node):
                node.label = namer.name("lib_copy", node.label, taken)

    descend(None)
