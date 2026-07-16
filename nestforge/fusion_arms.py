"""The Phase-2 agent's fusion tool surface: enumerate the legal fusion moves on an SDFG and apply one.

Three arms, all validated single-pair DaCe transformations:
  * ``fuse-loops``          -- two adjacent same-range sequential ``LoopRegion`` s (``FuseLoops``)
  * ``fuse-map-vertical``   -- a producer map feeding a consumer map through a transient (``MapFusionVertical``)
  * ``fuse-map-horizontal`` -- two sibling maps over the same range (``MapFusionHorizontal``)

Every move is legality-gated by the transformation's own ``can_be_applied_to``; the agent only ever sees
moves that are semantics-preserving. :func:`apply_fusion` commits one, re-verifying immediately before
apply (the map-fusion transforms assume a fresh ``can_be_applied`` and node references go stale after any
mutation -- so the agent loop is: enumerate -> apply one -> re-enumerate).

Fission (the other half of the Phase-2 lever) lives separately; this module is the fusion half.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Type

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.dataflow.map_fusion_horizontal import MapFusionHorizontal
from dace.transformation.dataflow.map_fusion_vertical import MapFusionVertical
from dace.transformation.interstate.fuse_loops import FuseLoops


@dataclass
class FusionMove:
    """One legal fusion the agent may apply. ``where`` maps the transformation's ``PatternNode`` names to
    the matched nodes (the ``apply_to`` / ``can_be_applied_to`` keyword arguments)."""
    kind: str
    where: Dict[str, nodes.Node]
    xform: Type = field(repr=False)

    def label(self) -> str:
        return f"{self.kind}({', '.join(str(n) for n in self.where.values())})"


def loop_fusion_moves(sdfg: dace.SDFG) -> List[FusionMove]:
    """Adjacent ``LoopRegion`` pairs FuseLoops accepts (single sequencing edge, same range, legal)."""
    moves: List[FusionMove] = []
    for cfg in sdfg.all_control_flow_regions(recursive=True):
        for first in cfg.nodes():
            if not isinstance(first, LoopRegion):
                continue
            out = cfg.out_edges(first)
            if len(out) != 1:
                continue
            second = out[0].dst
            if isinstance(second, LoopRegion) and second is not first and \
                    FuseLoops.can_be_applied_to(sdfg, first=first, second=second):
                moves.append(FusionMove("fuse-loops", {"first": first, "second": second}, FuseLoops))
    return moves


def vertical_map_moves(sdfg: dace.SDFG) -> List[FusionMove]:
    """Producer->consumer map pairs through a TRANSIENT intermediate that MapFusionVertical accepts. A
    non-transient intermediate is a real output -- fusing it away would drop a live result -- so only
    transients are offered."""
    moves: List[FusionMove] = []
    for state in sdfg.all_states():
        for node in state.nodes():
            if not (isinstance(node, nodes.AccessNode) and sdfg.arrays[node.data].transient):
                continue
            producers = [e.src for e in state.in_edges(node) if isinstance(e.src, nodes.MapExit)]
            consumers = [e.dst for e in state.out_edges(node) if isinstance(e.dst, nodes.MapEntry)]
            for mx in producers:
                for me in consumers:
                    if MapFusionVertical.can_be_applied_to(sdfg, first_map_exit=mx, array=node, second_map_entry=me):
                        moves.append(
                            FusionMove("fuse-map-vertical", {
                                "first_map_exit": mx,
                                "array": node,
                                "second_map_entry": me
                            }, MapFusionVertical))
    return moves


def horizontal_map_moves(sdfg: dace.SDFG) -> List[FusionMove]:
    """Sibling map pairs (same scope, parallel, same range) that MapFusionHorizontal accepts."""
    moves: List[FusionMove] = []
    for state in sdfg.all_states():
        scope = state.scope_dict()
        entries = [n for n in state.nodes() if isinstance(n, nodes.MapEntry)]
        for i, first in enumerate(entries):
            for second in entries[i + 1:]:
                if scope[first] is not scope[second]:
                    continue
                if MapFusionHorizontal.can_be_applied_to(sdfg,
                                                         first_parallel_map_entry=first,
                                                         second_parallel_map_entry=second):
                    moves.append(
                        FusionMove("fuse-map-horizontal", {
                            "first_parallel_map_entry": first,
                            "second_parallel_map_entry": second
                        }, MapFusionHorizontal))
    return moves


def enumerate_fusions(sdfg: dace.SDFG) -> List[FusionMove]:
    """Every legal fusion move on ``sdfg`` right now, across all three arms. The agent picks one, applies it
    (:func:`apply_fusion`), and re-enumerates -- applying a fusion invalidates the other moves' node
    references."""
    return loop_fusion_moves(sdfg) + vertical_map_moves(sdfg) + horizontal_map_moves(sdfg)


def apply_fusion(sdfg: dace.SDFG, move: FusionMove) -> None:
    """Commit one fusion move. Re-verifies (``verify=True``) immediately before applying -- the map-fusion
    transforms assume a fresh ``can_be_applied`` -- so pass only a move from a CURRENT
    :func:`enumerate_fusions` on this exact SDFG state."""
    move.xform.apply_to(sdfg, verify=True, annotate=False, save=False, **move.where)
