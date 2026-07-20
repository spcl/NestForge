"""Region-structure moves: merge the control-flow CONTAINERS, the level above nest fusion.

A nest (map/loop) fuses only with a sibling in the SAME region. Two maps in different States, or two
loops under different parent regions, are separated by a control-flow dependency and cannot fuse until
their enclosing regions merge. That merge is a distinct decision from fusing the nests themselves, so it
is a distinct API:

  * MAP barrier -- two adjacent ``SDFGState`` s. Merged by ``StateFusion`` (:func:`state_fusion_moves`).
    After the merge the maps that were in separate states are siblings and :mod:`nestforge.fusion_arms`
    can fuse them.
  * LOOP barrier -- two ``LoopRegion`` s under different parents. Merged by fusing the ENCLOSING loops,
    which is already a ``fuse-loops`` move in :func:`nestforge.fusion_arms.enumerate_fusions` (that
    enumerator recurses every control-flow region). So loop-region fusion needs no new transform -- fuse
    the outer loops first, then the inner ones.

So this module surfaces exactly the one primitive nest fusion cannot reach on its own: ``StateFusion``.
Every move is legality-gated by ``StateFusion.can_be_applied_to``, same contract as the fusion arms.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Type

import dace
from dace.sdfg.state import SDFGState
from dace.transformation.interstate.state_fusion import StateFusion


@dataclass
class RegionMove:
    """One legal region merge. ``where`` maps the transformation's ``PatternNode`` names to the matched
    control-flow blocks."""
    kind: str
    where: Dict[str, object]
    xform: Type = field(repr=False)

    def label(self) -> str:
        return f"{self.kind}({', '.join(str(b) for b in self.where.values())})"


def state_fusion_moves(sdfg: dace.SDFG) -> List[RegionMove]:
    """Adjacent ``SDFGState`` pairs ``StateFusion`` accepts -- the merge that dissolves the map barrier so
    cross-state maps become fusable. Enumerated across every control-flow region (recursive), mirroring
    :func:`nestforge.fusion_arms.enumerate_fusions`."""
    moves: List[RegionMove] = []
    for cfg in sdfg.all_control_flow_regions(recursive=True):
        for edge in cfg.edges():
            first, second = edge.src, edge.dst
            if isinstance(first, SDFGState) and isinstance(second, SDFGState) and first is not second and \
                    StateFusion.can_be_applied_to(sdfg, first_state=first, second_state=second):
                moves.append(RegionMove("fuse-states", {"first_state": first, "second_state": second}, StateFusion))
    return moves


def enumerate_region_fusions(sdfg: dace.SDFG) -> List[RegionMove]:
    """Every legal region merge right now. Today that is the state-fusion set; loop-region merges ride the
    fusion arms (fuse the enclosing loops). Applying one stales the rest -- re-enumerate after each."""
    return state_fusion_moves(sdfg)


def apply_region_fusion(sdfg: dace.SDFG, move: RegionMove) -> None:
    """Commit one region merge (from a CURRENT :func:`enumerate_region_fusions`). Re-verifies legality
    before applying, same as :func:`nestforge.fusion_arms.apply_fusion`."""
    move.xform.apply_to(sdfg, verify=True, annotate=False, save=False, **move.where)
