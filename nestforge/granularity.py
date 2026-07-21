"""Fusion granularity as a search axis (paper Axis 1) -- Phase 1 of the 4-phase optimizer.

Place in the pipeline:

  * **P0 (given, not searched):** canonicalize the kernel to a deterministic normal form -- the SAME
    start point for every kernel, backend, and optimizer (``tsvc.build_sdfg(kernel, "canonicalize")``,
    the statement-level normal form). Canonicalization is a separate paper; here it only fixes the start.
  * **P1 (this module):** from the canonical start, choose a granularity -- a partition of the canonical
    statement-atoms into kernels -- by applying dace-gated fusion moves. The lattice runs between two
    endpoints:
      - ``atoms``   -- finest: the canonical statement-atoms themselves (no re-fusion).
      - ``maximal`` -- coarsest: fuse everything legal (what a compiler picks blindly).
  * **P2:** the chosen partition's nests are externalized (``offload``); **P3** optimizes each nest;
    **P4** feeds measurements back to re-choose the P1 partition.

The claim (C1) is that the performance-optimal partition is *between* the endpoints and
*backend-dependent*, so it is measured, not fixed. A :class:`GranularityPoint` mutates any SDFG to a
chosen partition by first normalizing to the canonical atoms (:func:`to_canonical_atoms`, idempotent
when the SDFG is already the P0 canonical form), then applying a fixed number of fusion moves -- so the
start point is identical no matter the incoming granularity. This module enumerates a BOUNDED, named
ladder from atoms to maximal for the sweep; the traditional/agentic search explores the same move space
to find the measured best between rungs.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, List

import dace
from dace.transformation.passes.canonicalize.normalize_floor_division import NormalizeFloorDivision
from dace.sdfg.state import LoopRegion

from nestforge.fission_arms import fission_to_statements
from nestforge.fusion import apply_fusion, enumerate_fusions
from nestforge.strategies import top_level_map_entries


def to_canonical_atoms(sdfg: dace.SDFG) -> None:
    """Normalize ``sdfg`` in place to the canonical statement-atoms -- the P0 start point every P1 ladder
    is measured from. Idempotent: when ``sdfg`` is already the canonical (statement-level) form, this
    leaves it unchanged; otherwise it fissions to that same finest partition, so the ladder's base is
    identical regardless of how the SDFG was built."""
    fission_to_statements(sdfg)


@dataclass
class GranularityPoint:
    """One partition on the atoms->maximal lattice: a name plus a policy that mutates an SDFG to it.

    ``apply`` is idempotent w.r.t. the STARTING granularity -- it first normalizes to the canonical atoms
    (:func:`to_canonical_atoms`), then re-fuses a fixed number of moves -- so the same point is reached
    regardless of the SDFG's incoming granularity."""
    name: str
    apply: Callable[[dace.SDFG], None]


def count_nests(sdfg: dace.SDFG) -> int:
    """How many nests (top-level map-nests + loop-nests) the SDFG currently holds -- the coarseness of its
    partition. Atoms give the most; maximal fusion the fewest."""
    loops = sum(1 for cfg in sdfg.all_control_flow_regions(recursive=True) for n in cfg.nodes()
                if isinstance(n, LoopRegion))
    maps = sum(len(top_level_map_entries(state)) for state in sdfg.all_states())
    return loops + maps


def fuse_first_k(k: int) -> Callable[[dace.SDFG], None]:
    """A policy: normalize to the canonical atoms, then apply the first ``k`` legal fusion moves greedily
    (re-enumerating after each, since applying one stales the rest). ``k=0`` is atoms; large ``k``
    saturates at maximal."""

    def apply(sdfg: dace.SDFG) -> None:
        to_canonical_atoms(sdfg)
        for _ in range(k):
            moves = enumerate_fusions(sdfg)
            if not moves:
                break
            apply_fusion(sdfg, moves[0])
        # Forced: fission/fusion build indices with python `//` on sympy expressions, which is
        # sympy floor() -- distributed by sympy and printed without the floor, so the index
        # truncates term by term. Every rung is normalized before anything measures it.
        NormalizeFloorDivision().apply_pass(sdfg, {})

    return apply


def fusion_depth(sdfg: dace.SDFG) -> int:
    """The number of greedy fusion moves from atoms to the maximal (fixed-point) partition -- the height of
    the ladder for this SDFG. Computed on a copy, so ``sdfg`` is untouched."""
    probe = copy.deepcopy(sdfg)
    to_canonical_atoms(probe)
    depth = 0
    while True:
        moves = enumerate_fusions(probe)
        if not moves:
            return depth
        apply_fusion(probe, moves[0])
        depth += 1


def granularity_ladder(sdfg: dace.SDFG, max_points: int = 0) -> List[GranularityPoint]:
    """The named ladder of granularity points from ``atoms`` to ``maximal`` for ``sdfg``.

    Always includes both endpoints. Intermediate rungs are the partitions after ``k`` greedy fusions
    (``k = 1 .. depth-1``). ``max_points`` (>0) evenly subsamples the ladder to at most that many points
    -- keeping the endpoints -- so the sweep stays bounded on deep lattices; ``0`` keeps every rung.
    """
    depth = fusion_depth(sdfg)
    if depth == 0:  # nothing fuses: atoms IS maximal, one point
        return [GranularityPoint("atoms", fuse_first_k(0))]
    if max_points == 1:
        # a one-point budget cannot hold both endpoints; take the coarsest (what a compiler picks blindly),
        # so the single rung is the meaningful comparison point. Special-cased because the even-subsample
        # below divides by (max_points - 1).
        return [GranularityPoint(name_for(depth, depth), fuse_first_k(depth))]
    ks = list(range(depth + 1))  # 0 (atoms) .. depth (maximal)
    if max_points and max_points < len(ks):
        ks = sorted({round(i * depth / (max_points - 1)) for i in range(max_points)})
    return [GranularityPoint(name_for(k, depth), fuse_first_k(k)) for k in ks]


def name_for(k: int, depth: int) -> str:
    if k == 0:
        return "atoms"
    if k >= depth:
        return "maximal"
    return f"fuse-{k}"
