"""Pluggable detection strategies.

A strategy is ``Callable[[SDFG], List[Tuple[SDFG, NestNode]]]`` returning the nests to extract,
each paired with the **parent SDFG** it lives in (nested SDFGs are recursive, so the lowering
pass must know which SDFG to operate on). ``outer`` is the default. New strategies register via
:func:`register_strategy` and resolve by name with :func:`get_strategy`.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion

from nestforge.extract import NestNode

Strategy = Callable[[dace.SDFG], List[Tuple[dace.SDFG, NestNode]]]

_REGISTRY: Dict[str, Strategy] = {}


def register_strategy(name: str, fn: Strategy) -> None:
    _REGISTRY[name] = fn


def get_strategy(name: str) -> Strategy:
    if name not in _REGISTRY:
        raise KeyError(f"unknown strategy {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def _top_level_map_entries(state: dace.SDFGState) -> List[nodes.MapEntry]:
    """MapEntry nodes at the top of a state's scope tree (not nested inside another map)."""
    return [n for n in state.scope_children()[None] if isinstance(n, nodes.MapEntry)]


def outer(sdfg: dace.SDFG) -> List[Tuple[dace.SDFG, NestNode]]:
    """Outermost nests of the root SDFG: top-level map-nests + top-level CFG loop regions.

    Does not descend into nested SDFGs (those are already 'inside'); other strategies may.
    """
    refs: List[Tuple[dace.SDFG, NestNode]] = []
    for block in sdfg.nodes():
        if isinstance(block, LoopRegion):
            refs.append((sdfg, block))
        elif isinstance(block, dace.SDFGState):
            for me in _top_level_map_entries(block):
                refs.append((sdfg, me))
    return refs


register_strategy("outer", outer)
