# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Extract EVERY compute nest a detection strategy finds, each from a FRESH SDFG.

A kernel may split into several loop-nests. :func:`nestforge.extract.extract_nest_to_sdfg` mutates the
parent SDFG in place, so a ``(parent, node)`` ref captured from one build goes stale after the first
extraction. This helper rebuilds a fresh SDFG and re-runs the (deterministic) strategy once per nest, so
``refs_i`` aligns positionally with the initial ``refs`` and the idx-th nest is extracted from its own
untouched copy. The three "secondary" perf drivers (tsvc_arena / crosslang_xl / calloverhead) use it to
measure multi-nest kernels; :mod:`nestforge.perf.tsvc_full` has its own equivalent (``build_opt_context``).
"""
from __future__ import annotations

import copy
from typing import Callable, List, Tuple

import dace

from nestforge.extract import Boundary, extract_nest_to_sdfg
from nestforge.strategies import get_strategy


def extract_all_nests(build_fn: Callable[[], dace.SDFG], strategy_name: str,
                      key: str) -> List[Tuple[int, str, str, Boundary]]:
    """Extract every nest ``strategy_name`` finds, each from a FRESH SDFG.

    ``build_fn()`` must return a NEW SDFG on every call (each extraction mutates its parent in place).
    Returns ``[(idx, name, symbol, boundary), ...]``: for a SINGLE-nest kernel ``name == key`` and
    ``symbol == f"{key}_fp64"`` (identical to the old single-nest path, so those kernels' emitted symbol
    and existing tests are unchanged); for a MULTI-nest kernel ``name == f"{key}_n{idx}"`` and
    ``symbol == f"{name}_fp64"`` so each nest binds a distinct entry point. An empty list means the
    strategy found no compute nest (the caller skips the kernel).
    """
    strategy = get_strategy(strategy_name)
    # Build ONCE and deepcopy per nest. ``build_fn`` runs the DaCe frontend plus canonicalize, which
    # measured 70-2368 ms against 1.9-37.8 ms to copy the result -- 30-90x typically, up to 1175x --
    # and it was called once per nest plus once more just to count them. The contract that each
    # extraction gets a NEW SDFG is what a deepcopy provides; experiment_e1 already does exactly this
    # substitution for the same reason.
    master = build_fn()
    refs = strategy(master)
    single = len(refs) == 1
    out: List[Tuple[int, str, str, Boundary]] = []
    for idx in range(len(refs)):
        refs_i = strategy(copy.deepcopy(master))  # fresh SDFG per nest: extraction below mutates it
        parent, node = refs_i[idx]
        name = key if single else f"{key}_n{idx}"
        boundary = extract_nest_to_sdfg(parent, node, name=name)
        out.append((idx, name, f"{name}_fp64", boundary))
    return out
