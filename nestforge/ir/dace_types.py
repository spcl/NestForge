# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Narrow DaCe properties whose annotations are looser than their values. The memlet accessors assert what the caller
relies on, so a surprise fails here instead of deep in emission; ``bounds`` and ``strings`` only restate the type."""

from __future__ import annotations

from typing import cast

import dace
import sympy
from dace import subsets

#: One dimension of a range: begin, inclusive end, step.
Bounds = tuple[sympy.Expr, sympy.Expr, sympy.Expr]


def memlet_data(memlet: dace.Memlet) -> str:
    """The container a non-empty memlet moves."""
    assert memlet.data is not None, "an empty memlet moves no data"
    return memlet.data


def memlet_subset(memlet: dace.Memlet) -> subsets.Range | None:
    """A memlet's subset as a ``Range``, or ``None`` when it has none."""
    subset = memlet.subset
    assert subset is None or isinstance(subset, subsets.Range), f"subset {subset!r} is not a Range"
    return subset


def memlet_range(memlet: dace.Memlet) -> subsets.Range:
    """The subset of a memlet that has one."""
    subset = memlet_subset(memlet)
    assert subset is not None, f"memlet {memlet} has no subset"
    return subset


def other_subset(memlet: dace.Memlet) -> subsets.Range | None:
    """A memlet's other subset as a ``Range``, or ``None``."""
    subset = memlet.other_subset
    assert subset is None or isinstance(subset, subsets.Range), f"other subset {subset!r} is not a Range"
    return subset


def bounds(subset: subsets.Range) -> list[Bounds]:
    # Range stores sympy expressions; its annotation also admits bare ints
    return cast(list[Bounds], subset.ranges)


def strings(values: object) -> list[str]:
    """A ``ListProperty(element_type=str)``, which DaCe annotates as a list of ``type[str]``."""
    return cast(list[str], values)
