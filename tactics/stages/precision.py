# Copyright 2021 ETH Zurich and the Tactics authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The ``precision`` family: ``set-fp-mode`` fixes the floating-point flags every kernel is built with, and
``change-precision`` stores one internal array in another floating-point type. Both change results within rounding,
so neither is proven: the reference checks the program at ``get_performance``, as it checks a written kernel."""

from collections.abc import Iterator

import dace
from dace.sdfg import nodes
from dace.sdfg.state import SDFGState

from tactics.build.flags import CUDA_FP, FP_LEVELS
from tactics.records import Refusal, Refused

#: The element types ``change-precision`` takes, by the name the agent writes.
DTYPES: dict[str, dace.typeclass] = {"float32": dace.float32, "float64": dace.float64}

#: What an applied precision move tells the agent: its result is tested, not proven.
PRECISION_WARNING = (
    "changes results within rounding: get_performance checks them against the reference; a wrong result is not kept"
)


def fp_modes(gpu: bool) -> tuple[str, ...]:
    """The modes ``set-fp-mode`` offers: those every device of the target builds (nvcc and hipcc have no fast-math
    mode matching the CPU's)."""
    return tuple(level for level in FP_LEVELS if not gpu or level in CUDA_FP)


def bindings(sdfg: dace.SDFG, name: str) -> Iterator[tuple[SDFGState, nodes.AccessNode]]:
    """Every access node of ``name`` in the states of ``sdfg`` (not of its nests)."""
    for state in sdfg.states():
        for node in state.data_nodes():
            if node.data == name:
                yield state, node


def use_refusal(sdfg: dace.SDFG, name: str) -> str:
    """Empty when only statements read and write ``name``, directly or through nests; else what else touches it.
    A copy to or from another array, or a library call, would read the bytes in the old type."""
    for state, node in bindings(sdfg, name):
        for edge in state.all_edges(node):
            path = state.memlet_path(edge)
            other = path[-1].dst if path[0].src is node else path[0].src
            if isinstance(other, nodes.Tasklet):
                continue
            if isinstance(other, nodes.NestedSDFG):
                inner = path[-1].dst_conn if path[-1].dst is other else path[0].src_conn
                reason = use_refusal(other.sdfg, inner)
                if reason:
                    return reason
                continue
            return f"{name} is copied to or read by {other.label}, not only by statements"
    return ""


def precision_refusal(sdfg: dace.SDFG, container: str | None, dtype: str | None) -> Refused | None:
    """Why ``change-precision`` cannot store ``container`` as ``dtype``, or ``None``."""
    if container is None or dtype not in DTYPES:
        return Refused(f"change-precision takes container and dtype, one of {list(DTYPES)}.", Refusal.ARGUMENT)
    desc = sdfg.arrays.get(container)
    if desc is None:
        return Refused(f"the program has no array {container!r}.", Refusal.ARGUMENT)
    if not desc.transient:
        return Refused(f"{container} is an argument: change its type with relayout cast.", Refusal.ARGUMENT)
    if desc.dtype not in DTYPES.values():
        return Refused(f"{container} holds {desc.dtype.to_string()}, not a floating-point type.", Refusal.ARGUMENT)
    if desc.dtype == DTYPES[dtype]:
        return Refused(f"{container} is stored as {dtype} already.", Refusal.ARGUMENT)
    reason = use_refusal(sdfg, container)
    return Refused(reason + ".", Refusal.ARGUMENT) if reason else None


def set_precision(sdfg: dace.SDFG, container: str, dtype: str) -> None:
    """Store ``container`` as ``dtype`` in ``sdfg`` and in every nest it is passed to, under its name there."""
    sdfg.arrays[container].dtype = DTYPES[dtype]
    for state, node in bindings(sdfg, container):
        for edge in state.all_edges(node):
            path = state.memlet_path(edge)
            for end, conn in ((path[-1].dst, path[-1].dst_conn), (path[0].src, path[0].src_conn)):
                if isinstance(end, nodes.NestedSDFG) and conn in end.sdfg.arrays:
                    if end.sdfg.arrays[conn].dtype != DTYPES[dtype]:
                        set_precision(end.sdfg, conn, dtype)


def precision_candidates(sdfg: dace.SDFG) -> list[str]:
    """The internal float64 arrays ``change-precision`` could store as float32, for ``list_moves``."""
    return [
        name
        for name, desc in sdfg.arrays.items()
        if desc.transient
        and isinstance(desc, dace.data.Array)
        and desc.dtype == dace.float64
        and not use_refusal(sdfg, name)
        and any(True for _ in bindings(sdfg, name))
    ]
