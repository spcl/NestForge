# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The ``expand-library`` move: a library node becomes the maps and tasklets of DaCe's vendor-free expansion, in
canonical parallel form, so the other moves reach inside it."""

from __future__ import annotations

import dace
from dace import cpf_lowering
from dace.codegen import cpf
from dace.codegen.common import get_gpu_backend
from dace.libraries.standard.nodes.external_call import ExternalCall
from dace.sdfg import nodes
from dace.sdfg.state import SDFGState

from nestforge.stages.canonicalize import Targets, canonicalize

#: The expansions free of a vendor call, in the order tried: maps and tasklets, else a specialization into other
#: library nodes (a matmul into a gemm), which are expanded in turn.
EXPANSIONS = ("pure", "specialize")


def expansion(node: nodes.LibraryNode) -> str | None:
    """The implementation ``node`` expands with, or ``None`` when every one it has calls a vendor library. A kernel
    NestForge built (``ExternalCall``) never expands."""
    if isinstance(node, ExternalCall):
        return None
    return next((name for name in EXPANSIONS if name in node.implementations), None)


def implementations(node: nodes.LibraryNode, gpu: bool) -> tuple[str, ...]:
    """The implementations ``node`` may be set to: the ones a kernel unit renders (loops, the host vendor library,
    with a GPU target the device library of this machine's GPU), besides the one already chosen."""
    name = type(node).__name__
    offered = (
        *cpf.VENDOR_IMPLEMENTATIONS_HOST,
        *cpf.RENDERABLE_BY_NODE.get(name, ()),
        *cpf.RENDERABLE_IMPLEMENTATIONS,
        *(device_implementations(name) if gpu else ()),
    )
    chosen = node.implementation if not node.auto_select_implementation else None
    return tuple(dict.fromkeys(i for i in offered if i in node.implementations and i != chosen))


def device_implementations(name: str) -> tuple[str, ...]:
    """The device library calls and device expansions a GPU kernel of library node type ``name`` renders, in the
    dialect of this machine's GPU."""
    dialect = (
        cpf_lowering.Dialect.STANDALONE_HIP if get_gpu_backend() == "hip" else cpf_lowering.Dialect.STANDALONE_CUDA
    )
    return (*cpf.VENDOR_IMPLEMENTATIONS_DEVICE[dialect], *cpf.RENDERABLE_BY_NODE_DEVICE.get(name, ()))


def is_device_implementation(implementation: str) -> bool:
    """Whether ``implementation`` is a device library call, so its kernel runs on the GPU."""
    return any(implementation in names for names in cpf.VENDOR_IMPLEMENTATIONS_DEVICE.values())


def is_host_implementation(implementation: str) -> bool:
    """Whether ``implementation`` is a host library call, so its kernel runs on the CPU."""
    return implementation in cpf.VENDOR_IMPLEMENTATIONS_HOST


def set_implementation(node: nodes.LibraryNode, implementation: str) -> None:
    """Pin ``node`` to ``implementation``: no automatic selection (finalization, the kernel unit) overrides it."""
    node.implementation = implementation
    node.auto_select_implementation = False


def expand_library(state: SDFGState, node: nodes.LibraryNode) -> list[nodes.NestedSDFG]:
    """Expand ``node`` and every library node its expansion makes; returns the nested SDFGs it left in ``state``."""
    implementation = expansion(node)
    if implementation is None:
        raise ValueError(f"{node.label} ({type(node).__name__}) has no vendor-free expansion")
    before = {id(n) for n in state.nodes()}
    node.expand(state, implementation)
    made: list[nodes.NestedSDFG] = []
    for new in [n for n in state.nodes() if id(n) not in before]:
        if isinstance(new, nodes.LibraryNode) and expansion(new) is not None:
            made += expand_library(state, new)
        elif isinstance(new, nodes.NestedSDFG):
            made.append(new)
            expand_within(new.sdfg)
    return made


def expand_within(sdfg: dace.SDFG) -> None:
    """Expand every library node in ``sdfg`` that has a vendor-free expansion, nested SDFGs included."""
    for node, parent in list(sdfg.all_nodes_recursive()):
        if isinstance(node, nodes.LibraryNode) and expansion(node) is not None:
            expand_library(parent, node)


def settle(made: list[nodes.NestedSDFG], targets: Targets) -> None:
    """Canonicalize each nested SDFG an expansion made, then expand again what canonicalization lifts back into a
    library node (a gemm's maps into an einsum, a zero fill into a memset)."""
    for nsdfg in made:
        canonicalize(nsdfg.sdfg, targets)
        expand_within(nsdfg.sdfg)
