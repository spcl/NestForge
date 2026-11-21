# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The host/device copies placement reports (:func:`device_copies`) on hand-built programs: an empty memlet is no copy,
and a copy inside a nested SDFG or made by a ``CopyLibraryNode`` is one. Placing a real program is
``test_gpu_placement_copies``; nothing here is built or patched."""

import dace
import pytest
from dace.libraries.standard.nodes.copy import CopyLibraryNode

from nestforge.stages.placement import device_copies

N = dace.symbol("N", dtype=dace.int64)


def host_and_device(name: str, storage: dace.StorageType = dace.StorageType.GPU_Global) -> dace.SDFG:
    """An SDFG with a host array ``a`` and a transient ``a_dev`` in ``storage``."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("a_dev", [N], dace.float64, storage=storage, transient=True)
    return sdfg


def test_an_empty_memlet_between_memory_spaces_is_no_copy() -> None:
    """An empty memlet only orders its two accesses: the GPU default of a map feeding a recurrence has one, which
    ``finalize_scope`` answered as a copy (and counting its bytes failed every measurement, job 5372503)."""
    sdfg = host_and_device("ordered_only")
    state = sdfg.add_state()
    state.add_nedge(state.add_access("a"), state.add_access("a_dev"), dace.Memlet())
    copied = sdfg.add_state_after(state)
    copied.add_nedge(copied.add_access("a"), copied.add_access("a_dev"), dace.Memlet("a[0:N]"))

    assert device_copies(sdfg) == [("a", "a_dev")], "only the second state's edge moves data"


def test_a_copy_inside_a_nested_sdfg_is_a_copy_of_the_program() -> None:
    inner = host_and_device("inner")
    inner_state = inner.add_state()
    inner_state.add_nedge(inner_state.add_access("a"), inner_state.add_access("a_dev"), dace.Memlet("a[0:N]"))
    outer = dace.SDFG("outer")
    outer.add_array("x", [N], dace.float64)
    state = outer.add_state()
    nested = state.add_nested_sdfg(inner, {"a"}, set(), symbol_mapping={"N": "N"})
    state.add_edge(state.add_access("x"), None, nested, "a", dace.Memlet("x[0:N]"))

    assert device_copies(outer) == [("a", "a_dev")]


def copy_node_program(storage: dace.StorageType) -> dace.SDFG:
    """``a`` copied into ``a_dev`` (in ``storage``) by a ``CopyLibraryNode``: AccessNode -> CopyLibraryNode ->
    AccessNode, with no access-to-access edge."""
    sdfg = host_and_device(f"copy_node_{storage.name}", storage)
    state = sdfg.add_state()
    node = CopyLibraryNode("copy_a")
    state.add_node(node)
    state.add_edge(state.add_access("a"), None, node, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.Memlet("a[0:N]"))
    state.add_edge(
        node, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, state.add_access("a_dev"), None, dace.Memlet("a_dev[0:N]")
    )
    return sdfg


@pytest.mark.parametrize(
    ("storage", "expected"),
    [(dace.StorageType.GPU_Global, [("a", "a_dev")]), (dace.StorageType.CPU_Heap, [])],
    ids=["host-to-device", "host-to-host"],
)
def test_a_copy_library_node_is_a_copy_when_it_crosses_memory_spaces(
    storage: dace.StorageType, expected: list[tuple[str, str]]
) -> None:
    assert device_copies(copy_node_program(storage)) == expected
