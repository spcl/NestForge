"""The normal form the agent's tree is projected from (:mod:`nestforge.normalize`).

Every fixture is built by the DaCe Python frontend rather than by hand: the properties under test --
frontend labels carrying source line numbers, nested SDFGs at the top level, statements landing
outside any map -- are things a frontend produces and a hand-built graph does not, so a hand-built
fixture would test the pass against a shape it never meets.
"""
import re

import numpy as np
import pytest

pytest.importorskip("dace")

import dace as dc

from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion

from nestforge.normalize import (WRAP_PARAM, block_kind, free_tasklets, in_order, inline_top_level_nsdfgs,
                                 normalize_for_tree, rename_map_params, rename_transient_data, top_level_nsdfgs,
                                 wrap_free_tasklets, wrap_groups)

LABEL = re.compile(r"^(state|for|while|if|block|continue|break|return)(\d+)_(\d+)$")
KERNEL = re.compile(r"^kernel(\d+)_(\d+)$")


@dc.program
def branchy(A: dc.float64[20], B: dc.float64[20], out: dc.float64[20]):
    """A loop around a conditional -- exercises for/if/state and free scalar tasklets in a branch."""
    for i in range(1, 20):
        if A[i] > 0.0:
            B[i] = A[i] * 2.0
        else:
            B[i] = -A[i]
    out[:] = B[:]


@dc.program
def two_maps(A: dc.float64[20], B: dc.float64[20]):
    """Two sibling map nests plus a scalar statement between them."""
    for i in dc.map[0:20]:
        B[i] = A[i] + 1.0
    s = B[0] * 2.0
    for i in dc.map[0:20]:
        B[i] = B[i] + s


@dc.program
def inner_body(A: dc.float64[20], B: dc.float64[20]):
    for i in dc.map[0:20]:
        for j in dc.map[0:4]:
            B[i] += A[i] * j


@dc.program
def nested_call(A: dc.float64[20], B: dc.float64[20]):
    """A call, so the frontend leaves a NestedSDFG at the top level."""
    inner_body(A, B)


@dc.program
def descending(A: dc.float64[20], B: dc.float64[20]):
    """A loop that counts DOWN -- must come out of normalization with a positive unit step."""
    for i in range(19, -1, -1):
        B[i] = A[i] * 2.0


def reaches(graph, source, target) -> bool:
    """Whether ``target`` is downstream of ``source`` -- dace's own ``bfs_nodes``, which a CFG and a
    state both provide."""
    return target in graph.bfs_nodes(source)


def all_blocks(sdfg):
    return [b for sd in sdfg.all_sdfgs_recursive() for b in sd.all_control_flow_blocks()]


def all_maps(sdfg):
    return [
        n for sd in sdfg.all_sdfgs_recursive() for st in sd.all_states() for n in st.nodes()
        if isinstance(n, nodes.MapEntry)
    ]


# --- labels ----------------------------------------------------------------------------------------


def test_every_block_and_map_gets_a_canonical_name():
    sdfg = branchy.to_sdfg(simplify=True)
    normalize_for_tree(sdfg)
    for block in all_blocks(sdfg):
        assert LABEL.match(block.label), f"block kept a non-canonical label {block.label!r}"
    for entry in all_maps(sdfg):
        assert KERNEL.match(entry.map.label), f"map kept a non-canonical label {entry.map.label!r}"


def test_labels_are_globally_unique():
    sdfg = branchy.to_sdfg(simplify=True)
    normalize_for_tree(sdfg)
    labels = [b.label for b in all_blocks(sdfg)]
    assert len(labels) == len(set(labels)), f"duplicate block label in {sorted(labels)}"
    kernels = [e.map.label for e in all_maps(sdfg)]
    assert len(kernels) == len(set(kernels)), f"duplicate kernel label in {sorted(kernels)}"


def test_index_is_typed_so_one_kind_numbers_from_zero_at_each_depth():
    """Five kernels at depth 3 are kernel3_0..kernel3_4 -- the index counts that KIND at that level, so
    it never depends on how many blocks of another kind happen to sit beside them."""
    sdfg = two_maps.to_sdfg(simplify=True)
    normalize_for_tree(sdfg)
    by_kind_level = {}
    for block in all_blocks(sdfg):
        kind, level, index = LABEL.match(block.label).groups()
        by_kind_level.setdefault((kind, int(level)), []).append(int(index))
    for entry in all_maps(sdfg):
        level, index = KERNEL.match(entry.map.label).groups()
        by_kind_level.setdefault(("kernel", int(level)), []).append(int(index))
    for key, indices in by_kind_level.items():
        assert sorted(indices) == list(range(len(indices))), f"{key} is numbered {sorted(indices)}"


def test_a_frontend_source_line_label_does_not_survive():
    """Frontend labels embed the source line (``inner_9_4``), so an edit ABOVE a nest renames a nest
    that did not change. That is the whole reason the tree cannot use them as ids."""
    sdfg = nested_call.to_sdfg(simplify=False)
    before = [e.map.label for e in all_maps(sdfg)]
    assert any(re.search(r"_\d+$", label) for label in before), f"fixture stopped being representative: {before}"
    normalize_for_tree(sdfg)
    assert all(KERNEL.match(e.map.label) for e in all_maps(sdfg))


def test_a_counted_loop_is_a_for_and_a_conditional_loop_is_a_while():
    sdfg = branchy.to_sdfg(simplify=True)
    loops = [b for b in all_blocks(sdfg) if isinstance(b, LoopRegion)]
    assert loops, "fixture no longer has a LoopRegion"
    for loop in loops:
        assert block_kind(loop) == "for"
        loop.update_statement = None
        assert block_kind(loop) == "while"


def test_two_builds_of_one_program_get_the_same_names():
    """Determinism is the point of the tie-break: a topological order alone is not unique, and two
    orders give two label assignments for one program -- so an agent's saved id would not survive a
    rebuild."""
    first, second = branchy.to_sdfg(simplify=True), branchy.to_sdfg(simplify=True)
    normalize_for_tree(first)
    normalize_for_tree(second)
    assert [b.label for b in all_blocks(first)] == [b.label for b in all_blocks(second)]
    assert [e.map.label for e in all_maps(first)] == [e.map.label for e in all_maps(second)]


def test_normalizing_twice_is_normalizing_once():
    """The agent re-normalizes after every fusion move, so a second pass must not renumber anything."""
    sdfg = branchy.to_sdfg(simplify=True)
    normalize_for_tree(sdfg)
    once = ([b.label for b in all_blocks(sdfg)], [e.map.label for e in all_maps(sdfg)])
    normalize_for_tree(sdfg)
    assert ([b.label for b in all_blocks(sdfg)], [e.map.label for e in all_maps(sdfg)]) == once


def test_in_order_breaks_ties_by_insertion_order():
    sdfg = two_maps.to_sdfg(simplify=True)
    blocks = sdfg.nodes()
    order = in_order(sdfg)
    assert sorted(id(b) for b in order) == sorted(id(b) for b in blocks), "in_order dropped or added a block"
    # Independent blocks (no path between them) must come out in the order they were inserted.
    positions = {id(b): i for i, b in enumerate(order)}
    inserted = {id(b): i for i, b in enumerate(blocks)}
    for first in blocks:
        for second in blocks:
            if first is second or reaches(sdfg, first, second) or reaches(sdfg, second, first):
                continue
            assert (positions[id(first)] < positions[id(second)]) == (inserted[id(first)] < inserted[id(second)])


# --- every computation inside a map ----------------------------------------------------------------


def test_no_free_tasklet_survives():
    sdfg = two_maps.to_sdfg(simplify=True)
    assert any(free_tasklets(st) for st in sdfg.all_states()), "fixture has no free tasklet to wrap"
    normalize_for_tree(sdfg)
    assert not any(free_tasklets(st) for st in sdfg.all_states())


def test_a_wrap_map_is_one_iteration_and_sequential():
    sdfg = two_maps.to_sdfg(simplify=True)
    normalize_for_tree(sdfg)
    wraps = [e for e in all_maps(sdfg) if WRAP_PARAM in e.map.params]
    assert wraps, "nothing was wrapped"
    for entry in wraps:
        assert entry.map.range.num_elements() == 1
        assert entry.map.schedule == dc.ScheduleType.Sequential


def test_a_library_node_is_not_wrapped():
    """A library node already IS a kernel. Wrapping it would bury it under a map scope and hide that,
    and its expansion picks its own schedule."""

    @dc.program
    def gemm(A: dc.float64[8, 8], B: dc.float64[8, 8], C: dc.float64[8, 8]):
        C[:] = A @ B

    sdfg = gemm.to_sdfg(simplify=True)
    libnodes = [n for st in sdfg.all_states() for n in st.nodes() if isinstance(n, nodes.LibraryNode)]
    assert libnodes, "fixture no longer produces a library node"
    normalize_for_tree(sdfg)
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, nodes.LibraryNode):
                assert state.entry_node(node) is None, f"{node.label} was buried in a map scope"


def test_grouping_uses_the_fewest_maps_the_dependencies_allow():
    """A group must be an antichain -- merging two tasklets one can reach from the other closes a cycle
    through whatever sits between them. The minimum number of such groups is the longest chain
    (Mirsky), which is what the levelling produces."""
    sdfg = two_maps.to_sdfg(simplify=True)
    for state in sdfg.all_states():
        groups = wrap_groups(state)
        if not groups:
            continue
        placed = [t for group in groups for t in group]
        assert sorted(id(t) for t in placed) == sorted(id(t) for t in free_tasklets(state))
        for group in groups:  # no member reaches another
            for first in group:
                for second in group:
                    assert first is second or not reaches(state, first, second)
        longest = longest_free_chain(state)
        assert len(groups) == longest, f"{len(groups)} groups for a longest chain of {longest}"


def longest_free_chain(state) -> int:
    """The most free tasklets on any one dependency path through ``state`` -- the lower bound on how
    many maps the wrap can use."""
    free = {id(t) for t in free_tasklets(state)}
    best = {}
    longest = 0
    for node in in_order(state):
        reaching = max((best[id(e.src)] for e in state.in_edges(node) if id(e.src) in best), default=0)
        best[id(node)] = reaching + 1 if id(node) in free else reaching
        longest = max(longest, best[id(node)])
    return longest


def test_wrapping_nothing_changes_nothing():
    """A pass that does not apply must not mutate."""
    sdfg = nested_call.to_sdfg(simplify=False)
    inline_top_level_nsdfgs(sdfg)
    wrap_free_tasklets(sdfg)
    before = sdfg.to_json()
    assert wrap_free_tasklets(sdfg) == 0
    assert sdfg.to_json() == before


# --- no top-level nested SDFG ----------------------------------------------------------------------


def test_a_top_level_nested_sdfg_is_inlined():
    """Its states are real control flow; left nested they are an opaque box the agent cannot fuse
    across."""
    sdfg = nested_call.to_sdfg(simplify=False)
    assert top_level_nsdfgs(sdfg), "fixture no longer has a top-level nested SDFG"
    normalize_for_tree(sdfg)
    assert not top_level_nsdfgs(sdfg)


def test_a_nested_sdfg_inside_a_map_is_left_alone():
    """That one is a kernel body, not structure."""
    sdfg = nested_call.to_sdfg(simplify=False)
    normalize_for_tree(sdfg)
    inside = [(st, n) for st in sdfg.all_states() for n in st.nodes()
              if isinstance(n, nodes.NestedSDFG) and st.entry_node(n) is not None]
    assert inside, "fixture no longer has a nested SDFG inside a map"


def test_inlining_bails_out_without_touching_an_already_inlined_sdfg():
    sdfg = nested_call.to_sdfg(simplify=False)
    normalize_for_tree(sdfg)
    before = sdfg.to_json()
    assert inline_top_level_nsdfgs(sdfg) == 0
    assert sdfg.to_json() == before


# --- canonical iteration domains -------------------------------------------------------------------


def test_a_descending_loop_comes_out_with_a_positive_unit_step():
    sdfg = descending.to_sdfg(simplify=True)
    normalize_for_tree(sdfg)
    for loop in [b for b in all_blocks(sdfg) if isinstance(loop_or_none(b), LoopRegion)]:
        assert "-1" not in loop.update_statement.as_string, f"step stayed negative: {loop.update_statement.as_string}"


def loop_or_none(block):
    return block if isinstance(block, LoopRegion) else None


# --- the numerics ----------------------------------------------------------------------------------


@pytest.mark.e2e
def test_normalization_preserves_the_result_through_a_branch_and_a_loop():
    sdfg = branchy.to_sdfg(simplify=True)
    normalize_for_tree(sdfg)
    sdfg.validate()
    A = np.linspace(-1.0, 1.0, 20).copy()
    B, out = np.zeros(20), np.zeros(20)
    sdfg(A=A, B=B, out=out)
    expected = np.zeros(20)
    for i in range(1, 20):
        expected[i] = A[i] * 2.0 if A[i] > 0.0 else -A[i]
    assert np.array_equal(out, expected)


@pytest.mark.e2e
def test_normalization_preserves_the_result_through_an_inlined_nested_sdfg():
    sdfg = nested_call.to_sdfg(simplify=False)
    normalize_for_tree(sdfg)
    sdfg.validate()
    A = np.linspace(0.5, 2.0, 20).copy()
    B = np.zeros(20)
    sdfg(A=A, B=B)
    expected = np.zeros(20)
    for i in range(20):
        for j in range(4):
            expected[i] += A[i] * j
    assert np.array_equal(B, expected)


@pytest.mark.e2e
def test_normalization_preserves_the_result_of_a_descending_loop():
    sdfg = descending.to_sdfg(simplify=True)
    normalize_for_tree(sdfg)
    sdfg.validate()
    A = np.linspace(1.0, 5.0, 20).copy()
    B = np.zeros(20)
    sdfg(A=A, B=B)
    assert np.array_equal(B, A * 2.0)


# --- canonical data names --------------------------------------------------------------------------


def test_transient_data_is_renamed_but_the_interface_is_not():
    """A frontend qualifies a transient with its whole module path, which is most of the width of a
    tree line and names nothing the agent can act on. Non-transients are the program's interface --
    the boundary, the manifest and the emitted numpy signature all name them -- so they stay."""
    sdfg = nested_call.to_sdfg(simplify=False)
    interface = {n for n, desc in sdfg.arrays.items() if not desc.transient}
    normalize_for_tree(sdfg)
    assert interface <= set(sdfg.arrays), "a non-transient was renamed"
    transients = {n: d for n, d in sdfg.arrays.items() if d.transient}
    for name, desc in transients.items():
        prefix = "s" if isinstance(desc, dc.data.Scalar) else "t"
        assert re.fullmatch(rf"{prefix}\d+", name), f"uncanonical transient {name} ({type(desc).__name__})"


def test_renaming_never_collides_with_a_name_that_survives():
    """A survivor already called ``t0`` would be clobbered by whatever is renamed to ``t0``."""

    @dc.program
    def collide(t0: dc.float64[20], B: dc.float64[20]):
        tmp = t0 * 2.0
        B[:] = tmp + 1.0

    sdfg = collide.to_sdfg(simplify=True)
    normalize_for_tree(sdfg)
    assert "t0" in sdfg.arrays and not sdfg.arrays["t0"].transient, "the interface array t0 was overwritten"


def test_map_params_are_canonical():
    sdfg = branchy.to_sdfg(simplify=True)
    normalize_for_tree(sdfg)
    for entry in all_maps(sdfg):
        if WRAP_PARAM in entry.map.params:
            continue
        assert entry.map.params == [f"i{axis}" for axis in range(len(entry.map.params))]


def test_renaming_nothing_changes_nothing():
    sdfg = branchy.to_sdfg(simplify=True)
    normalize_for_tree(sdfg)
    before = sdfg.to_json()
    assert rename_transient_data(sdfg) == {}
    rename_map_params(sdfg)
    assert sdfg.to_json() == before


@pytest.mark.e2e
def test_renaming_preserves_the_result():
    """The renames touch every memlet that names a transient, so this is the check that matters."""
    sdfg = two_maps.to_sdfg(simplify=True)
    A = np.linspace(1.0, 3.0, 20).copy()
    expected = A + 1.0
    expected = expected + expected[0] * 2.0
    normalize_for_tree(sdfg)
    sdfg.validate()
    got = np.zeros(20)
    sdfg(A=A, B=got)
    assert np.array_equal(got, expected)
