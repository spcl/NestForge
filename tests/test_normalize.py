# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The normal form the agent's tree is projected from (:mod:`nestforge.ir.names`).

Every fixture is built by the DaCe Python frontend rather than by hand: the properties under test --
frontend labels carrying source line numbers, nested SDFGs at the top level, statements landing
outside any map -- are things a frontend produces and a hand-built graph does not, so a hand-built
fixture would test the pass against a shape it never meets.
"""

import re

import dace as dc
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion

from nestforge.ir.names import Namer, block_kind, in_order, inline_top_level_nsdfgs, normalize_labels, top_level_nsdfgs

LABEL = re.compile(r"^(state|for|if|block|continue|break|return)(\d+)$")
KERNEL = re.compile(r"^doall(\d+)$")


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


def reaches(graph, source, target) -> bool:
    """Whether ``target`` is downstream of ``source`` -- dace's own ``bfs_nodes``, which a CFG and a
    state both provide."""
    return target in graph.bfs_nodes(source)


def all_blocks(sdfg):
    return [b for sd in sdfg.all_sdfgs_recursive() for b in sd.all_control_flow_blocks()]


def all_maps(sdfg):
    return [
        n
        for sd in sdfg.all_sdfgs_recursive()
        for st in sd.states()
        for n in st.nodes()
        if isinstance(n, nodes.MapEntry)
    ]


# labels


def test_every_block_and_map_gets_a_canonical_name():
    sdfg = branchy.to_sdfg(simplify=True)
    normalize_labels(sdfg)
    for block in all_blocks(sdfg):
        assert LABEL.match(block.label), f"block kept a non-canonical label {block.label!r}"
    for entry in all_maps(sdfg):
        assert KERNEL.match(entry.map.label), f"map kept a non-canonical label {entry.map.label!r}"


def test_labels_are_globally_unique():
    sdfg = branchy.to_sdfg(simplify=True)
    normalize_labels(sdfg)
    labels = [b.label for b in all_blocks(sdfg)]
    assert len(labels) == len(set(labels)), f"duplicate block label in {sorted(labels)}"
    kernels = [e.map.label for e in all_maps(sdfg)]
    assert len(kernels) == len(set(kernels)), f"duplicate kernel label in {sorted(kernels)}"


def test_each_kind_is_numbered_from_zero_across_the_program():
    """One counter per kind: the doalls are doall0, doall1, ..., whatever depth they sit at and whatever blocks of
    another kind sit beside them."""
    sdfg = two_maps.to_sdfg(simplify=True)
    normalize_labels(sdfg)
    by_kind = {}
    for block in all_blocks(sdfg):
        kind, index = LABEL.match(block.label).groups()
        by_kind.setdefault(kind, []).append(int(index))
    by_kind["doall"] = [int(KERNEL.match(entry.map.label).group(1)) for entry in all_maps(sdfg)]
    for kind, indices in by_kind.items():
        assert sorted(indices) == list(range(len(indices))), f"{kind} is numbered {sorted(indices)}"


def test_a_retired_name_is_never_handed_out_again():
    sdfg = two_maps.to_sdfg(simplify=True)
    namer = Namer()
    normalize_labels(sdfg, namer)
    first = all_maps(sdfg)[0]
    retired = first.map.label

    namer.retire((retired,))
    normalize_labels(sdfg, namer)

    assert first.map.label != retired
    assert retired not in [entry.map.label for entry in all_maps(sdfg)]


def test_a_name_survives_a_second_pass_of_its_namer_after_a_new_row_appears():
    sdfg = two_maps.to_sdfg(simplify=True)
    namer = Namer()
    normalize_labels(sdfg, namer)
    before = [entry.map.label for entry in all_maps(sdfg)]
    state = next(iter(sdfg.states()))
    state.add_mapped_tasklet("fresh", {"k": "0:4"}, {}, "o = 1", {"o": dc.Memlet("A[k]")})

    normalize_labels(sdfg, namer)

    after = [entry.map.label for entry in all_maps(sdfg)]
    assert set(before) < set(after), (before, after)


def test_a_frontend_source_line_label_does_not_survive():
    """Frontend labels embed the source line (``inner_9_4``), so an edit above a nest renames a nest
    that did not change. That is the whole reason the tree cannot use them as ids."""
    sdfg = nested_call.to_sdfg(simplify=False)
    before = [e.map.label for e in all_maps(sdfg)]
    assert any(re.search(r"_\d+$", label) for label in before), f"fixture stopped being representative: {before}"
    normalize_labels(sdfg)
    assert all(KERNEL.match(e.map.label) for e in all_maps(sdfg))


def test_a_counted_and_a_conditional_loop_are_both_a_loop():
    """The tree names every sequential loop ``loop``; only a kernel is a do-all loop."""
    sdfg = branchy.to_sdfg(simplify=True)
    loops = [b for b in all_blocks(sdfg) if isinstance(b, LoopRegion)]
    assert loops, "fixture no longer has a LoopRegion"
    for loop in loops:
        assert block_kind(loop) == "for"
        loop.update_statement = None
        assert block_kind(loop) == "for"


def test_two_builds_of_one_program_get_the_same_names():
    """Determinism is the point of the tie-break: a topological order alone is not unique, and two
    orders give two label assignments for one program -- so an agent's saved id would not survive a
    rebuild."""
    first, second = branchy.to_sdfg(simplify=True), branchy.to_sdfg(simplify=True)
    normalize_labels(first)
    normalize_labels(second)
    assert [b.label for b in all_blocks(first)] == [b.label for b in all_blocks(second)]
    assert [e.map.label for e in all_maps(first)] == [e.map.label for e in all_maps(second)]


def test_normalizing_twice_is_normalizing_once():
    """The session relabels after every fusion move, so a second pass must not renumber anything."""
    sdfg = branchy.to_sdfg(simplify=True)
    normalize_labels(sdfg)
    once = ([b.label for b in all_blocks(sdfg)], [e.map.label for e in all_maps(sdfg)])
    normalize_labels(sdfg)
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


# no top-level nested SDFG


def test_a_top_level_nested_sdfg_is_inlined():
    """Its states are real control flow; left nested they are an opaque box the agent cannot fuse
    across."""
    sdfg = nested_call.to_sdfg(simplify=False)
    assert top_level_nsdfgs(sdfg), "fixture no longer has a top-level nested SDFG"
    inline_top_level_nsdfgs(sdfg)
    assert not top_level_nsdfgs(sdfg)


def test_a_nested_sdfg_inside_a_map_is_left_alone():
    """That one is a kernel body, not structure."""
    sdfg = nested_call.to_sdfg(simplify=False)
    inline_top_level_nsdfgs(sdfg)
    inside = [
        (st, n)
        for st in sdfg.states()
        for n in st.nodes()
        if isinstance(n, nodes.NestedSDFG) and st.entry_node(n) is not None
    ]
    assert inside, "fixture no longer has a nested SDFG inside a map"


def test_inlining_bails_out_without_touching_an_already_inlined_sdfg():
    sdfg = nested_call.to_sdfg(simplify=False)
    inline_top_level_nsdfgs(sdfg)
    before = sdfg.to_json()
    assert inline_top_level_nsdfgs(sdfg) == 0
    assert sdfg.to_json() == before
