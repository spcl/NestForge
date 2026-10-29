# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Scheduling moves addressed by tree labels: :meth:`Session.list_moves` and :meth:`Session.apply_move`."""

import copy
import hashlib
import json
import os
import re
from collections import Counter

import dace
import numpy as np
import pytest
from dace.libraries.blas import Dot
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, LoopRegion, SDFGState

from nestforge.ir.names import normalize_labels
from nestforge.records import MoveStatus
from nestforge.session import MoveResult, Session
from nestforge.stages.scopes import top_level_map_entries

N = dace.symbol("N", dtype=dace.int64)
T = dace.symbol("T", dtype=dace.int64)
K = dace.symbol("K", dtype=dace.int64)
SIZE = 7
STEPS = 5


@dace.program
def recur(x: dace.float64[N], out: dace.float64[N]):
    for i in range(1, N):
        out[i] = out[i - 1] + x[i]


@dace.program
def calls_then_loops(a: dace.float64[N], b: dace.float64[N], oa: dace.float64[N], ob: dace.float64[N]):
    recur(a, oa)
    recur(b, ob)
    for i in range(1, N):
        a[i] = a[i - 1] + oa[i]
    for i in range(1, N):
        b[i] = b[i - 1] + ob[i]


@dace.program
def two_loops(a: dace.float64[N], b: dace.float64[N]):
    for i in range(1, N):
        a[i] = a[i - 1] + 1.0
    for i in range(1, N):
        b[i] = b[i - 1] + a[i]


@dace.program
def read_ahead(a: dace.float64[N], b: dace.float64[N]):
    for i in range(1, N - 1):
        a[i] = a[i - 1] + 1.0
    for i in range(1, N - 1):
        b[i] = b[i - 1] + a[i + 1]


@dace.program
def vertical_pair(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    tmp = np.empty_like(A)
    for i in dace.map[0:N]:
        tmp[i] = A[i] + B[i]
    for i in dace.map[0:N]:
        C[i] = tmp[i] * 2.0


@dace.program
def two_indep(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N], D: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] * 2.0
    for i in dace.map[0:N]:
        D[i] = B[i] * 3.0


@dace.program
def one_statement(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[i] * 2.0 + 1.0


@dace.program
def two_statements(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[i] + 1.0
        c[i] = a[i] * 2.0


@dace.program
def nested(A: dace.float64[N, N], B: dace.float64[N, N]):
    for i in dace.map[0:N]:
        for j in dace.map[0:N]:
            B[i, j] = A[i, j] * 2.0


@dace.program
def triangle(A: dace.float64[N, N], B: dace.float64[N, N]):
    for i in dace.map[0:N]:
        for j in dace.map[i:N]:
            B[i, j] = A[i, j] * 2.0


@dace.program
def sweep(A: dace.float64[T, N]):
    for t in range(1, T):
        for i in dace.map[0:N]:
            A[t, i] = A[t - 1, i] + 1.0


@dace.program
def lane_cross(A: dace.float64[T, N]):
    for t in range(1, T):
        for i in dace.map[1:N]:
            A[t, i] = A[t - 1, i - 1] + 1.0


@dace.program
def sweep_then_scale(A: dace.float64[T, N], B: dace.float64[N]):
    for t in range(1, T):
        for i in dace.map[0:N]:
            A[t, i] = A[t - 1, i] + 1.0
    for i in dace.map[0:N]:
        B[i] = A[1, i] * 2.0


@dace.program
def two_recurrences(a: dace.float64[N], b: dace.float64[N]):
    for i in range(1, N):
        a[i] = a[i - 1] + 1.0
        b[i] = b[i - 1] * 0.5


@dace.program
def coupled_recurrences(a: dace.float64[N], b: dace.float64[N]):
    for i in range(1, N):
        a[i] = a[i - 1] + b[i - 1]
        b[i] = a[i] * 0.5


@dace.program
def long_body(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        t = a[i] * 2.0
        for k in range(3):
            t = t + a[i]
        b[i] = t
        c[i] = b[i] + t


@dace.program
def map_loop(A: dace.float64[T, N]):
    for i in dace.map[0:N]:
        for t in range(1, T):
            A[t, i] = A[t - 1, i] + 1.0


@dace.program
def map_triangle_loop(A: dace.float64[N, N]):
    for i in dace.map[0:N]:
        for t in range(i + 1, N):
            A[t, i] = A[t - 1, i] + 1.0


@dace.program
def guarded_loop(a: dace.float64[N]):
    if K > 0:
        for i in range(1, N):
            a[i] = a[i - 1] + 1.0


@dace.program
def loop_guard(a: dace.float64[N]):
    for i in range(1, N):
        if K > 0:
            a[i] = a[i - 1] + 1.0


@dace.program
def loop_variant_guard(a: dace.float64[N]):
    for i in range(1, N):
        if i > K:
            a[i] = a[i - 1] + 1.0


@dace.program
def steps_inside_a_map(A: dace.float64[N, T], C: dace.float64[N, T]):
    for i in dace.map[0:N]:
        for t in range(1, T):
            tmp = np.empty(T, dace.float64)
            for j in dace.map[0:T]:
                tmp[j] = A[i, j] * t
            for j in dace.map[0:T]:
                C[i, j] = C[i, j] + tmp[j]


@dace.program
def scale_in_place_then_read(A: dace.float64[N], B: dace.float64[N]):
    for i in dace.map[0:N]:
        A[i] = A[i] * 2
    for i in dace.map[0:N]:
        B[i] = A[i] + 1


@dace.program
def scale_then_neighbors(A: dace.float64[N], B: dace.float64[N]):
    t = np.empty(N, dace.float64)
    for i in dace.map[0:N]:
        t[i] = A[i] * 2
    for i in dace.map[1 : N - 1]:
        B[i] = t[i - 1] + t[i + 1]


@dace.program
def two_sums(A: dace.float64[N], S: dace.float64[1], R: dace.float64[1]):
    for i in dace.map[0:N]:
        S[0] += A[i] * 2
    for i in dace.map[0:N]:
        R[0] += A[i] + 1


def tree_labels(sdfg: dace.SDFG) -> list[str]:
    return [
        node.label
        for node, _ in sdfg.all_nodes_recursive()
        if isinstance(node, (ControlFlowBlock, nodes.MapEntry, nodes.LibraryNode))
    ]


def duplicate_labels(sdfg: dace.SDFG) -> dict[str, int]:
    return {label: count for label, count in Counter(tree_labels(sdfg)).items() if count > 1}


def digest(sdfg: dace.SDFG) -> str:
    return hashlib.sha256(json.dumps(sdfg.to_json(), sort_keys=True, default=str).encode()).hexdigest()


def loops(sdfg: dace.SDFG) -> list[LoopRegion]:
    return [
        b for cfg in sdfg.all_control_flow_regions(recursive=True) for b in cfg.nodes() if isinstance(b, LoopRegion)
    ]


def conditionals(sdfg: dace.SDFG) -> list[ConditionalBlock]:
    return [
        b
        for cfg in sdfg.all_control_flow_regions(recursive=True)
        for b in cfg.nodes()
        if isinstance(b, ConditionalBlock)
    ]


def top_level_maps(sdfg: dace.SDFG) -> list[nodes.MapEntry]:
    return [entry for state in sdfg.states() for entry in top_level_map_entries(state)]


def map_writes(sdfg: dace.SDFG) -> list[list[str]]:
    """The arrays each map writes, over every map in the SDFG hierarchy."""
    return sorted(
        sorted({e.data.data for e in state.out_edges(state.exit_node(node))})
        for node, state in sdfg.all_nodes_recursive()
        if isinstance(node, nodes.MapEntry)
    )


def session_and_reference(program, simplify: bool = True) -> tuple[Session, dace.SDFG]:
    sdfg = program.to_sdfg(simplify=simplify)
    # one build folder per xdist worker: two workers compiling one program name race on it
    sdfg.name = f"{sdfg.name}_{os.getpid()}"
    reference = copy.deepcopy(sdfg)
    reference.name = f"{sdfg.name}_reference"
    return Session(sdfg), reference


def apply_listed(session: Session, kind: str) -> MoveResult:
    (move,) = session.list_moves(kind)
    return session.apply_move(move.kind, move.labels, move.tile_sizes, move.implementation)


def random_arrays(**shapes: tuple[int, ...]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {name: rng.random(shape) for name, shape in shapes.items()}


def assert_same_values(reference: dace.SDFG, moved: dace.SDFG, arrays: dict[str, np.ndarray], **symbols: int) -> None:
    expected = {name: value.copy() for name, value in arrays.items()}
    actual = {name: value.copy() for name, value in arrays.items()}
    reference(**expected, **symbols)
    moved(**actual, **symbols)
    for name in arrays:
        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-13, err_msg=name)


# labels


def test_a_helper_called_twice_repeats_labels_that_the_session_makes_unique():
    sdfg = calls_then_loops.to_sdfg(simplify=False)
    repeated = duplicate_labels(sdfg)
    assert "init" in repeated and any(label.startswith("for_") for label in repeated), repeated

    Session(sdfg)

    assert duplicate_labels(sdfg) == {}


def test_labels_stay_unique_after_a_fusion_move():
    sut, reference = session_and_reference(calls_then_loops, simplify=False)
    loop_count = len(loops(sut.sdfg))

    result = apply_listed(sut, "for-fusion")

    assert result.status is MoveStatus.APPLIED
    assert len(loops(sut.sdfg)) == loop_count - 1
    assert duplicate_labels(sut.sdfg) == {}
    assert_same_values(reference, sut.sdfg, random_arrays(a=(SIZE,), b=(SIZE,), oa=(SIZE,), ob=(SIZE,)), N=SIZE)


def test_a_session_keeps_the_labels_normalize_labels_already_gave():
    sdfg = sweep_then_scale.to_sdfg(simplify=True)
    normalize_labels(sdfg)
    labels = tree_labels(sdfg)

    Session(sdfg)

    assert tree_labels(sdfg) == labels


def test_library_calls_get_unique_kernel_family_labels_and_keep_name_equal_to_label():
    sdfg = dace.SDFG("two_dots")
    state = sdfg.add_state()
    first, second = Dot("dot"), Dot("dot")
    state.add_node(first)
    state.add_node(second)

    Session(sdfg)

    assert re.fullmatch(r"lib_blas\d+", first.label), first.label
    assert re.fullmatch(r"lib_blas\d+", second.label), second.label
    assert first.label != second.label
    assert (first.name, second.name) == (first.label, second.label)


def test_the_tree_names_its_program_on_the_first_line():
    sut = Session(vertical_pair.to_sdfg(simplify=True))
    assert sut.describe().splitlines()[0] == f"SDFG '{sut.sdfg.label}'"


def test_listed_moves_name_rows_of_the_current_tree():
    sut = Session(sweep_then_scale.to_sdfg(simplify=True))
    sut.expand([label for label, (obj, _) in sut.row_index().items() if isinstance(obj, SDFGState)])
    tree = sut.describe()

    moves = sut.list_moves()

    assert moves, "the fixture offers no move"
    for label in {label for move in moves for label in move.labels}:
        assert re.search(rf"[-`] {label}\b", tree), (label, tree)


# loop fusion


def test_loop_fusion_merges_two_loops_into_one_that_writes_both_arrays():
    sut, reference = session_and_reference(two_loops)

    result = apply_listed(sut, "for-fusion")

    assert (result.status, result.reason) == (MoveStatus.APPLIED, "LoopFusion")
    (loop,) = loops(sut.sdfg)
    assert {"a", "b"} <= set(loop.read_and_write_sets()[1])
    assert_same_values(reference, sut.sdfg, random_arrays(a=(SIZE,), b=(SIZE,)), N=SIZE)


def test_loop_fusion_that_would_read_ahead_is_illegal_and_changes_nothing():
    sut = Session(read_ahead.to_sdfg(simplify=True))
    before = digest(sut.sdfg)

    result = sut.apply_move("for-fusion", ["for0", "for1"])

    assert result.status is MoveStatus.ILLEGAL and "LoopFusion" in result.reason, result
    assert digest(sut.sdfg) == before
    assert sut.list_moves("for-fusion") == []


# map fusion


def test_map_fusion_through_a_transient_is_vertical_and_leaves_one_map():
    sut, reference = session_and_reference(vertical_pair)

    result = apply_listed(sut, "doall-fusion")

    assert (result.status, result.reason) == (MoveStatus.APPLIED, "MapFusionVertical")
    assert len(top_level_maps(sut.sdfg)) == 1
    assert_same_values(reference, sut.sdfg, random_arrays(A=(SIZE,), B=(SIZE,), C=(SIZE,)), N=SIZE)


def test_map_fusion_of_independent_siblings_is_horizontal_and_leaves_one_map():
    sut, reference = session_and_reference(two_indep)

    result = apply_listed(sut, "doall-fusion")

    assert (result.status, result.reason) == (MoveStatus.APPLIED, "MapFusionHorizontal")
    assert len(top_level_maps(sut.sdfg)) == 1
    arrays = random_arrays(A=(SIZE,), B=(SIZE,), C=(SIZE,), D=(SIZE,))
    assert_same_values(reference, sut.sdfg, arrays, N=SIZE)


def test_a_map_fusion_inside_a_nested_sdfg_is_listed_and_applies():
    """``apply_move`` reaches rows in nested SDFGs, so ``list_moves`` must offer the moves found there too."""
    sut, reference = session_and_reference(steps_inside_a_map)

    result = apply_listed(sut, "doall-fusion")

    assert result.status is MoveStatus.APPLIED, result
    assert_same_values(reference, sut.sdfg, random_arrays(A=(SIZE, STEPS), C=(SIZE, STEPS)), N=SIZE, T=STEPS)


@pytest.mark.parametrize(
    "program, fusion",
    [
        (scale_in_place_then_read, "MapFusionVertical"),
        (scale_then_neighbors, "OTFMapFusion"),
        (two_sums, "MapFusionHorizontal"),
    ],
)
def test_map_fusion_fuses_in_place_on_the_fly_and_two_reductions_into_one_map(program, fusion):
    """In place: the first map updates the array the second reads. On the fly: the second reads a window of the
    first's output, recomputed per element. Two reductions: two nested bodies, each leaving through ``-wcr->``."""
    sut, reference = session_and_reference(program)
    sut.canonicalize()

    result = apply_listed(sut, "doall-fusion")

    assert (result.status, result.reason) == (MoveStatus.APPLIED, fusion)
    assert len(top_level_maps(sut.sdfg)) == 1
    arrays = random_arrays(A=(SIZE,), **({"S": (1,), "R": (1,)} if program is two_sums else {"B": (SIZE,)}))
    assert_same_values(reference, sut.sdfg, arrays, N=SIZE)


def test_map_fusion_across_states_is_illegal_and_names_the_state_barrier():
    sut = Session(two_indep.to_sdfg(simplify=False))
    labels = [entry.label for entry in top_level_maps(sut.sdfg)]
    before = digest(sut.sdfg)

    result = sut.apply_move("doall-fusion", labels)

    assert result.status is MoveStatus.ILLEGAL and "different states" in result.reason, result
    assert digest(sut.sdfg) == before


# map fission


def test_map_fission_splits_a_two_statement_map_into_maps_that_each_write_one_array():
    sut, reference = session_and_reference(two_statements, simplify=False)
    assert map_writes(sut.sdfg) == [["b", "c"]]

    result = apply_listed(sut, "doall-fission")

    assert (result.status, result.reason) == (MoveStatus.APPLIED, "MapFission")
    # MapFission moves the split maps into the nested SDFG that was the map's body
    assert top_level_maps(sut.sdfg) == []
    writes = map_writes(sut.sdfg)
    assert len(writes) > 1 and all(len(arrays) == 1 for arrays in writes), writes
    assert_same_values(reference, sut.sdfg, random_arrays(a=(SIZE,), b=(SIZE,), c=(SIZE,)), N=SIZE)


def test_map_fission_splits_a_flat_two_statement_map_by_its_outputs():
    sut, reference = session_and_reference(two_statements)
    (entry,) = top_level_maps(sut.sdfg)

    result = sut.apply_move("doall-fission", [entry.label])

    assert (result.status, result.reason) == (MoveStatus.APPLIED, "MapFission")
    assert len(top_level_maps(sut.sdfg)) > 1, "the map split into its independent components"
    assert_same_values(reference, sut.sdfg, random_arrays(a=(SIZE,), b=(SIZE,), c=(SIZE,)), N=SIZE)


def test_map_fission_of_a_flat_map_writing_one_array_is_illegal():
    sut = Session(one_statement.to_sdfg(simplify=True))
    (entry,) = top_level_maps(sut.sdfg)
    before = digest(sut.sdfg)

    result = sut.apply_move("doall-fission", [entry.label])

    assert result.status is MoveStatus.ILLEGAL and "MapFission" in result.reason, result
    assert digest(sut.sdfg) == before


def test_map_fission_of_a_nested_body_of_one_computation_is_illegal():
    sut = Session(one_statement.to_sdfg(simplify=True))
    sut.canonicalize()
    (entry,) = top_level_maps(sut.sdfg)

    result = sut.apply_move("doall-fission", [entry.label])

    assert result.status is MoveStatus.ILLEGAL and "one computation" in result.reason, result


def test_canonicalization_leaves_every_map_body_one_nested_sdfg_and_a_reduction_outside_it():
    """``map { nested SDFG } -> access node -wcr->``: no tasklet beside the body, no reduction inside it."""
    sut = Session(two_sums.to_sdfg(simplify=True))

    sut.canonicalize()

    for state in sut.sdfg.states():
        for entry in top_level_map_entries(state):
            exit_node = state.exit_node(entry)
            body = [n for n in state.scope_children()[entry] if n is not exit_node]
            (nest,) = [n for n in body if isinstance(n, nodes.NestedSDFG)]
            (leaves,) = [n for n in body if n is not nest]
            assert isinstance(leaves, nodes.AccessNode)
            assert [e.data.wcr is not None for e in state.out_edges(leaves)] == [True]
            assert not any(e.data.wcr for inner in nest.sdfg.states() for e in inner.edges())


# interchange


def test_map_map_interchange_puts_the_inner_parameter_outside():
    sut, reference = session_and_reference(nested)

    result = apply_listed(sut, "interchange-doall-doall")

    assert (result.status, result.reason) == (MoveStatus.APPLIED, "MapInterchange")
    (outer,) = top_level_maps(sut.sdfg)
    state = next(s for s in sut.sdfg.states() if outer in s.nodes())
    (inner,) = [n for n in state.scope_children()[outer] if isinstance(n, nodes.MapEntry)]
    assert (outer.map.params, inner.map.params) == (["j"], ["i"])
    assert_same_values(reference, sut.sdfg, random_arrays(A=(SIZE, SIZE), B=(SIZE, SIZE)), N=SIZE)


def test_map_map_interchange_of_a_triangular_nest_is_illegal():
    sut = Session(triangle.to_sdfg(simplify=True))
    before = digest(sut.sdfg)

    result = sut.apply_move("interchange-doall-doall", ["doall0", "doall1"])

    assert result.status is MoveStatus.ILLEGAL and "MapInterchange" in result.reason, result
    assert digest(sut.sdfg) == before


def test_naming_the_inner_map_first_is_illegal():
    sut = Session(nested.to_sdfg(simplify=True))
    before = digest(sut.sdfg)

    result = sut.apply_move("interchange-doall-doall", ["doall1", "doall0"])

    assert result.status is MoveStatus.ILLEGAL and "not directly inside" in result.reason, result
    assert digest(sut.sdfg) == before


def test_loop_map_interchange_moves_the_loop_inside_the_map():
    sut, reference = session_and_reference(sweep)

    result = apply_listed(sut, "interchange-for-doall")

    assert (result.status, result.reason) == (MoveStatus.APPLIED, "MoveLoopIntoMap")
    assert not [block for block in sut.sdfg.nodes() if isinstance(block, LoopRegion)]
    (entry,) = top_level_maps(sut.sdfg)
    state = next(s for s in sut.sdfg.states() if entry in s.nodes())
    (body,) = [n for n in state.scope_children()[entry] if isinstance(n, nodes.NestedSDFG)]
    assert [block for block in body.sdfg.nodes() if isinstance(block, LoopRegion)]
    assert_same_values(reference, sut.sdfg, random_arrays(A=(STEPS, SIZE)), N=SIZE, T=STEPS)


def test_loop_map_interchange_that_would_cross_map_iterations_is_illegal():
    sut = Session(lane_cross.to_sdfg(simplify=True))
    before = digest(sut.sdfg)

    result = sut.apply_move("interchange-for-doall", ["for0", "doall0"])

    assert result.status is MoveStatus.ILLEGAL and "MoveLoopIntoMap" in result.reason, result
    assert digest(sut.sdfg) == before


def test_loop_map_interchange_with_a_map_outside_the_loop_is_illegal():
    sut = Session(sweep_then_scale.to_sdfg(simplify=True))
    (loop,) = loops(sut.sdfg)
    (outside,) = [n for n in top_level_maps(sut.sdfg) if all(n not in s.nodes() for s in loop.states())]
    before = digest(sut.sdfg)

    result = sut.apply_move("interchange-for-doall", [loop.label, outside.label])

    assert result.status is MoveStatus.ILLEGAL and "not the one map directly inside" in result.reason, result
    assert digest(sut.sdfg) == before


# loop fission


def test_loop_fission_splits_independent_recurrences_into_one_loop_each():
    sut, reference = session_and_reference(two_recurrences)

    result = apply_listed(sut, "for-fission")

    assert (result.status, result.reason) == (MoveStatus.APPLIED, "LoopFission")
    writes = [{"a", "b"} & set(loop.read_and_write_sets()[1]) for loop in loops(sut.sdfg)]
    assert writes == [{"a"}, {"b"}], writes
    assert_same_values(reference, sut.sdfg, random_arrays(a=(SIZE,), b=(SIZE,)), N=SIZE)


def test_loop_fission_of_coupled_recurrences_is_illegal_and_changes_nothing():
    sut = Session(coupled_recurrences.to_sdfg(simplify=True))
    (loop,) = loops(sut.sdfg)
    before = digest(sut.sdfg)

    result = sut.apply_move("for-fission", [loop.label])

    assert result.status is MoveStatus.ILLEGAL and "LoopFission" in result.reason, result
    assert digest(sut.sdfg) == before
    assert sut.list_moves("for-fission") == []


# subgraph fission


@pytest.mark.parametrize("cut", [0, 1])
def test_subgraph_fission_splits_a_map_body_into_two_maps_at_the_named_block(cut):
    """The body is ``t = 2a; loop: t += a; b = t; c = b + t``: either cut hands ``t`` across, so it has to become
    one element per map iteration."""
    sut, reference = session_and_reference(long_body)
    moves = sut.list_moves("subgraph-fission")
    assert len(moves) == 2, moves

    result = sut.apply_move(moves[cut].kind, moves[cut].labels)

    assert (result.status, result.reason) == (MoveStatus.APPLIED, "SubgraphFission")
    assert len(top_level_maps(sut.sdfg)) == 2
    assert [str(extent) for extent in sut.sdfg.arrays["t"].shape] == ["N"]
    assert_same_values(reference, sut.sdfg, random_arrays(a=(SIZE,), b=(SIZE,), c=(SIZE,)), N=SIZE)


def test_subgraph_fission_after_the_last_block_is_illegal_and_changes_nothing():
    sut = Session(long_body.to_sdfg(simplify=True))
    (entry,) = top_level_maps(sut.sdfg)
    state = next(s for s in sut.sdfg.states() if entry in s.nodes())
    (body,) = [n for n in state.scope_children()[entry] if isinstance(n, nodes.NestedSDFG)]
    (last,) = body.sdfg.sink_nodes()
    before = digest(sut.sdfg)

    result = sut.apply_move("subgraph-fission", [entry.map.label, last.label])

    assert result.status is MoveStatus.ILLEGAL and "before the last" in result.reason, result
    assert digest(sut.sdfg) == before


# map-loop interchange


def test_map_loop_interchange_puts_the_loop_outside_the_map():
    sut, reference = session_and_reference(map_loop)

    result = apply_listed(sut, "interchange-doall-for")

    assert (result.status, result.reason) == (MoveStatus.APPLIED, "MapLoopInterchange")
    (loop,) = [block for block in sut.sdfg.nodes() if isinstance(block, LoopRegion)]
    (state,) = loop.nodes()
    assert [n for n in state.scope_children()[None] if isinstance(n, nodes.MapEntry)]
    assert_same_values(reference, sut.sdfg, random_arrays(A=(STEPS, SIZE)), N=SIZE, T=STEPS)


def test_map_loop_interchange_undoes_loop_map_interchange():
    sut, reference = session_and_reference(sweep)
    apply_listed(sut, "interchange-for-doall")

    result = apply_listed(sut, "interchange-doall-for")

    assert result.status is MoveStatus.APPLIED, result
    assert [type(block) for block in sut.sdfg.nodes()] == [LoopRegion]
    assert_same_values(reference, sut.sdfg, random_arrays(A=(STEPS, SIZE)), N=SIZE, T=STEPS)


def test_map_loop_interchange_of_a_loop_whose_bound_reads_the_map_parameter_is_illegal():
    """Each map iteration would run a different trip count, which one loop outside the map cannot express."""
    sut = Session(map_triangle_loop.to_sdfg(simplify=True))
    (entry,) = top_level_maps(sut.sdfg)
    (loop,) = loops(sut.sdfg)
    before = digest(sut.sdfg)

    result = sut.apply_move("interchange-doall-for", [entry.map.label, loop.label])

    assert result.status is MoveStatus.ILLEGAL and "varies across map iterations" in result.reason, result
    assert digest(sut.sdfg) == before


# if-loop interchange


def test_a_guard_moves_into_the_loop_it_guards():
    sut, reference = session_and_reference(guarded_loop)

    result = apply_listed(sut, "interchange-if-for")

    assert (result.status, result.reason) == (MoveStatus.APPLIED, "MoveIfIntoLoop")
    assert [type(block) for block in sut.sdfg.nodes()] == [LoopRegion]
    for k in (0, 3):
        assert_same_values(reference, sut.sdfg, random_arrays(a=(SIZE,)), N=SIZE, K=k)


def test_a_loop_invariant_guard_moves_out_of_its_loop():
    sut, reference = session_and_reference(loop_guard)

    result = apply_listed(sut, "interchange-for-if")

    assert (result.status, result.reason) == (MoveStatus.APPLIED, "MoveLoopInvariantIfUp")
    assert [type(block) for block in sut.sdfg.nodes()] == [ConditionalBlock]
    for k in (0, 3):
        assert_same_values(reference, sut.sdfg, random_arrays(a=(SIZE,)), N=SIZE, K=k)


def test_a_guard_that_reads_the_loop_variable_cannot_leave_the_loop():
    sut = Session(loop_variant_guard.to_sdfg(simplify=True))
    (loop,), (cond,) = loops(sut.sdfg), conditionals(sut.sdfg)
    before = digest(sut.sdfg)

    result = sut.apply_move("interchange-for-if", [loop.label, cond.label])

    assert result.status is MoveStatus.ILLEGAL and "MoveLoopInvariantIfUp" in result.reason, result
    assert digest(sut.sdfg) == before
    assert sut.list_moves("interchange-for-if") == []


# refusals that never touch the graph


def test_an_unknown_label_is_not_found_and_changes_nothing():
    sut = Session(vertical_pair.to_sdfg(simplify=True))
    before = digest(sut.sdfg)

    result = sut.apply_move("doall-fusion", ["doall0", "doall99"])

    assert result.status is MoveStatus.NOT_FOUND and "doall99" in result.reason, result
    assert digest(sut.sdfg) == before


def test_labels_a_move_retired_are_refused_and_change_nothing():
    sut = Session(vertical_pair.to_sdfg(simplify=True))
    (move,) = sut.list_moves("doall-fusion")
    sut.apply_move(move.kind, move.labels)
    after_move = digest(sut.sdfg)

    result = sut.apply_move(move.kind, move.labels)

    assert result.status is MoveStatus.NOT_FOUND and "changed; read the tree again" in result.reason, result
    assert digest(sut.sdfg) == after_move


def test_an_unknown_kind_is_a_malformed_call():
    sut = Session(vertical_pair.to_sdfg(simplify=True))
    with pytest.raises(ValueError, match="unknown move kind"):
        sut.apply_move("loop-tiling", ["for0"])
    with pytest.raises(ValueError, match="unknown move kind"):
        sut.list_moves("loop-tiling")


def test_a_wrong_label_count_is_a_malformed_call():
    sut = Session(vertical_pair.to_sdfg(simplify=True))
    with pytest.raises(ValueError, match="takes 2 label"):
        sut.apply_move("doall-fusion", ["doall1"])


def test_a_move_read_on_a_session_applies_to_its_trial_and_leaves_the_session_alone():
    """``performance`` and the agent's what-ifs run on a trial; the labels must carry over."""
    session, _ = session_and_reference(two_indep)
    session.bump()  # a session past its first change, as after canonicalization
    (move,) = session.list_moves("doall-fusion")
    before = digest(session.sdfg)

    result = session.trial().apply_move(move.kind, move.labels)

    assert result.status is MoveStatus.APPLIED, result
    assert digest(session.sdfg) == before


NLEV, NSPEC = (dace.symbol(name, dtype=dace.int64) for name in ("NLEV", "NSPEC"))


@dace.program
def species_sedimentation(q: dace.float64[NLEV, N, NSPEC], pr: dace.float64[N]):
    for jl in dace.map[0:N]:
        pr[jl] = q[0, jl, 0] * 0.5
        for js in range(NSPEC):
            for jk in range(1, NLEV):
                q[jk, jl, js] = q[jk, jl, js] + 0.1 * q[jk - 1, jl, js]


def test_a_cut_leaves_both_maps_in_one_state_each_with_its_own_nested_body():
    """The cleanup after a move fuses the states the cut made, and keeps every map body one nested SDFG with no
    write-conflict resolution inside it."""
    sut, _ = session_and_reference(species_sedimentation)
    sut.canonicalize()
    (cut,) = sut.list_moves("subgraph-fission")

    sut.apply_move(cut.kind, cut.labels)

    (state,) = sut.sdfg.nodes()
    entries = [n for n in state.nodes() if isinstance(n, nodes.MapEntry) and state.entry_node(n) is None]
    bodies = [state.scope_children()[entry] for entry in entries]
    assert len(entries) == 2
    assert all(len([n for n in body if isinstance(n, nodes.NestedSDFG)]) == len(body) - 1 for body in bodies)
    nested = [n for body in bodies for n in body if isinstance(n, nodes.NestedSDFG)]
    assert not any(e.data.wcr for n in nested for inner in n.sdfg.states() for e in inner.edges())


def test_a_level_loop_under_a_column_map_moves_out_once_the_map_is_cut_around_it():
    """#29: the level loop under the column map stayed innermost. Cut the map body around the loop, then the
    loop-map interchange applies (DaCe had called a scalar the canonical form reads twice loop-carried)."""
    sut, reference = session_and_reference(species_sedimentation)
    sut.canonicalize()
    (cut,) = sut.list_moves("subgraph-fission")
    sut.apply_move(cut.kind, cut.labels)

    result = apply_listed(sut, "interchange-doall-for")

    assert (result.status, result.reason) == (MoveStatus.APPLIED, "MapLoopInterchange")
    arrays = random_arrays(q=(STEPS, SIZE, 3), pr=(SIZE,))
    assert_same_values(reference, sut.sdfg, arrays, N=SIZE, NLEV=STEPS, NSPEC=3)
