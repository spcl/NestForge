# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``auto-vectorize-scope`` vectorizes the one doall it names; ``view_tree`` prints bodies as Python or as C, which
shows a vectorized body's tile functions. Printing never changes the program."""

import dace
import numpy as np

from nestforge.records import MoveStatus
from nestforge.session import Session
from nestforge.stages.performance import Reference
from nestforge.stages.vectorize import AUTO_VECTORIZE, DEFAULT_LANES, tile_lanes

N = dace.symbol("N", dace.int64, positive=True)
SIZE = 37  # not a multiple of the lanes: the masked remainder tile runs too


@dace.program
def axpy_then_reverse(x: dace.float64[N], y: dace.float64[N], z: dace.float64[N]):
    for i in dace.map[0:N]:
        y[i] = 2.0 * x[i] + y[i]
    for i in dace.map[0:N]:
        z[i] = y[N - 1 - i] * 3.0


T = dace.symbol("T", dace.int64, positive=True)


@dace.program
def accumulate_steps(x: dace.float64[N], y: dace.float64[N]):
    for _ in range(T):
        for i in dace.map[0:N]:
            y[i] = 2.0 * x[i] + y[i]


def session(tmp_path) -> Session:
    sut = Session(axpy_then_reverse.to_sdfg(simplify=True), work_dir=str(tmp_path), sizes={"N": SIZE})
    sut.canonicalize()
    return sut


def state_of(sut: Session, label: str) -> str:
    """The label of the state that holds the doall ``label``."""
    state = sut.row_index()[label][1]
    assert state is not None
    return state.label


def param(sut: Session, label: str) -> str:
    entry = sut.row_index()[label][0]
    assert isinstance(entry, dace.nodes.MapEntry)
    return str(entry.map.params[0])


def doall_writing(sut: Session, name: str) -> str:
    """The label of the top-level doall that writes ``name``."""
    for label, (obj, state) in sut.row_index().items():
        if isinstance(obj, dace.nodes.MapEntry) and state is not None and state.entry_node(obj) is None:
            if name in {e.data.data for e in state.out_edges(state.exit_node(obj))}:
                return label
    raise AssertionError(f"no top-level doall writes {name}")


def tile_call(node: dace.nodes.Node) -> bool:
    """Whether ``node`` is a C++ tasklet calling a tile function: what a vectorized body is made of."""
    return (
        isinstance(node, dace.nodes.Tasklet) and node.language is dace.Language.CPP and "tile_" in node.code.as_string
    )


def tile_calls(sdfg: dace.SDFG) -> int:
    return sum(tile_call(n) for n, _ in sdfg.all_nodes_recursive())


def correct(sut: Session) -> bool:
    x, y = np.arange(1.0, SIZE + 1), np.full(SIZE, 0.5)
    y_out = 2.0 * x + y
    expected = Reference({"y": y_out, "z": 3.0 * y_out[::-1]}, 1e-14, 0.0)
    report = sut.performance({"x": x, "y": y, "z": np.zeros(SIZE)}, expected, codegen="first")
    assert report.structured.error == "", report.structured
    return report.structured.correct is True


def test_vectorizing_one_doall_tiles_it_and_leaves_the_other_scalar(tmp_path):
    sut = session(tmp_path)
    axpy, reverse = doall_writing(sut, "y"), doall_writing(sut, "z")
    listed = {m.labels[0]: m.tile_sizes for m in sut.list_moves(AUTO_VECTORIZE)}
    assert listed == {axpy: (DEFAULT_LANES,), reverse: (DEFAULT_LANES,)}

    result = sut.apply_move(AUTO_VECTORIZE, [axpy], tile_sizes=[4], isa="scalar")

    assert (result.status, result.reason) == (MoveStatus.APPLIED, "VectorizeMap"), result
    assert tile_calls(sut.sdfg) > 0
    obj, state = sut.row_index()[reverse]
    assert isinstance(obj, dace.nodes.MapEntry) and obj.map.range.ranges[0][2] == 1, "the other doall was tiled"
    assert not any(tile_call(n) for n in state.scope_subgraph(obj).nodes())
    assert correct(sut)


def test_an_unknown_isa_or_lane_count_is_refused_and_changes_nothing(tmp_path):
    sut = session(tmp_path)
    axpy = doall_writing(sut, "y")

    wrong_isa = sut.apply_move(AUTO_VECTORIZE, [axpy], tile_sizes=[4], isa="mmx")
    no_lanes = sut.apply_move(AUTO_VECTORIZE, [axpy], tile_sizes=[], isa="avx2")

    assert wrong_isa.status is MoveStatus.ILLEGAL and "avx512" in wrong_isa.reason, wrong_isa
    assert no_lanes.status is MoveStatus.ILLEGAL and "tile_sizes" in no_lanes.reason, no_lanes
    assert tile_calls(sut.sdfg) == 0


def test_a_scalable_isa_takes_its_own_lanes_and_a_fixed_one_the_given_ones():
    assert tile_lanes("sve", [3]) == (DEFAULT_LANES,)
    assert tile_lanes("sve", [3, 5]) == (DEFAULT_LANES, DEFAULT_LANES)
    assert tile_lanes("avx512", [8, 2]) == (8, 2)


def test_the_c_view_prints_a_scalar_body_with_its_accesses(tmp_path):
    sut = session(tmp_path)
    axpy = doall_writing(sut, "y")
    i = param(sut, axpy)
    before = sut.sdfg.hash_sdfg()

    tree = sut.view(expand=[state_of(sut, axpy)], c=[state_of(sut, axpy)])

    assert f"y[{i}] = (__tmp0[0] + y[{i}]);" in tree, tree
    assert "auto " not in tree, tree
    assert sut.sdfg.hash_sdfg() == before, "printing changed the program"
    assert f"y[{i}] = (__tmp0[0] + y[{i}]);" not in sut.view(python=[state_of(sut, axpy)])


def test_the_c_view_shows_a_vectorized_body_as_tile_functions(tmp_path):
    sut = session(tmp_path)
    axpy = doall_writing(sut, "y")
    sut.apply_move(AUTO_VECTORIZE, [axpy], tile_sizes=[4], isa="scalar")
    state = state_of(sut, doall_writing(sut, "z"))

    c_tree, python_tree = sut.view(expand=[state], c=[state]), sut.view(python=[state])

    assert "tile_load" in c_tree and "tile_store" in c_tree and "<not emitted" not in c_tree, c_tree
    assert "<not emitted" in python_tree, "the Python view spells no C++ tasklet"


def test_a_row_inside_a_c_row_keeps_its_own_language(tmp_path):
    sut = session(tmp_path)
    axpy, reverse = doall_writing(sut, "y"), doall_writing(sut, "z")
    i, j = param(sut, axpy), param(sut, reverse)
    state = state_of(sut, axpy)

    tree = sut.view(expand=[state], c=[state], python=[reverse])

    assert f"y[{i}] = (__tmp0[0] + y[{i}]);" in tree, tree
    assert f"z[{j}] = (y[" not in tree and f"z[{j}] = y[N - {j} - 1] * 3.0" in tree, tree
    refused = sut.set_language(["nowhere"], "c")
    assert refused.startswith("illegal") and "nowhere" in refused


def test_a_doall_nested_in_another_prints_its_state_once_and_its_accumulation_in_c(tmp_path):
    sut = Session(accumulate_steps.to_sdfg(simplify=True), work_dir=str(tmp_path), sizes={"N": SIZE, "T": 3})
    sut.canonicalize()
    states = [label for label, (obj, _) in sut.row_index().items() if isinstance(obj, dace.SDFGState)]

    tree = sut.view(expand=states, c=states)

    rows = [line.split("- ", 1)[1].split()[0] for line in tree.splitlines() if "- " in line and ": " not in line]
    assert len(rows) == len(set(rows)), tree
    assert any("+= _wcr_priv" in line for line in tree.splitlines()), tree


def test_rollback_restores_the_body_language_the_program_had(tmp_path):
    sut = session(tmp_path)
    axpy = doall_writing(sut, "y")
    state = state_of(sut, axpy)
    sut.set_language([state], "c")
    sut.apply_move(AUTO_VECTORIZE, [axpy], tile_sizes=[4], isa="scalar")
    sut.set_language([state], "python")

    sut.apply_move("rollback", [])

    assert sut.languages == {state: "c"}
    assert tile_calls(sut.sdfg) == 0
