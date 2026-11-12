# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unrolling moves (a loop or an outermost doall of a constant trip count, up to the agent's cap) and
``propagate-config``, which specializes the program for argument values behind a dispatch on them."""

import pathlib
import tempfile

import dace
import numpy as np
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, LoopRegion

from nestforge.records import MoveStatus
from nestforge.session import Session

N = dace.symbol("N", dtype=dace.int64)
K = dace.symbol("K", dtype=dace.int64)
SIZE = 6


@dace.program
def short_scan(a: dace.float64[8]):
    for k in range(1, 4):
        a[k] = a[k - 1] + a[k]


@dace.program
def long_scan(a: dace.float64[24]):
    for k in range(1, 20):
        a[k] = a[k - 1] + a[k]


@dace.program
def sized_scan(a: dace.float64[N]):
    for k in range(1, N):
        a[k] = a[k - 1] + a[k]


@dace.program
def short_scale(a: dace.float64[4], b: dace.float64[4]):
    for i in dace.map[0:4]:
        b[i] = 2.0 * a[i]


@dace.program
def long_scale(a: dace.float64[20], b: dace.float64[20]):
    for i in dace.map[0:20]:
        b[i] = 2.0 * a[i]


@dace.program
def sized_scale(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = 2.0 * a[i]


@dace.program
def scale_then_shift(a: dace.float64[4], b: dace.float64[4]):
    tmp = np.empty_like(a)
    for i in dace.map[0:4]:
        tmp[i] = 2.0 * a[i]
    for i in dace.map[0:4]:
        b[i] = b[i] + tmp[i]


@dace.program
def repeat(a: dace.float64[N]):
    for k in range(K):
        for i in dace.map[0:N]:
            a[i] = a[i] * 0.5 + 1.0


def session(program, tmp_path: pathlib.Path) -> Session:
    return Session(program.to_sdfg(simplify=True), work_dir=str(tmp_path))


def labels_of(sut: Session, kind: type) -> list[str]:
    return [label for label, (obj, _) in sut.row_index().items() if isinstance(obj, kind)]


def count(sdfg: dace.SDFG, kind: type) -> int:
    return sum(isinstance(n, kind) for n, _ in sdfg.all_nodes_recursive())


def loops(sdfg: dace.SDFG) -> list[LoopRegion]:
    return [b for b in sdfg.all_control_flow_blocks(recursive=True) if isinstance(b, LoopRegion)]


def conditionals(sdfg: dace.SDFG) -> list[ConditionalBlock]:
    return [b for b in sdfg.all_control_flow_blocks(recursive=True) if isinstance(b, ConditionalBlock)]


def repeated(a: np.ndarray, steps: int) -> np.ndarray:
    for _ in range(steps):
        a = a * 0.5 + 1.0
    return a


# unroll-for


def test_a_constant_loop_unrolls_into_one_body_copy_per_iteration(tmp_path):
    sut = session(short_scan, tmp_path)
    tasklets = count(sut.sdfg, nodes.Tasklet)
    (loop,) = labels_of(sut, LoopRegion)

    result = sut.apply_move("unroll-for", [loop])

    assert (result.status, result.reason) == (MoveStatus.APPLIED, "LoopUnroll"), result
    assert loops(sut.sdfg) == []
    assert count(sut.sdfg, nodes.Tasklet) == 3 * tasklets
    a = np.random.default_rng(0).random(8)
    expected = a.copy()
    expected[:4] = np.cumsum(a[:4])
    sut.sdfg(a=a)
    np.testing.assert_allclose(a, expected, rtol=1e-14)


def test_a_loop_longer_than_the_cap_is_refused_and_unrolls_under_a_larger_cap(tmp_path):
    sut = session(long_scan, tmp_path)
    (loop,) = labels_of(sut, LoopRegion)

    refused = sut.apply_move("unroll-for", [loop])

    assert refused.status is MoveStatus.ILLEGAL and "19 iterations" in refused.reason, refused
    assert sut.list_moves("unroll-for") == []
    assert sut.apply_move("unroll-for", [loop], max_iterations=19).status is MoveStatus.APPLIED
    assert loops(sut.sdfg) == []


def test_a_loop_of_symbolic_trip_count_is_refused(tmp_path):
    sut = session(sized_scan, tmp_path)
    (loop,) = labels_of(sut, LoopRegion)

    result = sut.apply_move("unroll-for", [loop], max_iterations=1000)

    assert result.status is MoveStatus.ILLEGAL and "no constant trip count" in result.reason, result
    assert len(loops(sut.sdfg)) == 1


# unroll-doall


def test_a_constant_doall_unrolls_into_one_body_copy_per_iteration(tmp_path):
    sut = session(short_scale, tmp_path)
    tasklets = count(sut.sdfg, nodes.Tasklet)
    (doall,) = labels_of(sut, nodes.MapEntry)

    result = sut.apply_move("unroll-doall", [doall])

    assert (result.status, result.reason) == (MoveStatus.APPLIED, "MapUnroll"), result
    assert count(sut.sdfg, nodes.MapEntry) == 0
    assert count(sut.sdfg, nodes.Tasklet) == 4 * tasklets
    a, b = np.random.default_rng(1).random(4), np.zeros(4)
    sut.sdfg(a=a, b=b)
    np.testing.assert_allclose(b, 2.0 * a, rtol=0, atol=0)


def test_a_doall_longer_than_the_cap_is_refused_and_unrolls_under_a_larger_cap(tmp_path):
    sut = session(long_scale, tmp_path)
    (doall,) = labels_of(sut, nodes.MapEntry)

    refused = sut.apply_move("unroll-doall", [doall])

    assert refused.status is MoveStatus.ILLEGAL and "20 iterations" in refused.reason, refused
    assert sut.list_moves("unroll-doall") == []
    assert sut.apply_move("unroll-doall", [doall], max_iterations=20).status is MoveStatus.APPLIED
    assert count(sut.sdfg, nodes.MapEntry) == 0


def test_a_doall_of_symbolic_range_is_refused(tmp_path):
    sut = session(sized_scale, tmp_path)
    (doall,) = labels_of(sut, nodes.MapEntry)

    result = sut.apply_move("unroll-doall", [doall], max_iterations=1000)

    assert result.status is MoveStatus.ILLEGAL and "no constant trip count" in result.reason, result
    assert count(sut.sdfg, nodes.MapEntry) == 1


def test_a_doall_linked_to_another_computation_of_its_state_is_refused(tmp_path):
    sut = session(scale_then_shift, tmp_path)
    first, _ = labels_of(sut, nodes.MapEntry)

    result = sut.apply_move("unroll-doall", [first])

    assert result.status is MoveStatus.ILLEGAL and "shares arrays" in result.reason, result
    assert count(sut.sdfg, nodes.MapEntry) == 2


# propagate-config


def test_propagate_config_is_offered_on_no_row_only_when_there_is_a_value_to_give(tmp_path):
    offered = session(repeat, tmp_path / "a").list_moves("propagate-config")
    fixed = session(short_scan, tmp_path / "b").list_moves("propagate-config")

    assert [(move.kind, move.labels) for move in offered] == [("propagate-config", ())]
    assert fixed == []


def test_propagate_config_dispatches_to_a_specialized_body_whose_constant_loop_then_unrolls(tmp_path):
    sut = session(repeat, tmp_path)

    result = sut.apply_move("propagate-config", [], config_values={"K": 3})

    assert (result.status, result.reason) == (MoveStatus.APPLIED, "ConfigPropagation"), result
    (dispatch,) = conditionals(sut.sdfg)
    assert len(dispatch.branches) == 2 and dispatch.branches[1][0] is None  # the generic program is the else
    both = loops(sut.sdfg)
    assert len(both) == 2 and all(loop.sdfg is sut.sdfg for loop in both)  # inlined: rows of the tree
    (unroll,) = sut.list_moves("unroll-for")  # the specialized loop; the generic one runs K times
    assert sut.apply_move(unroll.kind, unroll.labels).status is MoveStatus.APPLIED
    assert len(loops(sut.sdfg)) == 1
    a = np.random.default_rng(2).random(SIZE)
    for steps in (3, 5):
        b = a.copy()
        sut.sdfg(a=b, N=SIZE, K=steps)
        np.testing.assert_allclose(b, repeated(a, steps), rtol=1e-14, err_msg=f"K={steps}")


def test_propagate_config_keeps_the_signature_and_the_codegen_flags(tmp_path):
    sut = session(repeat, tmp_path)
    sut.canonicalize()
    names, reductions = list(sut.sdfg.arg_names), sut.sdfg.openmp_array_reductions
    assert reductions, "canonicalization no longer sets openmp_array_reductions; this test pins nothing"

    sut.apply_move("propagate-config", [], config_values={"K": 3})

    assert (sut.sdfg.arg_names, sut.sdfg.openmp_array_reductions) == (names, reductions)


def test_propagate_config_without_values_with_unknown_names_or_values_dace_refuses_is_illegal(tmp_path):
    sut = session(repeat, tmp_path)

    cases = {
        "pass config_values": {},
        "['Q'] name no argument": {"Q": 1},
        "blocked by ConfigPropagation": {"a": 1.0},  # an array the program writes
    }

    for reason, values in cases.items():
        result = sut.apply_move("propagate-config", [], config_values=values)
        assert result.status is MoveStatus.ILLEGAL and reason in result.reason, (values, result)
    assert conditionals(sut.sdfg) == []


def test_propagate_config_is_closed_once_the_program_has_kernels(tmp_path):
    sut = session(repeat, tmp_path)
    sut.run_defaults("scopes")

    result = sut.apply_move("propagate-config", [], config_values={"K": 3})

    assert result.status is MoveStatus.ILLEGAL and "has kernels" in result.reason, result
    assert sut.list_moves("propagate-config") == []


def test_undo_takes_propagate_config_back_to_the_program_before_it(tmp_path):
    sut = session(repeat, tmp_path)
    sut.apply_move("propagate-config", [], config_values={"K": 3})
    assert len(conditionals(sut.sdfg)) == 1

    sut.undo()

    assert conditionals(sut.sdfg) == [] and len(loops(sut.sdfg)) == 1


if __name__ == "__main__":
    for test in (
        test_a_constant_loop_unrolls_into_one_body_copy_per_iteration,
        test_a_loop_longer_than_the_cap_is_refused_and_unrolls_under_a_larger_cap,
        test_a_loop_of_symbolic_trip_count_is_refused,
        test_a_constant_doall_unrolls_into_one_body_copy_per_iteration,
        test_a_doall_longer_than_the_cap_is_refused_and_unrolls_under_a_larger_cap,
        test_a_doall_of_symbolic_range_is_refused,
        test_a_doall_linked_to_another_computation_of_its_state_is_refused,
        test_propagate_config_is_offered_on_no_row_only_when_there_is_a_value_to_give,
        test_propagate_config_dispatches_to_a_specialized_body_whose_constant_loop_then_unrolls,
        test_propagate_config_keeps_the_signature_and_the_codegen_flags,
        test_propagate_config_without_values_with_unknown_names_or_values_dace_refuses_is_illegal,
        test_propagate_config_is_closed_once_the_program_has_kernels,
        test_undo_takes_propagate_config_back_to_the_program_before_it,
    ):
        test(pathlib.Path(tempfile.mkdtemp()))
