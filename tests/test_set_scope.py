# Copyright 2021 ETH Zurich and the Tactics authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Session.set_scope: one kernel over regions no scheduling move fuses, so the kernel written in stage 5 fuses
them instead."""

import copy
import os
import warnings
from pathlib import Path

import dace
import numpy as np
import pytest
from dace.frontend.python.parser import DaceProgram
from dace.libraries.standard.nodes.external_call import ExternalCall, external_calls
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion, SDFGState
from test_fusion_fuzz import load_program
from test_placement import chain
from test_rollback_equivalence import content_of

from tactics.records import MoveStatus
from tactics.server import Setup, started_session
from tactics.session import Session
from tactics.stages.canonicalize import Targets
from tactics.stages.scopes import kernel_arguments, parallel_top_level_maps

N = dace.symbol("N", dtype=dace.int64)
SIZE = 9


@dace.program
def live_intermediate(A: dace.float64[N], T: dace.float64[N], C: dace.float64[N]) -> None:
    for i in dace.map[0:N]:
        T[i] = A[i] + 1.0
    for i in dace.map[0:N]:
        C[i] = T[N - 1 - i] * 2.0  # another iteration's element: no fusion accepts the pair


@dace.program
def three_stages(A: dace.float64[N], C: dace.float64[N]) -> None:
    T = np.empty_like(A)
    U = np.empty_like(A)
    for i in dace.map[0:N]:
        T[i] = A[i] + 1.0
    for i in dace.map[0:N]:
        U[i] = T[i] * 2.0
    for i in dace.map[0:N]:
        C[i] = U[i] - 3.0


@dace.program
def dot_scale(A: dace.float64[N], C: dace.float64[N]) -> None:
    s = np.zeros([1])
    for i in dace.map[0:N]:
        s[0] += A[i] * A[i]
    for i in dace.map[0:N]:
        C[i] = A[i] / s[0]


@dace.program
def column_prefix(A: dace.float64[N, N], B: dace.float64[N, N]) -> None:
    for i in range(1, N):
        for j in range(N):
            A[i, j] = A[i - 1, j] + B[i, j]


@dace.program
def running_sum(x: dace.float64[N], out: dace.float64[N]) -> None:
    for i in range(1, N):
        out[i] = out[i - 1] + x[i]


@dace.program
def smooth_steps(A: dace.float64[N], B: dace.float64[N]) -> None:
    for t in range(3):
        for i in range(1, N - 1):
            B[i] = (A[i - 1] + A[i] + A[i + 1]) / 3.0
        A[1 : N - 1] = B[1 : N - 1] + 1.0


@dace.program
def time_loop_of_maps(A: dace.float64[N], B: dace.float64[N]) -> None:
    for _ in range(3):
        for i in dace.map[1 : N - 1]:
            B[i] = (A[i - 1] + A[i] + A[i + 1]) / 3.0
        for i in dace.map[1 : N - 1]:
            A[i] = B[i] + 1.0


@dace.program
def map_then_matmul(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]) -> None:
    T = np.empty_like(A)
    for i, j in dace.map[0:N, 0:N]:
        T[i, j] = A[i, j] * 2.0
    C[:] = T @ B


def session_and_reference(program: DaceProgram, simplify: bool = True) -> tuple[Session, dace.SDFG]:
    sdfg = program.to_sdfg(simplify=simplify)
    sdfg.name = f"{sdfg.name}_{os.getpid()}"  # one build folder per xdist worker
    reference = copy.deepcopy(sdfg)
    reference.name = f"{sdfg.name}_reference"
    return Session(sdfg), reference


def map_labels(session: Session) -> list[str]:
    session.describe()
    return [label for label, (obj, _) in session.row_index().items() if isinstance(obj, nodes.MapEntry)]


def map_state_labels(session: Session) -> list[str]:
    session.describe()
    return [
        label
        for label, (obj, state) in session.row_index().items()
        if state is None and isinstance(obj, dace.SDFGState) and any(isinstance(n, nodes.MapEntry) for n in obj.nodes())
    ]


def assert_same_values(reference: dace.SDFG, grouped: dace.SDFG, names: tuple[str, ...], rtol: float = 0.0) -> None:
    rng = np.random.default_rng(0)
    arrays = {name: rng.random(SIZE) for name in names}
    expected = {name: value.copy() for name, value in arrays.items()}
    actual = {name: value.copy() for name, value in arrays.items()}
    reference(**expected, N=SIZE)
    grouped(**actual, N=SIZE)
    for name in names:
        np.testing.assert_allclose(actual[name], expected[name], rtol=rtol, atol=0, err_msg=name)


def test_two_maps_no_fusion_accepts_become_one_kernel() -> None:
    """``T`` is a program output, so vertical fusion refuses the pair; one kernel may still compute both."""
    session, reference = session_and_reference(live_intermediate)
    assert session.list_moves("doall-fusion") == [], "the fixture must offer no fusion, else it tests nothing"
    labels = map_labels(session)

    result = session.set_scope(labels)

    assert result.status is MoveStatus.APPLIED, result
    (kernel,) = external_calls(session.sdfg)
    assert session.kernel(result.reason) is kernel
    assert sorted(kernel_arguments(kernel)[1]) == ["C", "T"]
    assert_same_values(reference, session.sdfg, ("A", "T", "C"))


def test_maps_in_different_states_are_grouped_by_their_blocks() -> None:
    session, reference = session_and_reference(live_intermediate, simplify=False)
    states = map_state_labels(session)
    assert len(states) == 2, states

    result = session.set_scope(states)

    assert result.status is MoveStatus.APPLIED, result
    assert len(external_calls(session.sdfg)) == 1
    assert_same_values(reference, session.sdfg, ("A", "T", "C"))


def test_a_map_running_between_the_named_maps_is_refused() -> None:
    """The middle map reads the first and feeds the last, so it would have to run inside the kernel."""
    session, _ = session_and_reference(three_stages)
    first, _, last = map_labels(session)
    before = session.sdfg.to_json()

    result = session.set_scope([first, last])

    assert result.status is MoveStatus.ILLEGAL and "runs between" in result.reason, result
    assert session.sdfg.to_json() == before and external_calls(session.sdfg) == []


def test_blocks_that_do_not_follow_each_other_are_refused() -> None:
    session, _ = session_and_reference(three_stages, simplify=False)
    first, _, last = map_state_labels(session)

    result = session.set_scope([first, last])

    assert result.status is MoveStatus.ILLEGAL and "straight run" in result.reason, result
    assert external_calls(session.sdfg) == []


def test_default_scopes_lowers_the_maps_left_after_a_group() -> None:
    session, reference = session_and_reference(three_stages)
    first, second, _ = map_labels(session)
    grouped = session.set_scope([first, second])
    assert grouped.status is MoveStatus.APPLIED, grouped
    grouped_kernel = session.kernel(grouped.reason)

    rest = session.default_scopes()

    names = [ext.name for ext in external_calls(session.sdfg)]
    assert len(rest) == 1 and len(names) == 2 and len(set(names)) == 2, names
    assert kernel_arguments(grouped_kernel)[:2] == (["A"], ["U"]), "T stays inside the grouped kernel"
    assert_same_values(reference, session.sdfg, ("A", "C"))


def test_a_kernels_problem_sized_scope_array_is_a_heap_allocation_not_a_stack_array() -> None:
    """``T`` lives inside the grouped kernel and has the problem's size ``N``: on the stack it would be a
    variable-length array that grows with the input until the stack overflows."""
    session, reference = session_and_reference(three_stages)
    first, second, _ = map_labels(session)
    session.set_scope([first, second])
    session.default_scopes()
    (grouped,) = [ext for ext in external_calls(session.sdfg) if "T" in ext.standalone_sdfg.arrays]

    assert grouped.standalone_sdfg.arrays["T"].storage == dace.StorageType.CPU_Heap
    assert "double T[" not in session.kernel_source(grouped.name), "T is no variable-length array in the unit"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session.sdfg(A=np.ones(SIZE), C=np.zeros(SIZE), N=SIZE)
    assert not [w for w in caught if "Variable-length array" in str(w.message)]
    assert_same_values(reference, session.sdfg, ("A", "C"))


@pytest.mark.parametrize("simplify", [True, False])
def test_a_single_state_becomes_one_top_level_kernel(simplify: bool) -> None:
    """DaCe nests a lone state as itself, so the state's contents must become the kernel, not a map body."""
    session, reference = session_and_reference(live_intermediate, simplify=simplify)
    state_label = map_state_labels(session)[-1]

    result = session.set_scope([state_label])

    assert result.status is MoveStatus.APPLIED, result
    kernel = session.kernel(result.reason)
    (state,) = [s for s in session.sdfg.states() if kernel in s.nodes()]
    assert state.entry_node(kernel) is None, "the kernel sits inside a map"
    assert_same_values(reference, session.sdfg, ("A", "T", "C"))


def test_a_length1_transient_the_group_writes_first_is_not_an_input() -> None:
    """``s`` is written by the first map before the second reads it, so it never crosses the kernel boundary."""
    session, reference = session_and_reference(dot_scale, simplify=False)
    session.describe()
    blocks = [
        label for label, (obj, state) in session.row_index().items() if state is None and obj in session.sdfg.nodes()
    ]

    result = session.set_scope(blocks)

    assert result.status is MoveStatus.APPLIED, result
    # the parallel sum into s adds in thread order
    assert_same_values(reference, session.sdfg, ("A", "C"), rtol=1e-14)


def test_a_loop_and_the_state_after_it_inside_a_time_loop_become_one_kernel_in_the_loop_body() -> None:
    """A scope is any straight run of one region, not only of the top level: the kernel replaces the inner loop and
    the copy state and is called once per time step."""
    session, reference = session_and_reference(smooth_steps)
    session.describe()
    rows = session.row_index()
    time_loop = rows["for0"][0]
    after_inner = next(block.label for block in time_loop.nodes() if isinstance(block, SDFGState))

    result = session.set_scope(["for1", after_inner])

    assert result.status is MoveStatus.APPLIED, result
    kernel = session.kernel(result.reason)
    (body_state,) = time_loop.nodes()
    assert kernel in body_state.nodes(), "the kernel left the time loop"
    assert not any(isinstance(block, LoopRegion) for block in time_loop.all_control_flow_blocks())
    assert_same_values(reference, session.sdfg, ("A", "B"))


def test_blocks_at_different_levels_are_refused() -> None:
    session, _ = session_and_reference(smooth_steps)
    session.describe()
    before = session.sdfg.to_json()

    result = session.set_scope(["for0", "for1"])

    assert result.status is MoveStatus.ILLEGAL, result
    assert "one level" in result.reason
    assert session.sdfg.to_json() == before and external_calls(session.sdfg) == []


def test_no_labels_is_refused() -> None:
    session, _ = session_and_reference(live_intermediate)
    before = session.sdfg.to_json()

    result = session.set_scope([])

    assert result.status is MoveStatus.ILLEGAL, result
    assert session.sdfg.to_json() == before and external_calls(session.sdfg) == []


def test_labels_a_scope_consumed_are_refused() -> None:
    session, _ = session_and_reference(live_intermediate)
    labels = map_labels(session)
    session.set_scope(labels)

    assert session.set_scope(labels).status is MoveStatus.NOT_FOUND


@pytest.mark.e2e
def test_a_grouped_kernel_builds_its_library_and_the_program_still_computes(tmp_path: Path) -> None:
    """Stage 5 takes a grouped kernel like any other: its CPF unit holds both maps, and the linked library
    reproduces the program."""
    sdfg = live_intermediate.to_sdfg(simplify=True)
    sdfg.name = f"{sdfg.name}_{os.getpid()}_lib"
    reference = copy.deepcopy(sdfg)
    reference.name = f"{sdfg.name}_reference"
    session = Session(sdfg, work_dir=str(tmp_path))
    name = session.set_scope(map_labels(session)).reason

    info = session.optimize_kernel(name)

    assert session.kernel(name).implementation == "ExternCall"
    assert sorted(info.abi_order) == ["A", "C", "N", "T"]
    assert_same_values(reference, session.sdfg, ("A", "T", "C"))


def test_unsetting_a_scope_inlines_the_kernel_back_and_the_program_still_computes() -> None:
    sdfg = live_intermediate.to_sdfg(simplify=True)
    reference = copy.deepcopy(sdfg)
    reference.name = f"{sdfg.name}_reference"
    session = Session(sdfg)
    name = session.set_scope(map_labels(session)).reason

    result = session.unset_scope(name)

    assert (result.status, result.reason) == (MoveStatus.APPLIED, name)
    assert not external_calls(session.sdfg)
    assert len(map_labels(session)) == 2
    assert_same_values(reference, session.sdfg, ("A", "T", "C"))


def test_unsetting_a_scope_brings_back_the_containers_the_kernel_kept_to_itself() -> None:
    session, reference = session_and_reference(three_stages)
    name = session.set_scope(map_labels(session)).reason

    result = session.unset_scope(name)

    assert result.status is MoveStatus.APPLIED, result
    assert {"T", "U"} <= {array for array, desc in session.sdfg.arrays.items() if desc.transient}
    assert_same_values(reference, session.sdfg, ("A", "C"))


def test_unsetting_no_kernel_changes_nothing() -> None:
    session = Session(live_intermediate.to_sdfg(simplify=True))
    name = session.set_scope(map_labels(session)).reason

    missing = session.unset_scope("extcall_9")

    assert missing.status is MoveStatus.NOT_FOUND
    assert [ext.name for ext in external_calls(session.sdfg)] == [name]


# loop nests no move can reshape


def top_level_kernels(session: Session) -> list[ExternalCall]:
    return [n for state in session.sdfg.states() for n in state.nodes() if isinstance(n, ExternalCall)]


def test_a_loop_nest_no_interchange_applies_to_becomes_one_kernel_that_computes_the_column_prefix_sum() -> None:
    session, _ = session_and_reference(column_prefix)
    rng = np.random.default_rng(0)
    A, B = rng.random((SIZE, SIZE)), rng.random((SIZE, SIZE))
    expected = A.copy()
    expected[1:] = A[0] + np.cumsum(B[1:], axis=0)

    result = session.set_scope(["for0"])

    assert result.status is MoveStatus.APPLIED, result
    (kernel,) = top_level_kernels(session)
    assert session.kernel(result.reason) is kernel
    assert not any(isinstance(block, LoopRegion) for block in session.sdfg.nodes()), "the loop nest left the program"
    session.sdfg(A=A, B=B, N=SIZE)
    np.testing.assert_allclose(A, expected, rtol=1e-12)


def test_a_single_sequential_loop_nest_can_be_scoped_explicitly() -> None:
    session, _ = session_and_reference(running_sum)
    x, out = np.arange(SIZE, dtype=np.float64), np.zeros(SIZE)
    assert parallel_top_level_maps(session.sdfg) == [], "no map: the loop is the only region to name"

    result = session.set_scope(["for0"])

    assert result.status is MoveStatus.APPLIED, result
    assert len(top_level_kernels(session)) == 1
    assert not any(isinstance(block, LoopRegion) for block in session.sdfg.nodes()), "the loop left the program"
    assert not session.list_kernels()[0].parallel, "a loop kernel is sequential, so placement keeps it on the CPU"
    session.sdfg(x=x, out=out, N=SIZE)
    np.testing.assert_array_equal(out, [0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0, 36.0])


# a loop that holds parallel maps, and a map next to a library call


def test_a_time_loop_holding_parallel_maps_becomes_one_kernel_called_once() -> None:
    """The default makes a kernel of each map inside the loop, so the loop calls two kernels per step; naming the loop
    makes the whole loop one kernel, called once."""
    default, _ = session_and_reference(time_loop_of_maps)
    assert len(default.default_scopes()) == 2, (
        "the fixture must have a kernel per map by default, else it tests nothing"
    )
    session, reference = session_and_reference(time_loop_of_maps)
    session.describe()

    result = session.set_scope(["for0"])

    assert result.status is MoveStatus.APPLIED, result
    (kernel,) = top_level_kernels(session)
    assert session.kernel(result.reason) is kernel
    scoped = session.sdfg
    assert scoped is not None
    assert not any(isinstance(block, LoopRegion) for block in scoped.nodes()), "the loop left the program"
    assert not session.list_kernels()[0].parallel, "a kernel holding a time loop is sequential"
    assert_same_values(reference, scoped, ("A", "B"))


def test_a_map_and_the_library_call_after_it_become_one_kernel_that_computes_the_product() -> None:
    session, reference = session_and_reference(map_then_matmul)
    session.describe()
    rows = session.row_index()
    labels = [label for label, (obj, _) in rows.items() if isinstance(obj, (nodes.MapEntry, nodes.LibraryNode))]
    assert len(labels) == 2, labels
    rng = np.random.default_rng(0)
    A, B = rng.random((SIZE, SIZE)), rng.random((SIZE, SIZE))

    result = session.set_scope(labels)

    assert result.status is MoveStatus.APPLIED, result
    (kernel,) = top_level_kernels(session)
    assert session.kernel(result.reason) is kernel
    expected, actual = np.zeros((SIZE, SIZE)), np.zeros((SIZE, SIZE))
    scoped = session.sdfg
    assert scoped is not None
    reference(A=A.copy(), B=B.copy(), C=expected, N=SIZE)
    scoped(A=A.copy(), B=B.copy(), C=actual, N=SIZE)
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
    np.testing.assert_allclose(actual, (A * 2.0) @ B, rtol=1e-12)


# Reproducer: unset_scope left an invalid nested SDFG (#97, fuzz seed 1702)
def test_unsetting_a_kernel_whose_body_stays_nested_leaves_a_valid_program(tmp_path: Path) -> None:
    """Issue #97 (the fusion fuzz program of seed 1702): one doall scoped and unset again. The kernel's body is not
    inlined; the nested SDFG that stays keeps its arguments, named as the connectors (``_out_...``), non-transient."""
    program, _ = load_program(tmp_path, 1702)
    session = Session(program.to_sdfg(simplify=True), targets=Targets(gpu=False), work_dir=str(tmp_path / "work"))
    session.canonicalize()
    session.describe()
    doalls = [label for label, (obj, _) in session.row_index().items() if isinstance(obj, nodes.MapEntry)]
    kernel = session.set_scope(doalls[1:2]).reason

    assert session.unset_scope(str(kernel)).status is MoveStatus.APPLIED
    session.sdfg.validate()


def test_naming_an_empty_state_is_refused_and_changes_nothing(tmp_path: Path) -> None:
    """Inlining a kernel leaves its state empty, and the tree still names it; scoping it crashed in DaCe's
    ``nest_state_subgraph`` with ``KeyError: -1`` (the placement sequence fuzz, seed 1)."""
    session = Session(chain.to_sdfg(simplify=True), targets=Targets(gpu=False), work_dir=str(tmp_path))
    session.canonicalize()
    for kernel in session.default_scopes():
        assert session.unset_scope(kernel.name).status is MoveStatus.APPLIED
    session.describe()
    empty = [label for label, (obj, _) in session.row_index().items() if isinstance(obj, SDFGState) and not obj.nodes()]
    assert empty, "the fixture must leave an empty state, else it tests nothing"
    before = content_of(session.sdfg)

    result = session.set_scope(empty[:1])

    assert result.status is MoveStatus.ILLEGAL, result
    assert "empty" in str(result.reason)
    assert content_of(session.sdfg) == before and not external_calls(session.sdfg)


# Reproducer: scoped step state dropped the scalar an edge reads (#111, cg)
def test_a_scoped_state_keeps_the_scalar_the_convergence_test_after_it_reads(tmp_path: Path) -> None:
    """spcl/tactics#111: ``cg``'s step state writes ``sqrt_rsnew`` and the edge after it tests it; the kernel the state
    becomes still writes it where the edge reads it, so the program validates after the scopes are finalized."""
    sut = started_session(Setup("cg", "S", "float64", True, str(tmp_path)), codegen="first")
    sut.run_defaults("moves")
    (label,) = [
        label
        for label, (obj, parent) in sut.row_index().items()
        if parent is None
        and isinstance(obj, SDFGState)
        and any("sqrt_rsnew" in edge.data.free_symbols for edge in obj.parent_graph.out_edges(obj))
    ]

    result = sut.set_scope([label])
    sut.run_defaults("scopes")
    finalized = sut.finalize_scope()

    assert (result.status, finalized.status) == (MoveStatus.APPLIED, MoveStatus.APPLIED), (result, finalized)
    sut.sdfg.validate()
