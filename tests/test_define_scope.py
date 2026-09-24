# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Session.define_scope: one kernel over regions no scheduling move fuses, so the kernel written in phase 4 fuses
them instead."""

import copy
import os

import numpy as np
import pytest

import dace
from dace.sdfg import nodes

from nestforge.ir.libnode import ExternalCall, external_calls
from nestforge.session import Session

N = dace.symbol("N", dtype=dace.int64)
SIZE = 9


@dace.program
def live_intermediate(A: dace.float64[N], T: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        T[i] = A[i] + 1.0
    for i in dace.map[0:N]:
        C[i] = T[i] * 2.0


@dace.program
def three_stages(A: dace.float64[N], C: dace.float64[N]):
    T = np.empty_like(A)
    U = np.empty_like(A)
    for i in dace.map[0:N]:
        T[i] = A[i] + 1.0
    for i in dace.map[0:N]:
        U[i] = T[i] * 2.0
    for i in dace.map[0:N]:
        C[i] = U[i] - 3.0


def session_and_reference(program, simplify: bool = True) -> tuple[Session, dace.SDFG]:
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


def assert_same_values(reference: dace.SDFG, grouped: dace.SDFG, names: tuple[str, ...]) -> None:
    rng = np.random.default_rng(0)
    arrays = {name: rng.random(SIZE) for name in names}
    expected = {name: value.copy() for name, value in arrays.items()}
    actual = {name: value.copy() for name, value in arrays.items()}
    reference(**expected, N=SIZE)
    grouped(**actual, N=SIZE)
    for name in names:
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)


def test_two_maps_no_fusion_accepts_become_one_kernel():
    """``T`` is a program output, so vertical fusion refuses the pair; one kernel may still compute both."""
    session, reference = session_and_reference(live_intermediate)
    assert session.list_moves("map-fusion") == [], "the fixture must offer no fusion, else it tests nothing"
    labels = map_labels(session)

    result = session.define_scope(labels, session.epoch)

    assert result.status == "applied", result
    (kernel,) = external_calls(session.sdfg)
    assert session.resolve(result.reason, "kernel") is kernel
    assert sorted(session.kernel_boundary(result.reason)["outputs"]) == ["C", "T"]
    assert_same_values(reference, session.sdfg, ("A", "T", "C"))


def test_maps_in_different_states_are_grouped_by_their_blocks():
    session, reference = session_and_reference(live_intermediate, simplify=False)
    states = map_state_labels(session)
    assert len(states) == 2, states

    result = session.define_scope(states, session.epoch)

    assert result.status == "applied", result
    assert len(external_calls(session.sdfg)) == 1
    assert_same_values(reference, session.sdfg, ("A", "T", "C"))


def test_a_map_running_between_the_named_maps_is_refused():
    """The middle map reads the first and feeds the last, so it would have to run inside the kernel."""
    session, _ = session_and_reference(three_stages)
    first, _, last = map_labels(session)
    before = session.epoch

    result = session.define_scope([first, last], before)

    assert result.status == "illegal" and "runs between" in result.reason, result
    assert session.epoch == before and external_calls(session.sdfg) == []


def test_blocks_that_do_not_follow_each_other_are_refused():
    session, _ = session_and_reference(three_stages, simplify=False)
    first, _, last = map_state_labels(session)

    result = session.define_scope([first, last], session.epoch)

    assert result.status == "illegal" and "straight run" in result.reason, result
    assert external_calls(session.sdfg) == []


def test_define_scopes_lowers_the_maps_left_after_a_group():
    session, reference = session_and_reference(three_stages)
    first, second, _ = map_labels(session)
    grouped = session.define_scope([first, second], session.epoch)
    assert grouped.status == "applied", grouped

    rest = session.define_scopes()

    names = [ext.name for ext in external_calls(session.sdfg)]
    assert len(rest) == 1 and len(names) == 2 and len(set(names)) == 2, names
    assert all(isinstance(ext, ExternalCall) for ext in external_calls(session.sdfg))
    assert_same_values(reference, session.sdfg, ("A", "C"))


def test_labels_from_an_old_epoch_are_stale():
    session, _ = session_and_reference(live_intermediate)
    labels = map_labels(session)
    session.normalize()

    assert session.define_scope(labels, 0).status == "stale"


@pytest.mark.e2e
def test_a_grouped_kernel_builds_its_library_and_the_program_still_computes(tmp_path):
    """Phase 4 takes a grouped kernel like any other: its CPF unit holds both maps, and the linked library
    reproduces the program."""
    sdfg = live_intermediate.to_sdfg(simplify=True)
    sdfg.name = f"{sdfg.name}_{os.getpid()}_lib"
    reference = copy.deepcopy(sdfg)
    reference.name = f"{sdfg.name}_reference"
    session = Session(sdfg, work_dir=str(tmp_path))
    kernel_id = session.define_scope(map_labels(session), session.epoch).reason

    info = session.optimize_kernel(kernel_id)

    assert session.resolve(kernel_id, "kernel").implementation == "ExternCall"
    assert sorted(info["abi_order"]) == ["A", "C", "N", "T"]
    assert_same_values(reference, session.sdfg, ("A", "T", "C"))
