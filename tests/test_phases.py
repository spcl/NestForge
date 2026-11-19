# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""NestForge's own phases: ``next_phase`` runs the defaults of the phases between, ``prev_phase`` takes the current
phase back completely, ``goto_phase`` returns to an earlier one, and every tool works only in its phases."""

import dace
import numpy as np
from dace.libraries.standard.nodes.external_call import external_calls
from dace.sdfg import nodes

from nestforge.server import answer
from nestforge.session import Session

N = dace.symbol("N", dtype=dace.int64)


@dace.program
def three_stages(A: dace.float64[N], C: dace.float64[N]):
    T = np.empty_like(A)
    for i in dace.map[0:N]:
        T[i] = A[i] + 1.0
    for i in dace.map[0:N]:
        C[i] = T[i] * 2.0


def fresh(agent_phases: tuple[str, ...] = ()) -> Session:
    sut = Session(three_stages.to_sdfg(simplify=True), agent_phases=agent_phases)
    sut.canonicalize()
    sut.enter_first_phase()
    return sut


def doalls(sut: Session) -> list[str]:
    sut.describe()
    return [label for label, (obj, _) in sut.row_index().items() if isinstance(obj, nodes.MapEntry)]


def test_prev_phase_takes_back_the_scopes_and_the_placement_the_way_in_ran():
    sut = fresh()
    structured_tree = sut.describe()
    sut.next_phase()
    sut.set_scope([doalls(sut)[0]])
    scoped_kernels = {ext.name for ext in external_calls(sut.sdfg)}

    into_scoped = sut.next_phase()
    back_to_scopes = sut.prev_phase()
    kernels_as_left = {ext.name for ext in external_calls(sut.sdfg)}
    unplaced_as_left = sut.unplaced is None
    back_to_structured = sut.prev_phase()

    assert (into_scoped.status, into_scoped.phase) == ("moved", "scoped")
    assert any(ran.startswith("placement") for ran in into_scoped.ran), "leaving scopes places every kernel"
    assert (back_to_scopes.phase, back_to_scopes.undone) == ("scopes", ("scoped",))
    assert kernels_as_left == scoped_kernels and unplaced_as_left, "the scopes as the agent left them, not placed"
    assert back_to_structured.phase == "structured"
    assert not external_calls(sut.sdfg) and sut.describe() == structured_tree


def test_goto_phase_returns_to_an_earlier_phase_and_refuses_a_later_one():
    sut = fresh()
    tree = sut.describe()
    for _ in range(3):
        sut.next_phase()
    last = sut.next_phase()

    later = sut.goto_phase("kernels")
    back = sut.goto_phase("structured")

    assert (last.status, last.phase) == ("refused", "kernels")
    assert later.status == "refused"
    assert (back.status, back.undone) == ("moved", ("kernels", "scoped", "scopes"))
    assert sut.describe() == tree


def test_a_packet_without_the_structured_phase_starts_in_scopes_after_the_default_moves():
    sut = fresh(("scopes", "kernels"))

    first = sut.prev_phase()
    step = sut.next_phase()

    assert (first.status, first.phase) == ("refused", "scopes"), "the agent's first phase has no way back"
    assert (step.phase, step.ran) == (
        "kernels",
        ("placement: every kernel placed", "scoped: no move inside the kernels"),
    )


def test_a_tool_of_another_phase_is_refused_and_moves_inside_a_kernel_wait_for_the_scoped_phase():
    sut = fresh()

    early_scope = answer(sut, "set_scope", {"labels": [doalls(sut)[0]]}, None)
    scoped_move = answer(sut, "list_moves", {"scope": "anything"}, None)
    sut.next_phase()
    late_move = answer(sut, "apply_move", {"kind": "doall-fusion", "labels": doalls(sut)[:2]}, None)

    assert "phase scopes" in early_scope.error and "structured" in early_scope.error
    assert "scoped" in scoped_move.error
    assert "phase scoped, structured" in late_move.error and "in the phase scopes" in late_move.error


def test_leaving_a_phase_summarizes_its_moves_and_the_kernels():
    sut = fresh()
    (move, *_) = sut.list_moves("doall-fusion")
    sut.apply_move(move.kind, move.labels)

    left_structured = sut.next_phase()
    left_scopes = sut.next_phase()

    assert left_structured.summary.splitlines()[:2] == [
        "structured: 1 move(s) applied",
        f"  doall-fusion {' '.join(move.labels)}",
    ]
    assert left_scopes.summary.startswith("scopes: 0 move(s) applied")
    assert "kernels: " in left_scopes.summary and " on cpu" in left_scopes.summary
