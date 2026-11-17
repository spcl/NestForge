# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tools by target (design D21, option (c)): on a session without a GPU target the placement tools have no device to
choose, so ``place``, ``insert_copies`` and ``unplace`` answer that and change nothing, while ``next_phase``'s default
placement still fixes the kernels. A GPU session keeps every tool working. Transformations only: no kernel is built."""

import dace
import numpy as np
import pytest
from dace.libraries.standard.nodes.external_call import external_calls
from dace.sdfg import nodes

from nestforge.records import MoveStatus
from nestforge.server import CALLS, NO_DEVICE_CHOICE, SCOPES_FIXED, SCOPES_FIXED_ON_CPU
from nestforge.session import MoveResult, PlaceResult, Session
from nestforge.stages.canonicalize import Targets

N = dace.symbol("N", dtype=dace.int64)

#: The tools that choose a device, and the arguments of one valid call of each.
PLACEMENT_CALLS: dict[str, dict[str, object]] = {"place": {}, "insert_copies": {}, "unplace": {}}


@dace.program
def chain(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    t = np.empty_like(a)
    for i in dace.map[0:N]:
        t[i] = a[i] + b[i]
    for i in dace.map[0:N]:
        c[i] = t[i] * 2.0


def sdfg_of(session: Session) -> dace.SDFG:
    """The session's program; a session always holds one."""
    assert session.sdfg is not None, "a session holds its program"
    return session.sdfg


def answer(session: Session, tool: str, **arguments: object) -> MoveResult | PlaceResult:
    """A tool's answer on ``session``, typed: the tools of this phase answer a move or a placement."""
    result = CALLS[tool](session, **arguments)
    assert isinstance(result, MoveResult | PlaceResult), result
    return result


def fresh(gpu: bool, tmp_path) -> Session:
    return Session(chain.to_sdfg(simplify=True), targets=Targets(gpu=gpu), work_dir=str(tmp_path))


def doalls(session: Session) -> list[str]:
    session.describe()
    return [label for label, (obj, _) in session.row_index().items() if isinstance(obj, nodes.MapEntry)]


def kernel_names(session: Session) -> list[str]:
    return [ext.name for ext in external_calls(sdfg_of(session))]


@pytest.mark.parametrize("tool", list(PLACEMENT_CALLS))
def test_a_placement_tool_on_a_cpu_session_answers_that_there_is_no_device_and_changes_nothing(tmp_path, tool):
    session = fresh(False, tmp_path)
    first, _ = doalls(session)
    answer(session, "set_scope", labels=[first])
    before, applied = sdfg_of(session).to_json(), list(session.applied)

    result = answer(session, tool, **PLACEMENT_CALLS[tool])

    assert result.status is MoveStatus.ILLEGAL and result.reason == NO_DEVICE_CHOICE
    assert sdfg_of(session).to_json() == before and session.applied == applied
    assert session.unplaced is None, "nothing is placed, so the scopes stay open"


def test_a_cpu_session_refuses_a_device_choice_naming_the_gpu_as_it_refuses_any_other(tmp_path):
    session = fresh(False, tmp_path)
    session.run_defaults("scopes")

    result = answer(session, "place", devices=dict.fromkeys(kernel_names(session), "gpu"))

    assert result.status is MoveStatus.ILLEGAL and result.reason == NO_DEVICE_CHOICE


def test_a_cpu_session_scopes_again_after_a_refused_place(tmp_path):
    session = fresh(False, tmp_path)
    first, second = doalls(session)
    answer(session, "set_scope", labels=[first])
    answer(session, "place")

    result = answer(session, "set_scope", labels=[second])

    assert result.status is MoveStatus.APPLIED and len(kernel_names(session)) == 2


def test_next_phase_still_places_a_cpu_session_on_the_cpu(tmp_path):
    session = fresh(False, tmp_path)

    CALLS["run_defaults"](session, stage="placement")

    assert session.unplaced is not None and kernel_names(session)
    assert set(session.devices().values()) == {"cpu"}


def test_a_placed_cpu_session_names_a_new_round_not_unplace_when_it_refuses_a_scope(tmp_path):
    session = fresh(False, tmp_path)
    first, _ = doalls(session)
    CALLS["run_defaults"](session, stage="placement")

    with pytest.raises(ValueError) as refused:
        answer(session, "set_scope", labels=[first])

    assert str(refused.value) == SCOPES_FIXED_ON_CPU and "unplace" not in str(refused.value)


def test_a_placed_gpu_session_names_unplace_when_it_refuses_a_scope(tmp_path):
    session = fresh(True, tmp_path)
    CALLS["run_defaults"](session, stage="placement")

    with pytest.raises(ValueError) as refused:
        answer(session, "unset_scope", name=kernel_names(session)[0])

    assert str(refused.value) == SCOPES_FIXED


def test_a_gpu_session_keeps_every_placement_tool(tmp_path):
    session = fresh(True, tmp_path)
    session.run_defaults("scopes")

    placed = answer(session, "place")
    copied = answer(session, "insert_copies")
    unplaced = answer(session, "unplace")

    assert isinstance(placed, PlaceResult) and isinstance(copied, PlaceResult)
    assert placed.status is MoveStatus.APPLIED and "gpu" in placed.devices.values()
    assert copied.status is MoveStatus.APPLIED and copied.copies
    assert unplaced.status is MoveStatus.APPLIED and session.unplaced is None


def test_the_kernel_work_gate_comes_before_the_target_on_a_cpu_session(tmp_path):
    session = fresh(False, tmp_path)
    CALLS["run_defaults"](session, stage="placement")
    CALLS["focus_kernel"](session, name=kernel_names(session)[0])

    with pytest.raises(ValueError, match="kernel work has started"):
        answer(session, "unplace")
