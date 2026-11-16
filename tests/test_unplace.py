# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""unplace: from the devices back to the scopes inside one phase. It drops the placement and its copies, keeps the
kernels and the moves inside them, reopens set_scope and unset_scope, and is refused once the kernel work started."""

import json

import dace
import numpy as np
import pytest
from dace.libraries.standard.helper import GPU_RESIDENT_STORAGES
from dace.libraries.standard.nodes.external_call import external_calls
from dace.sdfg import nodes
from test_failed_calls import program, without_guids

from nestforge.records import MoveStatus
from nestforge.server import CALLS
from nestforge.session import PlaceResult, Session
from nestforge.stages.canonicalize import Targets

N = dace.symbol("N", dtype=dace.int64)

#: The two targets the session's own calls run on: placement is a choice on the GPU and a no-op on the CPU.
TARGETS = [pytest.param(False, id="cpu"), pytest.param(True, id="gpu")]
#: The target the tools choose a device on; on a CPU session they change nothing (tests/test_tools_by_target.py), so
#: a journal never holds them there.
TOOL_TARGET = [pytest.param(True, id="gpu")]


@dace.program
def chain(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    t = np.empty_like(a)
    for i in dace.map[0:N]:
        t[i] = a[i] + b[i]
    for i in dace.map[0:N]:
        c[i] = t[i] * 2.0


def fresh(gpu: bool, tmp_path) -> Session:
    return Session(chain.to_sdfg(simplify=True), targets=Targets(gpu=gpu), work_dir=str(tmp_path))


def doalls(session: Session) -> list[str]:
    session.describe()
    return [label for label, (obj, _) in session.row_index().items() if isinstance(obj, nodes.MapEntry)]


def device_storages(session: Session) -> list[str]:
    return [name for name, desc in session.sdfg.arrays.items() if desc.storage in GPU_RESIDENT_STORAGES]


def unnamed_states(value: object) -> object:
    """``value`` without the names DaCe's offloading gives the copy states it adds, which count every state it made."""
    if isinstance(value, dict):
        named = value.get("type") == "SDFGState"
        return {key: unnamed_states(item) for key, item in value.items() if not (named and key == "label")}
    return [unnamed_states(item) for item in value] if isinstance(value, list) else value


def structure(session: Session) -> str:
    return json.dumps(unnamed_states(without_guids(session.sdfg.to_json())), sort_keys=True, default=str)


def replay(gpu: bool, tmp_path, calls: list[tuple[str, dict]]) -> Session:
    """A fresh session repeating ``calls`` through the tools, as the route rebuilds one from its journal."""
    session = fresh(gpu, tmp_path)
    for tool, arguments in calls:
        CALLS[tool](session, **arguments)
    return session


@pytest.mark.parametrize("gpu", TARGETS)
def test_unplace_reopens_the_scopes_and_keeps_the_kernels(tmp_path, gpu):
    session = fresh(gpu, tmp_path)
    first, second = doalls(session)
    kernel = session.set_scope([first]).reason
    session.place()  # gives second its default scope first
    kernels = [ext.name for ext in external_calls(session.sdfg)]

    result = session.unplace()

    assert result.status is MoveStatus.APPLIED and session.unplaced is None
    assert [ext.name for ext in external_calls(session.sdfg)] == kernels and kernel in kernels and len(kernels) == 2
    (default,) = set(kernels) - {kernel}
    assert session.unset_scope(default).status is MoveStatus.APPLIED
    assert CALLS["set_scope"](session, labels=[second]).status is MoveStatus.APPLIED
    assert len(external_calls(session.sdfg)) == 2


@pytest.mark.parametrize("gpu", TARGETS)
def test_the_first_place_scopes_what_no_scope_covers_and_keeps_the_devices_it_was_given(tmp_path, gpu):
    session = fresh(gpu, tmp_path)
    first, _ = doalls(session)
    kernel = session.set_scope([first]).reason
    default = fresh(gpu, tmp_path / "default")
    default.run_defaults("placement")

    assert session.place({kernel: "cpu"}).status is MoveStatus.APPLIED
    assert doalls(session) == [] and len(session.devices()) == len(default.devices())
    assert session.devices()[kernel] == "cpu"
    again = session.run_defaults("placement")  # on a GPU it inserts the copies; on a CPU nothing is left to do
    assert isinstance(again, PlaceResult) and again.status is (MoveStatus.APPLIED if gpu else MoveStatus.ILLEGAL)
    assert session.devices()[kernel] == "cpu"


def test_unplace_drops_the_device_memory_and_schedules_of_inserted_copies(tmp_path):
    session = fresh(True, tmp_path)
    session.run_defaults("scopes")
    session.place()
    session.insert_copies()
    assert device_storages(session)

    session.unplace()

    assert device_storages(session) == []
    assert all(ext.schedule != dace.ScheduleType.GPU_Device for ext in external_calls(session.sdfg))


@pytest.mark.parametrize("gpu", TOOL_TARGET)
def test_unplace_then_place_again_is_the_program_of_placing_once(tmp_path, gpu):
    first, _ = doalls(fresh(gpu, tmp_path))
    once = [("set_scope", {"labels": [first]}), ("place", {}), ("insert_copies", {})]
    detour = replay(gpu, tmp_path / "detour", [*once, ("unplace", {}), ("place", {}), ("insert_copies", {})])
    direct = replay(gpu, tmp_path / "direct", once)

    assert detour.devices() == direct.devices()
    assert sorted(device_storages(detour)) == sorted(device_storages(direct))
    assert structure(detour) == structure(direct)


@pytest.mark.parametrize("gpu", TOOL_TARGET)
def test_unplace_then_scope_then_place_puts_the_kernels_where_scoping_first_does(tmp_path, gpu):
    first, second = doalls(fresh(gpu, tmp_path))
    detour = replay(
        gpu,
        tmp_path / "detour",
        [
            ("set_scope", {"labels": [first]}),
            ("place", {}),
            ("insert_copies", {}),
            ("unplace", {}),
            ("unset_scope", {"name": "extcall_doall1"}),  # the default scope the first place gave second
            ("set_scope", {"labels": [second]}),
            ("place", {}),
            ("insert_copies", {}),
        ],
    )
    direct = replay(
        gpu,
        tmp_path / "direct",
        [("set_scope", {"labels": [first]}), ("set_scope", {"labels": [second]}), ("place", {}), ("insert_copies", {})],
    )

    # unset_scope leaves the kernel's boundary containers in the program (as on origin/main), so the SDFGs differ there
    assert detour.devices() == direct.devices()
    assert sorted(device_storages(detour)) == sorted(device_storages(direct))


@pytest.mark.parametrize("gpu", TOOL_TARGET)
def test_a_journal_with_unplace_rebuilds_the_same_program(tmp_path, gpu):
    first, second = doalls(fresh(gpu, tmp_path))
    calls = [("set_scope", {"labels": [first]}), ("place", {}), ("unplace", {}), ("set_scope", {"labels": [second]})]

    once, again = replay(gpu, tmp_path / "once", calls), replay(gpu, tmp_path / "again", calls)

    assert program(once) == program(again) and once.applied == again.applied


@pytest.mark.parametrize("gpu", TARGETS)
def test_unplace_before_any_placement_is_refused_and_changes_nothing(tmp_path, gpu):
    session = fresh(gpu, tmp_path)
    session.run_defaults("scopes")
    before = session.sdfg.to_json()

    result = session.unplace()

    assert result.status is MoveStatus.ILLEGAL and "no kernel is placed" in result.reason
    assert session.sdfg.to_json() == before


@pytest.mark.parametrize("gpu", TARGETS)
def test_set_scope_on_a_placed_program_is_refused_and_keeps_the_placement(tmp_path, gpu):
    session = fresh(gpu, tmp_path)
    first, second = doalls(session)
    session.set_scope([first])
    session.place()
    before, unplaced = session.sdfg.to_json(), session.unplaced

    result = session.set_scope([second])

    assert result.status is MoveStatus.ILLEGAL and "call unplace first" in result.reason
    assert session.sdfg.to_json() == before and session.unplaced is unplaced


@pytest.mark.parametrize("gpu", TARGETS)
def test_a_refused_first_place_changes_nothing(tmp_path, gpu):
    session = fresh(gpu, tmp_path)
    doalls(session)
    before, applied, unscoped = structure(session), list(session.applied), session.unscoped

    result = session.place({"no_such_kernel": "cpu"})

    assert isinstance(result, PlaceResult) and result.status is MoveStatus.ILLEGAL
    assert "no kernel is named no_such_kernel" in result.reason
    # the default scopes the first place gives are taken back with the refusal
    assert external_calls(session.sdfg) == [] and structure(session) == before
    assert session.applied == applied and session.unscoped is unscoped and session.unplaced is None
    assert len(doalls(session)) == 2 and session.place().status is MoveStatus.APPLIED


def test_a_refused_first_place_keeps_the_scopes_already_set(tmp_path):
    session = fresh(True, tmp_path)
    first, _ = doalls(session)
    kernel = session.set_scope([first]).reason
    before = structure(session)

    result = session.place({kernel: "fpga"})

    assert result.status is MoveStatus.ILLEGAL and "a device is 'cpu' or 'gpu'" in result.reason
    assert [ext.name for ext in external_calls(session.sdfg)] == [kernel] and structure(session) == before


def test_the_scopes_gate_names_unplace(tmp_path):
    session = fresh(True, tmp_path)
    first, _ = doalls(session)
    CALLS["run_defaults"](session, stage="placement")

    with pytest.raises(ValueError, match="scopes are fixed; unplace drops the placement"):
        CALLS["set_scope"](session, labels=[first])


def test_unplace_is_refused_once_the_kernel_work_started(tmp_path):
    session = fresh(False, tmp_path)
    CALLS["run_defaults"](session, stage="placement")
    CALLS["focus_kernel"](session, name=external_calls(session.sdfg)[0].name)

    with pytest.raises(ValueError, match="kernel work has started"):
        CALLS["unplace"](session)
    assert session.unplaced is not None
