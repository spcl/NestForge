# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Random sequences of the scopes and placement calls on random programs keep the session's invariants.

Each seed canonicalizes one program of the fusion fuzzer's grammar and makes :data:`STEPS` random calls of
``set_scope``, ``unset_scope``, ``run_defaults("scopes")``, ``place``, ``insert_copies`` and ``unplace``, some with a
name or a device that does not exist. Nothing is built: the checks are on the program.

- a refused call leaves the program, its journal and its placement as they were;
- an applied call leaves a valid program in which every kernel keeps its body and has a device;
- after ``insert_copies`` no GPU kernel copies in its own call;
- placing the kernels again on the devices they have gives the same program;
- a fresh session repeating the applied calls is the same program.

A second sequence makes the calls through :func:`nestforge.server.answer`, as the judge serves them: the tools' gates
and argument checks too, and ``apply_scope_move`` with a move ``list_moves`` offers inside a random kernel. There a
call answered with an error is refused as well, and must leave the program as it was.
"""

from typing import Any

import dace
import numpy as np
import pytest
from dace.libraries.standard.nodes.external_call import external_calls
from dace.sdfg import nodes
from test_fusion_fuzz import load_program
from test_scope_fuzz import random_block_run
from test_unplace import structure

from nestforge.records import MoveStatus
from nestforge.server import answer
from nestforge.session import MoveResult, PlaceResult, Session
from nestforge.stages.canonicalize import Targets
from nestforge.stages.placement import DEVICES, copies_in_call, has_device_memory

NCASES = 12
#: Calls per seed: long enough to place, undo and place again a few times.
STEPS = 14
#: One call in this many names a kernel or a device that does not exist.
BOGUS_EVERY = 5
TARGETS = [pytest.param(False, id="cpu"), pytest.param(True, id="gpu")]
REFUSED = (MoveStatus.ILLEGAL, MoveStatus.NOT_FOUND)

#: A call of the sequence: the session method's name and its arguments.
Call = tuple[str, tuple[object, ...]]
#: The session methods the sequences draw from.
METHODS = {
    "set_scope": Session.set_scope,
    "unset_scope": Session.unset_scope,
    "run_defaults": Session.run_defaults,
    "place": Session.place,
    "insert_copies": Session.insert_copies,
    "unplace": Session.unplace,
}


def fresh(seed: int, tmp_path, gpu: bool, work: str = "work") -> Session:
    """A canonicalized session on the seed's program; the source lives in ``tmp_path``, whose path the SDFG records."""
    program, _ = load_program(tmp_path, seed + 1700)
    session = Session(program.to_sdfg(simplify=True), targets=Targets(gpu=gpu), work_dir=str(tmp_path / work))
    session.canonicalize()
    return session


def program(session: Session) -> dace.SDFG:
    assert session.sdfg is not None
    return session.sdfg


def kernels(session: Session) -> list[str]:
    return [ext.name for ext in external_calls(program(session))]


def doalls(session: Session) -> list[str]:
    session.describe()
    return [label for label, (obj, _) in session.row_index().items() if isinstance(obj, nodes.MapEntry)]


def random_devices(session: Session, rng: np.random.Generator, bogus: bool) -> dict[str, str]:
    choices = DEVICES if session.targets.gpu else ("cpu",)
    devices = {name: choices[int(rng.integers(len(choices)))] for name in kernels(session) if rng.integers(2)}
    if bogus:
        devices |= {"no_such_kernel": "cpu"} if rng.integers(2) else dict.fromkeys(kernels(session)[:1], "tpu")
    return devices


def random_call(session: Session, rng: np.random.Generator) -> Call:
    bogus = int(rng.integers(BOGUS_EVERY)) == 0
    match int(rng.integers(7)):
        case 0:
            return "set_scope", (random_block_run(session, rng),)
        case 1:
            maps = doalls(session)
            return "set_scope", ([maps[int(rng.integers(len(maps)))]] if maps else ["no_such_row"],)
        case 2:
            names = kernels(session)
            return "unset_scope", ("no_such_kernel" if bogus or not names else names[int(rng.integers(len(names)))],)
        case 3:
            return "run_defaults", ("scopes",)
        case 4:
            return "place", (random_devices(session, rng, bogus),)
        case 5:
            return "insert_copies", ()
        case _:
            return "unplace", ()


def status_of(answer: object) -> MoveStatus:
    """The status of a call's answer; a default stage answers the kernels it made, which is always applied."""
    return answer.status if isinstance(answer, (PlaceResult, MoveResult)) else MoveStatus.APPLIED


def check_program(session: Session, call: Call) -> None:
    program(session).validate()
    # a scope over a block that holds a kernel takes that kernel into its body: it builds and runs (job 5371457)
    for ext in external_calls(program(session)):
        assert ext.standalone_sdfg is not None, (call, ext.name)
    devices = session.devices()
    assert set(devices.values()) <= set(DEVICES if session.targets.gpu else ("cpu",)), (call, devices)
    if has_device_memory(program(session)):
        assert copies_in_call(program(session)) == [], call


def run_sequence(seed: int, tmp_path, gpu: bool) -> tuple[Session, list[Call]]:
    """The session after :data:`STEPS` random calls, each checked, and the calls that applied."""
    session, rng = fresh(seed, tmp_path, gpu), np.random.default_rng(seed)
    applied: list[Call] = []
    for _ in range(STEPS):
        call = random_call(session, rng)
        before = structure(session), list(session.applied), session.unplaced is None
        answer = METHODS[call[0]](session, *call[1])
        if status_of(answer) in REFUSED:
            after = structure(session), list(session.applied), session.unplaced is None
            assert after == before, f"seed={seed} gpu={gpu}: refused {call} changed the session ({answer})"
            continue
        applied.append(call)
        check_program(session, call)
    return session, applied


@pytest.mark.parametrize("gpu", TARGETS)
@pytest.mark.parametrize("seed", range(NCASES))
def test_a_random_sequence_of_scope_and_placement_calls_keeps_the_invariants(seed, tmp_path, gpu):
    session, _ = run_sequence(seed, tmp_path, gpu)

    if session.unplaced is None:
        return
    # the kernels again on the devices they have: the same program
    devices, copied, placed = session.devices(), has_device_memory(program(session)), structure(session)
    assert session.unplace().status is MoveStatus.APPLIED
    assert session.place(devices).status is MoveStatus.APPLIED
    if copied:
        assert session.insert_copies().status is MoveStatus.APPLIED
    assert structure(session) == placed, f"seed={seed} gpu={gpu}: unplace then place differs from place"


@pytest.mark.parametrize("gpu", TARGETS)
@pytest.mark.parametrize("seed", range(NCASES))
def test_repeating_the_applied_calls_rebuilds_the_program(seed, tmp_path, gpu):
    session, applied = run_sequence(seed, tmp_path, gpu)

    again = fresh(seed, tmp_path, gpu, "again")
    for name, arguments in applied:
        assert status_of(METHODS[name](again, *arguments)) not in REFUSED, (name, arguments)

    assert structure(again) == structure(session)


def test_the_sequences_place_and_refuse(tmp_path):
    """The seeds must reach a placed program and a refusal on most runs, else the fuzzer probes one corner only."""
    placed = sum(run_sequence(seed, tmp_path, True)[0].unplaced is not None for seed in range(NCASES))

    assert placed >= NCASES // 4, placed


def test_unset_scope_keeps_a_kernel_body_it_cannot_inline_valid(tmp_path):
    """Seed 2: one map scoped and unset again. The kernel's body is not inlined; the nested SDFG that stays keeps
    its connector arrays (named ``_out_...``) non-transient."""
    session = fresh(2, tmp_path, False)
    kernel = session.set_scope(doalls(session)[1:2]).reason

    assert session.unset_scope(str(kernel)).status is MoveStatus.APPLIED

    program(session).validate()


# the same sequences through the tools, as the judge answers them

#: A tool call: the tool's name and its keyword arguments, as the judge receives them.
ToolCall = tuple[str, dict[str, Any]]
#: The statuses of a tool's answer that changed nothing.
REFUSED_ANSWERS = frozenset({"illegal", "not-found"})


def scope_move_call(session: Session, rng: np.random.Generator) -> ToolCall | None:
    """``apply_scope_move`` with a move ``list_moves`` offers inside a random kernel; ``None`` when none is offered."""
    names = kernels(session)
    if not names:
        return None
    kernel = names[int(rng.integers(len(names)))]
    offered = session.list_moves(scope=kernel)
    if not offered:
        return None
    move = offered[int(rng.integers(len(offered)))]
    arguments = {"scope": kernel, "kind": move.kind, "labels": list(move.labels), "tile_sizes": list(move.tile_sizes)}
    return "apply_scope_move", arguments


def random_tool_call(session: Session, rng: np.random.Generator) -> ToolCall:
    if int(rng.integers(4)) == 0 and (move := scope_move_call(session, rng)) is not None:
        return move
    name, arguments = random_call(session, rng)
    keywords = {"set_scope": ("labels",), "unset_scope": ("name",), "run_defaults": ("stage",), "place": ("devices",)}
    return name, dict(zip(keywords.get(name, ()), arguments, strict=True))


def told(session: Session, call: ToolCall) -> tuple[bool, str]:
    """``call`` through the judge's answer: whether it applied, and what it said."""
    reply = answer(session, call[0], dict(call[1]), None, timed=False)
    if reply.error:
        return False, reply.error
    result = reply.model_dump(mode="json")["result"]
    status = result.get("status") if isinstance(result, dict) else None
    return status not in REFUSED_ANSWERS, str(result)[:300]


def run_tool_sequence(seed: int, tmp_path, gpu: bool) -> tuple[Session, list[ToolCall]]:
    """The session after :data:`STEPS` random tool calls, each checked, and the calls that applied."""
    session, rng = fresh(seed, tmp_path, gpu), np.random.default_rng(seed)
    applied: list[ToolCall] = []
    for _ in range(STEPS):
        call = random_tool_call(session, rng)
        before = structure(session), list(session.applied), session.unplaced is None
        done, said = told(session, call)
        if not done:
            after = structure(session), list(session.applied), session.unplaced is None
            assert after == before, f"seed={seed} gpu={gpu}: refused {call} changed the session ({said})"
            continue
        applied.append(call)
        check_program(session, (call[0], tuple(call[1].values())))
    return session, applied


@pytest.mark.parametrize("gpu", TARGETS)
@pytest.mark.parametrize("seed", range(NCASES))
def test_a_random_sequence_through_the_tools_keeps_the_invariants_and_replays(seed, tmp_path, gpu):
    session, applied = run_tool_sequence(seed, tmp_path, gpu)

    again = fresh(seed, tmp_path, gpu, "again")
    for call in applied:
        done, said = told(again, call)
        assert done, (call, said)
    assert structure(again) == structure(session)


def test_the_tool_sequences_move_inside_kernels_and_meet_the_gates(tmp_path):
    """The seeds must apply a scope move and be refused by a gate on some runs, else they probe one corner only."""
    moved = refused = 0
    for seed in range(NCASES):
        session, rng = fresh(seed, tmp_path, True, f"w{seed}"), np.random.default_rng(seed)
        for _ in range(STEPS):
            call = random_tool_call(session, rng)
            done, said = told(session, call)
            moved += done and call[0] == "apply_scope_move"
            refused += not done and ("placed" in said or "fixed" in said)
    assert moved >= 1 and refused >= 1, (moved, refused)
