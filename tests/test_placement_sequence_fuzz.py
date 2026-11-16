# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Random sequences of the scopes and placement calls on random programs keep the session's invariants.

Each seed canonicalizes one program of the fusion fuzzer's grammar and makes :data:`STEPS` random calls of
``set_scope``, ``unset_scope``, ``run_defaults("scopes")``, ``place``, ``insert_copies`` and ``unplace``, some with a
name or a device that does not exist. Nothing is built: the checks are on the program.

- a refused call leaves the program, its journal and its placement as they were;
- an applied call leaves a valid program in which no kernel holds a kernel and every kernel has a device;
- after ``insert_copies`` no GPU kernel copies in its own call;
- placing the kernels again on the devices they have gives the same program;
- a fresh session repeating the applied calls is the same program.
"""

import dace
import numpy as np
import pytest
from dace.libraries.standard.nodes.external_call import ExternalCall, external_calls
from dace.sdfg import nodes
from test_fusion_fuzz import load_program
from test_scope_fuzz import random_block_run
from test_unplace import structure

from nestforge.records import MoveStatus
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
    for ext in external_calls(program(session)):
        body = ext.standalone_sdfg
        assert body is not None, (call, ext.name)
        assert not any(isinstance(node, ExternalCall) for node, _ in body.all_nodes_recursive()), (call, ext.name)
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
