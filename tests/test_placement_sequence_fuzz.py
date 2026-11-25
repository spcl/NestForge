# Copyright 2021 ETH Zurich and the Tactics authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Random sequences of the scopes phase's calls on random programs keep the session's invariants.

Each seed canonicalizes one program of the fusion fuzzer's grammar and makes :data:`STEPS` random calls of
``set_scope``, ``unset_scope``, ``run_defaults("scopes")``, a ``run-on-*`` move ``list_scope_moves`` offers,
``rollback`` and ``finalize_scope``, some with a kernel or a device that does not exist. Nothing is built: the checks
are on the program.

- a refused call leaves the program, its journal and its placement as they were; through the tools, no call crashes;
- an applied call leaves a valid program in which every kernel keeps its body and has a device;
- after ``finalize_scope`` no GPU kernel takes host memory in its call (a scalar passed by value is no memory);
- a ``run-on-*`` move then ``rollback`` gives back the program before the move;
- a fresh session repeating the applied calls is the same program.

A second sequence makes the calls through :func:`tactics.server.answer`, as the judge serves them: the tools' gates
(scopes are fixed once placed) and argument checks too, and ``apply_scope_move`` with a schedule ``list_scope_moves``
offers inside a random kernel. There a call answered with an error is refused as well, and must leave the program as
it was.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import dace
import numpy as np
import pytest
from dace.libraries.standard.helper import GPU_RESIDENT_STORAGES
from dace.libraries.standard.nodes.external_call import ExternalCall, call_site, external_calls
from dace.sdfg import nodes
from test_fusion_fuzz import load_program
from test_rollback_equivalence import state_of
from test_scope_fuzz import random_block_run

from tactics.records import MoveStatus
from tactics.server import answer
from tactics.session import MoveResult, PlaceResult, Session
from tactics.stages.canonicalize import Targets
from tactics.stages.moves import RUN_ON_KINDS
from tactics.stages.placement import DEVICES, has_device_memory, kernel_device

NCASES = 12
#: Calls per seed: long enough to place, take back and place again a few times.
STEPS = 14
#: One call in this many names a kernel or a device that does not exist.
BOGUS_EVERY = 5
TARGETS = [pytest.param(False, id="cpu"), pytest.param(True, id="gpu")]
REFUSED = (MoveStatus.ILLEGAL, MoveStatus.NOT_FOUND)

#: A call of the sequence: the session method's name and its arguments.
Call = tuple[str, tuple[Any, ...]]
#: The program as the agent sees it (:func:`state_of`), whether it is placed, and the moves applied.
Snapshot = tuple[object, bool, list[MoveResult]]


def apply_scope_move(session: Session, kind: str, labels: list[str]) -> MoveResult:
    return session.apply_move(kind, labels)


#: The session methods the sequences draw from.
METHODS: dict[str, Callable[..., object]] = {
    "set_scope": Session.set_scope,
    "unset_scope": Session.unset_scope,
    "run_defaults": Session.run_defaults,
    "apply_scope_move": apply_scope_move,
    "finalize_scope": Session.finalize_scope,
}


def fresh(seed: int, tmp_path: Path, gpu: bool, work: str = "work") -> Session:
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


def snapshot(session: Session) -> Snapshot:
    return state_of(session), session.unplaced is None, list(session.applied)


def random_run_on(session: Session, rng: np.random.Generator, bogus: bool) -> Call:
    """A ``run-on-*`` move ``list_scope_moves`` offers; with ``bogus``, one on no kernel or on a GPU the run lacks."""
    if bogus:
        if rng.integers(2) or not kernels(session):
            return "apply_scope_move", ("run-on-cpu", ["no_such_kernel"])
        return "apply_scope_move", ("run-on-gpu", kernels(session)[:1])  # refused on a CPU target only
    offered = [move for move in session.list_scope_moves() if move.kind in RUN_ON_KINDS]
    if not offered:
        return "apply_scope_move", ("run-on-cpu", ["no_such_kernel"])
    move = offered[int(rng.integers(len(offered)))]
    return "apply_scope_move", (move.kind, list(move.labels))


def random_call(session: Session, rng: np.random.Generator, gated: bool = False) -> Call:
    """A random call. Unless ``gated`` (the judge's gates refuse it), no scope call (draws 0 to 2) on a placed program:
    the session itself takes one as a change of the placed program's kernels, which the agent cannot make."""
    bogus = int(rng.integers(BOGUS_EVERY)) == 0
    match int(rng.integers(3 if session.unplaced is not None and not gated else 0, 7)):
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
            return random_run_on(session, rng, bogus)
        case 5:
            return "finalize_scope", ()
        case _:
            return "apply_scope_move", ("rollback", [])


def status_of(reply: object) -> MoveStatus:
    """The status of a call's answer; a default stage answers the kernels it made, which is always applied."""
    return reply.status if isinstance(reply, (PlaceResult, MoveResult)) else MoveStatus.APPLIED


def takes_host_memory(ext: ExternalCall, state: dace.SDFGState) -> bool:
    """Whether the call passes ``ext`` host memory; a scalar passed by value is no memory, on either device."""
    by_value = call_site(ext, state).by_value
    edges = [(e.dst_conn, e.data.data) for e in state.in_edges(ext)]
    edges += [(e.src_conn, e.data.data) for e in state.out_edges(ext)]
    return any(
        data is not None and conn not in by_value and state.sdfg.arrays[data].storage not in GPU_RESIDENT_STORAGES
        for conn, data in edges
    )


def gpu_kernels_on_host_memory(sdfg: dace.SDFG) -> list[str]:
    return [
        ext.name
        for ext, state in sdfg.all_nodes_recursive()
        if isinstance(ext, ExternalCall)
        and isinstance(state, dace.SDFGState)
        and kernel_device(ext) == "gpu"
        and takes_host_memory(ext, state)
    ]


def check_program(session: Session, call: Call) -> None:
    program(session).validate()
    # a scope over a block that holds a kernel takes that kernel into its body: it builds and runs (job 5371457)
    for ext in external_calls(program(session)):
        assert ext.standalone_sdfg is not None, (call, ext.name)
    devices = session.devices()
    assert set(devices.values()) <= set(DEVICES if session.targets.gpu else ("cpu",)), (call, devices)
    if has_device_memory(program(session)):
        assert gpu_kernels_on_host_memory(program(session)) == [], call


def check_run_on_rolls_back(session: Session, call: Call, before: Snapshot) -> None:
    """``rollback`` of the run-on move just applied gives back ``before``; the move is applied again after."""
    after = snapshot(session)
    assert session.apply_move("rollback", []).status is MoveStatus.APPLIED, call
    assert snapshot(session) == before, f"rollback of {call} differs from the program before it"
    assert status_of(METHODS[call[0]](session, *call[1])) is MoveStatus.APPLIED, call
    assert snapshot(session)[:2] == after[:2], f"{call} applied again differs from its first time"


def run_sequence(seed: int, tmp_path: Path, gpu: bool) -> tuple[Session, list[Call]]:
    """The session after :data:`STEPS` random calls, each checked, and the calls that applied."""
    session, rng = fresh(seed, tmp_path, gpu), np.random.default_rng(seed)
    applied: list[Call] = []
    for _ in range(STEPS):
        call = random_call(session, rng)
        before = snapshot(session)
        reply = METHODS[call[0]](session, *call[1])
        if status_of(reply) in REFUSED:
            assert snapshot(session) == before, f"seed={seed} gpu={gpu}: refused {call} changed the session ({reply})"
            continue
        applied.append(call)
        check_program(session, call)
        if call[0] == "apply_scope_move" and call[1][0] in RUN_ON_KINDS:
            check_run_on_rolls_back(session, call, before)
    return session, applied


@pytest.mark.parametrize("gpu", TARGETS)
@pytest.mark.parametrize("seed", range(NCASES))
def test_a_random_sequence_keeps_the_invariants_and_repeating_its_applied_calls_rebuilds_the_program(
    seed: int, tmp_path: Path, gpu: bool
) -> None:
    session, applied = run_sequence(seed, tmp_path, gpu)  # checks every call on the way

    again = fresh(seed, tmp_path, gpu, "again")
    for name, arguments in applied:
        assert status_of(METHODS[name](again, *arguments)) not in REFUSED, (name, arguments)

    assert snapshot(again)[:2] == snapshot(session)[:2]


def test_the_sequences_place_finalize_and_refuse(tmp_path: Path) -> None:
    """The seeds must place, finalize and be refused on some runs, else the fuzzer probes one corner only."""
    placed = finalized = refused = 0
    for seed in range(NCASES):
        session, rng = fresh(seed, tmp_path, True, f"w{seed}"), np.random.default_rng(seed)
        for _ in range(STEPS):
            call = random_call(session, rng)
            status = status_of(METHODS[call[0]](session, *call[1]))
            placed += status is MoveStatus.APPLIED and call[0] == "apply_scope_move" and call[1][0] in RUN_ON_KINDS
            finalized += status is MoveStatus.APPLIED and call[0] == "finalize_scope"
            refused += status in REFUSED

    assert placed >= NCASES // 4 and finalized >= 1 and refused >= 1, (placed, finalized, refused)


# the same sequences through the tools, as the judge answers them

#: A tool call: the tool's name and its keyword arguments, as the judge receives them.
ToolCall = tuple[str, dict[str, Any]]
#: The statuses of a tool's answer that changed nothing.
REFUSED_ANSWERS = frozenset({"illegal", "not-found"})
#: The exceptions the judge's refusals come as (a gate, an argument check); any other error is a crash.
REFUSAL_ERRORS = ("ValueError", "ValidationError")
#: The schedules ``set-schedule-cpu`` takes, and thread blocks for ``set-schedule-gpu``.
CPU_SCHEDULES = ("default", "static", "dynamic", "guided")
GPU_BLOCKS = ((64,), (128,), (32, 8))
#: The keyword arguments of each tool :func:`random_call` draws, in its argument order.
KEYWORDS = {
    "set_scope": ("labels",),
    "unset_scope": ("name",),
    "run_defaults": ("stage",),
    "apply_scope_move": ("kind", "labels"),
}


def schedule_move_call(session: Session, rng: np.random.Generator) -> ToolCall | None:
    """``apply_scope_move`` with a move ``list_scope_moves`` offers inside a random kernel; ``None`` when none is."""
    names = kernels(session)
    if not names:
        return None
    kernel = names[int(rng.integers(len(names)))]
    offered = [move for move in session.list_scope_moves(scope=kernel) if move.kind not in RUN_ON_KINDS]
    if not offered:
        return None
    move = offered[int(rng.integers(len(offered)))]
    arguments: dict[str, Any] = {"scope": kernel, "kind": move.kind, "labels": list(move.labels)}
    if move.kind == "set-schedule-cpu":
        arguments["schedule"] = CPU_SCHEDULES[int(rng.integers(len(CPU_SCHEDULES)))]
    if move.kind == "set-schedule-gpu":
        arguments["block"] = list(move.block or GPU_BLOCKS[int(rng.integers(len(GPU_BLOCKS)))])
    return "apply_scope_move", arguments


def random_tool_call(session: Session, rng: np.random.Generator) -> ToolCall:
    if int(rng.integers(4)) == 0 and (move := schedule_move_call(session, rng)) is not None:
        return move
    name, arguments = random_call(session, rng, gated=True)
    return name, dict(zip(KEYWORDS.get(name, ()), arguments, strict=True))


def told(session: Session, call: ToolCall) -> tuple[bool, str]:
    """``call`` through the judge's answer: whether it applied, and what it said; a crash fails the test."""
    reply = answer(session, call[0], dict(call[1]), None, timed=False)
    if reply.error:
        assert reply.error.startswith(REFUSAL_ERRORS), f"{call} crashed: {reply.error}"
        return False, reply.error
    result = reply.model_dump(mode="json")["result"]
    status = result.get("status") if isinstance(result, dict) else None
    return status not in REFUSED_ANSWERS, str(result)[:300]


def run_tool_sequence(seed: int, tmp_path: Path, gpu: bool) -> tuple[Session, list[ToolCall]]:
    """The session after :data:`STEPS` random tool calls, each checked, and the calls that applied."""
    session, rng = fresh(seed, tmp_path, gpu), np.random.default_rng(seed)
    applied: list[ToolCall] = []
    for _ in range(STEPS):
        call = random_tool_call(session, rng)
        before = snapshot(session)
        done, said = told(session, call)
        if not done:
            assert snapshot(session) == before, f"seed={seed} gpu={gpu}: refused {call} changed the session ({said})"
            continue
        applied.append(call)
        check_program(session, (call[0], tuple(call[1].values())))
    return session, applied


@pytest.mark.parametrize("gpu", TARGETS)
@pytest.mark.parametrize("seed", range(NCASES))
def test_a_random_sequence_through_the_tools_keeps_the_invariants_and_replays(
    seed: int, tmp_path: Path, gpu: bool
) -> None:
    session, applied = run_tool_sequence(seed, tmp_path, gpu)

    again = fresh(seed, tmp_path, gpu, "again")
    for call in applied:
        done, said = told(again, call)
        assert done, (call, said)
    assert snapshot(again)[:2] == snapshot(session)[:2]


def test_the_tool_sequences_move_inside_kernels_and_meet_the_gates(tmp_path: Path) -> None:
    """The seeds must apply a schedule move and be refused by a gate on some runs, else they probe one corner only."""
    moved = refused = 0
    for seed in range(NCASES):
        session, rng = fresh(seed, tmp_path, True, f"w{seed}"), np.random.default_rng(seed)
        for _ in range(STEPS):
            call = random_tool_call(session, rng)
            done, said = told(session, call)
            moved += done and call[0] == "apply_scope_move" and "scope" in call[1]
            refused += not done and ("placed" in said or "fixed" in said)
    assert moved >= 1 and refused >= 1, (moved, refused)


#: The calls of seed 11 through the tools, before the sequences drew no scope call on a placed program (the rollbacks
#: take back the placement and the third ``run_defaults``, so the gate admits the ``set_scope``).
SEED_11_CALLS: list[Call] = [
    ("set_scope", (["state8"],)),
    ("apply_scope_move", ("run-on-gpu", ["extcall_0"])),
    ("run_defaults", ("scopes",)),
    ("run_defaults", ("scopes",)),
    ("run_defaults", ("scopes",)),
    ("finalize_scope", ()),
    ("apply_scope_move", ("rollback", [])),
    ("apply_scope_move", ("rollback", [])),
    ("set_scope", (["doall2"],)),
    ("apply_scope_move", ("run-on-gpu", ["extcall_1"])),
]


def test_finalize_scope_after_a_rolled_back_placement_on_seed_11(tmp_path: Path) -> None:
    session = fresh(11, tmp_path, True)
    for name, arguments in SEED_11_CALLS:
        assert status_of(METHODS[name](session, *arguments)) is MoveStatus.APPLIED, (name, arguments)

    assert session.finalize_scope().status is MoveStatus.APPLIED
    check_program(session, ("finalize_scope", ()))
