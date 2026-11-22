# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Every legal move of a communication row keeps what every rank computes: each fixture of
``tests/comm_rank_programs.py`` is canonicalized, every ``lib-before`` and ``lib-after`` the session lists for a
communication row is applied to a copy, and each copy runs under ``mpirun`` at 1, 2 and 4 ranks against the program
without the move, output by output and bit for bit."""

import copy
import os
import shutil
import subprocess
import sys
from pathlib import Path

import dace
import numpy as np
import pytest
from comm_rank_programs import FIXTURES
from dace.libraries.comm.nodes.node import CommNode

from nestforge.records import MoveStatus
from nestforge.session import Move, Session
from nestforge.stages.lib_motion import LIB_AFTER, LIB_BEFORE, SPLIT_BLOCKING

pytestmark = [pytest.mark.mpi, pytest.mark.integration]

#: The rank counts every program runs at.
RANKS = (1, 2, 4)
#: Seconds one ``mpirun`` may take: a moved wait that deadlocks fails instead of hanging the run.
RUN_TIMEOUT_S = 300
#: The fixtures whose blocking ring needs a neighbor that is not the rank itself.
NEEDS_TWO = frozenset({"ring"})
#: The fixtures with a blocking row that has a start-and-wait form.
SPLITTABLE = ("redistribute", "residual_loop", "ring")

RUNNER = Path(__file__).with_name("comm_rank_runner.py")


def comm_moves(session: Session) -> list[Move]:
    """The ``lib-before`` and ``lib-after`` moves the session lists whose row is a communication node."""
    rows = session.row_index()
    return [
        move
        for kind in (LIB_BEFORE, LIB_AFTER)
        for move in session.list_moves(kind)
        if isinstance(rows[move.labels[0]][0], CommNode)
    ]


def moved_programs(name: str, split: bool = False) -> tuple[dace.SDFG, list[tuple[str, dace.SDFG]]]:
    """The canonical program of fixture ``name``, and a copy of it per legal move of a communication row; with
    ``split``, every blocking row is split into its start and wait first, and the copies hold the split too."""
    session = Session(FIXTURES[name].program.to_sdfg(simplify=True))
    session.canonicalize()
    base = copy.deepcopy(session.sdfg)
    moved: list[tuple[str, dace.SDFG]] = []
    while split and (splits := session.list_moves(SPLIT_BLOCKING)):
        assert session.apply_move(SPLIT_BLOCKING, splits[0].labels).status is MoveStatus.APPLIED
    if split:
        assert session.history, f"{name}: no blocking row was split"
        moved.append((SPLIT_BLOCKING, copy.deepcopy(session.sdfg)))
    for move in comm_moves(session):
        result = session.apply_move(move.kind, move.labels)
        assert result.status is MoveStatus.APPLIED, (move, result.reason)
        moved.append((f"{move.kind} {' '.join(move.labels)}", copy.deepcopy(session.sdfg)))
        assert session.undo().status is MoveStatus.APPLIED
    return base, moved


def run_on_ranks(sdfg: dace.SDFG, name: str, ranks: int, folder: Path) -> list[dict[str, np.ndarray]]:
    """Each rank's outputs of ``sdfg`` (fixture ``name``) at ``ranks`` ranks."""
    launcher = shutil.which("mpirun")
    assert launcher is not None, "mpirun is on PATH (Open MPI)"
    build = folder / "build"
    sdfg.build_folder = str(build)
    sdfg.compile()
    prefix = folder / f"out{ranks}"
    command = [
        launcher,
        "--oversubscribe",
        "-n",
        str(ranks),
        sys.executable,
        str(RUNNER),
        str(build),
        name,
        str(prefix),
    ]
    # one thread per rank: the fixtures' reductions are atomic adds, whose order varies across threads
    env = os.environ | {"OMP_NUM_THREADS": "1"}
    subprocess.run(command, check=True, timeout=RUN_TIMEOUT_S, cwd=RUNNER.parent, env=env)
    return [dict(np.load(f"{prefix}.{rank}.npz")) for rank in range(ranks)]


def outputs_match_on_every_rank(name: str, base: dace.SDFG, moved: list[tuple[str, dace.SDFG]], tmp_path: Path) -> None:
    ranks = [count for count in RANKS if count > 1 or name not in NEEDS_TWO]
    for count in ranks:
        base.name = f"{name}_base"
        want = run_on_ranks(base, name, count, tmp_path / "base")
        assert any(np.any(value != 0) for value in want[0].values()), "the fixture computes something"
        for index, (label, program) in enumerate(moved):
            program.name = f"{name}_moved{index}"
            got = run_on_ranks(program, name, count, tmp_path / f"moved{index}")
            for rank, (expected, actual) in enumerate(zip(want, got, strict=True)):
                for output in expected:
                    assert np.array_equal(expected[output], actual[output]), (label, count, rank, output)


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_every_legal_move_of_a_communication_row_keeps_every_ranks_outputs(name: str, tmp_path: Path) -> None:
    base, moved = moved_programs(name)
    outputs_match_on_every_rank(name, base, moved, tmp_path)


@pytest.mark.parametrize("name", SPLITTABLE)
def test_split_blocking_rows_and_every_move_of_their_starts_and_waits_keep_every_ranks_outputs(
    name: str, tmp_path: Path
) -> None:
    base, moved = moved_programs(name, split=True)
    outputs_match_on_every_rank(name, base, moved, tmp_path)


def exchanged_locals(session: Session) -> list[str]:
    """The local arrays a communication row of the session's program sends or receives, sorted."""
    sdfg = session.sdfg
    return sorted(
        {
            str(edge.data.data)
            for state in sdfg.states()
            for node in state.nodes()
            if isinstance(node, CommNode)
            for edge in state.all_edges(node)
            if not edge.data.is_empty()
            and sdfg.arrays[str(edge.data.data)].transient
            and len(sdfg.arrays[str(edge.data.data)].shape) > 1
        }
    )


def relaid_programs(name: str) -> tuple[dace.SDFG, list[tuple[str, dace.SDFG]]]:
    """The canonical program of fixture ``name``, and a copy per exchanged local array with its dimensions reversed:
    relayout packs the buffer before the send and unpacks it after the receive (12.8)."""
    session = Session(FIXTURES[name].program.to_sdfg(simplify=True))
    session.canonicalize()
    base = copy.deepcopy(session.sdfg)
    relaid: list[tuple[str, dace.SDFG]] = []
    for array in exchanged_locals(session):
        order = [f"i{d}" for d in reversed(range(len(session.sdfg.arrays[array].shape)))]
        operation = {"action": "permute", "array": array, "order": order}
        result = session.apply_move("relayout", [], operation=operation)
        assert result.status is MoveStatus.APPLIED, (array, result.reason)
        relaid.append((f"relayout {array}", copy.deepcopy(session.sdfg)))
        assert session.undo().status is MoveStatus.APPLIED
    assert relaid, f"{name}: no local array is exchanged"
    return base, relaid


def test_reversing_the_layout_of_an_exchanged_local_array_keeps_every_ranks_outputs(tmp_path: Path) -> None:
    base, relaid = relaid_programs("redistribute")
    outputs_match_on_every_rank("redistribute", base, relaid, tmp_path)


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_every_fixture_lists_a_move_of_a_communication_row(name: str) -> None:
    """The fixtures leave the rows somewhere to go, so the test above checks moves, not an empty list."""
    _, moved = moved_programs(name)
    assert moved, f"{name}: no legal move of a communication row is listed"
