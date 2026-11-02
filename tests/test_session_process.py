# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A session process that stops answering is ended after the call's timeout, one that died says so, and a measuring
call stops when asked and leaves its session answering; it starts in the bench's llvm OpenMP context, whose libomp is
the one runtime NestForge links."""

import json
import multiprocessing
import os
import pathlib
import signal
import tempfile
import threading
import time
from multiprocessing.connection import Connection
from typing import Any

import pytest
from hpcagent_agent import plugins as agent_plugins
from hpcagent_bench import plugins

from nestforge import server
from nestforge.paths import REPO_ROOT
from nestforge.server import STOP_SIGNAL, Reply, SessionProcess, Setup
from nestforge.session import Stopped, Stopper

SETUP = Setup(kernel="none", preset="S", datatype="float64", gpu=False)
#: Seconds a test waits for what should happen at once.
PROMPT_S = 10.0


def gone_within(pid: int, seconds: float) -> bool:
    """Whether process ``pid`` is gone, or a zombie, within ``seconds``."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            if pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z":
                return True
        except OSError:
            return True
        time.sleep(0.05)
    return False


def ready_then_hang(connection: Connection, setup: Setup) -> None:
    """A session that starts, then never answers a call and ignores being asked to stop."""
    del setup
    signal.signal(STOP_SIGNAL, signal.SIG_IGN)
    connection.send(Reply())
    connection.recv()
    time.sleep(600)


def ready_then_stop_when_asked(connection: Connection, setup: Setup) -> None:
    """A session that measures until asked to stop, and then answers the next call."""
    del setup
    stopper = Stopper()
    signal.signal(STOP_SIGNAL, lambda signum, frame: stopper.ask())
    connection.send(Reply())
    connection.recv()
    try:
        with stopper.armed():
            time.sleep(600)
    except Stopped:
        connection.send(Reply(stopped=True))
    connection.recv()
    connection.send(Reply(result="still here"))


def ready_then_die(connection: Connection, setup: Setup) -> None:
    """A session that starts, then dies on its first call."""
    del setup
    connection.send(Reply())
    connection.recv()
    raise SystemExit(3)


def ready_then_report_library_path(connection: Connection, setup: Setup) -> None:
    """A session that starts, then answers its first call with its ``LD_LIBRARY_PATH``."""
    del setup
    connection.send(Reply())
    connection.recv()
    connection.send(Reply(result=os.environ.get("LD_LIBRARY_PATH", "")))


def never_ready(connection: Connection, setup: Setup) -> None:
    """A session whose canonicalization hangs."""
    del connection, setup
    time.sleep(600)


def ready_then_die_leaving_a_child(connection: Connection, setup: Setup) -> None:
    """A session killed in a call, its forked child (named in ``shared_root``) still holding the pipe."""
    connection.send(Reply())
    connection.recv()
    child = os.fork()
    if child == 0:
        time.sleep(600)
    pathlib.Path(setup.shared_root).write_text(str(child))
    os.kill(os.getpid(), signal.SIGKILL)


def ready_then_leave_a_file(connection: Connection, setup: Setup) -> None:
    """A session that names its temporary folder in its first answer, then waits."""
    connection.send(Reply())
    connection.recv()
    tempfile.mkdtemp(prefix="work")
    connection.send(Reply(result=tempfile.gettempdir()))
    connection.recv()


def busy_session_in_a_judge(marker: pathlib.Path) -> None:
    """A judge whose session is in a call that will not end."""
    session = SessionProcess(SETUP, target=ready_then_hang)
    session.connection.send(("view_tree", {}))
    marker.write_text(str(session.process.pid))
    time.sleep(600)


def test_a_session_that_never_reports_ready_is_ended_within_the_budget(monkeypatch):
    monkeypatch.setattr(server, "TOOL_TIMEOUT_S", 0.5)
    monkeypatch.setattr(server, "CLOSE_TIMEOUT_S", 0.1)
    started = time.monotonic()

    with pytest.raises(LookupError, match="did not start"):
        SessionProcess(SETUP, target=never_ready)

    assert time.monotonic() - started < PROMPT_S


def test_a_session_killed_between_calls_answers_that_it_died_and_does_not_raise():
    session = SessionProcess(SETUP, target=ready_then_hang)
    os.kill(session.process.pid, signal.SIGKILL)
    session.process.join(PROMPT_S)

    reply = session.call("view_tree", {})

    assert "the session process died" in reply.error, reply
    assert not session.alive


def test_a_session_killed_in_a_call_is_noticed_while_its_forked_child_holds_the_pipe(tmp_path):
    marker = tmp_path / "child"
    started = time.monotonic()
    session = SessionProcess(
        Setup(kernel="none", preset="S", datatype="float64", gpu=False, shared_root=str(marker)),
        target=ready_then_die_leaving_a_child,
    )

    reply = session.call("view_tree", {})

    assert "the session process died" in reply.error, reply
    assert time.monotonic() - started < 60.0, "the child's 600 s do not hold the call"
    assert gone_within(int(marker.read_text()), PROMPT_S)


def test_a_busy_session_ends_with_its_killed_judge(tmp_path):
    marker = tmp_path / "session"
    spawn = multiprocessing.get_context("spawn")
    judge = spawn.Process(target=busy_session_in_a_judge, args=(marker,))
    judge.start()
    deadline = time.monotonic() + 60.0
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    session = int(marker.read_text())

    judge.kill()
    judge.join()

    assert gone_within(session, PROMPT_S)


def test_the_session_keeps_its_files_in_a_folder_the_judge_removes_even_after_a_sigkill():
    session = SessionProcess(SETUP, target=ready_then_leave_a_file)
    scratch = pathlib.Path(session.call("view_tree", {}).result)
    assert scratch.is_dir() and any(scratch.iterdir())

    os.kill(session.process.pid, signal.SIGKILL)
    session.process.join(PROMPT_S)
    session.close()

    assert not scratch.exists()


def report_pin(connection: Connection, setup: Setup) -> None:
    """A session answering, per call, what a child it forks and one it spawns would run on after :func:`server.pin`."""
    connection.send(Reply())
    cores, environment = os.sched_getaffinity(0), dict(os.environ)
    while True:
        try:
            _, _, slot, _ = connection.recv()
        except EOFError:
            return
        server.pin(slot, setup.gpu, cores, environment)
        reader, writer = os.pipe()
        child = os.fork()
        if child == 0:
            os.write(writer, json.dumps(placement()).encode())
            os._exit(0)
        os.close(writer)
        forked = json.loads(os.read(reader, 1 << 16))
        os.waitpid(child, 0)
        with multiprocessing.get_context("spawn").Pool(1) as pool:
            connection.send(Reply(result=(forked, pool.apply(placement))))


def placement() -> dict[str, Any]:
    """Where this process runs and what an OpenMP team and a GPU runtime see here."""
    keys = ("OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
    return {"cpus": sorted(os.sched_getaffinity(0)), **{key: os.environ.get(key) for key in keys}}


def test_a_slot_pins_what_the_call_starts_and_the_next_call_starts_from_the_whole_mask(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "96")
    monkeypatch.setenv("HPCAGENT_BENCH_JUDGE_GPUS_PER_NODE", "0")
    monkeypatch.setenv("HPCAGENT_BENCH_JUDGE_CPU_SLOTS_PER_NODE", "2")
    session = SessionProcess(SETUP, target=report_pin)

    unpinned = session.call("view_tree", {}).result[0]
    first = session.call("get_performance", {}, slot=0).result
    second = session.call("get_performance", {}, slot=1).result
    again = session.call("view_tree", {}).result[0]
    session.close()

    assert unpinned == again and unpinned["OMP_NUM_THREADS"] == "96", "a call without a slot gets everything back"
    for forked, spawned in (first, second):
        assert forked == spawned, "a fork and a spawn inherit the same pin"
        assert forked["OMP_NUM_THREADS"] == str(len(forked["cpus"])) and forked["OMP_PLACES"] == "cores"
        assert set(forked["cpus"]) < set(unpinned["cpus"])
    assert not set(first[0]["cpus"]) & set(second[0]["cpus"]), "two slots share no core"


def test_a_call_that_does_not_answer_in_time_ends_the_session(monkeypatch):
    monkeypatch.setattr(server, "STOP_GRACE_S", 1.0)
    session = SessionProcess(SETUP, target=ready_then_hang)
    started = time.monotonic()

    reply = session.call("view_tree", {}, timeout_s=2.0)

    assert "did not answer within 2 s" in reply.error
    assert time.monotonic() - started < 30.0
    assert not session.alive


def test_a_measuring_call_whose_client_left_stops_and_the_session_answers_the_next_call():
    session = SessionProcess(SETUP, target=ready_then_stop_when_asked)
    gone = threading.Event()
    gone.set()
    started = time.monotonic()

    reply = session.call("get_performance", {}, gone)

    assert reply.stopped and "call it again" in reply.error, reply
    assert time.monotonic() - started < 10.0, "the slot comes back at once, not at the end of the work"
    assert session.alive and session.call("view_tree", {}).result == "still here"
    session.close()


def test_a_call_out_of_budget_is_stopped_with_what_to_do_and_keeps_the_session():
    session = SessionProcess(SETUP, target=ready_then_stop_when_asked)

    reply = session.call("get_performance", {}, timeout_s=0.5)

    assert reply.stopped and "Your moves are kept" in reply.error, reply
    assert session.alive
    session.close()


def test_a_call_that_holds_no_slot_finishes_after_its_client_left():
    session = SessionProcess(SETUP, target=ready_then_report_library_path)
    gone = threading.Event()
    gone.set()

    reply = session.call("apply_move", {}, gone)

    assert not reply.stopped and not reply.error, reply
    session.close()


def test_a_session_that_died_reports_its_exit_and_is_not_alive():
    session = SessionProcess(SETUP, target=ready_then_die)

    reply = session.call("view_tree", {})

    assert "the session process died" in reply.error
    session.process.join(10.0)
    assert not session.alive


def test_the_session_starts_in_the_llvm_openmp_context_and_the_caller_keeps_its_own(tmp_path, monkeypatch):
    lib = tmp_path / "llvm" / "lib"
    lib.mkdir(parents=True)
    monkeypatch.setenv("HPCAGENT_BENCH_RUNTIME_OMP_CONTEXT_ROOT", str(tmp_path))
    monkeypatch.setattr(plugins, "register", lambda: agent_plugins.find([REPO_ROOT / "plugin" / "nestforge"]))
    before = os.environ.get("LD_LIBRARY_PATH")
    session = SessionProcess(SETUP, target=ready_then_report_library_path)

    reply = session.call("view_tree", {})

    assert reply.result.split(os.pathsep)[0] == str(lib)
    assert os.environ.get("LD_LIBRARY_PATH") == before


if __name__ == "__main__":
    test_a_session_that_died_reports_its_exit_and_is_not_alive()
