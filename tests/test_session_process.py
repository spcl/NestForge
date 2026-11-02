# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A session process that stops answering is ended after the call's timeout, and one that died says so; it starts
in the bench's llvm OpenMP context, whose libomp is the one runtime NestForge links."""

import os
import time
from multiprocessing.connection import Connection

from hpcagent_agent import plugins as agent_plugins
from hpcagent_bench import plugins

from nestforge.paths import REPO_ROOT
from nestforge.server import Reply, SessionProcess, Setup

SETUP = Setup(kernel="none", preset="S", datatype="float64", gpu=False)


def ready_then_hang(connection: Connection, setup: Setup) -> None:
    """A session that starts, then never answers a call."""
    del setup
    connection.send(Reply())
    connection.recv()
    time.sleep(600)


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


def test_a_call_that_does_not_answer_in_time_ends_the_session():
    session = SessionProcess(SETUP, target=ready_then_hang)
    started = time.monotonic()

    reply = session.call("view_tree", {}, timeout_s=2.0)

    assert "did not answer within 2 s" in reply.error
    assert time.monotonic() - started < 30.0
    assert not session.alive


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
    test_a_call_that_does_not_answer_in_time_ends_the_session()
    test_a_session_that_died_reports_its_exit_and_is_not_alive()
