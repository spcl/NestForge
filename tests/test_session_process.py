# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A session process that stops answering is ended after the call's timeout, and one that died says so."""

import time
from multiprocessing.connection import Connection

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


if __name__ == "__main__":
    test_a_call_that_does_not_answer_in_time_ends_the_session()
    test_a_session_that_died_reports_its_exit_and_is_not_alive()
