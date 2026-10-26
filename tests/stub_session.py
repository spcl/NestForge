# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A stand-in for the judge's session process: every tool answers a predefined reply, so the route, the agent tools
and a scripted agent are tested without DaCe. :func:`install` swaps it into the loaded route module."""

import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import pytest
from hpcagent_bench.harness.routes import ROUTES

from nestforge.server import Reply, Setup

#: Each tool's predefined answer, shaped as the session's own.
REPLIES: dict[str, Reply] = {
    "get_cpf": Reply(result="epoch 1\n`- state0_0\n   |- kernel map_0 [i=0:N]\n   `- kernel map_1 [i=0:N]"),
    "view_tree": Reply(result="epoch 1\n`- state0_0"),
    "list_moves": Reply(result=[{"kind": "fuse", "labels": ["map_0", "map_1"], "epoch": 1}]),
    "apply_move": Reply(result={"status": "applied", "reason": "", "epoch": 2}),
    "run_defaults": Reply(result={"status": "applied", "reason": "", "epoch": 2}),
    "set_scope": Reply(result={"status": "applied", "reason": "kernel_0", "epoch": 2}),
    "unset_scope": Reply(result={"status": "applied", "reason": "", "epoch": 3}),
    "list_kernels": Reply(result=[{"name": "kernel_0", "device": "cpu", "arguments": ["A", "B"]}]),
    "read_kernel": Reply(result="void kernel_0(double* A, double* B) {}"),
    "write_kernel": Reply(result={"status": "ok", "reason": ""}),
    "tune_kernel": Reply(result={"kernel": "kernel_0", "cells": 4, "winner": "gcc-O3", "config": {}}),
    "get_performance": Reply(result=None),
    "finalize": Reply(result={"library": "/shared/nestforge/libfuse_diamond.so", "workspace_bytes": "0"}),
}

#: Seconds a stub start and a stub call take, so concurrent requests overlap unless the route serializes them.
DELAY_S = 0.05


@dataclass(slots=True)
class Stub:
    """What the stub sessions were asked: the sessions started, every call, and the most calls in flight at once.
    ``dies_on`` names a tool whose call ends its session, as a crash in generated code does."""

    replies: dict[str, Reply] = field(default_factory=lambda: dict(REPLIES))
    dies_on: str = ""
    started: list[Setup] = field(default_factory=list)
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    in_flight: int = 0
    most_in_flight: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class StubSession:
    """A :class:`~nestforge.server.SessionProcess` answering from its :class:`Stub`."""

    __slots__ = ("alive", "stub")

    def __init__(self, stub: Stub, setup: Setup) -> None:
        self.stub = stub
        stub.started.append(setup)
        time.sleep(DELAY_S)
        self.alive = True

    def call(self, tool: str, arguments: dict[str, Any]) -> Reply:
        stub = self.stub
        with stub.lock:
            stub.in_flight += 1
            stub.most_in_flight = max(stub.most_in_flight, stub.in_flight)
        time.sleep(DELAY_S)
        with stub.lock:
            stub.in_flight -= 1
            stub.calls.append((tool, arguments))
        if tool == stub.dies_on:
            self.alive = False
            return Reply(error="the session process died (exit code -9)")
        return stub.replies[tool]

    def close(self) -> None:
        self.alive = False


def install(monkeypatch: pytest.MonkeyPatch, stub: Stub) -> Stub:
    """Serve every NestForge session of the loaded route from ``stub`` until ``monkeypatch`` undoes it."""
    route = sys.modules[ROUTES["nestforge"].handler.__module__]
    monkeypatch.setattr(route, "SessionProcess", lambda setup: StubSession(stub, setup))
    return stub
