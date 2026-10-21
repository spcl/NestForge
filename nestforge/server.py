# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A :class:`Session` in a process of its own, driven over a pipe: the caller's threads never fork, and a crash in
generated code ends this process, not the caller. The HPCAgent-Bench plugin serves each episode's kernel with one.
HPCAgent-Bench is imported in the child only, so importing nestforge never loads it."""

from __future__ import annotations

import multiprocessing
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any

import numpy as np

from nestforge.corpus.bench import corpus_kernel, preset_sizes
from nestforge.records import Record
from nestforge.session import Session
from nestforge.stages.canonicalize import Targets
from nestforge.stages.performance import Performance, Reference

#: Tool -> the session call it makes, the tool's validated arguments passed as keywords.
CALLS: dict[str, Callable[..., object]] = {
    "get_cpf": Session.describe,
    "view_tree": Session.view,
    "list_moves": Session.list_moves,
    "apply_move": Session.apply_move,
    "run_defaults": Session.run_defaults,
    "set_scope": Session.define_scope,
    "unset_scope": Session.unset_scope,
    "get_performance": lambda session: None,
}

#: Tools whose answer carries :meth:`Session.performance`.
TIMED = frozenset({"apply_move", "run_defaults", "get_performance"})

#: Seconds a closing session process gets before it is terminated.
CLOSE_TIMEOUT_S = 5.0


@dataclass(frozen=True, slots=True)
class Setup:
    """Which kernel a session serves: a bench kernel name, the size preset, the graded datatype and the device."""

    kernel: str
    preset: str
    datatype: str
    gpu: bool


class Reply(Record):
    """One call's answer: the session's result and, for a timed tool, its performance; ``error`` instead on failure."""

    result: Any = None
    performance: Performance | None = None
    error: str = ""


def bench_data(setup: Setup) -> tuple[dict[str, np.ndarray], Reference | None]:
    """The bench's inputs at ``setup``'s preset, and its reference outputs with the grading tolerance (``None`` when
    the bench has none for this kernel)."""
    from hpcagent_bench.categories import Preset
    from hpcagent_bench.frameworks.benchmark import Benchmark
    from hpcagent_bench.frameworks.test import kernel_tolerances
    from hpcagent_bench.harness.grading import ReferenceUnavailable
    from hpcagent_bench.numerical_oracle import compiled_reference
    from hpcagent_bench.support.bindings.contract import graded_datatype

    kernel = corpus_kernel(setup.kernel)
    assert kernel is not None, "start() checked the kernel"
    data = Benchmark(kernel.short_name).get_data(preset=Preset.parse(setup.preset))
    inputs = {key: value for key, value in data.items() if isinstance(value, np.ndarray)}
    try:
        expected = compiled_reference(kernel.spec, dict(data))
    except ReferenceUnavailable:
        return inputs, None
    rtol, atol = kernel_tolerances(kernel.spec, graded_datatype(kernel.spec, setup.datatype))
    return inputs, Reference(expected, rtol, atol)


def start(setup: Setup) -> tuple[Session, dict[str, np.ndarray], Reference | None]:
    """The canonicalized session of ``setup``'s kernel, and the data its whole program runs on."""
    kernel = corpus_kernel(setup.kernel)
    if kernel is None:
        raise LookupError(f"{setup.kernel!r} has no DaCe program NestForge can start from")
    session = Session(kernel.to_sdfg(), Targets(gpu=setup.gpu), sizes=preset_sizes(kernel, setup.preset))
    session.canonicalize()
    inputs, reference = bench_data(setup)
    return session, inputs, reference


def failure(err: BaseException) -> Reply:
    return Reply(error=f"{type(err).__name__}: {err}")


def serve(connection: Connection, setup: Setup) -> None:
    """The child: build the session, report ready (or why not), then answer ``(tool, arguments)`` until closed."""
    try:
        session, inputs, reference = start(setup)
    except Exception as err:  # noqa: BLE001 -- the caller reads why the session could not start
        connection.send(failure(err))
        return
    connection.send(Reply())
    while True:
        try:
            tool, arguments = connection.recv()
        except EOFError:
            return
        try:
            result = CALLS[tool](session, **arguments)
            performance = session.performance(inputs, reference) if tool in TIMED else None
            connection.send(Reply(result=result, performance=performance))
        except Exception as err:  # noqa: BLE001 -- the caller reads it as the tool's error
            connection.send(failure(err))


class SessionProcess:
    """The caller's handle on one session process."""

    __slots__ = ("__weakref__", "connection", "process")

    def __init__(self, setup: Setup) -> None:
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        # not a daemon: the session forks and spawns children of its own; the finalizer ends it with the caller
        self.process = context.Process(target=serve, args=(child, setup))
        self.process.start()
        child.close()
        weakref.finalize(self, self.process.kill)
        ready = self.receive()
        if ready.error:
            self.close()
            raise LookupError(ready.error)

    def receive(self) -> Reply:
        try:
            return self.connection.recv()
        except EOFError:
            return Reply(error=f"the session process died (exit code {self.process.exitcode})")

    def call(self, tool: str, arguments: dict[str, object]) -> Reply:
        """Run ``tool`` (a key of :data:`CALLS`) with ``arguments`` in the session process."""
        self.connection.send((tool, arguments))
        return self.receive()

    def close(self) -> None:
        self.connection.close()
        self.process.join(CLOSE_TIMEOUT_S)
        if self.process.is_alive():
            self.process.kill()
