# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A :class:`Session` in a process of its own, driven over a pipe: the caller's threads never fork, and a crash in
generated code ends this process, not the caller. The HPCAgent-Bench plugin serves each episode's kernel with one.
HPCAgent-Bench is imported in the child only, so importing nestforge never loads it."""

from __future__ import annotations

import multiprocessing
import sys
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import dace
import numpy as np

from nestforge.corpus.bench import corpus_kernel, preset_sizes
from nestforge.records import Record
from nestforge.session import Codegen, Session
from nestforge.stages.canonicalize import Targets
from nestforge.stages.finalize import BenchTarget, Final
from nestforge.stages.moves import move_kinds
from nestforge.stages.performance import Performance, Reference


def finalize(session: Session) -> Final:
    """The session's program assembled into the folder the judge and the agent share, one folder per session."""
    from hpcagent_bench.harness.sandbox import shared_dir

    return session.finalize(Path(shared_dir()) / "nestforge" / session.work_dir.name)


#: Tool -> the session call it makes, the tool's validated arguments passed as keywords.
CALLS: dict[str, Callable[..., object]] = {
    "get_cpf": Session.describe,
    "view_tree": Session.view,
    "list_move_kinds": lambda session: move_kinds(),
    "list_moves": Session.list_moves,
    "apply_move": Session.apply_move,
    "run_defaults": Session.run_defaults,
    "set_scope": Session.set_scope,
    "unset_scope": Session.unset_scope,
    "list_kernels": Session.list_kernels,
    "read_kernel": Session.kernel_source,
    "write_kernel": Session.set_kernel_source,
    "tune_kernel": Session.sweep,
    "get_performance": lambda session: None,
    "finalize": finalize,
}

#: Tools whose answer carries :meth:`Session.performance`.
TIMED = frozenset({"apply_move", "run_defaults", "write_kernel", "tune_kernel", "get_performance"})

#: Seconds a closing session process gets before it is terminated.
CLOSE_TIMEOUT_S = 5.0

#: Seconds one tool call may take (a build of the whole program included) before its session is ended.
TOOL_TIMEOUT_S = 1800.0


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
    # the program names an argument the generator renamed by its new name; unmapped, it would run on zeros
    renames = kernel.renames()
    # a scalar argument (cloudsc's timestep ptsphy) is a 0-d array; dropping it would zero it
    inputs = {
        renames.get(key, key): np.asarray(value)
        for key, value in data.items()
        if isinstance(value, (np.ndarray, np.generic, int, float))
    }
    try:
        expected = compiled_reference(kernel.spec, dict(data))
    except ReferenceUnavailable as err:
        print(f"nestforge: {setup.kernel}: no reference, outputs go unchecked: {err}", file=sys.stderr, flush=True)
        return inputs, None
    rtol, atol = kernel_tolerances(kernel.spec, graded_datatype(kernel.spec, setup.datatype))
    return inputs, Reference({renames.get(k, k): v for k, v in expected.items()}, rtol, atol)


def start(setup: Setup, codegen: Codegen = "sweep") -> tuple[Session, dict[str, np.ndarray], Reference | None]:
    """The canonicalized session of ``setup``'s kernel, and the data its whole program runs on. The agent drives the
    moves; scopes, placement and code generation (``codegen``, the stage 6 sweep by default) run on their defaults."""
    kernel = corpus_kernel(setup.kernel)
    if kernel is None:
        raise LookupError(f"{setup.kernel!r} has no DaCe program NestForge can start from")
    session = Session(
        kernel.to_sdfg(), Targets(gpu=setup.gpu), sizes=preset_sizes(kernel, setup.preset), codegen=codegen
    )
    session.bench = BenchTarget(kernel.short_name, setup.datatype, setup.gpu)
    session.canonicalize()
    inputs, reference = bench_data(setup)
    # after canonicalization: it is what promotes a scalar input to a symbol
    session.sizes = symbol_inputs(session.sdfg, inputs) | session.sizes
    return session, inputs, reference


def symbol_inputs(sdfg: dace.SDFG, inputs: dict[str, np.ndarray]) -> dict[str, int]:
    """The integer scalars ``initialize()`` returns that the program reads as symbols, not data (TSVC s162's ``k``,
    which canonicalization promotes to a symbol): only sizes bind a symbol, so each becomes one. A size the
    preset already gives keeps the preset's value."""
    free = {str(s) for s in sdfg.free_symbols}
    return {
        name: int(value)
        for name, value in inputs.items()
        if name in free and value.ndim == 0 and np.issubdtype(value.dtype, np.integer)
    }


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

    def __init__(self, setup: Setup, target: Callable[[Connection, Setup], None] = serve) -> None:
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        # not a daemon: the session forks and spawns children of its own; the finalizer ends it with the caller
        self.process = context.Process(target=target, args=(child, setup))
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

    @property
    def alive(self) -> bool:
        """Whether the session process still runs; a dead one answers no further call."""
        return self.process.is_alive()

    def call(self, tool: str, arguments: dict[str, object], timeout_s: float = TOOL_TIMEOUT_S) -> Reply:
        """Run ``tool`` (a key of :data:`CALLS`) with ``arguments`` in the session process; a call that does not
        answer within ``timeout_s`` ends the session."""
        self.connection.send((tool, arguments))
        if not self.connection.poll(timeout_s):
            self.close()
            return Reply(error=f"{tool} did not answer within {timeout_s:.0f} s; the session was ended")
        return self.receive()

    def close(self) -> None:
        self.connection.close()
        self.process.join(CLOSE_TIMEOUT_S)
        if self.process.is_alive():
            self.process.kill()
            self.process.join()
