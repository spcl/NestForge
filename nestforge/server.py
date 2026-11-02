# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A :class:`Session` in a process of its own, driven over a pipe: the caller's threads never fork, and a crash in
generated code ends this process, not the caller. The HPCAgent-Bench plugin serves each episode's kernel with one.
HPCAgent-Bench is imported when a session starts, so importing nestforge never loads it.

A call that builds or runs code stops when its caller asks (:data:`STOP_SIGNAL`), keeping the session and everything
it already built and measured, so the same call goes on from there."""

from __future__ import annotations

import contextlib
import hashlib
import json
import multiprocessing
import os
import signal
import sys
import threading
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import dace
import numpy as np

from nestforge.corpus.bench import corpus_kernel, preset_sizes
from nestforge.records import KernelStatus, MoveStatus, Record
from nestforge.session import Codegen, KernelOutcome, MoveResult, PlaceResult, Session, Stopped
from nestforge.stages.canonicalize import Targets
from nestforge.stages.finalize import BenchTarget, Final
from nestforge.stages.moves import move_kinds
from nestforge.stages.performance import Performance, Reference


def finalize(session: Session) -> Final:
    """The session's program assembled into the folder the judge and the agent share, one folder per session."""
    from hpcagent_bench.harness.sandbox import shared_dir

    root = session.shared_root or shared_dir()

    final = session.finalize(Path(root) / "nestforge" / session.work_dir.name)
    # every score and submit finalizes first, so each graded library's moves are on record beside it
    digest = hashlib.sha256(final.library.read_bytes()).hexdigest()
    moves = [move.model_dump(mode="json") for move in session.applied]
    line = {"library": str(final.library), "sha256": digest, "moves": moves}
    with (final.library.parent / "applied_moves.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps(line) + "\n")
    return final


#: Tool -> the session call it makes, the tool's validated arguments passed as keywords.
CALLS: dict[str, Callable[..., object]] = {
    "get_cpf": Session.describe,
    "view_tree": Session.view,
    "list_move_kinds": lambda session: move_kinds(),
    "list_moves": Session.list_moves,
    "apply_move": Session.apply_move,
    "applied_moves": lambda session: session.applied,
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

#: Tools whose answer carries :meth:`Session.performance`; a move answers at once and is measured on request.
TIMED = frozenset({"write_kernel", "tune_kernel", "get_performance"})

#: Tools that build or run code, so they hold a device slot and need the bench's data. Everything else is Python on
#: the session's program: a read or a move takes a fraction of a second and no device.
MEASURING = TIMED | {"finalize"}

#: Tools whose calls change the session's program; one that fails, its timed answer included, leaves it as it was.
CHANGING = frozenset({"apply_move", "run_defaults", "set_scope", "unset_scope", "write_kernel", "tune_kernel"})

#: What a fresh session repeats to be the program the agent left. Not ``tune_kernel``: finalize sweeps anyway.
REPEATED = CHANGING - {"tune_kernel"}

#: The size rung a kernel falls back to when its NumPy oracle cannot finish at the graded rung.
CHECK_RUNG = "S"

#: How the program's feedback builds its kernels: stage 5's first configuration of each kernel, as cheap to repeat
#: after every move as a comparison needs. The compiler sweep of stage 6 stays with finalize, what is graded.
FEEDBACK_CODEGEN: Codegen = "first"

#: Seconds a closing session process gets before it is terminated.
CLOSE_TIMEOUT_S = 5.0

#: Seconds of work one tool call may take before it is stopped; below the agents' own limit on a call (1800 s), so an
#: agent reads why and what to do instead of its client's timeout.
TOOL_TIMEOUT_S = 1500.0

#: How often a waiting caller looks for its client and the session's answer.
POLL_S = 0.25

#: Seconds a stopped call gets to answer before its session is ended (a long NumPy call takes signals when it returns).
STOP_GRACE_S = 30.0

#: What asks a session process to stop its call.
STOP_SIGNAL = signal.SIGUSR1


#: The bench setting that points an ML kernel's PyTorch reference at a KernelBench checkout; a pip install of the
#: bench does not ship the submodule.
KERNELBENCH_DIR_VAR = "HPCAGENT_BENCH_ML_KERNELBENCH_DIR"


@dataclass(frozen=True, slots=True)
class Setup:
    """Which kernel a session serves: a bench kernel name, the size preset, the graded datatype and the device."""

    kernel: str
    preset: str
    datatype: str
    gpu: bool
    #: The episode\'s folder, where finalize builds; the shared folder when empty.
    shared_root: str = ""


class Reply(Record):
    """One call's answer: the session's result and, for a timed tool, its performance; ``error`` instead on failure.
    ``stopped``: the call was stopped, nothing is wrong and the session keeps what it did. ``ended``: the session
    process is gone, this call's change with it."""

    result: Any = None
    performance: Performance | None = None
    error: str = ""
    stopped: bool = False
    ended: bool = False


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
    data = Benchmark(kernel.short_name).get_data(preset=Preset.parse(size_rung(setup.preset)))
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
        hint = f"; set {KERNELBENCH_DIR_VAR} to a KernelBench checkout" if "KernelBench" in str(err) else ""
        print(
            f"nestforge: {setup.kernel}: no reference, outputs go unchecked: {err}{hint}", file=sys.stderr, flush=True
        )
        return inputs, None
    rtol, atol = kernel_tolerances(kernel.spec, graded_datatype(kernel.spec, setup.datatype))
    return inputs, Reference({renames.get(k, k): v for k, v in expected.items()}, rtol, atol)


def size_rung(preset: str) -> str:
    """The rung a session of the judge's ``preset`` token runs on: a fuzzed token (``XL+fuzz``, ``fuzzed``) draws its
    sizes around a rung, and a session measures on that rung."""
    from hpcagent_bench.spec import parse_preset

    return parse_preset(preset).anchor.text


def start(setup: Setup, codegen: Codegen = "sweep") -> tuple[Session, dict[str, np.ndarray], Reference | None]:
    """The canonicalized session of ``setup``'s kernel, and the data its whole program runs on."""
    session = started_session(setup, codegen)
    return session, *session_data(setup, session)


def started_session(setup: Setup, codegen: Codegen = "sweep") -> Session:
    """The canonicalized session of ``setup``'s kernel, before any data. The agent drives the moves; scopes, placement
    and code generation (``codegen``, the stage 6 sweep by default) run on their defaults."""
    kernel = corpus_kernel(setup.kernel)
    if kernel is None:
        raise LookupError(f"{setup.kernel!r} has no DaCe program NestForge can start from")
    session = Session(
        kernel.to_sdfg(), Targets(gpu=setup.gpu), sizes=preset_sizes(kernel, size_rung(setup.preset)), codegen=codegen
    )
    session.bench = BenchTarget(kernel.short_name, setup.datatype, setup.gpu)
    session.shared_root = setup.shared_root
    session.canonicalize()
    return session


def session_data(setup: Setup, session: Session) -> tuple[dict[str, np.ndarray], Reference | None]:
    """The data ``session``'s whole program runs on; binds the sizes that come from it. Only code that runs needs it,
    so a read never waits for the reference. Call it after canonicalization: that promotes a scalar input to a symbol."""
    inputs, reference = bench_data(setup)
    session.sizes = symbol_inputs(session.sdfg, inputs) | session.sizes
    session.check_sizes = check_rung_sizes(setup, session)
    return inputs, reference


def check_rung_sizes(setup: Setup, session: Session) -> dict[str, int]:
    """:attr:`Session.sizes` at :data:`CHECK_RUNG` where the rung gives a value."""
    from hpcagent_bench.categories import Preset
    from hpcagent_bench.frameworks.benchmark import Benchmark

    kernel = corpus_kernel(setup.kernel)
    assert kernel is not None, "the session started from it"
    data = Benchmark(kernel.short_name).get_data(preset=Preset.parse(CHECK_RUNG))
    renames = kernel.renames()
    # a scalar argument comes back as an array in the program; a rung that names no value keeps the graded one
    scalars = {renames.get(k, k): np.asarray(v) for k, v in data.items() if isinstance(v, (np.generic, int, float))}
    return session.sizes | symbol_inputs(session.sdfg, scalars) | preset_sizes(kernel, CHECK_RUNG)


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


def stopped_text(tool: str, timeout_s: float) -> str:
    """What the agent reads when ``tool`` ran out of its budget: nothing was lost, and repeating it continues."""
    return (
        f"{tool} ran out of its {timeout_s:.0f} s budget and stopped. Your moves are kept, and every kernel it already "
        "built and measured is cached: call it again to go on from there."
    )


def failure(err: BaseException) -> Reply:
    return Reply(error=f"{type(err).__name__}: {err}")


def applied(reply: Reply) -> bool:
    """Whether a change's answer changed the program: a refused move or kernel is an answer and changes nothing."""
    result = reply.result
    status = result.status if isinstance(result, (MoveResult, PlaceResult, KernelOutcome)) else MoveStatus.APPLIED
    return not reply.error and status in (MoveStatus.APPLIED, KernelStatus.OK)


def timed_call(
    session: Session,
    tool: str,
    arguments: dict[str, Any],
    data: tuple[dict[str, np.ndarray], Reference | None] | None,
) -> Reply:
    """``tool``'s result on ``session``, and its performance when it is timed. A stop during that measurement does not
    undo the tool: its result stands, without the feedback."""
    result = CALLS[tool](session, **arguments)
    if tool not in TIMED:
        return Reply(result=result)
    assert data is not None, "a timed tool measures, so its data was loaded"
    try:
        return Reply(result=result, performance=session.performance(*data, codegen=FEEDBACK_CODEGEN))
    except Stopped:
        if tool == "get_performance":
            raise
        return Reply(result=result)


def answer(
    session: Session,
    tool: str,
    arguments: dict[str, Any],
    data: tuple[dict[str, np.ndarray], Reference | None] | None,
) -> Reply:
    """The reply to one call. A change stays only if its whole answer succeeds, and its error says so."""
    if tool not in CHANGING:
        try:
            return timed_call(session, tool, arguments, data)
        except Exception as err:  # noqa: BLE001 -- the caller reads it as the tool's error
            return failure(err)
    try:
        with session.atomic():
            return timed_call(session, tool, arguments, data)
    except Exception as err:  # noqa: BLE001 -- the caller reads it as the tool's error
        return Reply(error=f"{failure(err).error} (the program is unchanged)")


def serve(connection: Connection, setup: Setup) -> None:
    """The child: build the session, report ready (or why not), then answer ``(tool, arguments)`` until closed."""
    try:
        session = started_session(setup)
    except Exception as err:  # noqa: BLE001 -- the caller reads why the session could not start
        connection.send(failure(err))
        return
    signal.signal(STOP_SIGNAL, lambda signum, frame: session.stopper.ask())
    connection.send(Reply())
    data: tuple[dict[str, np.ndarray], Reference | None] | None = None
    while True:
        try:
            tool, arguments = connection.recv()
        except EOFError:
            return
        session.stopper.clear()
        try:
            if data is None and tool in MEASURING:
                data = session_data(setup, session)
            connection.send(answer(session, tool, arguments, data))
        except Stopped:
            connection.send(Reply(stopped=True))
        except Exception as err:  # noqa: BLE001 -- the caller reads it as the tool's error
            connection.send(failure(err))


def own_group(target: Callable[[Connection, Setup], None], connection: Connection, setup: Setup) -> None:
    """``target`` in a process group of its own, so that ending the session ends the builds and runs it started."""
    os.setpgrp()
    target(connection, setup)


def kill_group(pid: int | None) -> None:
    """Kill the session process ``pid`` and everything it started; a group already gone is fine."""
    assert pid is not None, "a started process has a pid"
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGKILL)


class SessionProcess:
    """The caller's handle on one session process."""

    __slots__ = ("__weakref__", "connection", "process")

    def __init__(self, setup: Setup, target: Callable[[Connection, Setup], None] = serve) -> None:
        from hpcagent_bench.plugins import openmp_environment

        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        # not a daemon: the session forks and spawns children of its own; the finalizer ends it with the caller
        self.process = context.Process(target=own_group, args=(target, child, setup))
        # libomp, the one runtime NestForge links (plugin.toml openmp), with the BLAS and FFTW built against it
        with openmp_environment("nestforge"):
            self.process.start()
        child.close()
        weakref.finalize(self, kill_group, self.process.pid)
        ready = self.receive()
        if ready.error:
            self.close()
            raise LookupError(ready.error)

    def received(self, tool: str, timeout_s: float) -> Reply:
        reply = self.receive()
        return Reply(error=stopped_text(tool, timeout_s), stopped=True) if reply.stopped else reply

    def receive(self) -> Reply:
        try:
            return self.connection.recv()
        except EOFError:
            return Reply(error=f"the session process died (exit code {self.process.exitcode})")

    @property
    def alive(self) -> bool:
        """Whether the session process still runs; a dead one answers no further call."""
        return self.process.is_alive()

    def call(
        self,
        tool: str,
        arguments: dict[str, object],
        gone: threading.Event | None = None,
        timeout_s: float = TOOL_TIMEOUT_S,
    ) -> Reply:
        """Run ``tool`` (a key of :data:`CALLS`) with ``arguments`` in the session process. A measuring call is stopped
        once ``gone`` is set, and any call after ``timeout_s``.

        :param gone: Set when the client left; a call that holds no device slot finishes either way, for its retry."""
        self.connection.send((tool, arguments))
        deadline = time.monotonic() + timeout_s
        while not self.connection.poll(POLL_S):
            client_left = gone is not None and gone.is_set() and tool in MEASURING
            if client_left or time.monotonic() > deadline:
                return self.stop(tool, timeout_s)
        return self.received(tool, timeout_s)

    def stop(self, tool: str, timeout_s: float) -> Reply:
        """Stop the running ``tool``, a signal per poll (one sent before the session took the call would be lost); a
        session that does not answer is ended, and only then its process is lost."""
        pid, give_up = self.process.pid, time.monotonic() + STOP_GRACE_S
        assert pid is not None, "a started process has a pid"
        while time.monotonic() < give_up:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, STOP_SIGNAL)
            if self.connection.poll(POLL_S):
                return self.received(tool, timeout_s)
        self.close()
        return Reply(error=f"{tool} did not answer within {timeout_s:.0f} s; the session was ended")

    def close(self) -> None:
        self.connection.close()
        self.process.join(CLOSE_TIMEOUT_S)
        if self.process.is_alive():
            kill_group(self.process.pid)
            self.process.join()
