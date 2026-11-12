# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A :class:`Session` in a process of its own, driven over a pipe: the caller's threads never fork, and a crash in
generated code ends this process, not the caller. The HPCAgent-Bench plugin serves each episode's kernel with one.
HPCAgent-Bench is imported when a session starts, so importing nestforge never loads it.

A call that builds or runs code stops when its caller asks (:data:`STOP_SIGNAL`), keeping the session and everything
it already built and measured, so the same call goes on from there."""

from __future__ import annotations

import contextlib
import copy
import functools
import hashlib
import importlib
import inspect
import json
import multiprocessing
import os
import shutil
import signal
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.util import Finalize
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dace
import numpy as np
from dace.frontend.python.parser import DaceProgram
from dace.libraries.standard.nodes.external_call import external_calls

from nestforge.build.isolation import CHILD_CPUS, SEAL_KEEP_ENV, restrict_cpus
from nestforge.build.toolchain import SymbolValue
from nestforge.corpus.bench import corpus_kernel, preset_sizes
from nestforge.records import KernelStatus, MoveStatus, Record
from nestforge.session import Codegen, KernelOutcome, MoveResult, PlaceResult, Session, Stage, Stopped
from nestforge.stages.canonicalize import Targets
from nestforge.stages.finalize import BenchTarget, Final
from nestforge.stages.moves import move_kinds
from nestforge.stages.performance import BENCH_REFERENCE, NUMPY_REFERENCE, Performance, Reference

if TYPE_CHECKING:
    from hpcagent_bench.harness.judge_scheduler import DeviceSlot


def finalize(session: Session) -> Final:
    """The session's program assembled into the folder the judge and the agent share, one folder per session."""
    from hpcagent_bench.harness.sandbox import shared_dir

    folder = Path(session.shared_root or shared_dir()) / "nestforge" / session.work_dir.name
    folder.mkdir(parents=True, exist_ok=True)
    # a folder per finalize: a grade still copying an earlier library must not read this one
    final = session.finalize(Path(tempfile.mkdtemp(prefix="final-", dir=folder)))
    # every score and submit finalizes first, so each graded library's moves are on record beside it
    digest = hashlib.sha256(final.library.read_bytes()).hexdigest()
    moves = [move.model_dump(mode="json") for move in session.applied]
    device_code = bool(final.device_source)
    line = {
        "library": str(final.library),
        "sha256": digest,
        "moves": moves,
        "kernels": final.kernels,
        "device_code": device_code,
    }
    with (folder / "applied_moves.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps(line) + "\n")
    return final


def has_kernels(session: Session) -> bool:
    return bool(external_calls(session.sdfg))


def is_placed(session: Session) -> bool:
    return session.unplaced is not None


#: Why a move is refused: moves take the rows of the program before its scopes.
MOVES_CLOSED = "the program has kernels: moves are closed; next_phase reaches a new round, which reopens them"
#: Why a scope is refused: the agent's kernel work builds on the kernels placement fixed.
SCOPES_FIXED = "the kernels are placed: scopes are fixed; next_phase reaches a new round, which reopens them"


def until(closed: Callable[[Session], bool], why: str, call: Callable[..., object]) -> Callable[..., object]:
    """``call``, refused with ``why`` while ``closed`` holds of the session."""

    @functools.wraps(call)
    def run(session: Session, **arguments: object) -> object:
        if closed(session):
            raise ValueError(why)
        return call(session, **arguments)

    return run


def run_defaults(session: Session, stage: Stage) -> object:
    if stage in ("moves", "finish") and has_kernels(session):
        raise ValueError(MOVES_CLOSED)
    return session.run_defaults(stage)


#: Tool -> the session call it makes, the tool's validated arguments passed as keywords.
CALLS: dict[str, Callable[..., object]] = {
    "get_cpf": Session.describe,
    "view_tree": Session.view,
    "list_move_kinds": lambda session: move_kinds(),
    "list_moves": Session.list_moves,
    "apply_move": until(has_kernels, MOVES_CLOSED, Session.apply_move),
    "undo_move": until(has_kernels, MOVES_CLOSED, Session.undo),
    "applied_moves": lambda session: session.applied,
    "run_defaults": run_defaults,
    "set_scope": until(is_placed, SCOPES_FIXED, Session.set_scope),
    "unset_scope": until(is_placed, SCOPES_FIXED, Session.unset_scope),
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
#: A fresh session repeats them to be the program the agent left: a sweep too, finalize sweeps no written kernel.
CHANGING = frozenset(
    {"apply_move", "undo_move", "run_defaults", "set_scope", "unset_scope", "write_kernel", "tune_kernel"}
)

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
        if isinstance(value, (np.ndarray, np.generic, int, float, complex))
    }
    source = BENCH_REFERENCE
    try:
        expected = compiled_reference(kernel.spec, dict(data))
    except ReferenceUnavailable as err:
        hint = f"; set {KERNELBENCH_DIR_VAR} to a KernelBench checkout" if "KernelBench" in str(err) else ""
        try:
            expected = numpy_outputs(kernel.spec, dict(data))
        except ReferenceUnavailable as fallback:
            print(
                f"nestforge: {setup.kernel}: no reference, outputs go unchecked: {err}{hint}; {fallback}",
                file=sys.stderr,
                flush=True,
            )
            return inputs, None
        source = NUMPY_REFERENCE
        print(
            f"nestforge: {setup.kernel}: the bench has no reference ({err}{hint}); outputs are graded against the "
            "kernel's NumPy function instead",
            file=sys.stderr,
            flush=True,
        )
    rtol, atol = kernel_tolerances(kernel.spec, graded_datatype(kernel.spec, setup.datatype))
    return inputs, Reference({renames.get(k, k): v for k, v in expected.items()}, rtol, atol, source)


def numpy_outputs(spec: Any, data: dict[str, Any]) -> dict[str, np.ndarray]:
    """The outputs of the kernel's own NumPy function (``<module>_numpy.py`` of the bench) on a copy of ``data``,
    by output name. Raises ``ReferenceUnavailable`` when the kernel has none or it does not run on ``data``."""
    from hpcagent_bench.harness.grading import ReferenceUnavailable
    from hpcagent_bench.harness.scoring import ieee_reference_data
    from hpcagent_bench.spec import numpy_reference_path

    path = numpy_reference_path(spec.relative_path, spec.module_name)
    if path is None:
        raise ReferenceUnavailable(f"{spec.short_name} has no NumPy function")
    module_path = f"hpcagent_bench.benchmarks.{spec.relative_path.replace('/', '.')}.{path.stem}"
    try:
        function = vars(importlib.import_module(module_path))[spec.func_name]
        names = list(inspect.signature(function).parameters)
        widened = ieee_reference_data(dict(data))
        # a function writes its outputs in place: give it copies, so the session's inputs stay as drawn
        args = {name: copy.deepcopy(widened[name]) for name in names}
        result = function(**args)
        returned = [] if result is None else list(result) if isinstance(result, (tuple, list)) else [result]
        unwritten = [name for name in spec.output_args if name not in args]
        if len(unwritten) != len(returned):
            raise ReferenceUnavailable(f"{path.name}: {len(returned)} returned values for outputs {unwritten}")
        values = {name: args[name] for name in spec.output_args if name in args} | dict(zip(unwritten, returned))
        return {name: np.asarray(values[name]) for name in spec.output_args}
    except ReferenceUnavailable:
        raise
    except Exception as err:  # noqa: BLE001 -- any failure of the function is "no reference", named
        raise ReferenceUnavailable(f"NumPy function of {spec.short_name}: {type(err).__name__}: {err}") from err


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
    kernel = corpus_kernel(setup.kernel)
    assert kernel is not None, "the session started from it"
    inputs |= closure_inputs(kernel.program(), session.sdfg)
    session.sizes = symbol_inputs(session.sdfg, inputs) | session.sizes
    session.check_sizes = check_rung_sizes(setup, session)
    return inputs, reference


def check_rung_sizes(setup: Setup, session: Session) -> dict[str, SymbolValue]:
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


def symbol_inputs(sdfg: dace.SDFG, inputs: dict[str, np.ndarray]) -> dict[str, SymbolValue]:
    """The numeric scalars ``initialize()`` returns that the program reads as symbols, not data (TSVC s162's integer
    ``k``, cp2k_density_matrix_trs4's float ``threshold``; canonicalization promotes both to symbols): only sizes
    bind a symbol, so each becomes one, a float kept a float. A size the preset already gives keeps the preset's
    value."""
    free = {str(s) for s in sdfg.free_symbols}
    return {
        name: value.item()
        for name, value in inputs.items()
        if name in free and value.ndim == 0 and numeric_scalar(value.dtype)
    }


def closure_inputs(program: DaceProgram, sdfg: dace.SDFG) -> dict[str, np.ndarray]:
    """The arrays ``program`` computed when it was parsed, which ``sdfg`` takes as arguments: DaCe evaluates a
    subscripted NumPy call on constants (cp2k_density_matrix_trs4's ``np.arange(3, dtype=np.int64)[None, :]``) once,
    at parse time, and passes the result in as a closure array. Only the program holds its value, and the bench's
    inputs do not name it: left out, it would be zero-filled with the outputs and the program would run on zeros.
    ``program`` must have been parsed (``to_sdfg``), as :func:`started_session` does."""
    return {
        name: np.asarray(value)
        for name, value in program.__sdfg_closure__().items()
        if name in sdfg.arrays and isinstance(value, np.ndarray)
    }


def numeric_scalar(dtype: np.dtype) -> bool:
    """A bool, an integer or a float: the kinds a symbol takes (vexx_k's ``gamma_only`` flag becomes one)."""
    return np.issubdtype(dtype, np.bool_) or np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating)


def stopped_text(tool: str, timeout_s: float) -> str:
    """What the agent reads when ``tool`` ran out of its budget: nothing was lost, and repeating it continues."""
    return (
        f"{tool} ran out of its {timeout_s:.0f} s budget and stopped. Your moves are kept, and every kernel it already "
        "built and measured is cached: call it again to go on from there."
    )


def failure(err: BaseException) -> Reply:
    return Reply(error=f"{type(err).__name__}: {err}")


def move_text(arguments: dict[str, Any]) -> str:
    """How an answer names the move ``apply_move`` was called with: its kind and labels."""
    return f"{arguments['kind']} on {', '.join(arguments['labels'])}"


def applied(reply: Reply) -> bool:
    """Whether a change's answer changed the program: a refused move or kernel is an answer and changes nothing."""
    result = reply.result
    status = result.status if isinstance(result, (MoveResult, PlaceResult, KernelOutcome)) else MoveStatus.APPLIED
    return not reply.error and status in (MoveStatus.APPLIED, KernelStatus.OK, KernelStatus.UNCHECKED)


def timed_call(
    session: Session,
    tool: str,
    arguments: dict[str, Any],
    data: tuple[dict[str, np.ndarray], Reference | None] | None,
    timed: bool = True,
) -> Reply:
    """``tool``'s result on ``session``, and its performance when it is timed. A stop during that measurement does not
    undo the tool: its result stands, without the feedback.

    :param timed: False for a rebuild's repeat: nobody reads its performance."""
    result = CALLS[tool](session, **arguments)
    if tool not in TIMED or not timed:
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
    timed: bool = True,
) -> Reply:
    """The reply to one call. A change stays only if its whole answer succeeds, and its error says so."""
    if tool not in CHANGING:
        try:
            return timed_call(session, tool, arguments, data, timed)
        except Exception as err:  # noqa: BLE001 -- the caller reads it as the tool's error
            return failure(err)
    try:
        with session.atomic():
            return timed_call(session, tool, arguments, data, timed)
    except Exception as err:  # noqa: BLE001 -- the caller reads it as the tool's error
        if tool == "apply_move":
            error = f"transformation {move_text(arguments)} crashed: {failure(err).error}"
            return Reply(error=f"{error}. It was rolled back; the session is intact, try another move.")
        return Reply(
            error=f"{failure(err).error} (the program is unchanged; the session is intact, other calls still work)"
        )


def pin(slot: DeviceSlot | None, gpu: bool, cores: set[int], environment: dict[str, str]) -> None:
    """What this call starts runs as a graded child of device slot ``slot`` does: on the slot's cores and memory, with
    a team of their size, and on its GPU if it is a GPU slot; on all of ``cores`` without a slot. Each call starts from
    the session's first ``cores`` and ``environment``, as the bench narrows the current mask and device list."""
    from hpcagent_bench import flags
    from hpcagent_bench.harness import native_call

    restrict_cpus(cores)
    os.environ.clear()
    os.environ.update(environment)
    if slot is None:
        return
    cpus = native_call.grading_cpus(slot.index)
    if cpus:
        restrict_cpus(cpus)
        native_call.bind_memory(native_call.numa_nodes(cpus))
        os.environ.update(flags.cpu_env(flags.Mode.MULTI_CORE, threads=native_call.slot_threads(cpus)))
        os.environ.update(native_call.CHILD_BIND_ENV)
    if gpu and slot.kind == "gpu":  # a CPU slot's index is no GPU ordinal
        native_call.restrict_visible_device(os.environ, slot.index)


def serve(connection: Connection, setup: Setup) -> None:
    """The child: build the session, report ready (or why not), then answer ``(tool, arguments, slot, timed)`` until
    closed: ``slot`` is the device slot the call holds, ``timed`` as :func:`timed_call` takes it."""
    try:
        session = started_session(setup)
    except Exception as err:  # noqa: BLE001 -- the caller reads why the session could not start
        connection.send(failure(err))
        return
    signal.signal(STOP_SIGNAL, lambda signum, frame: session.stopper.ask())
    connection.send(Reply())
    data: tuple[dict[str, np.ndarray], Reference | None] | None = None
    # the session's own mask, not the current one: libomp may have bound this thread to one CPU already
    cores, environment = set(CHILD_CPUS.cpus), dict(os.environ)
    while True:
        try:
            tool, arguments, slot, timed = connection.recv()
        except EOFError:
            return
        session.stopper.clear()
        if tool == "write_kernel":
            environment[SEAL_KEEP_ENV] = tempfile.gettempdir()  # the session holds agent code from here on
        try:
            pin(slot, setup.gpu, cores, environment)
            if data is None and tool in MEASURING:
                with session.stopper.armed():
                    data = session_data(setup, session)
            connection.send(answer(session, tool, arguments, data, timed))
        except Stopped:
            connection.send(Reply(stopped=True))
        except Exception as err:  # noqa: BLE001 -- the caller reads it as the tool's error
            connection.send(failure(err))


def end_with_parent(parent: int, scratch: str) -> None:
    """End the session once ``parent``, the judge, is gone: only the judge bounds its calls. Not ``PR_SET_PDEATHSIG``,
    which fires when the creating thread ends, and a handler thread ends with its connection."""
    while os.getppid() == parent:
        time.sleep(POLL_S)
    shutil.rmtree(scratch, ignore_errors=True)
    os.killpg(0, signal.SIGKILL)


def own_group(
    target: Callable[[Connection, Setup], None], connection: Connection, setup: Setup, parent: int, scratch: str
) -> None:
    """``target`` in a process group of its own, so that ending the session ends the builds and runs it started, and
    with its temporary files in ``scratch``, which outlives a SIGKILL of the session."""
    os.setpgrp()
    tempfile.tempdir = os.environ["TMPDIR"] = scratch
    threading.Thread(target=end_with_parent, args=(parent, scratch), daemon=True).start()
    target(connection, setup)


def kill_group(pid: int | None) -> None:
    """Kill the session process ``pid`` and everything it started; a group already gone is fine."""
    assert pid is not None, "a started process has a pid"
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGKILL)


def end_session(pid: int | None, scratch: str) -> None:
    """Kill the session process ``pid`` with what it started, and remove its files."""
    kill_group(pid)
    shutil.rmtree(scratch, ignore_errors=True)


class SessionProcess:
    """The caller's handle on one session process."""

    __slots__ = ("__weakref__", "connection", "end", "process")

    def __init__(self, setup: Setup, target: Callable[[Connection, Setup], None] = serve) -> None:
        from hpcagent_bench.plugins import openmp_environment

        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        scratch = tempfile.mkdtemp(prefix="nfsession_")
        # not a daemon: the session forks and spawns children of its own; the finalizer ends it with the caller, ahead
        # of the join of every child that multiprocessing makes at exit, which a weakref.finalize may come after
        self.process = context.Process(target=own_group, args=(target, child, setup, os.getpid(), scratch))
        # libomp, the one runtime NestForge links (plugin.toml openmp), with the BLAS and FFTW built against it
        with openmp_environment("nestforge"):
            self.process.start()
        child.close()
        self.end = Finalize(self, end_session, args=(self.process.pid, scratch), exitpriority=0)
        deadline = time.monotonic() + TOOL_TIMEOUT_S
        while not self.connection.poll(POLL_S):
            if time.monotonic() > deadline:
                self.close()
                raise LookupError(f"the session did not start within {TOOL_TIMEOUT_S:.0f} s")
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
        except (EOFError, ConnectionResetError):
            self.process.join(CLOSE_TIMEOUT_S)
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
        slot: DeviceSlot | None = None,
        timed: bool = True,
    ) -> Reply:
        """Run ``tool`` (a key of :data:`CALLS`) with ``arguments`` in the session process. A measuring call is stopped
        once ``gone`` is set, and any call after ``timeout_s``.

        :param gone: Set when the client left; a call that holds no device slot finishes either way, for its retry.
        :param slot: The device slot the call holds, which what it starts runs on.
        :param timed: False to leave out the performance of a timed tool."""
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):  # a dead session answers EOF below
            self.connection.send((tool, arguments, slot, timed))
        deadline = time.monotonic() + timeout_s
        while not self.connection.poll(POLL_S):
            if not self.alive:  # its forked child keeps the pipe open, so no EOF comes
                kill_group(self.process.pid)
                return self.receive()
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
        self.end()
        self.process.kill()  # one still starting has no group yet
        self.process.join()
