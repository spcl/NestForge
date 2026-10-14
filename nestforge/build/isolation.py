# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Run generated code in a child process, so a crash or a runaway loop becomes a :class:`Failure` instead of
taking down the caller. A child's answer is a :class:`Result` model, validated when it crosses back."""

from __future__ import annotations

import ctypes
import faulthandler
import multiprocessing
import os
import select
import signal
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

#: OpenMP runtimes whose thread pool must be torn down before a fork.
OMP_RUNTIME_SONAMES = ("libgomp.so.1", "libomp.so.5", "libomp.so", "libiomp5.so")

#: Sonames of LLVM libomp (``libiomp5`` is its Intel alias).
LIBOMP_SONAMES = ("libomp.so.5", "libomp.so", "libiomp5.so")

#: ``omp_soft_pause`` of ``omp_pause_resource_t`` (OpenMP 5.0); unlike a hard pause it keeps threadprivate data.
OMP_PAUSE_SOFT = 1

#: Longest exception message a child reports back, after its type name.
ERROR_CHARS = 4000

#: Wall-clock seconds an isolated child may run before it counts as hung.
RUN_TIMEOUT_S = 900.0

#: The shared-memory file LLVM libomp registers per process and removes only at a normal exit; a child that
#: leaves through ``os._exit``, a signal or a kill leaves it behind.
OPENMP_REGISTRATION = "/dev/shm/__KMP_REGISTERED_LIB_{pid}_{uid}"

#: Bytes read from a child's result pipe at a time.
PIPE_CHUNK = 65536


def pause_openmp_pools() -> None:
    """Pause every loaded OpenMP runtime before a fork; a live libgomp pool deadlocks the child."""
    paused: dict[int | None, None] = {}  # one runtime loads under several sonames; a second pause fails
    for soname in OMP_RUNTIME_SONAMES:
        try:
            lib = ctypes.CDLL(soname, mode=os.RTLD_NOLOAD)  # only pause a runtime already mapped
        except OSError:
            continue
        try:
            pause = lib.omp_pause_resource_all
        except AttributeError:
            warnings.warn(
                f"{soname}: no omp_pause_resource_all (pre-OpenMP-5.0); its pool stays up across the fork, "
                "safe only if it installs a pthread_atfork handler (libgomp does not)"
            )
            continue
        address = ctypes.cast(pause, ctypes.c_void_p).value
        if address in paused:
            continue
        paused[address] = None
        pause.argtypes = [ctypes.c_int]
        pause.restype = ctypes.c_int
        # LLVM libomp refuses only when it has no pool (never initialized) or already paused: nothing to pause
        if pause(OMP_PAUSE_SOFT) != 0 and soname not in LIBOMP_SONAMES:
            warnings.warn(f"{soname}: omp_pause_resource_all(soft) failed; its pool stays up across the fork")


def drop_openmp_registration(pid: int) -> None:
    """Remove the libomp registration a finished child ``pid`` may have left in shared memory."""
    Path(OPENMP_REGISTRATION.format(pid=pid, uid=os.getuid())).unlink(missing_ok=True)


class Result(BaseModel):
    """What a child sends back; a subclass names its fields."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


class Failure(Result):
    """A child that raised, crashed, hung or sent nothing usable."""

    error: str


def error_result(e: BaseException) -> Failure:
    """A child's exception as a result, its message cut to :data:`ERROR_CHARS`."""
    return Failure(error=f"{type(e).__name__}: {str(e)[:ERROR_CHARS]}")


def parse_result[T: Result](buf: bytes, result: type[T]) -> T | Failure:
    """The child's JSON as a :class:`Failure` or as ``result``; anything else is a malformed result."""
    if not buf:
        return Failure(error="child produced no result")
    for model in (Failure, result):
        try:
            return model.model_validate_json(buf)
        except ValidationError:
            continue
    return Failure(error="child produced malformed result (crashed mid-write)")


def run_spawned[T: Result](target: Callable[[Any], T], payload: Any, timeout: float = RUN_TIMEOUT_S) -> T | Failure:
    """:func:`run_isolated` in a freshly spawned interpreter: a CUDA context does not survive a fork. ``target``
    and ``payload`` must pickle."""
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    child = context.Process(target=spawned_entry, args=(target, payload, sender))
    child.start()
    assert child.pid is not None, "a started process has a pid"
    sender.close()
    try:
        return spawned_result(receiver, child, timeout)
    finally:
        receiver.close()
        drop_openmp_registration(child.pid)


def spawned_result(receiver: Any, child: Any, timeout: float) -> Any:
    """The spawned child's result, or a :class:`Failure` when it runs past ``timeout`` or dies before sending one."""
    if not receiver.poll(timeout):
        child.kill()
        child.join()
        return Failure(error=f"timeout after {timeout:.0f}s (runaway kernel)")
    try:
        result = receiver.recv()
    except EOFError:  # the child exited, or was killed by a signal, without sending
        child.join()
        return Failure(error=f"crashed (exit code {child.exitcode})")
    child.join()
    return result


def spawned_entry(target: Callable[[Any], Result], payload: Any, sender: Any) -> None:
    """The spawned child's body; a Python exception comes back as an error."""
    faulthandler.disable()  # a child's segfault must not dump the parent's stack
    try:
        result = target(payload)
    except BaseException as e:
        result = error_result(e)
    sender.send(result)
    sender.close()


def run_isolated[T: Result](work_fn: Callable[[], T], result: type[T], timeout: float = RUN_TIMEOUT_S) -> T | Failure:
    """``work_fn()`` from a forked child, or a :class:`Failure` on an exception, crash, timeout or bad result."""
    pause_openmp_pools()
    r, w = os.pipe()
    with warnings.catch_warnings():
        # the fork is the point: the child runs only work_fn, and the OpenMP pools were paused above
        warnings.filterwarnings(
            "ignore", message=".*use of fork\\(\\) may lead to deadlocks", category=DeprecationWarning
        )
        pid = os.fork()
    if pid == 0:
        os.close(r)
        faulthandler.disable()  # a child's segfault must not dump the parent's stack
        try:
            payload = work_fn().model_dump_json()
        except BaseException as e:
            payload = error_result(e).model_dump_json()
        try:
            os.write(w, payload.encode())
        finally:
            os.close(w)
            os._exit(0)
    os.close(w)
    try:
        outcome = forked_result(pid, r, timeout)
        return outcome if isinstance(outcome, Failure) else parse_result(outcome, result)
    finally:
        drop_openmp_registration(pid)


def forked_result(pid: int, r: int, timeout: float) -> bytes | Failure:
    """What the forked child wrote to pipe ``r``, or a :class:`Failure` when it crashes or hangs past ``timeout``;
    the child is reaped either way."""
    start, buf, timed_out = time.perf_counter(), b"", True
    try:
        while True:
            remaining = timeout - (time.perf_counter() - start)
            if remaining <= 0:
                break
            ready, _, _ = select.select([r], [], [], remaining)
            if not ready:
                break
            chunk = os.read(r, PIPE_CHUNK)
            if not chunk:  # the child closed the pipe: done, or dead
                timed_out = False
                break
            buf += chunk
    finally:
        os.close(r)
    if timed_out:
        reaped, status = os.waitpid(pid, os.WNOHANG)
        if reaped == 0:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            return Failure(error=f"timeout after {timeout:.0f}s (runaway kernel)")
    else:
        _, status = os.waitpid(pid, 0)  # the child is exiting, so this returns
    if os.WIFSIGNALED(status):
        return Failure(error=f"crashed (signal {os.WTERMSIG(status)})")
    return buf
