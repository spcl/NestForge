# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Run generated code in a child process, so a crash or a runaway loop becomes an ``{"error": ...}`` result
instead of taking down the caller."""

from __future__ import annotations

import faulthandler
import ctypes
import json
import multiprocessing
import os
import select
import signal
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

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


def error_result(e: BaseException) -> dict:
    """A child's exception as a result, its message cut to :data:`ERROR_CHARS`."""
    return {"error": f"{type(e).__name__}: {str(e)[:ERROR_CHARS]}"}


def run_spawned(target: Callable[[Any], dict], payload: Any, timeout: float = RUN_TIMEOUT_S) -> dict:
    """:func:`run_isolated` in a freshly spawned interpreter: a CUDA context does not survive a fork. ``target``
    and ``payload`` must pickle."""
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    child = context.Process(target=spawned_entry, args=(target, payload, sender))
    child.start()
    sender.close()
    try:
        return spawned_result(receiver, child, timeout)
    finally:
        receiver.close()
        drop_openmp_registration(child.pid)


def spawned_result(receiver: Any, child: Any, timeout: float) -> dict:
    """The spawned child's dict, or an error when it runs past ``timeout`` or dies before sending one."""
    if not receiver.poll(timeout):
        child.kill()
        child.join()
        return {"error": f"timeout after {timeout:.0f}s (runaway kernel)"}
    try:
        result = receiver.recv()
    except EOFError:  # the child exited, or was killed by a signal, without sending
        child.join()
        return {"error": f"crashed (exit code {child.exitcode})"}
    child.join()
    return result


def spawned_entry(target: Callable[[Any], dict], payload: Any, sender: Any) -> None:
    """The spawned child's body; a Python exception comes back as an error."""
    faulthandler.disable()  # a child's segfault must not dump the parent's stack
    try:
        result = target(payload)
    except BaseException as e:
        result = error_result(e)
    sender.send(result)
    sender.close()


def run_isolated(work_fn: Callable[[], dict], timeout: float = RUN_TIMEOUT_S) -> dict:
    """``work_fn()`` from a forked child, or ``{"error": ...}`` on an exception, crash, timeout or bad result."""
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
            payload = json.dumps(work_fn())
        except BaseException as e:
            payload = json.dumps(error_result(e))
        try:
            os.write(w, payload.encode())
        finally:
            os.close(w)
            os._exit(0)
    os.close(w)
    try:
        return forked_result(pid, r, timeout)
    finally:
        drop_openmp_registration(pid)


def forked_result(pid: int, r: int, timeout: float) -> dict:
    """The forked child's dict read from pipe ``r``, or an error when it crashes, hangs past ``timeout`` or writes
    no valid result; the child is reaped either way."""
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
            return {"error": f"timeout after {timeout:.0f}s (runaway kernel)"}
    else:
        _, status = os.waitpid(pid, 0)  # the child is exiting, so this returns
    if os.WIFSIGNALED(status):
        return {"error": f"crashed (signal {os.WTERMSIG(status)})"}
    try:
        return json.loads(buf) if buf else {"error": "child produced no result"}
    except json.JSONDecodeError:
        return {"error": "child produced malformed result (crashed mid-write)"}
