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
from typing import Any

#: OpenMP runtimes whose thread pool must be torn down before a fork.
OMP_RUNTIME_SONAMES = ("libgomp.so.1", "libomp.so.5", "libomp.so", "libiomp5.so")

#: ``omp_pause_resource_t`` (OpenMP 5.0); ``hard`` also frees threadprivate data.
OMP_PAUSE_SOFT = 1
OMP_PAUSE_HARD = 2

OMP_PAUSE_MODES = {"soft": OMP_PAUSE_SOFT, "hard": OMP_PAUSE_HARD}

#: Longest error text a child reports back.
ERROR_CHARS = 4000


def pause_openmp_pools(mode: int = OMP_PAUSE_SOFT) -> None:
    """Pause every loaded OpenMP runtime before a fork; a live libgomp pool deadlocks the child."""
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
        pause.argtypes = [ctypes.c_int]
        pause.restype = ctypes.c_int
        if pause(mode) != 0:
            warnings.warn(f"{soname}: omp_pause_resource_all(mode={mode}) failed; its pool stays up across the fork")


def quiet_fatal_signals() -> None:
    """Disable an inherited faulthandler, so a child's segfault does not dump the parent's stack."""
    faulthandler.disable()


def run_spawned(target: Callable[[Any], dict], payload: Any, timeout: float = 900.0) -> dict:
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
    quiet_fatal_signals()
    try:
        result = target(payload)
    except BaseException as e:
        result = {"error": f"{type(e).__name__}: {str(e)[:ERROR_CHARS]}"}
    sender.send(result)
    sender.close()


def run_isolated(work_fn: Callable[[], dict], timeout: float = 900.0) -> dict:
    """``work_fn()`` from a forked child, or ``{"error": ...}`` on an exception, crash, timeout or bad result."""
    pause_openmp_pools()
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        quiet_fatal_signals()
        try:
            payload = json.dumps(work_fn())
        except BaseException as e:
            payload = json.dumps({"error": f"{type(e).__name__}: {str(e)[:ERROR_CHARS]}"})
        try:
            os.write(w, payload.encode())
        finally:
            os.close(w)
            os._exit(0)
    os.close(w)
    start, buf, timed_out = time.perf_counter(), b"", True
    try:
        while True:
            remaining = timeout - (time.perf_counter() - start)
            if remaining <= 0:
                break
            ready, _, _ = select.select([r], [], [], remaining)
            if not ready:
                break
            chunk = os.read(r, 65536)
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
