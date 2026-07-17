"""Run a compiled kernel in a forked child, so a **segfault** or a **runaway loop** in freshly-compiled
code cannot take down the parent (a pytest run or a sweep rank).

Every place that compiles arbitrary code and then calls it via ctypes -- the arena, the TSVC driver,
any test that builds-and-runs -- routes the call through :func:`run_isolated`. ``os.fork`` shares the
parent's memory copy-on-write, so no numpy array or ctypes type is pickled; the child ``dlopen``s the
``.so``, and when it ``_exit``s that mapping is released, so the parent never accumulates library
handles. ``work_fn`` must return a small JSON-able dict (the summary: ``ok``/``maxdiff``/``time_us``);
a crash, timeout, or malformed result comes back as an ``{"error": ...}`` sentinel instead.
"""
from __future__ import annotations

import ctypes
import json
import os
import select
import signal
import time
from typing import Callable, Dict

#: OpenMP runtimes whose thread pool must be torn down before a fork (see :func:`pause_openmp_pools`).
#: Probed by the sonames a linked node library actually records in DT_NEEDED.
OMP_RUNTIME_SONAMES = ("libgomp.so.1", "libomp.so.5", "libomp.so", "libiomp5.so", "libnvomp.so")

#: ``omp_pause_resource_t`` (OpenMP 5.0). Both tear the thread pool down, which is what makes the fork
#: safe; they differ in what ELSE is freed:
#:   * ``soft`` -- the pool goes, the runtime may keep other resources, and threadprivate data SURVIVES.
#:   * ``hard`` -- everything is freed, including threadprivate data (which the spec says is then lost).
#: Both were measured to make a forked child's parallel region run where it otherwise hung. ``soft`` is
#: the default: it is the weaker claim that still buys fork safety, so it cannot destroy state a caller
#: expected to keep. ``hard`` is available for a caller that wants the runtime fully reset.
OMP_PAUSE_SOFT = 1
OMP_PAUSE_HARD = 2

#: name -> ``omp_pause_resource_t`` value, for a config/CLI knob.
OMP_PAUSE_MODES = {"soft": OMP_PAUSE_SOFT, "hard": OMP_PAUSE_HARD}


def pause_openmp_pools(mode: int = OMP_PAUSE_SOFT) -> None:
    """Tear down the thread pool of every OpenMP runtime ALREADY loaded here, so the coming fork is safe.

    ``fork()`` duplicates only the calling thread, so a child that enters a parallel region while the
    parent's pool is live blocks forever waiting on pool threads that no longer exist -- libgomp
    deadlocks exactly this way, and installs no ``pthread_atfork`` handler to recover (LLVM's libomp
    does, which is the only reason the default runtime survives this at all). Since a node library may
    legitimately link either -- the runtime is a configurable axis, and libgomp is a valid gnu-only
    choice -- ``run_isolated`` must not depend on which one happens to be loaded.

    ``omp_pause_resource_all`` is the OpenMP 5.0 API for precisely this: it destroys the pool, leaving
    the runtime free to re-initialise on the next parallel region -- in the parent or in the child.
    Measured: with a live libgomp pool the forked child HUNG; after this call it ran, under BOTH
    ``mode`` values (see :data:`OMP_PAUSE_MODES`), and the parent's own next region simply spins the
    pool back up. The pool is a cache, so tearing it down costs a re-spin, not correctness.

    Best effort by construction:
      * ``RTLD_NOLOAD`` -- only ever ask a runtime that is ALREADY mapped. Plain ``CDLL`` would LOAD it,
        which would both add a runtime this process never needed and defeat the purpose.
      * a runtime without the OMP_5.0 symbol (or that refuses -- the call is invalid from inside a
        parallel region, and returns non-zero) is skipped: pausing is a hardening step, and a failure
        here must not break a fork that would otherwise have been fine.
    """
    for soname in OMP_RUNTIME_SONAMES:
        try:
            lib = ctypes.CDLL(soname, mode=os.RTLD_NOLOAD)
        except OSError:
            continue  # not loaded in this process: nothing to pause
        try:
            pause = lib.omp_pause_resource_all
        except AttributeError:
            continue  # pre-OpenMP-5.0 runtime: no way to ask
        pause.argtypes = [ctypes.c_int]
        pause.restype = ctypes.c_int
        try:
            pause(mode)
        except OSError:
            pass


def quiet_fatal_signals() -> None:
    """In the forked child, drop the inherited faulthandler. pytest installs faulthandler, which the child
    inherits across ``os.fork``; when freshly-compiled code then segfaults, that dumps a Python traceback
    (misleadingly showing the PARENT's call stack up to the fork) into the captured output. The parent
    already reports the crash via the child's exit signal, so let the child die quietly instead."""
    try:
        import faulthandler
        faulthandler.disable()
    except Exception:
        pass


def run_isolated(work_fn: Callable[[], Dict], timeout: float = 900.0) -> Dict:
    """Run ``work_fn`` in a forked child and return its dict, or an ``{"error": ...}`` sentinel on
    crash / timeout / malformed output. The parent always survives.

    ``timeout`` guards against a runaway (never-terminating) kernel, so it must comfortably exceed the
    longest legitimate run: the default is generous, and large-problem jobs (e.g. the XL cross-language
    job timing 268M-element kernels over many reps) pass an even larger value explicitly."""
    # An OpenMP pool that is live across the fork deadlocks the child on its first parallel region --
    # which is every kernel built from a parallel nest. Tear the pools down first (see the function).
    pause_openmp_pools()
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        os.close(r)
        quiet_fatal_signals()
        try:
            payload = json.dumps(work_fn())
        except BaseException as e:  # any Python-level failure comes back as an error (a segfault does not)
            payload = json.dumps({"error": f"{type(e).__name__}: {str(e)[:200]}"})
        try:
            os.write(w, payload.encode())
        finally:
            os.close(w)
            os._exit(0)
    # parent: read the result with a wall-clock deadline, then reap (killing a hung child)
    os.close(w)
    start, buf, timed_out = time.perf_counter(), b"", True
    try:
        while True:
            remaining = timeout - (time.perf_counter() - start)
            if remaining <= 0:
                break  # deadline hit -> timed_out stays True
            ready, _, _ = select.select([r], [], [], remaining)
            if not ready:
                break  # deadline hit
            chunk = os.read(r, 65536)
            if not chunk:  # EOF: the child closed the pipe (finished writing, or died)
                timed_out = False
                break
            buf += chunk
    finally:
        os.close(r)
    if timed_out:
        reaped, status = os.waitpid(pid, os.WNOHANG)
        if reaped == 0:  # genuinely still running -> runaway; kill and reap
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            return {"error": f"timeout after {timeout:.0f}s (runaway kernel)"}
    else:
        _, status = os.waitpid(pid, 0)  # EOF seen: the child is exiting -> a blocking reap is safe
    if os.WIFSIGNALED(status):  # a segfault etc. never reached the os.write, so buf is empty
        return {"error": f"crashed (signal {os.WTERMSIG(status)})"}
    try:
        return json.loads(buf) if buf else {"error": "child produced no result"}
    except json.JSONDecodeError:
        return {"error": "child produced malformed result (crashed mid-write)"}
