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

import json
import os
import select
import signal
import time
from typing import Callable, Dict


def run_isolated(work_fn: Callable[[], Dict], timeout: float = 900.0) -> Dict:
    """Run ``work_fn`` in a forked child and return its dict, or an ``{"error": ...}`` sentinel on
    crash / timeout / malformed output. The parent always survives.

    ``timeout`` guards against a runaway (never-terminating) kernel, so it must comfortably exceed the
    longest legitimate run: the default is generous, and large-problem jobs (e.g. the XL cross-language
    job timing 268M-element kernels over many reps) pass an even larger value explicitly."""
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        os.close(r)
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
