# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``run_isolated`` turns a crash, a hang or a malformed result of fresh generated code into a result the sweep can
record, and the parent survives each one. The assertions read the returned dict, since a dying child takes its
coverage with it.
"""

import ctypes
import os
import signal
import threading
import time
import warnings

import pytest
from pathlib import Path

from nestforge.build.isolation import ERROR_CHARS, OPENMP_REGISTRATION, run_isolated


#: Seconds the background thread of the warning test stays alive; it only needs to outlive one fork.
ISOLATION_WAIT_S = 5.0

#: Isolated runs of the registration-leak test; one leak per run would show as that many files.
ISOLATED_RUNS = 3


def test_a_result_comes_back_from_the_child_unchanged():
    assert run_isolated(lambda: {"value": 41 + 1, "name": "ok"}, timeout=60) == {"value": 42, "name": "ok"}


def test_an_exception_in_the_child_returns_an_error_instead_of_killing_the_sweep():
    """A kernel that raises during marshalling must not take the parent with it: the sweep records
    the reason and moves to the next cell."""

    def boom():
        raise ValueError("bad argument shape")

    result = run_isolated(boom, timeout=60)
    assert result["error"].startswith("ValueError: bad argument shape")


def test_the_error_text_is_truncated_so_one_bad_kernel_cannot_flood_the_report():
    """A C++ template error pasted into an exception can run to megabytes; the record is a table
    cell, not a log."""

    def verbose():
        raise RuntimeError("x" * (ERROR_CHARS * 3))

    result = run_isolated(verbose, timeout=60)
    assert len(result["error"]) <= ERROR_CHARS + len("RuntimeError: ")


def test_a_segfault_is_reported_as_a_signal_and_the_parent_survives():
    """the case the fork exists for: freshly-compiled code faults. The signal number is kept because
    it distinguishes a genuine memory fault from a kernel the runtime aborted."""

    def crash():
        os.kill(os.getpid(), signal.SIGSEGV)
        return {"unreachable": True}

    result = run_isolated(crash, timeout=60)
    assert result["error"] == f"crashed (signal {int(signal.SIGSEGV)})"
    assert run_isolated(lambda: {"next": True}, timeout=60) == {"next": True}  # the sweep moves on


def test_a_child_that_exits_without_writing_is_an_error_not_an_empty_success():
    """An empty pipe must not decode as "no differences found" -- an absent result is a failure."""

    def vanish():
        os._exit(0)

    assert run_isolated(vanish, timeout=60) == {"error": "child produced no result"}


def test_a_runaway_child_is_killed_and_reported_rather_than_hanging_the_sweep():
    """The timeout guards an infinite loop in generated code. It must also reap: a sweep that leaks a
    zombie per runaway kernel exhausts the process table long before it finishes."""

    def runaway():
        time.sleep(120)
        return {}

    start = time.perf_counter()
    result = run_isolated(runaway, timeout=1.0)
    elapsed = time.perf_counter() - start
    assert "timeout after" in result["error"]
    assert elapsed < 30, "the deadline did not fire; it waited for the child instead"
    with pytest.raises(ChildProcessError):
        os.waitpid(-1, os.WNOHANG)  # nothing left unreaped


def test_a_child_returning_something_unserialisable_reports_the_reason():
    """``work_fn`` promises a JSON-able dict. Breaking that promise is the author's bug, and it comes
    back named rather than as a silent empty result."""
    result = run_isolated(lambda: {"array": object()}, timeout=60)
    assert "error" in result and "TypeError" in result["error"]


def test_isolation_warns_nothing_when_openmp_is_loaded():
    """A loaded, initialized libomp is paused once and the fork is deliberate, so neither warns."""
    libomp = ctypes.CDLL("libomp.so.5")  # RTLD_LOCAL: later tests must not bind GOMP_* to it
    libomp.omp_get_max_threads()  # initialize the runtime
    worker = threading.Thread(target=threading.Event().wait, args=(ISOLATION_WAIT_S,), daemon=True)
    worker.start()  # a live thread is what makes Python warn about fork
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert run_isolated(lambda: {"ok": 1}) == {"ok": 1}


def test_a_child_using_libomp_leaves_no_registration_behind():
    """libomp registers every process in shared memory and unregisters only at a normal exit; the isolated child
    leaves through ``os._exit``, so the parent removes what it left."""
    pids: list[int] = []

    def work() -> dict:
        ctypes.CDLL("libomp.so.5").omp_get_max_threads()  # initializes libomp, which registers this process
        return {"pid": os.getpid()}

    for _ in range(ISOLATED_RUNS):
        pids.append(run_isolated(work)["pid"])
    left = [pid for pid in pids if Path(OPENMP_REGISTRATION.format(pid=pid, uid=os.getuid())).exists()]
    assert not left, f"children {left} left their libomp registration in shared memory"
