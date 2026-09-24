# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``run_isolated`` must survive a live OpenMP pool in the parent, whichever runtime holds it.

``fork()`` copies only the calling thread, so a child entering a parallel region waits forever on the parent's
pool threads. libgomp has no ``pthread_atfork`` handler and hangs this way; libomp recovers. These tests start a
pool in the parent on purpose and check that the child still runs.
"""

import ctypes
import inspect
import os
import select
import shutil
import subprocess

import numpy as np
import pytest

from nestforge.build.toolchain import OpenMPRuntime, lib_linkable
from nestforge.build.isolation import (
    ERROR_CHARS,
    OMP_PAUSE_MODES,
    OMP_PAUSE_SOFT,
    OMP_RUNTIME_SONAMES,
    pause_openmp_pools,
    run_isolated,
)

OMP_SRC = """#include <omp.h>
void kern(double *a, int n) {
  #pragma omp parallel for
  for (int i = 0; i < n; i++) a[i] += 1.0;
}
"""

N = 4096

#: Whether omp_pause_resource_all tears the pool down, measured as threads in /proc/self/task:
#:     libgomp soft 16->1   libgomp hard 16->1   libomp soft 16->16   libomp hard 16->2
#: libomp's soft pause keeps the pool and still returns 0; its fork is safe through its atfork handler instead.


def thread_count():
    """Live threads of this process; a live OpenMP pool adds one per worker."""
    return len(os.listdir("/proc/self/task"))


def build(tmp_path, runtime):
    """A kernel with an OpenMP region, linked against ``runtime`` (``gomp`` or ``omp``) with the flags the builds
    use; gcc alone does not find libomp under /usr/lib/llvm-N/lib. A missing runtime fails, never skips."""
    assert shutil.which("gcc"), "gcc is required to build the OpenMP kernel this file's regression needs"
    src = tmp_path / "k.c"
    src.write_text(OMP_SRC)
    so = tmp_path / f"k_{runtime}.so"
    extra = []
    if runtime != "gomp":  # gcc links libgomp by default
        assert lib_linkable(runtime, "gcc"), f"lib{runtime}.so is not linkable by gcc; install it (libomp-dev)"
        extra = OpenMPRuntime(name=f"lib{runtime}", soname=runtime).link_flags("gcc")
    proc = subprocess.run(
        ["gcc", "-O2", "-fPIC", "-shared", "-fopenmp", str(src), *extra, "-o", str(so)], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    needed = subprocess.run(["readelf", "-d", str(so)], capture_output=True, text=True).stdout
    assert f"[lib{runtime}.so" in needed, f"expected lib{runtime} in DT_NEEDED, got:\n{needed}"
    return so


#: Worker threads the parent's region starts; set explicitly, since under OMP_NUM_THREADS=1 there is no pool and
#: every test here would pass vacuously.
POOL_THREADS = min(4, os.cpu_count() or 1)


def call_kernel(so, n=N, threads=POOL_THREADS):
    """Enter an OpenMP parallel region, in the parent or in the child alike."""
    lib = ctypes.CDLL(str(so))
    lib.omp_set_num_threads.argtypes = [ctypes.c_int]
    lib.omp_set_num_threads.restype = None
    lib.omp_set_num_threads(threads)
    lib.kern.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    lib.kern.restype = None
    a = np.zeros(n)
    lib.kern(a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n)
    return a


@pytest.mark.parametrize("runtime", ["gomp", "omp"])
def test_forked_child_runs_openmp_after_the_parent_already_did(tmp_path, runtime):
    """Without pausing the pool, the libgomp child hangs until the 60 s timeout with nothing to read."""
    so = build(tmp_path, runtime)
    call_kernel(so)  # the parent's pool is now live
    res = run_isolated(lambda: {"total": float(call_kernel(so).sum())}, timeout=60.0)
    assert "error" not in res, f"lib{runtime}: forked child failed after the parent used OpenMP: {res}"
    assert res["total"] == float(N), f"lib{runtime}: child computed {res['total']}, expected {N}"


@pytest.mark.parametrize("runtime", ["gomp", "omp"])
@pytest.mark.parametrize("mode", sorted(OMP_PAUSE_MODES))
def test_the_parent_can_still_use_openmp_after_pausing(tmp_path, runtime, mode):
    """A paused runtime restarts its pool on the next parallel region, so pausing costs the parent nothing."""
    so = build(tmp_path, runtime)
    call_kernel(so)
    pause_openmp_pools(OMP_PAUSE_MODES[mode])
    np.testing.assert_allclose(call_kernel(so), np.ones(N))  # pool rebuilt, still correct


@pytest.mark.parametrize("runtime", ["gomp", "omp"])
@pytest.mark.parametrize("mode", sorted(OMP_PAUSE_MODES))
def test_both_teardown_modes_make_the_fork_safe(tmp_path, runtime, mode):
    """Either pause mode makes the fork safe; the default is the weaker soft mode, so it has to be shown to."""
    so = build(tmp_path, runtime)
    call_kernel(so)
    pause_openmp_pools(OMP_PAUSE_MODES[mode])

    # fork by hand: run_isolated would pause again with its own mode
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            os.write(w, b"ok" if float(call_kernel(so).sum()) == float(N) else b"bad")
        finally:
            os.close(w)
            os._exit(0)
    os.close(w)
    ready, _, _ = select.select([r], [], [], 60.0)
    got = os.read(r, 8) if ready else b""
    if not ready:
        os.kill(pid, 9)
    os.waitpid(pid, 0)
    why = "produced nothing (hung on its parallel region, then was killed)" if not got else f"computed {got.decode()}"
    assert got == b"ok", f"lib{runtime} + omp_pause_{mode}: child {why}"


def test_pausing_is_safe_when_no_openmp_runtime_is_loaded():
    """Pausing must not load a runtime the process does not hold (hence RTLD_NOLOAD)."""
    before = mapped_omp()
    pause_openmp_pools()
    pause_openmp_pools()  # idempotent
    assert mapped_omp() == before, "pausing loaded an OpenMP runtime that was not already mapped"


def mapped_omp():
    with open("/proc/self/maps") as fh:
        maps = fh.read()
    return sorted({n for n in ("libgomp", "libomp", "libiomp5") if n + ".so" in maps})


def test_the_pause_drops_the_thread_count_for_the_default_runtime(tmp_path):
    """The thread count, not a child that happened not to hang, shows the default (gomp, soft) pause tears the
    pool down; the libomp outlier cannot be isolated with both runtimes loaded, so it is only documented."""
    so = build(tmp_path, "gomp")
    call_kernel(so)
    busy = thread_count()
    assert busy > 1, f"the poisoning region brought up no worker threads (count {busy}); nothing to tear down"
    assert thread_count() == busy, "the thread count moved with no pause -- the measurement is not stable"
    pause_openmp_pools()
    assert thread_count() < busy, f"(gomp, soft): pool not torn down, thread count stayed at {busy}"


def test_the_default_pause_mode_is_soft():
    """Soft keeps threadprivate data, which hard would discard on every fork."""
    default = inspect.signature(pause_openmp_pools).parameters["mode"].default
    assert default == OMP_PAUSE_SOFT, default


def test_a_mapped_runtime_without_the_pause_symbol_is_warned_not_silent(monkeypatch):
    """A pre-OpenMP-5.0 runtime cannot be paused; staying silent would hide the condition a child deadlocks on."""

    def fake_cdll(name, mode=0):
        if name == OMP_RUNTIME_SONAMES[0]:
            return object()  # loaded, without omp_pause_resource_all
        raise OSError("not loaded in this process")

    monkeypatch.setattr(ctypes, "CDLL", fake_cdll)
    with pytest.warns(UserWarning, match="omp_pause_resource_all"):
        pause_openmp_pools()


def test_a_child_exception_survives_the_pipe_intact():
    """The translator embeds 2000 characters of stderr in its error, and the diagnosis is at the end."""
    tail = "LAST-FRAME-MARKER"
    long_message = "x" * 2500 + tail

    def boom():
        raise RuntimeError(long_message)

    res = run_isolated(boom, timeout=60.0)
    assert tail in res["error"], f"message truncated to {len(res['error'])} chars"
    assert len(long_message) <= ERROR_CHARS
