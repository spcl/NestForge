"""``run_isolated`` must survive a live OpenMP pool in the parent -- whichever runtime holds it.

``fork()`` duplicates only the calling thread. A child that enters a parallel region while the parent's
pool is live blocks forever on pool threads that no longer exist, and the failure has no symptom beyond
a wall-clock timeout: no crash, no compiler running, the parent simply sits in ``select()``.

libgomp deadlocks exactly this way and installs no ``pthread_atfork`` handler; LLVM's libomp installs
one and recovers. Since the OpenMP runtime is a configurable axis -- libgomp is a valid gnu-only choice
-- ``run_isolated`` must not be safe merely because the default runtime happens to be the forgiving one.
:func:`pause_openmp_pools` is what makes that true, so these tests poison the parent on purpose and
assert the child still runs.
"""
import ctypes
import shutil
import subprocess

import numpy as np
import pytest

from nestforge.isolation import OMP_PAUSE_MODES, pause_openmp_pools, run_isolated

OMP_SRC = """#include <omp.h>
void kern(double *a, int n) {
  #pragma omp parallel for
  for (int i = 0; i < n; i++) a[i] += 1.0;
}
"""

N = 4096


def build(tmp_path, runtime):
    """A kernel with an OpenMP region, linked against ``runtime`` ("gomp" or "omp")."""
    if not shutil.which("gcc"):
        pytest.skip("no gcc")
    src = tmp_path / "k.c"
    src.write_text(OMP_SRC)
    so = tmp_path / f"k_{runtime}.so"
    extra = []
    if runtime != "gomp":  # gcc's default IS libgomp; anything else must be pinned at link
        libdir = subprocess.run(["gcc", f"-print-file-name=lib{runtime}.so"], capture_output=True,
                                text=True).stdout.strip()
        if not libdir.startswith("/"):
            pytest.skip(f"lib{runtime}.so not linkable by gcc here")
        extra = [f"-L{libdir.rsplit('/', 1)[0]}", f"-Wl,--push-state,--no-as-needed,-l{runtime},--pop-state"]
    proc = subprocess.run(["gcc", "-O2", "-fPIC", "-shared", "-fopenmp", *extra, str(src), "-o", str(so)],
                          capture_output=True,
                          text=True)
    assert proc.returncode == 0, proc.stderr[-800:]
    needed = subprocess.run(["readelf", "-d", str(so)], capture_output=True, text=True).stdout
    assert f"[lib{runtime}.so" in needed, f"expected lib{runtime} in DT_NEEDED, got:\n{needed}"
    return so


def call_kernel(so, n=N):
    """Enter an OpenMP parallel region. Used for BOTH the parent's poisoning and the child's work, so
    the two differ only in which side of the fork they run on."""
    lib = ctypes.CDLL(str(so))
    lib.kern.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    lib.kern.restype = None
    a = np.zeros(n)
    lib.kern(a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n)
    return a


@pytest.mark.parametrize("runtime", ["gomp", "omp"])
def test_forked_child_runs_openmp_after_the_parent_already_did(tmp_path, runtime):
    """THE regression: parent enters a parallel region, THEN forks a child that enters one.

    Without pause_openmp_pools the libgomp case hangs until the caller's timeout -- which is what took
    out two s000 cells in CI for 900s apiece, with no compiler running and no error to read. A short
    timeout here means a regression reports in seconds rather than wedging the suite.
    """
    so = build(tmp_path, runtime)
    call_kernel(so)  # poison: the parent's pool is now live
    res = run_isolated(lambda: {"total": float(call_kernel(so).sum())}, timeout=60.0)
    assert "error" not in res, f"lib{runtime}: forked child failed after the parent used OpenMP: {res}"
    assert res["total"] == float(N), f"lib{runtime}: child computed {res['total']}, expected {N}"


@pytest.mark.parametrize("runtime", ["gomp", "omp"])
@pytest.mark.parametrize("mode", sorted(OMP_PAUSE_MODES))
def test_the_parent_can_still_use_openmp_after_pausing(tmp_path, runtime, mode):
    """Pausing must not cost the parent anything, under EITHER tear-down mode: a paused runtime
    re-initialises on its next parallel region. Otherwise run_isolated would fix the fork by breaking
    every caller that later runs a kernel itself -- the pool is a cache, and tearing it down is not the
    same as disabling OpenMP."""
    so = build(tmp_path, runtime)
    call_kernel(so)
    pause_openmp_pools(OMP_PAUSE_MODES[mode])
    np.testing.assert_allclose(call_kernel(so), np.ones(N))  # pool rebuilt, still correct


@pytest.mark.parametrize("runtime", ["gomp", "omp"])
@pytest.mark.parametrize("mode", sorted(OMP_PAUSE_MODES))
def test_both_teardown_modes_make_the_fork_safe(tmp_path, runtime, mode):
    """BOTH omp_pause_resource_t options must buy fork safety -- the pool goes either way; they differ
    only in what else is freed (hard also discards threadprivate data, soft keeps it).

    Tested explicitly because the default is the WEAKER one: soft is chosen so pausing cannot destroy
    state a caller expected to keep, which is only defensible if soft genuinely tears the pool down.
    An unmeasured assumption there would silently return every libgomp fork to a 900s hang.
    """
    so = build(tmp_path, runtime)
    call_kernel(so)  # poison the parent
    pause_openmp_pools(OMP_PAUSE_MODES[mode])

    # fork by hand: run_isolated pauses internally, which would mask whether THIS mode did the work.
    import os
    import select
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
    assert got == b"ok", (f"lib{runtime} + omp_pause_{mode}: child "
                          f"{'HUNG (pool survived the pause)' if not got else 'computed ' + got.decode()}")


def test_pausing_is_safe_when_no_openmp_runtime_is_loaded():
    """The common case: most calls fork from a process holding no OpenMP runtime at all. Pausing must be
    a silent no-op there and, above all, must not LOAD a runtime to ask it (RTLD_NOLOAD) -- that would
    add a runtime the process never needed, on the very path meant to keep the fork clean."""
    before = mapped_omp()
    pause_openmp_pools()
    pause_openmp_pools()  # idempotent
    assert mapped_omp() == before, "pausing loaded an OpenMP runtime that was not already mapped"


def mapped_omp():
    with open("/proc/self/maps") as fh:
        maps = fh.read()
    return sorted({n for n in ("libgomp", "libomp", "libiomp5", "libnvomp") if n + ".so" in maps})
