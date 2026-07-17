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
import inspect
import os
import select
import shutil
import subprocess

import numpy as np
import pytest

from nestforge.perf import flags
from nestforge.isolation import OMP_PAUSE_MODES, OMP_PAUSE_SOFT, pause_openmp_pools, run_isolated

OMP_SRC = """#include <omp.h>
void kern(double *a, int n) {
  #pragma omp parallel for
  for (int i = 0; i < n; i++) a[i] += 1.0;
}
"""

N = 4096

#: (runtime, mode) -> does omp_pause_resource_all ACTUALLY tear the thread pool down? MEASURED here, not
#: assumed, by counting threads in /proc/self/task across the call:
#:     libgomp soft 16->1   libgomp hard 16->1   libomp soft 16->16   libomp hard 16->2
#: libomp's SOFT pause is the outlier: it leaves the pool fully up and still returns 0 ("success"), so the
#: return code cannot detect it -- only the thread count can. That cell's fork is safe anyway, but for a
#: different reason (libomp's pthread_atfork handler rebuilds the child's pool), which is precisely why
#: "the child didn't hang" is too weak an observable to pin tear-down with. The default -- libgomp + soft
#: -- is the one that must genuinely tear down, and does.
TEARS_DOWN_POOL = {("gomp", "soft"): True, ("gomp", "hard"): True, ("omp", "soft"): False, ("omp", "hard"): True}


def thread_count():
    """Live threads in THIS process, straight from procfs -- no ``ps`` shell-out. A live OpenMP pool shows
    up here as one thread per core beyond the baseline, so tear-down is directly observable rather than
    inferred from a fork that happened not to hang."""
    return len(os.listdir("/proc/self/task"))


# gcc and both OpenMP runtimes are hard requirements of this file, not optional extras: gcc is the repo's
# default compiler and libgomp is what it links by default, so "no gcc" or "no libgomp" is a broken
# environment to surface, never a skip to hide behind -- a skip here would silently retire THE regression
# this file exists to pin, and would fail CI anyway (the unit set runs under NESTFORGE_CI_NO_SKIP=1).
def build(tmp_path, runtime):
    """A kernel with an OpenMP region, linked against ``runtime`` ("gomp" or "omp").

    libomp is located with the SAME helper the product uses (:func:`~nestforge.perf.flags.runtime_dir`)
    rather than a bare ``gcc -print-file-name``. That is not a nicety: on the CI runner gcc does not know
    where libomp lives (libomp-18-dev puts it under /usr/lib/llvm-18/lib, off gcc's path -- the exact
    split runtime_dir exists to bridge), so the bare probe returns "libomp.so" and a hand-rolled check
    either wrongly skips or wrongly fails. If runtime_dir cannot find it either, the runtime genuinely is
    not installed and the test skips -- an absent toolchain is a skip, never a red test."""
    assert shutil.which("gcc"), "gcc is required to build the OpenMP kernel this file's regression needs"
    src = tmp_path / "k.c"
    src.write_text(OMP_SRC)
    so = tmp_path / f"k_{runtime}.so"
    extra = []
    if runtime != "gomp":  # gcc's default IS libgomp; anything else must be pinned at link
        lib_dir = flags.runtime_dir(runtime, "gcc")
        if lib_dir is None and not flags.lib_linkable(runtime, "gcc"):
            pytest.skip(f"lib{runtime}.so is not installed / linkable by gcc here")
        search = [f"-L{lib_dir}", f"-Wl,-rpath,{lib_dir}"] if lib_dir else []
        extra = [*search, f"-Wl,--push-state,--no-as-needed,-l{runtime},--pop-state"]
    proc = subprocess.run(
        ["gcc", "-O2", "-fPIC", "-shared", "-fopenmp", *extra,
         str(src), "-o", str(so)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-800:]
    needed = subprocess.run(["readelf", "-d", str(so)], capture_output=True, text=True).stdout
    # DT_NEEDED may show libomp.so.5 for a libiomp5 request (ABI-compat symlink); accept the resolved one.
    assert any(f"[lib{r}.so" in needed for r in (runtime, "omp")), f"expected lib{runtime} in DT_NEEDED, got:\n{needed}"
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
