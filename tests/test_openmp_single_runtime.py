# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Every kernel library and the program link one OpenMP runtime, whatever compiler built them.

A bare ``-fopenmp`` links each family's default (gcc libgomp, clang libomp, icx libiomp5), and a program mixing them
runs two thread pools. The flags of each family look right in isolation, so these tests read DT_NEEDED of real
libraries. Most only link; the one that runs both kernels does so in a fresh interpreter.
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

import nestforge
from nestforge.build.toolchain import (
    LIBOMP,
    lib_linkable,
    openmp_compile_flags,
    openmp_link_flags,
    runtime_library,
)

#: A minimal nest with an OpenMP region: enough to make the compiler link a runtime.
OMP_SRC = """#include <omp.h>
void kern(double *a, int n) {
  #pragma omp parallel for
  for (int i = 0; i < n; i++) a[i] += 1.0;
}
"""

#: A second, differently-shaped nest (a reduction lowers to ``parallel for reduction``), so a single kernel
#: cannot link one runtime by luck of its shape.
OMP_SRC_REDUCE = """#include <omp.h>
double kern2(const double *a, int n) {
  double s = 0.0;
  #pragma omp parallel for reduction(+:s)
  for (int i = 0; i < n; i++) s += a[i] * 2.0;
  return s;
}
"""

#: A g++ kernel reaching the OpenMP entries GCC lowers to distinct GOMP_* calls: simd reduction, collapsed
#: dynamic schedule, max reduction, single, task and taskwait.
OMP_SRC_ENTRIES = """#include <omp.h>
double kern3(double *a, const double *b, int n, int m) {
  double s = 0.0, mx = 0.0;
  #pragma omp parallel for simd reduction(+:s)
  for (int i = 0; i < n; i++) s += a[i] * b[i];
  #pragma omp parallel for collapse(2) schedule(dynamic)
  for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++) a[i] += b[j];
  #pragma omp parallel for reduction(max:mx)
  for (int i = 0; i < n; i++) mx = a[i] > mx ? a[i] : mx;
  #pragma omp parallel
  {
    #pragma omp single
    {
      #pragma omp task
      { a[0] += omp_get_thread_num(); }
      #pragma omp taskwait
    }
  }
  return s + mx;
}
"""

#: The OpenMP runtimes a linked object can name, by DT_NEEDED soname stem.
OMP_SONAMES = ("libgomp", "libomp", "libiomp5")

#: The C compilers of the families nest-forge sweeps.
COMPILERS = ("gcc", "clang", "icx")


def linked_openmp_runtimes(so):
    """The OpenMP runtimes in ``so``'s DT_NEEDED, as soname stems; libiomp5 may resolve to libomp."""
    out = subprocess.run(["readelf", "-d", str(so)], capture_output=True, text=True, check=True).stdout
    return {name for name in OMP_SONAMES if f"[{name}.so" in out}


#: The runtime call a compiler emits to open a parallel region: LLVM/Intel ``__kmpc_fork_call``, GNU
#: ``GOMP_parallel``. Its presence is the only proof the region survived compilation.
OMP_FORK_SYMBOLS = ("kmpc_fork", "GOMP_parallel")


def emits_parallel_region(so):
    """Whether ``so`` calls into an OpenMP runtime: ``clang -fopenmp=libgomp`` links libgomp yet emits only
    ``__kmpc_*`` calls, which libgomp lacks, and runs serially with correct results."""
    out = subprocess.run(["nm", "-u", str(so)], capture_output=True, text=True, check=True).stdout
    return any(sym in out for sym in OMP_FORK_SYMBOLS)


def available_compilers():
    """The C compilers present here. Never empty: the CI runner has gcc and clang."""
    return [cc for cc in COMPILERS if shutil.which(cc)]


def build_cell(tmp_path, compiler, src=OMP_SRC, tag="k"):
    """Link one nest the way the owned build does: the OpenMP compile flags before the source, the libomp link
    flags after the object."""
    assert lib_linkable(LIBOMP, compiler), f"libomp is not linkable by {compiler} (setup_apt.sh installs libomp-dev)"
    csrc = tmp_path / f"{tag}.c"
    csrc.write_text(src)
    so = tmp_path / f"{tag}_{compiler}.so"
    cmd = [compiler, "-O2", "-fPIC", "-shared", *openmp_compile_flags(compiler), str(csrc)]
    proc = subprocess.run([*cmd, *openmp_link_flags(compiler), "-o", str(so)], capture_output=True, text=True)
    assert proc.returncode == 0, f"{compiler} failed to link libomp:\n{proc.stderr[-1500:]}"
    # The pragma is in the source, so a cell that links a runtime but opens no region was silently serialized.
    assert emits_parallel_region(so), (
        f"{compiler}: the cell links {sorted(linked_openmp_runtimes(so))} but emits NO OpenMP fork call"
    )
    return so


def test_every_compiler_links_the_same_single_runtime(tmp_path):
    """Across every available compiler, a cell links exactly one OpenMP runtime, the same for all: a bare -fopenmp
    links libgomp under gcc and libomp under clang. The global runtime is libomp, which the owned build resolves
    for gcc, clang and icx."""
    seen = {}
    for cc in available_compilers():
        so = build_cell(tmp_path, cc, tag="a")
        rts = linked_openmp_runtimes(so)
        assert len(rts) == 1, f"{cc} linked {len(rts)} OpenMP runtimes ({sorted(rts)}), must be exactly 1"
        seen[cc] = rts
    assert seen, "no compiler could link a runtime -- the matrix would be vacuous"
    distinct = set().union(*seen.values())
    assert distinct == {"libomp"}, f"the single runtime must be libomp for every compiler: {seen}"


def test_two_different_nests_from_two_compilers_share_one_runtime(tmp_path):
    """Two unrelated nests (elementwise map + reduction), each built by a different compiler, as they would be
    when linked into one program. The union over the pair must still be one runtime."""
    compilers = available_compilers()
    assert len(compilers) >= 2, f"needs two compiler families, found {compilers} (setup_apt.sh installs gcc+clang)"
    union, built = set(), {}
    for cc, src, tag in ((compilers[0], OMP_SRC, "map"), (compilers[1], OMP_SRC_REDUCE, "red")):
        so = build_cell(tmp_path, cc, src=src, tag=tag)
        built[f"{cc}:{tag}"] = sorted(linked_openmp_runtimes(so))
        union |= linked_openmp_runtimes(so)
    assert len(union) == 1, f"two node libraries, two compilers, {len(union)} runtimes: {built}"


#: Loads both node libraries into one process, runs both nests, and reports what got mapped. Run via EXEC,
#: not fork: a forked child inherits runtimes other tests deliberately loaded.
RUN_BOTH_SRC = """
import ctypes, json, sys
import numpy as np
so_a, so_b, n = sys.argv[1], sys.argv[2], int(sys.argv[3])

def mapped():
    with open("/proc/self/maps") as fh:
        maps = fh.read()
    return sorted({x for x in ("libgomp", "libomp", "libiomp5") if x + ".so" in maps})

before = mapped()
a = np.arange(n, dtype=np.float64)
lib_a, lib_b = ctypes.CDLL(so_a), ctypes.CDLL(so_b)
lib_a.kern.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
lib_a.kern.restype = None
lib_b.kern2.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
lib_b.kern2.restype = ctypes.c_double
p = a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
lib_a.kern(p, n)                 # gcc's nest:   a[i] += 1.0, in place
total = lib_b.kern2(p, n)        # clang's nest: sum(a[i] * 2.0), over gcc's output
print(json.dumps({"a": a.tolist(), "total": float(total), "runtimes": mapped(), "before": before}))
"""


def run_both_in_a_clean_process(tmp_path, so_a, so_b, n):
    """Run both nests in a fresh interpreter and return its report (see :data:`RUN_BOTH_SRC`)."""
    script = tmp_path / "run_both.py"
    script.write_text(RUN_BOTH_SRC)
    proc = subprocess.run(
        [sys.executable, str(script), str(so_a), str(so_b), str(n)], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, f"running both nests together failed:\n{proc.stderr[-1500:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_two_compilers_nests_run_together_on_one_runtime_and_match_numpy(tmp_path):
    """One nest built by gcc and a different nest built by clang, loaded into one process and run -- sharing
    a single OpenMP runtime, and computing the right answer."""
    assert shutil.which("gcc") and shutil.which("clang"), (
        f"needs gcc AND clang, found {available_compilers()} (setup_apt.sh installs both)"
    )
    built = {}
    for cc, src, tag in (("gcc", OMP_SRC, "kern"), ("clang", OMP_SRC_REDUCE, "kern2")):
        so = build_cell(tmp_path, cc, src=src, tag=f"e2e_{tag}")
        built[cc] = so
        assert linked_openmp_runtimes(so) == linked_openmp_runtimes(built["gcc"]), "cells disagree on the runtime"

    n = 512
    res = run_both_in_a_clean_process(tmp_path, built["gcc"], built["clang"], n)

    assert not res["before"], f"the fresh interpreter already had an OpenMP runtime mapped: {res['before']}"
    assert len(res["runtimes"]) == 1, f"two compilers' node libraries loaded {res['runtimes']} into one process"
    expect_a = np.arange(n, dtype=np.float64) + 1.0
    np.testing.assert_allclose(np.array(res["a"]), expect_a, rtol=0, atol=0)
    np.testing.assert_allclose(res["total"], float(np.sum(expect_a * 2.0)), rtol=1e-12)


def test_a_kmpc_compiler_on_libgomp_would_be_caught_not_silently_serialized(tmp_path):
    """The trap itself: clang emitting kmpc, linked against gomp-only libgomp, and emits_parallel_region
    sees the serialization. If this ever emits a fork call, libgomp gained a kmpc layer and the prune can
    be revisited -- deliberately."""
    assert shutil.which("clang"), "no clang on PATH (setup_apt.sh installs it)"
    csrc = tmp_path / "mismatch.c"
    csrc.write_text(OMP_SRC)
    so = tmp_path / "mismatch.so"
    proc = subprocess.run(
        ["clang", "-O2", "-fPIC", "-shared", "-fopenmp=libgomp", str(csrc), "-o", str(so)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"clang -fopenmp=libgomp failed to link:\n{proc.stderr[-1500:]}"
    assert "libgomp" in linked_openmp_runtimes(so), "expected the mismatch to link libgomp"
    assert not emits_parallel_region(so), (
        "clang -fopenmp=libgomp emitted a fork call -- libgomp now has a kmpc "
        "layer, so the single-runtime prune for kmpc families can be revisited"
    )


def symbol_names(command):
    """The symbol names ``nm`` prints, without a symbol version suffix."""
    out = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    return {line.split()[-1].split("@")[0] for line in out.splitlines() if line.strip()}


def test_every_openmp_entry_a_gxx_object_calls_is_exported_by_libomp(tmp_path):
    """libomp serves g++ code through its GOMP compatibility layer. An entry a newer GCC emits that libomp lacks
    would fail at load time, so every OpenMP symbol the object leaves undefined must be a libomp export."""
    assert shutil.which("g++"), "no g++ on PATH (setup_apt.sh installs it)"
    source = tmp_path / "entries.c"
    source.write_text(OMP_SRC_ENTRIES)
    obj = tmp_path / "entries.o"
    subprocess.run(["g++", "-x", "c++", "-O2", "-fopenmp", "-c", str(source), "-o", str(obj)], check=True)
    libomp = runtime_library(LIBOMP, "g++")
    assert libomp is not None, "no libomp found for g++ (setup_apt.sh installs libomp-dev)"

    called = {name for name in symbol_names(["nm", "-u", str(obj)]) if name.startswith(("GOMP_", "omp_"))}
    exported = symbol_names(["nm", "-D", "--defined-only", str(libomp)])

    assert len(called) >= 5, f"the object calls too few OpenMP entries to cover GCC's lowering: {sorted(called)}"
    assert sorted(called - exported) == [], (
        f"libomp does not export these entries g++ calls: {sorted(called - exported)}"
    )


def test_every_link_search_path_is_paired_with_an_rpath():
    """A ``-L`` without its ``-Wl,-rpath`` links, but the library then fails to load without LD_LIBRARY_PATH."""
    package = Path(nestforge.__file__).parent
    offenders = []
    for path in sorted(package.rglob("*.py")):
        lines = path.read_text().splitlines()
        for num, line in enumerate(lines):
            if not re.search(r"""["']-L""", line):
                continue
            window = " ".join(lines[num : num + 2])  # the flag list may wrap onto the next line
            if "rpath" not in window:
                offenders.append(f"{path.name}:{num + 1}: {line.strip()}")
    assert not offenders, "a -L without a paired -Wl,-rpath:\n" + "\n".join(offenders)
