"""The cross-compiler / single-runtime support matrix, populated by TRYING -- not by reasoning.

The question this answers empirically: given the compilers and OpenMP runtimes actually installed, which
(set-of-compilers, one-runtime) combinations can build an N-loopnest program where EACH loop is compiled
parallel by a DIFFERENT compiler, LINK it against that ONE runtime, and RUN it to the right answer?

Reasoning about ABI tables gets the common cases right and the corners wrong: a runtime is linkable but
its ``.so`` will not load (icx needs libsvml off the default path); a flag is accepted but the back end
is inert (Ubuntu clang's Polly); ``libiomp5`` resolves to ``libomp`` via an ABI symlink; ``nvc -mp``
hard-links libnvomp and refuses every other runtime, so it can never join a shared-runtime program. The
only trustworthy matrix is the one where every cell was compiled, linked, loaded and checked. This module
builds that, populating absolute paths for the compilers, the OpenMP runtimes and the vector-math
libraries along the way -- the inputs a sweep needs and a figure documents.

Nothing here is imported by the hot arena path; it is a discovery/reporting tool (a CLI and the source of
the generated support table), so it may compile a few dozen tiny programs. Every kernel it builds is run
in a forked child via :func:`~nestforge.isolation.run_isolated`, so a bad combination crashes its child,
not the sweep.
"""
from __future__ import annotations

import ctypes
import itertools
import json
import os
import subprocess
import tempfile
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from nestforge.build import OPENMP_RUNTIMES, OpenMPRuntime, compiler_family, driver_lib_path
from nestforge.isolation import run_isolated
from nestforge.perf import flags
from nestforge.perf.tsvc_arena import Toolchain, discover_toolchains

#: Where the discovered toolchain config is cached. Populated ONCE (the first time nest-forge runs against
#: a machine), then loaded verbatim -- discovery compiles dozens of probe programs, which is wasteful to
#: repeat and, worse, non-deterministic to re-run mid-sweep. A machine's compilers do not move between
#: runs; when they do, the user deletes the file (or points NF_TOOLCHAIN_CACHE elsewhere) to force a
#: re-probe. Under the repo root's .cache so it is per-checkout and git-ignored.
DEFAULT_CACHE = Path(os.environ.get("NF_TOOLCHAIN_CACHE") or (Path(__file__).resolve().parents[2] / ".cache" /
                                                              "toolchains.json"))

#: The vector-math libraries whose absolute path we resolve per compiler (a compiler auto-links or is
#: pointed at exactly one; the arena's veclib axis selects among what is present). SVML ships with Intel,
#: libmvec with glibc, SLEEF as a separate package.
VECLIB_SONAMES = ("svml", "sleef", "mvec")


@dataclass(frozen=True)
class ToolPaths:
    """Absolute paths discovered for one compiler family: the driver, the OpenMP runtimes it can locate,
    and the vector-math libraries it can locate. Everything an arena cell or a figure needs, resolved
    once by asking the driver itself (``-print-file-name``) rather than guessing filesystem layout."""
    family: str  # OpenMP family: gnu | llvm | intel-classic | nvidia
    compiler: str  # absolute path to the C driver
    openmp: Dict[str, str] = field(default_factory=dict)  # runtime name -> absolute lib dir
    veclibs: Dict[str, str] = field(default_factory=dict)  # veclib soname -> absolute .so path


def resolve_tool_paths(tc: Toolchain) -> ToolPaths:
    """Resolve the absolute OpenMP-runtime and vector-lib paths ``tc`` can actually link, by asking its
    own driver. A runtime/lib the driver cannot locate is simply absent from the dict -- the matrix build
    is what decides whether a present one is usable end to end."""
    fam = compiler_family(tc.cc)
    omp: Dict[str, str] = {}
    for name, rt in OPENMP_RUNTIMES.items():
        found = driver_lib_path(rt.soname, tc.cc)
        if found is not None:
            omp[name] = str(found.parent)
    vec: Dict[str, str] = {}
    for soname in VECLIB_SONAMES:
        found = driver_lib_path(soname, tc.cc)
        if found is not None:
            vec[soname] = str(found)
    return ToolPaths(family=fam, compiler=tc.cc, openmp=omp, veclibs=vec)


#: One parallel loop, parameterised by a unique function name so N of them link into one program without
#: symbol clash. Each writes only its own output slice, so the whole program's result is checkable.
def loop_source(index: int) -> str:
    return (f"#include <omp.h>\n"
            f"void nest{index}(double *restrict a, const double *restrict b, int n) {{\n"
            f"  #pragma omp parallel for\n"
            f"  for (int i = 0; i < n; i++) a[i] = b[i] * 2.0 + {index}.0;\n"
            f"}}\n")


#: The driver that dlopens all N nests and runs them -- built once, links against nothing itself.
_DRIVER_SRC = """#include <stddef.h>
extern void {externs};
void run_all(double *a, const double *b, int n) {{
{calls}
}}
"""


def emits_fork_call(obj: str) -> bool:
    """True if object ``obj`` contains an OpenMP runtime fork call -- proof the loop was parallelised, not
    silently left serial (see :func:`nestforge.perf.flags.autopar_fires` for why linking is not enough)."""
    try:
        syms = subprocess.run(["nm", "-u", obj], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return any(s in syms for s in ("GOMP_parallel", "kmpc_fork"))


@dataclass
class MatrixCell:
    """One attempt: this tuple of compilers, each building one nest, linked against this one runtime."""
    runtime: str
    compilers: Tuple[str, ...]  # family label per nest, in nest order
    ok: bool
    loads: bool
    parallel: bool  # every nest object actually emitted a fork call
    correct: bool  # the linked program ran and matched numpy
    reason: str = ""


def try_combination(nests_by: List[Toolchain], runtime: OpenMPRuntime, workdir: Path) -> MatrixCell:
    """Compile each nest with its assigned compiler, link all against ONE runtime, load and run.

    This is the whole experiment. A nest is built to a ``.o`` with that compiler's ``omp-emit`` lane flags
    for ``runtime``; if any compiler cannot target the runtime the cell is a pruned skip. The objects link
    into one ``.so`` (the runtime search/rpath comes from the FIRST compiler that can supply it -- they
    all target the same soname). The driver dlopens it and runs every nest; the result is checked against
    numpy. Every stage that can fail is recorded, so the cell says not just pass/fail but WHERE it failed.
    """
    families = tuple(compiler_family(t.cc) for t in nests_by)
    label = tuple(t.name for t in nests_by)
    objs: List[str] = []
    parallel = True
    for idx, tc in enumerate(nests_by):
        f, reason = flags.lane_flags(tc.fp_family, "strict-ieee", "default", "omp-emit", "c", 2,
                                     compiler=tc.cc, openmp=runtime)
        if f is None:
            return MatrixCell(runtime.name, label, False, False, False, False, f"{tc.name}: {reason}")
        src = workdir / f"nest{idx}.c"
        src.write_text(loop_source(idx))
        obj = workdir / f"nest{idx}.o"
        # Compile ONLY (-c): linking happens once, below, so exactly one runtime enters the program.
        comp = [tc.cc, *[x for x in f if x not in ("-shared", )], "-c", str(src), "-o", str(obj)]
        proc = subprocess.run(comp, capture_output=True, text=True)
        if proc.returncode != 0:
            return MatrixCell(runtime.name, label, False, False, False, False,
                              f"{tc.name} compile: {proc.stderr.strip().splitlines()[-1][:80] if proc.stderr.strip() else '?'}")
        parallel = parallel and emits_fork_call(str(obj))
        objs.append(str(obj))

    # Link all nests + the runtime into one .so. Use the linking flags of the first compiler that can
    # supply the runtime search path -- every object needs the SAME soname, so any provider works.
    linker = nests_by[0]
    link_extra, _ = flags.openmp_runtime_flags(linker.cc, linker.fp_family, runtime)
    so = workdir / "program.so"
    link = [linker.cc, "-shared", *objs, *(link_extra or []), "-o", str(so)]
    proc = subprocess.run(link, capture_output=True, text=True)
    if proc.returncode != 0:
        return MatrixCell(runtime.name, label, False, False, parallel, False,
                          f"link: {proc.stderr.strip().splitlines()[-1][:80] if proc.stderr.strip() else '?'}")

    driver = workdir / "driver.c"
    externs = ", ".join(f"nest{i}(double*, const double*, int)" for i in range(len(nests_by)))
    calls = "\n".join(f"  nest{i}(a, b, n);" for i in range(len(nests_by)))
    driver.write_text(_DRIVER_SRC.format(externs=externs, calls=calls))
    prog = workdir / "prog.so"
    proc = subprocess.run([linker.cc, "-shared", "-fPIC", str(driver), str(so), "-o", str(prog),
                           f"-Wl,-rpath,{so.parent}"], capture_output=True, text=True)
    if proc.returncode != 0:
        return MatrixCell(runtime.name, label, True, False, parallel, False, "driver link failed")

    def work():
        lib = ctypes.CDLL(str(prog))
        lib.run_all.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        lib.run_all.restype = None
        n = 256
        a = np.zeros(n)
        b = np.arange(n, dtype=np.float64)
        lib.run_all(a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n)
        # each nest overwrites a[i] with b[i]*2 + its index; the LAST nest wins.
        expect = b * 2.0 + float(len(nests_by) - 1)
        return {"ok": bool(np.allclose(a, expect)), "maxdiff": float(np.max(np.abs(a - expect)))}

    res = run_isolated(work, timeout=60.0)
    if "error" in res:
        return MatrixCell(runtime.name, label, True, False, parallel, False, f"load/run: {res['error'][:80]}")
    return MatrixCell(runtime.name, label, True, True, parallel, bool(res["ok"]),
                      "" if res["ok"] else "ran but diverged from numpy")


def build_support_matrix(toolchains: Optional[List[Toolchain]] = None,
                         nests: int = 2,
                         drop_on_empty: Tuple[str, ...] = ("nvhpc", )) -> Tuple[List[MatrixCell], List[str]]:
    """Populate the support matrix by trying every (compiler-tuple, runtime) that could share a runtime.

    ``nests`` sets how many loops the trial program has (>=2 to exercise CROSS-compiler linkage). For each
    runtime, every ordered assignment of the compatible compilers to the nests is attempted; a cell
    SURVIVES when it links, loads, parallelises and runs correct. If NO cell survives for any runtime, the
    families in ``drop_on_empty`` are removed and the whole thing is retried once -- a single islanding
    compiler (nvc, which can only ever link its own libnvomp) must not be able to empty the matrix for the
    others. Returns ``(surviving_cells, notes)``.
    """
    if toolchains is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            toolchains = discover_toolchains("auto")
    notes: List[str] = []

    def attempt(tcs: List[Toolchain]) -> List[MatrixCell]:
        survivors: List[MatrixCell] = []
        for rt_name, rt in OPENMP_RUNTIMES.items():
            usable = [t for t in tcs if rt.compatible(t.cc)]
            if len(usable) < 1:
                continue
            # every ordered assignment of the usable compilers across the nests (with repetition: one
            # compiler building all nests is the same-compiler baseline the cross-compiler case is judged
            # against).
            for combo in itertools.product(usable, repeat=nests):
                with tempfile.TemporaryDirectory() as d:
                    cell = try_combination(list(combo), rt, Path(d))
                if cell.ok and cell.loads and cell.correct:
                    survivors.append(cell)
        return survivors

    survivors = attempt(toolchains)
    if not survivors and drop_on_empty:
        kept = [t for t in toolchains if t.name not in drop_on_empty]
        if len(kept) < len(toolchains):
            notes.append(f"no runtime supported any combination; dropped {drop_on_empty} and retried")
            survivors = attempt(kept)
    if not survivors:
        notes.append("no (compiler, runtime) combination survived -- OpenMP is unusable in this toolchain set")
    return survivors, notes


def surviving_runtimes(cells: List[MatrixCell]) -> List[str]:
    """The runtimes that support at least one working cross-compiler combination, best first -- the answer
    to 'which OpenMP runtime should the sweep standardise on for THIS machine'.

    Ranked by how many cross-compiler combinations each supports, then -- on a tie -- the portable default
    (libomp) wins over an equivalent that is really it under an ABI symlink (libiomp5 -> libomp) or a
    vendor-only runtime. Without that tiebreak the alphabetically-first name would win, picking libiomp5
    on a box where it is just Intel's copy of libomp."""
    by_rt: Dict[str, int] = {}
    for c in cells:
        if len(set(c.compilers)) > 1:  # cross-compiler cells are the ones that prove a SHARED runtime
            by_rt[c.runtime] = by_rt.get(c.runtime, 0) + 1
    default = flags.DEFAULT_OPENMP_RUNTIME.name
    return [rt for rt, _ in sorted(by_rt.items(), key=lambda kv: (-kv[1], kv[0] != default, kv[0]))]


def render_matrix(cells: List[MatrixCell], notes: List[str]) -> str:
    """A compact text table of the surviving cells plus any notes -- the CLI output and figure source."""
    lines = ["runtime    compilers                     parallel  correct"]
    lines.append("-" * 60)
    for c in sorted(cells, key=lambda c: (c.runtime, c.compilers)):
        lines.append(f"{c.runtime:10s} {'+'.join(c.compilers):28s}  {'yes' if c.parallel else 'NO ':8s}  "
                     f"{'yes' if c.correct else 'no'}")
    if not cells:
        lines.append("(none survived)")
    rts = surviving_runtimes(cells)
    lines.append("")
    lines.append(f"cross-compiler runtimes (best first): {', '.join(rts) if rts else '(none)'}")
    for n in notes:
        lines.append(f"note: {n}")
    return "\n".join(lines)


def machine_config(cache: Path = DEFAULT_CACHE, refresh: bool = False) -> Dict:
    """The discovered toolchain config for THIS machine: absolute compiler/runtime/veclib paths and the
    surviving cross-compiler runtimes. Probed ONCE and cached; loaded verbatim thereafter.

    This is the config the user asked for: run the discovery when nest-forge first ports to a machine,
    store it under ``.cache``, and on every later run LOAD it rather than re-probe. Discovery compiles
    dozens of tiny programs -- wasteful to repeat, and re-probing mid-project could silently shift which
    runtime a sweep standardises on. A machine's compilers do not move between runs; when they genuinely
    change, delete the cache (or pass ``refresh=True``) to re-probe. The file is human-readable JSON so a
    site can inspect -- or hand-edit -- exactly what was found.
    """
    if cache.exists() and not refresh:
        try:
            return json.loads(cache.read_text())
        except (OSError, ValueError):
            pass  # a corrupt cache re-probes rather than crashing the sweep that depends on it
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        toolchains = discover_toolchains("auto")
    cells, notes = build_support_matrix(toolchains)
    config = {
        "compilers": {t.name: {"cc": t.cc, "cxx": t.cxx, "version": list(t.version), "source": t.source}
                      for t in toolchains},
        "paths": {t.name: asdict(resolve_tool_paths(t)) for t in toolchains},
        "default_openmp_runtime": (surviving_runtimes(cells) or [flags.DEFAULT_OPENMP_RUNTIME.name])[0],
        "surviving_runtimes": surviving_runtimes(cells),
        "support_matrix": [asdict(c) for c in cells],
        "notes": notes,
    }
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(config, indent=2))
    except OSError:
        pass  # read-only checkout: still return the config, just do not persist it
    return config


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Discover the toolchain support matrix for this machine.")
    ap.add_argument("--refresh", action="store_true", help="re-probe even if the cache exists")
    ap.add_argument("--no-cache", action="store_true", help="probe and print, do not read or write the cache")
    args = ap.parse_args()
    if args.no_cache:
        cells, notes = build_support_matrix()
        print(render_matrix(cells, notes))
        return
    config = machine_config(refresh=args.refresh)
    print(f"cached at: {DEFAULT_CACHE}")
    print(f"default OpenMP runtime for this machine: {config['default_openmp_runtime']}")
    print(f"compilers: {', '.join(config['compilers'])}")
    print(f"surviving cross-compiler runtimes: {', '.join(config['surviving_runtimes']) or '(none)'}")


if __name__ == "__main__":
    main()
