"""Empirical cross-compiler / single-runtime support matrix: TRIES every (compilers, ONE runtime) combo --
compile, link, load, run -- rather than reasoning from ABI tables, which miss real corners (a runtime that
won't load, an inert backend, one islanded to its own compiler). Off the hot arena path; each probe runs
forked via :func:`~nestforge.isolation.run_isolated`."""
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

#: Discovered toolchain config cache -- probed once, then loaded verbatim until deleted (or refreshed).
DEFAULT_CACHE = Path(
    os.environ.get("NF_TOOLCHAIN_CACHE") or (Path(__file__).resolve().parents[2] / ".cache" / "toolchains.json"))

#: Vector-math libs whose absolute path we resolve per compiler: SVML (Intel), libmvec (glibc), SLEEF (separate pkg).
VECLIB_SONAMES = ("svml", "sleef", "mvec")


@dataclass(frozen=True)
class ToolPaths:
    """Absolute paths for one compiler family: driver, locatable OpenMP runtimes, locatable veclibs --
    resolved by asking the driver (``-print-file-name``), not by guessing filesystem layout."""
    family: str  # OpenMP family: gnu | llvm | intel-classic | nvidia
    compiler: str  # absolute path to the C driver
    openmp: Dict[str, str] = field(default_factory=dict)  # runtime name -> absolute lib dir
    veclibs: Dict[str, str] = field(default_factory=dict)  # veclib soname -> absolute .so path


def resolve_tool_paths(tc: Toolchain) -> ToolPaths:
    """Resolve the OpenMP-runtime and veclib paths ``tc`` can actually link, by asking its own driver;
    one it can't locate is simply absent from the dict."""
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


#: One parallel loop with a unique function name so N of them link into one program without symbol clash.
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
    """True if ``obj`` emits an OpenMP fork call -- proof the loop parallelised, not just linked (see
    :func:`nestforge.perf.flags.autopar_fires`)."""
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
    """Compile each nest with its assigned compiler, link all against ONE runtime, load and run -- the
    whole experiment; link flags come from the FIRST compiler that can supply the runtime path. Every
    failing stage is recorded, so the cell says WHERE it failed, not just pass/fail."""
    families = tuple(compiler_family(t.cc) for t in nests_by)
    label = tuple(t.name for t in nests_by)
    objs: List[str] = []
    parallel = True
    for idx, tc in enumerate(nests_by):
        f, reason = flags.lane_flags(tc.fp_family,
                                     "strict-ieee",
                                     "default",
                                     "omp-emit",
                                     "c",
                                     2,
                                     compiler=tc.cc,
                                     openmp=runtime)
        if f is None:
            return MatrixCell(runtime.name, label, False, False, False, False, f"{tc.name}: {reason}")
        src = workdir / f"nest{idx}.c"
        src.write_text(loop_source(idx))
        obj = workdir / f"nest{idx}.o"
        # Compile only (-c); link happens once below so exactly one runtime enters the program.
        comp = [tc.cc, *[x for x in f if x not in ("-shared", )], "-c", str(src), "-o", str(obj)]
        proc = subprocess.run(comp, capture_output=True, text=True)
        if proc.returncode != 0:
            return MatrixCell(
                runtime.name, label, False, False, False, False,
                f"{tc.name} compile: {proc.stderr.strip().splitlines()[-1][:80] if proc.stderr.strip() else '?'}")
        parallel = parallel and emits_fork_call(str(obj))
        objs.append(str(obj))

    # Link all nests + runtime into one .so via the first compiler's link flags -- all need the SAME soname.
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
    proc = subprocess.run(
        [linker.cc, "-shared", "-fPIC",
         str(driver), str(so), "-o",
         str(prog), f"-Wl,-rpath,{so.parent}"],
        capture_output=True,
        text=True)
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

    A cell SURVIVES when it links, loads, parallelises and runs correct. If none survive for any runtime,
    families in ``drop_on_empty`` are dropped and retried once, so an islanding compiler (nvc,
    libnvomp-only) can't empty the matrix for everyone else. Returns ``(surviving_cells, notes)``."""
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
            # every ordered assignment of usable compilers across nests (repeats = the same-compiler baseline)
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
    """Runtimes supporting >=1 working cross-compiler combination, best first: ranked by combo count,
    ties broken toward the portable default (libomp) over an ABI-symlinked equivalent (libiomp5) so the
    alphabetically-first name doesn't win by accident."""
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


@dataclass
class VeclibCell:
    """One (compiler-family, veclib) attempt: does a ``sin`` loop compile+link, does the object actually
    CALL the packed ``sin`` (not scalar), and match ``numpy.sin``? Veclib analogue of :class:`MatrixCell`
    -- a ``-fveclib=`` flag can be accepted while the vectorizer leaves the call scalar."""
    veclib: str
    compiler: str  # compiler-family label (gnu | llvm | intel-classic | nvidia)
    ok: bool  # compiled AND linked
    loads: bool  # the linked .so dlopened and ran
    vectorized: bool  # the object references the veclib's packed sin (nm -u fingerprint)
    correct: bool  # ran and matched numpy.sin
    reason: str = ""


#: One ``sin`` loop under ``#pragma omp simd``; at -O3/native+fast-math the vectorizer may swap the scalar
#: ``sin`` for a packed veclib routine, which :func:`vectorized_via` confirms via undefined symbols.
_SIN_SOURCE = """#include <math.h>
void sinloop(double *restrict a, const double *restrict b, int n) {
  #pragma omp simd
  for (int i = 0; i < n; i++) a[i] = sin(b[i]);
}
"""


def vectorized_via(veclib: str, obj_path: str) -> bool:
    """True if ``obj_path`` calls ``veclib``'s packed vector ``sin`` (via ``nm -u`` fingerprint) -- proof the
    vectorizer fired, not just that the flag was accepted. ``none`` is always False (scalar baseline);
    ``libmvec``/``sleef`` share the same glibc ``_ZGV*`` emission and differ only in the linked library;
    ``svml`` emits ``__svml_sin*``."""
    if veclib == "none":
        return False
    try:
        syms = subprocess.run(["nm", "-u", obj_path], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    if veclib in ("libmvec", "sleef"):
        return any(s in syms for s in (
            "_ZGVbN2v_sin",
            "_ZGVcN4v_sin",
            "_ZGVdN4v_sin",
            "_ZGVeN8v_sin",  # unmasked SSE/AVX/AVX2/AVX512
            "_ZGVbM2v_sin",
            "_ZGVcM4v_sin",
            "_ZGVdM4v_sin",
            "_ZGVeM8v_sin"))  # masked (omp-simd/AVX512)
    if veclib == "svml":
        return "__svml_sin" in syms
    return False


def try_veclib(tc: Toolchain, veclib: str, workdir: Path) -> VeclibCell:
    """Compile a ``sin`` loop with ``tc`` against ``veclib``, prove the object CALLS the packed routine,
    link it, and run it FORKED against ``numpy.sin`` -- veclib analogue of :func:`try_combination`.

    The ``none`` baseline OMITS ``-ffast-math``: WITH it, gcc emits libmvec calls this cell never links,
    so the scalar baseline would fail to load. Each failing stage forces later stages False."""
    fam = compiler_family(tc.cc)
    vec, reason = flags.veclib_flags(tc.cc, veclib)
    if vec is None:
        return VeclibCell(veclib, fam, False, False, False, False, reason or f"veclib {veclib} unsupported")
    # -ffast-math authorises the scalar->packed substitution; the baseline must omit it or gcc emits
    # unlinked libmvec calls (see docstring).
    base = ["-O3", "-march=native", "-fopenmp-simd", "-fPIC", "-shared"]
    if veclib != "none":
        base = ["-ffast-math", *base]
    src = workdir / "sinloop.c"
    src.write_text(_SIN_SOURCE)
    obj = workdir / "sinloop.o"
    comp = subprocess.run([tc.cc, *base, *vec, "-c", str(src), "-o", str(obj)], capture_output=True, text=True)
    if comp.returncode != 0:
        last = comp.stderr.strip().splitlines()[-1][:80] if comp.stderr.strip() else "?"
        return VeclibCell(veclib, fam, False, False, False, False, f"compile: {last}")
    vectorized = vectorized_via(veclib, str(obj))
    # Link object BEFORE veclib -l flags: under --as-needed a lib listed first is dropped from DT_NEEDED.
    so = workdir / "sinloop.so"
    link = subprocess.run([tc.cc, *base, str(obj), *vec, "-o", str(so)], capture_output=True, text=True)
    if link.returncode != 0:
        last = link.stderr.strip().splitlines()[-1][:80] if link.stderr.strip() else "?"
        return VeclibCell(veclib, fam, False, False, vectorized, False, f"link: {last}")

    def work():
        lib = ctypes.CDLL(str(so))
        lib.sinloop.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        lib.sinloop.restype = None
        n = 1024
        a = np.zeros(n)
        b = np.linspace(0.0, 6.0, n)
        lib.sinloop(a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n)
        return {"ok": bool(np.allclose(a, np.sin(b), atol=1e-6)), "maxdiff": float(np.max(np.abs(a - np.sin(b))))}

    res = run_isolated(work, timeout=60.0)
    if "error" in res:
        return VeclibCell(veclib, fam, True, False, vectorized, False, f"load/run: {res['error'][:80]}")
    return VeclibCell(veclib, fam, True, True, vectorized, bool(res["ok"]),
                      "" if res["ok"] else "ran but diverged from numpy.sin")


def probe_vector_libs(toolchains: List[Toolchain]) -> List[VeclibCell]:
    """Probe every (toolchain, veclib) empirically: compile+link+run a ``sin`` loop forked against
    ``numpy.sin``. ``none`` is the scalar baseline proving the harness works (veclib analogue of
    :func:`build_support_matrix`)."""
    cells: List[VeclibCell] = []
    for tc in toolchains:
        for veclib in flags.VECLIBS:
            with tempfile.TemporaryDirectory() as d:
                cells.append(try_veclib(tc, veclib, Path(d)))
    return cells


class MachineCompat:
    """Queryable view of the discovered support matrix -- what THIS machine actually supports, for a sweep
    to prune against instead of the static ABI table (:meth:`OpenMPRuntime.compatible`, which answers only
    what's possible in principle). Built from a :func:`machine_config` dict (the cache), read once per sweep."""

    def __init__(self, config: Dict):
        self.config = config
        self._cells = config.get("support_matrix", [])
        self._veclib_cells = config.get("veclib_matrix", [])

    def default_runtime(self) -> OpenMPRuntime:
        """The runtime to standardise on: the empirically-best cross-compiler survivor, or LIBOMP if
        nothing was discovered (no cache)."""
        name = self.config.get("default_openmp_runtime") or flags.DEFAULT_OPENMP_RUNTIME.name
        return OPENMP_RUNTIMES.get(name, flags.DEFAULT_OPENMP_RUNTIME)

    def is_supported(self, compiler_family: str, runtime_name: str) -> bool:
        """Did ``compiler_family`` linked against ``runtime_name`` build, load, parallelise and run correct
        here? The per-cell answer the arena prunes on."""
        return any(c["runtime"] == runtime_name and compiler_family in c["compilers"] and c["correct"] and c["parallel"]
                   for c in self._cells)

    def supported_runtimes(self, compiler_family: str) -> List[str]:
        """Runtimes ``compiler_family`` can use here, ranked with the machine default first, so a
        fallback picks the most-portable working runtime, not an arbitrary one."""
        order = self.config.get("surviving_runtimes", []) + list(OPENMP_RUNTIMES)
        seen, out = set(), []
        for rt in order:
            if rt not in seen and self.is_supported(compiler_family, rt):
                seen.add(rt)
                out.append(rt)
        return out

    def supported_compilers(self, runtime_name: str) -> List[str]:
        """Compiler families that can target ``runtime_name`` here -- who may join a sweep standardised
        on that one runtime."""
        fams = {
            f
            for c in self._cells if c["runtime"] == runtime_name and c["correct"] and c["parallel"]
            for f in c["compilers"]
        }
        return sorted(fams)

    def runtime_for(self, compiler_family: str) -> Optional[OpenMPRuntime]:
        """The runtime to build ``compiler_family``'s cells with: the machine default if supported (keeps
        the sweep on ONE runtime), else its own best runtime, else None if it can't parallelise here."""
        default = self.default_runtime()
        if self.is_supported(compiler_family, default.name):
            return default
        own = self.supported_runtimes(compiler_family)
        return OPENMP_RUNTIMES.get(own[0]) if own else None

    def supported_veclibs(self, compiler_family: str) -> List[str]:
        """Veclibs ``compiler_family`` ran correctly here (``none`` included -- always a valid choice).
        From the empirical probe (``supported_veclibs`` config field), not the ABI table."""
        return list(self.config.get("supported_veclibs", {}).get(compiler_family, []))

    def veclib_vectorizes(self, compiler_family: str, veclib: str) -> bool:
        """Did ``compiler_family`` x ``veclib`` emit a packed vector ``sin`` AND match numpy here?
        Separates a veclib that really vectorises from one whose flag was accepted but stayed scalar."""
        return any(c["compiler"] == compiler_family and c["veclib"] == veclib and c["vectorized"] and c["correct"]
                   for c in self._veclib_cells)


def machine_compat(cache: Path = DEFAULT_CACHE, refresh: bool = False) -> MachineCompat:
    """The compatibility view for this machine, from the cache (probing once if absent)."""
    return MachineCompat(machine_config(cache=cache, refresh=refresh))


def cached_default_runtime(cache: Path = DEFAULT_CACHE) -> OpenMPRuntime:
    """The machine's discovered default OpenMP runtime if a cache exists, else the static default.

    NEVER probes (unlike :func:`machine_compat`) -- safe on the sweep hot path, since discovery must
    stay an explicit once-per-machine step, never a surprise mid-sweep."""
    if cache.exists():
        try:
            return MachineCompat(json.loads(cache.read_text())).default_runtime()
        except (OSError, ValueError):
            pass
    return flags.DEFAULT_OPENMP_RUNTIME


def machine_config(cache: Path = DEFAULT_CACHE, refresh: bool = False) -> Dict:
    """The discovered toolchain config for this machine: absolute compiler/runtime/veclib paths +
    surviving cross-compiler runtimes. Probed ONCE and cached as human-readable JSON; loaded verbatim
    thereafter until the cache is deleted or ``refresh=True`` forces a re-probe."""
    if cache.exists() and not refresh:
        try:
            return json.loads(cache.read_text())
        except (OSError, ValueError):
            pass  # corrupt cache -> re-probe, never crash the sweep
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        toolchains = discover_toolchains("auto")
    cells, notes = build_support_matrix(toolchains)
    veclib_cells = probe_vector_libs(toolchains)
    supported_vec: Dict[str, List[str]] = {}
    for c in veclib_cells:
        if c.correct and c.veclib not in supported_vec.setdefault(c.compiler, []):
            supported_vec[c.compiler].append(c.veclib)
    config = {
        "compilers": {
            t.name: {
                "cc": t.cc,
                "cxx": t.cxx,
                "version": list(t.version),
                "source": t.source
            }
            for t in toolchains
        },
        "paths": {
            t.name: asdict(resolve_tool_paths(t))
            for t in toolchains
        },
        "default_openmp_runtime": (surviving_runtimes(cells) or [flags.DEFAULT_OPENMP_RUNTIME.name])[0],
        "surviving_runtimes": surviving_runtimes(cells),
        "support_matrix": [asdict(c) for c in cells],
        "veclib_matrix": [asdict(c) for c in veclib_cells],
        "supported_veclibs": supported_vec,
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
