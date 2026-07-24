# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-device characterization (stage A of the vectorization + veclib selection): what SIMD ISAs the host
supports, and how each installed vector-math library trades throughput against accuracy on this box.

Two independent axes share this one "measure the device once" stage:

  * **ISA detection** (:func:`host_isas`) -- builds on DaCe's own ``detect_host_isa`` (never re-parses
    ``/proc/cpuinfo``) and expands it to the tuple of tile-op ISAs the vectorization sweep should emit,
    always with ``SCALAR`` appended as the floor. An ``ARM_SVE`` host yields BOTH ``ARM_NEON`` and
    ``ARM_SVE`` (per the plan decision to measure both on SVE hardware).
  * **veclib ranking** (:func:`rank_veclibs`) -- for each INSTALLED, compiler-compatible vector-math
    library (SLEEF / libmvec / SVML), a compact microbenchmark over the elemental functions that actually
    route through it (:func:`pure_only_math_ops`) reports throughput AND max-ULP error vs a long-double
    reference. That converts veclib from a per-kernel combinatorial axis into a per-device *ranking*: the
    arena then carries only ``none`` + the accuracy-gated winner. A library that is absent or incompatible
    is skipped WITH a recorded reason, never silently dropped.

Nothing here mutates DaCe or emits an SDFG; it reads host capabilities and compiles tiny standalone probes
(run as separate processes, so a broken probe never touches the parent)."""
from __future__ import annotations

import functools
import platform
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dace.libraries.tileops._dispatch import detect_host_isa

from nestforge.build import VECTOR_LIBS, VectorMathLib, compiler_family, vectorlib_installed

#: ``detect_host_isa()`` result -> the tile-op ISAs the vectorization sweep emits for this host, DEFAULT
#: (widest) first, with ``SCALAR`` always appended as the floor. ``ARM_SVE`` keeps ``ARM_NEON`` too (both
#: measured on SVE hardware, per the plan). Keys are exactly ``detect_host_isa``'s return domain.
_ISA_LADDER: Dict[str, Tuple[str, ...]] = {
    "AVX512": ("AVX512", "SCALAR"),
    "AVX2": ("AVX2", "SCALAR"),
    "ARM_SVE": ("ARM_SVE", "ARM_NEON", "SCALAR"),
    "ARM_NEON": ("ARM_NEON", "SCALAR"),
    "SCALAR": ("SCALAR", ),
}


def host_isas() -> Tuple[str, ...]:
    """The tile-op ISAs to emit vectorization cells for on THIS host: :func:`dace...detect_host_isa`
    expanded through :data:`_ISA_LADDER` (widest first, ``SCALAR`` floor always present). Never re-sniffs
    the CPU -- it defers to DaCe's detector so the arena and the tile-op expansion agree on the host ISA."""
    return _ISA_LADDER.get(detect_host_isa(), ("SCALAR", ))


def pure_only_math_ops() -> frozenset:
    """The elemental math ops routed to the ``pure`` scalar tile loop specifically so the compiler's
    vector-math library captures them (``atan2``/``hypot``/``tan``/``asin``/``pow``/...). A nest that uses
    NONE of these gets no veclib axis (the library is inert for it). Sourced from DaCe so the two never
    drift; imported lazily because it is an internal of the tile-op dispatcher."""
    from dace.libraries.tileops._dispatch import _PURE_ONLY_MATH_OPS
    return _PURE_ONLY_MATH_OPS


# --- veclib characterization -------------------------------------------------------------------------
@dataclass(slots=True)
class VeclibProfile:
    """One vector-math library's measured trade-off on this device (for one compiler)."""
    name: str  # sleef | libmvec | svml
    compiler: str
    throughput_speedup: float  # veclib elems/s / scalar elems/s (>1 = the veclib is faster)
    max_ulp: float  # worst-case ULP error of the veclib result vs a long-double reference
    ok: bool
    reason: Optional[str] = None  # why it was skipped, when ok is False


def _probe_source() -> str:
    """A standalone C probe: fill an array, apply a math-heavy expression, print ``<seconds> <max_ulp>``.
    The expression lives in ONE ``EXPR`` macro instantiated twice -- ``double`` (the veclib target) and
    ``long double`` (the accuracy reference) -- so both compute exactly the same thing and max-ULP measures
    the vectorized library's error. ``n`` comes from argv and the results feed a checksum so nothing folds
    away. The probe is a separate process, so a crash never touches the parent."""
    return r"""#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define EXPR(S, FSIN, FCOS, FEXP, FLOG) (FSIN(S) + FCOS(S) + FEXP(0.01 * (S)) + FLOG(1.0 + (S) * (S)))
int main(int argc, char **argv) {
    long n = atol(argv[1]);
    double *x = malloc(n * sizeof(double)), *y = malloc(n * sizeof(double));
    for (long i = 0; i < n; i++) x[i] = 0.5 + 5.0 * ((double)i / (double)n);
    struct timespec a, b;
    clock_gettime(CLOCK_MONOTONIC, &a);
    for (long i = 0; i < n; i++) { double xv = x[i]; y[i] = EXPR(xv, sin, cos, exp, log); }
    clock_gettime(CLOCK_MONOTONIC, &b);
    double secs = (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) * 1e-9;
    double max_ulp = 0.0, checksum = 0.0;
    for (long i = 0; i < n; i++) {
        long double xlv = (long double)x[i];
        double refd = (double)EXPR(xlv, sinl, cosl, expl, logl);
        double u = nextafter(refd, refd * 2.0 + 1.0) - refd;
        double err = u > 0.0 ? fabs(y[i] - refd) / u : 0.0;
        if (err > max_ulp) max_ulp = err;
        checksum += y[i];
    }
    fprintf(stderr, "%.3f", checksum);  /* keep the loop live, off stdout */
    printf("%.9f %.3f\n", secs, max_ulp);
    return 0;
}
"""


def _run_probe(compiler: str, extra_flags: List[str], link_flags: List[str], n: int) -> Optional[Tuple[float, float]]:
    """Compile + run the probe once; return ``(seconds, max_ulp)`` or ``None`` on any compile/run failure.
    The probe is a separate PROCESS (subprocess), so a crash never touches this one."""
    with tempfile.TemporaryDirectory(prefix="nf_veclib_") as d:
        src = Path(d) / "probe.c"
        exe = Path(d) / "probe"
        src.write_text(_probe_source())
        cmd = [compiler, "-O3", "-march=native", *extra_flags, str(src), "-o", str(exe), "-lm", *link_flags]
        try:
            if subprocess.run(cmd, capture_output=True, text=True, timeout=120).returncode != 0:
                return None
            out = subprocess.run([str(exe), str(n)], capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError):
            return None
        if out.returncode != 0 or not out.stdout.strip():
            return None
        try:
            secs, ulp = out.stdout.split()
            return float(secs), float(ulp)
        except ValueError:
            return None


def characterize_veclib(compiler: str, name: str, n: int = 4_000_000) -> VeclibProfile:
    """Measure one vector-math library on this device: throughput speedup vs the scalar build and worst-case
    ULP error vs a long-double reference. Gated on install + compiler compatibility; a miss returns
    ``ok=False`` with a reason rather than raising."""
    vl: Optional[VectorMathLib] = VECTOR_LIBS.get(name)
    if vl is None:
        return VeclibProfile(name, compiler, 0.0, 0.0, False, f"unknown veclib {name!r}")
    if not vl.compatible(compiler):
        return VeclibProfile(name, compiler, 0.0, 0.0, False, f"{name} incompatible with {compiler_family(compiler)}")
    if not vectorlib_installed(vl):
        return VeclibProfile(name, compiler, 0.0, 0.0, False, f"{name} not installed (soname lib{vl.soname} not found)")
    scalar = _run_probe(compiler, [], [], n)
    # The veclib needs relaxed math semantics to substitute the packed calls, so the probe uses -ffast-math.
    veclib = _run_probe(compiler, ["-ffast-math", *vl.compile_flags(compiler)], vl.link_flags(compiler), n)
    if scalar is None or veclib is None:
        return VeclibProfile(name, compiler, 0.0, 0.0, False, f"{name} probe failed to compile/run")
    scalar_s, _ = scalar
    veclib_s, ulp = veclib
    speedup = (scalar_s / veclib_s) if veclib_s > 0 else 0.0
    return VeclibProfile(name, compiler, speedup, ulp, True)


def rank_veclibs(compiler: str, max_ulp: float = 4.0) -> List[VeclibProfile]:
    """Every installed, compatible veclib measured on this device, best (fastest within the ``max_ulp``
    accuracy budget) first. Libraries over the ULP budget sort last (accuracy gates speed). The caller
    takes ``[0]`` as the characterized winner for the ``none`` + winner arena axis."""
    profiles = [characterize_veclib(compiler, name) for name in VECTOR_LIBS]
    ok = [p for p in profiles if p.ok]
    ok.sort(key=lambda p: (p.max_ulp > max_ulp, -p.throughput_speedup))
    return ok + [p for p in profiles if not p.ok]


@dataclass(slots=True)
class DeviceProfile:
    """The cached per-device characterization consumed by the vectorization + veclib selection stages."""
    machine: str
    host_isas: Tuple[str, ...]
    veclib_ranking: List[VeclibProfile]  # empty when no veclib is installed for the given compiler


@functools.lru_cache(maxsize=8, typed=True)
def device_profile(compiler: str = "gcc") -> DeviceProfile:
    """Characterize this device ONCE per process (memoized per compiler): the host ISAs plus the veclib
    ranking for ``compiler``. Cross-run persistence is a future add (keyed by host + compiler); within a
    sweep rank this in-process memo already means "measured once"."""
    return DeviceProfile(machine=platform.machine(), host_isas=host_isas(), veclib_ranking=rank_veclibs(compiler))
