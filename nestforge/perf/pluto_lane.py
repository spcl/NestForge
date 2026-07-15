"""Pluto polyhedral lane helpers: emit-side plumbing for the Pluto (``polycc``) backend, kept apart from
the DaCe-touching orchestration in :mod:`nestforge.perf.tsvc_full` so the parse / gate / invoke logic is
unit-testable WITHOUT a compiler or ``polycc`` on the box.

Pluto is a distinct TOOLCHAIN, not a compiler flag (unlike Polly): ``polycc`` applies a polyhedral
source-to-source transform (tiling + auto-parallelization) to a ``#pragma scop``-wrapped C source
(``<base>_pluto_input.c`` -- numpyto_c emits it alongside the plain C source) and writes a DIFFERENT C
file we then compile. That transformed function keeps a **VLA-parameter signature with the size symbols
FIRST** (``void k(int64 N, int64 M, double a[restrict N][M], ...)``), which nest-forge's ``signature_order``
regex cannot parse -- a VLA dim ``a[restrict N][M]`` is not a bare ``name``. So the ABI (symbol + argument
ORDER) is taken from the authoritative ``<base>_pluto_binding.json`` numpyto_c emits for exactly this
reason, NOT re-derived from the source. At the C ABI a VLA array parameter decays to a pointer, so once the
right ORDER is known nest-forge's existing ``c_argtypes`` / ``call_c`` marshal it unchanged (arrays ->
pointer, size symbols -> int64 values passed first).

Pluto's polyhedral model is AFFINE-only: a non-affine subscript (gather / modulo / integer-division) is
outside it and ``polycc`` may silently miscompile rather than reject, so such a nest is SKIPPED via the
shared detector lifted into :func:`optarena.pluto_affine.scop_nonaffine_reason`. ``polycc`` is a separate
polyhedral toolchain almost always ABSENT off the optarena container, so every gate below returns a
RECORDED skip reason, never a crash -- the lane is opt-in (``--pluto``)."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from optarena.pluto_affine import scop_nonaffine_reason

#: The polyhedral compiler driver. A shell wrapper that execs the pluto solver + isl + a C compiler.
POLYCC = "polycc"

#: Pluto's transformed output is built with these ON TOP of the C base flags: ``_POSIX_C_SOURCE`` is
#: re-supplied because clan/pet drops the source's leading define, and ``-fopenmp`` turns on the
#: parallelization polycc emitted (mirrors optarena's ``_PLUTO_EXTRA_FLAGS``).
PLUTO_EXTRA_FLAGS: Tuple[str, ...] = ("-D_POSIX_C_SOURCE=199309L", "-fopenmp")


def polycc_available() -> bool:
    """True when ``polycc`` is on PATH. Off the optarena container it usually is not -- the lane then
    records ``skip:not-installed`` rather than attempting a transform."""
    return shutil.which(POLYCC) is not None


def find_pluto_input(sources: List[Path]) -> Optional[Path]:
    """The ``*_pluto_input.c`` scop among a nest's emitted sources, or ``None`` if numpyto_c emitted none
    (a kernel with no lowerable scop). numpyto_c writes it beside the plain ``<base>.c`` on every
    sequential C emit, so it is present whenever the C lane is."""
    for s in sources:
        if s.suffix == ".c" and s.name.endswith("_pluto_input.c"):
            return s
    return None


def pluto_binding_path(pluto_input: Path) -> Path:
    """The ``<base>_pluto_binding.json`` sibling of a ``<base>_pluto_input.c`` (numpyto_c emits both)."""
    return pluto_input.with_name(pluto_input.name.replace("_pluto_input.c", "_pluto_binding.json"))


def pluto_output_path(pluto_input: Path) -> Path:
    """Where ``polycc`` writes the transformed source for a ``<base>_pluto_input.c``: ``<base>_pluto.c``."""
    return pluto_input.with_name(pluto_input.name.replace("_pluto_input.c", "_pluto.c"))


def read_pluto_binding(pluto_input: Path) -> Optional[Dict]:
    """Load the ``_pluto_binding.json`` for a pluto input, or ``None`` when numpyto_c emitted no binding
    (older emit / non-C kernel). The binding is the AUTHORITATIVE ABI: ``symbols.c`` + the size-first
    ``args`` order the VLA signature needs."""
    pb = pluto_binding_path(pluto_input)
    if not pb.exists():
        return None
    return json.loads(pb.read_text())


def binding_symbol_and_order(binding: Dict) -> Tuple[str, List[str]]:
    """``(symbol, [arg_name, ...])`` from a pluto binding: the exported C symbol and the parameter names in
    the Pluto (size-symbols-first) declaration order the transformed function expects."""
    symbol = binding["symbols"]["c"]
    order = [a["name"] for a in binding["args"]]
    return symbol, order


def pluto_gate_reason(pluto_input: Optional[Path]) -> Optional[str]:
    """Why the Pluto lane must SKIP this nest, or ``None`` when it can proceed. Mirrors optarena's
    ``_run_pluto`` precedence: ``polycc`` absent FIRST (so a box without the toolchain reports one uniform
    ``skip:not-installed`` across every nest, not a confusing mix), then no scop emitted, then a non-affine
    subscript (outside Pluto's model -- skip rather than risk a silent miscompile). A returned string is a
    recorded ``skip:...`` reason, never an error."""
    if not polycc_available():
        return "skip:not-installed"
    if pluto_input is None:
        return "skip:unsupported:no-scop"
    nonaffine = scop_nonaffine_reason(pluto_input.read_text())
    if nonaffine:
        return f"skip:unsupported:non-affine:{nonaffine}"
    return None


def run_polycc(pluto_input: Path, out_c: Path, timeout: float) -> Tuple[bool, Optional[str]]:
    """Transform ``pluto_input`` with ``polycc --pet`` into ``out_c``. Returns ``(ok, reason)``; ``reason``
    is a recorded ``skip:...`` string on any non-success (polycc rejected a scop it cannot lower, crashed,
    or timed out), never a raise.

    ``--pet`` selects libpet as the scop PARSER (the default clan parser chokes on the emitted ``int64_t``
    loop counters); Pluto's default schedule is used as-is -- a miscompile is recorded as the correctness
    result, never papered over with fusion/tiling flags. ``cwd=out_c.parent`` confines polycc's scratch
    (``.pluto.cloog`` etc.) to the throwaway nest dir instead of the CWD."""
    try:
        proc = subprocess.run(
            [POLYCC, "--pet", str(pluto_input), "-o", str(out_c)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(out_c.parent))
    except subprocess.TimeoutExpired:
        return False, "skip:unsupported:polycc-timeout"
    except OSError as e:
        return False, f"skip:not-installed:{type(e).__name__}"
    if proc.returncode or not out_c.exists():
        # polycc rejected a non-affine scop or its solver crashed: attempted, not a nest-forge failure.
        return False, "skip:unsupported:polycc"
    return True, None
