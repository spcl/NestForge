# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pluto polyhedral lane helpers: emit-side plumbing for the Pluto (``polycc``) backend, kept apart from
the DaCe orchestration in :mod:`nestforge.perf.tsvc_full` so it's unit-testable without ``polycc``.

Pluto is a distinct toolchain, not a compiler flag: ``polycc`` transforms a ``#pragma scop`` C source
(``<base>_pluto_input.c``) into a different C file with a VLA signature (size symbols first) that
``signature_order``'s regex can't parse -- so the ABI is read from the authoritative
``<base>_pluto_binding.json`` numpyto_c emits, never re-derived from the transformed source.

Pluto's model is AFFINE-only; a non-affine subscript may be silently miscompiled rather than rejected,
so such a nest is SKIPPED via :func:`optarena.pluto_affine.scop_nonaffine_reason`. ``polycc`` is usually
absent off the optarena container, so every gate below returns a recorded skip reason, never a crash --
the lane is opt-in (``--pluto``)."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from hpcagent_bench.pluto_affine import scop_nonaffine_reason

#: The polyhedral compiler driver. A shell wrapper that execs the pluto solver + isl + a C compiler.
POLYCC = "polycc"

#: Pluto's transformed output is built with these ON TOP of the C base flags: ``_POSIX_C_SOURCE`` is
#: re-supplied because clan/pet drops the source's leading define, and ``-fopenmp`` turns on the
#: parallelization polycc emitted (mirrors hpcagent_bench's ``_PLUTO_EXTRA_FLAGS``).
PLUTO_EXTRA_FLAGS: Tuple[str, ...] = ("-D_POSIX_C_SOURCE=199309L", "-fopenmp")


def polycc_available() -> bool:
    """True when ``polycc`` is on PATH; absent off the optarena container, where the lane records
    ``skip:not-installed``."""
    return shutil.which(POLYCC) is not None


def find_pluto_input(sources: List[Path]) -> Optional[Path]:
    """The ``*_pluto_input.c`` scop among a nest's emitted sources, or ``None`` if numpyto_c emitted none
    (no lowerable scop)."""
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
    """Load the ``_pluto_binding.json`` for a pluto input, or ``None`` when none was emitted. The
    authoritative ABI: ``symbols.c`` + the size-first ``args`` order the VLA signature needs."""
    pb = pluto_binding_path(pluto_input)
    if not pb.exists():
        return None
    return json.loads(pb.read_text())


def binding_symbol_and_order(binding: Dict) -> Tuple[str, List[str]]:
    """``(symbol, [arg_name, ...])`` from a pluto binding: the exported symbol and the parameter names in
    Pluto's size-symbols-first order."""
    symbol = binding["symbols"]["c"]
    order = [a["name"] for a in binding["args"]]
    return symbol, order


def pluto_gate_reason(pluto_input: Optional[Path]) -> Optional[str]:
    """Why the Pluto lane must skip this nest, or ``None`` when it can proceed. Checks in order:
    ``polycc`` absent (uniform ``skip:not-installed`` across the whole box), no scop emitted, then a
    non-affine subscript (skip rather than risk a silent miscompile). Always a recorded reason, never
    an error."""
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
    is a recorded ``skip:...`` string on any non-success, never a raise.

    ``--pet`` selects libpet as the scop parser (the default clan parser chokes on ``int64_t`` loop
    counters). ``cwd=out_c.parent`` confines polycc's scratch files to the throwaway nest dir."""
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
        # polycc rejected the scop or crashed: attempted, not a nest-forge failure.
        return False, "skip:unsupported:polycc"
    return True, None
