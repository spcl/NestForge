# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The compile-flag matrix phase 5 sweeps: FP mode crossed with vectorizer cost model, per compiler family.
``intel`` is its own family because icx, icpx and ifx default to ``-fp-model=fast``."""

from __future__ import annotations

from collections.abc import Sequence

#: FP modes, strictest first.
FP_LEVELS: tuple[str, ...] = ("strict-ieee", "contract-fma", "fast-math")

#: Relative tolerance against the NumPy float64 oracle. ``strict-ieee`` evaluates in the oracle's order, so only the
#: dtype floor applies.
FP_RTOL: dict[str, float] = {
    "strict-ieee": 0.0,
    "contract-fma": 1e-13,
    "fast-math": 1e-5,
}

#: Relative tolerance floor per output dtype, about one ULP; the gate is ``max(mode, dtype)``.
DTYPE_RTOL: dict[str, float] = {
    "float64": 2.3e-16,
    "float32": 1.2e-7,
    "float16": 9.8e-4,
}

#: FP-mode flags per family and mode.
FP: dict[str, dict[str, list[str]]] = {
    "gnu": {
        "strict-ieee": ["-ffp-contract=off", "-fexcess-precision=standard"],
        "contract-fma": ["-ffp-contract=fast", "-fexcess-precision=standard"],
        "fast-math": ["-ffast-math", "-mrecip"],
    },
    "llvm": {
        "strict-ieee": ["-ffp-contract=off"],
        "contract-fma": ["-ffp-contract=fast"],
        "fast-math": ["-ffast-math", "-mrecip"],
    },
    "intel": {
        "strict-ieee": ["-fp-model=strict"],
        "contract-fma": ["-fp-model=precise"],
        "fast-math": ["-fp-model=fast=2", "-ftz"],
    },
}

#: Vectorizer cost models: the compiler's own, fewer vectorizations (gcc only), and none.
COST_MODELS: tuple[str, ...] = ("default", "cheap", "no-vec")


#: The prefix every cell shares.
BASE_FLAGS: tuple[str, ...] = ("-O3", "-march=native", "-fPIC", "-shared")


def fp_flags(family: str, level: str) -> list[str]:
    """FP-mode flags for ``family`` at ``level``."""
    return list(FP[family][level])


def cost_flags(family: str, model: str) -> list[str]:
    """Vectorizer cost-model flags for ``family``; empty where it has no such knob."""
    if model == "no-vec":
        return {
            "gnu": ["-fno-tree-vectorize"],
            "llvm": ["-fno-vectorize", "-fno-slp-vectorize"],
            "intel": ["-fno-vectorize", "-fno-slp-vectorize"],
        }.get(family, [])
    if model == "cheap":
        return {"gnu": ["-fvect-cost-model=cheap"]}.get(family, [])
    return []


def flag_matrix(family: str) -> list[tuple[str, str, list[str]]]:
    """``(fp_level, cost_model, flags)`` per cell for ``family``, one per distinct flag set."""
    matrix: list[tuple[str, str, list[str]]] = []
    seen: dict[tuple[str, ...], None] = {}
    base = list(BASE_FLAGS)
    for level in FP_LEVELS:
        for model in COST_MODELS:
            flags = base + fp_flags(family, level) + cost_flags(family, model)
            key = tuple(flags)
            if key in seen:
                continue
            seen[key] = None
            matrix.append((level, model, flags))
    return matrix


#: Device and host FP flags per mode a GPU kernel sweeps; ``--fmad`` fuses multiply-adds on the device. nvcc has
#: no fast-math mode that matches the CPU one.
CUDA_FP: dict[str, list[str]] = {
    "strict-ieee": ["--fmad=false", "-Xcompiler=-ffp-contract=off"],
    "contract-fma": ["--fmad=true", "-Xcompiler=-ffp-contract=fast"],
}

#: The cost model of a GPU cell: nvcc has none to sweep.
NO_COST_MODEL = "none"


def cuda_base_flags(build_flags: Sequence[str]) -> list[str]:
    """``build_flags`` (what a CPF CUDA unit needs), then the prefix every GPU cell shares."""
    return [*build_flags, "-O3", "-arch=native", "-Xcompiler=-fPIC", "-shared"]


def cuda_flag_matrix(build_flags: Sequence[str]) -> list[tuple[str, list[str]]]:
    """``(fp_level, flags)`` per GPU cell."""
    base = cuda_base_flags(build_flags)
    return [(level, base + fp) for level, fp in CUDA_FP.items()]
