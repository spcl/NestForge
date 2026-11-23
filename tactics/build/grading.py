# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A kernel's outputs graded against its NumPy oracle as HPCAgent-Bench grades a program's (``grade_outputs``): the
bench's comparator (``frameworks.utilities.compare_arrays``) at the band of each output's precision, with the
reassociation floor ``eps_acc * sqrt(l)`` at the output's accumulation length (:mod:`tactics.ir.accumulation`) and
the bench's grading knobs (``grading.fp_checks``: ``strict_small_l2``, ``strict_small_sign``, ``relative_atol``). The bench grades only program outputs,
against its own reference; a kernel has neither, so NestForge supplies the oracle and ``l`` and the bench the rule.

The bench is imported inside the functions: importing tactics never loads it."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from tactics.build.isolation import Result


@dataclass(frozen=True, slots=True)
class Expected:
    """What a kernel's outputs must reproduce: the oracle's outputs, and the accumulation length of each."""

    outputs: dict[str, np.ndarray]
    lengths: Mapping[str, int] = field(default_factory=dict)


class Grade(Result):
    """One kernel's outputs against :class:`Expected`. ``error`` is the bench's max relative error over every output;
    ``detail`` says which outputs fail and why; ``ungradeable`` is set when the bench refuses the band at an output's
    length (``eps_acc * sqrt(l) >= rtol``), which leaves the kernel to the whole program's check."""

    ok: bool
    error: float
    detail: str = ""
    ungradeable: str = ""


def output_precision(array: np.ndarray) -> object:
    """The bench ``Precision`` an output is graded at: its float format (a complex output's component's); fp64 for an
    integer or bool output, which compares exactly whatever the band."""
    from hpcagent_bench.precision import Precision, precision_from_datatype

    kind = array.dtype
    if not np.issubdtype(kind, np.inexact):
        return Precision.FP64
    name = np.finfo(kind).dtype.name if np.issubdtype(kind, np.complexfloating) else kind.name
    try:
        return precision_from_datatype(name)
    except ValueError:
        return Precision.FP64


def grade_kernel(expected: Expected, outputs: Mapping[str, np.ndarray]) -> Grade:
    """``outputs`` graded against ``expected`` by the bench's rule; a missing output, and a comparison of no
    elements at all, fail.

    :param expected: The oracle's outputs and their accumulation lengths; an output with no length takes the bench's
        default, its own size.
    :param outputs: The kernel's outputs, by name."""
    from hpcagent_bench import config
    from hpcagent_bench.frameworks.utilities import compare_arrays
    from hpcagent_bench.precision import UngradeableTolerance, accumulation_eps, tolerance_band
    from hpcagent_bench.strict_small import StrictSmall

    checks = config.current().grading.fp_checks
    gate = StrictSmall.of(checks.enabled, checks.sign_margin, checks.l2_factor)
    relative_atol = "relative_atol" in checks.enabled
    worst, failures = 0.0, []
    for name, want in expected.outputs.items():
        if name not in outputs:
            failures.append(f"{name}: missing output")
            worst = math.inf
            continue
        precision = output_precision(want)
        band = tolerance_band(precision)
        try:
            verdict = compare_arrays(
                want,
                outputs[name],
                rtol=band.rtol,
                atol=band.atol,
                accum_length=expected.lengths.get(name),
                eps_precision=accumulation_eps(precision),
                strict_small=gate,
                relative_atol=relative_atol,
            )
        except UngradeableTolerance as err:
            return Grade(ok=False, error=math.inf, ungradeable=f"{name}: {err}")
        worst = max(worst, float(verdict.error))
        if not verdict.ok:
            failures.append(f"{name}: {verdict.detail}")
    if not any(np.size(want) for want in expected.outputs.values()):
        return Grade(ok=False, error=math.inf, detail="no output element to compare")
    return Grade(ok=not failures, error=worst, detail="; ".join(failures))
