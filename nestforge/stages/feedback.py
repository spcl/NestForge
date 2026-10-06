# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage 7: a concise report for the agent. A fixed rule table turns raw evidence (validation, per-kernel times,
compiler remarks) into short hints, most important first."""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from nestforge.build.toolchain import COMPILE_TIMEOUT_S, compiler_family, openmp_compile_flags
from nestforge.stages.kernel import KernelSource, KernelVerdict

#: A kernel above this share of the program's kernel time is named as the one to work on.
HOT_SHARE = 0.5

#: Hints a report keeps.
REPORT_LINES = 8

#: Characters of an error line a hint quotes.
HINT_TEXT = 160

#: Distinct vectorizer reasons a kernel's hints quote.
VECTOR_REASONS = 2

#: Hint ranks, most important first.
FAILED, WRONG, HOT, SPILLS, NOT_VECTORIZED = range(5)

#: Vectorizer reasons, by a phrase they contain, and what to try.
VECTOR_ADVICE: tuple[tuple[str, str], ...] = (
    ("control flow", "move the condition out of the loop (interchange-loop-if) or make the body branch-free"),
    ("dependen", "fission the dependent statement into its own loop (loop-fission / map-fission)"),
    ("alias", "the kernel source can mark its pointers __restrict__"),
    ("call", "inline or replace the call in the loop body"),
    ("access", "interchange so the innermost loop walks contiguous memory"),
    ("stride", "interchange so the innermost loop walks contiguous memory"),
    ("bound", "give the loop a trip count the compiler can compute"),
)
DEFAULT_VECTOR_ADVICE = "simplify the loop body, or split it with fission"

#: A vectorizer remark that only reports the cost model's choice, which stage 6 already sweeps.
COST_MODEL_VERDICT = "not profitable"


@dataclass(frozen=True, slots=True)
class Evidence:
    """What one kernel's latest build showed."""

    kernel: str
    verdict: KernelVerdict
    remarks: str


def remark_flags(compiler: str) -> list[str]:
    """The flags that make ``compiler`` explain its vectorizer or register decisions on stderr."""
    if Path(compiler).name.startswith("nvcc"):
        return ["-Xptxas", "-v"]
    if compiler_family(compiler) == "llvm":
        return ["-Rpass-missed=loop-vectorize", "-Rpass-analysis=loop-vectorize"]
    return ["-fopt-info-vec-missed"]


def compiler_remarks(src: KernelSource, compiler: str, flags: list[str]) -> str:
    """The remarks of one extra compile of the kernel's unit, to an object that is thrown away; the build error
    text when it does not compile."""
    extra = [] if src.device == "gpu" else openmp_compile_flags(compiler)
    kept = [f for f in flags if f != "-shared"]
    with tempfile.TemporaryDirectory(prefix="nf_remarks_") as scratch:
        cmd = [compiler, *kept, *extra, *remark_flags(compiler), "-c", str(src.unit), "-o", f"{scratch}/k.o"]
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=COMPILE_TIMEOUT_S)
    return done.stderr


def vectorizer_reasons(remarks: str) -> list[str]:
    """Distinct reasons gcc or clang gave for a loop it did not vectorize; the cost model's own verdict is left to
    stage 6, which sweeps the cost model."""
    found = re.findall(r"not vectorized: ([^\[\n]+)", remarks)
    reasons = (reason.strip().rstrip(".") for reason in found)
    return list(dict.fromkeys(r for r in reasons if COST_MODEL_VERDICT not in r))


def vector_advice(reason: str) -> str:
    return next((advice for phrase, advice in VECTOR_ADVICE if phrase in reason.lower()), DEFAULT_VECTOR_ADVICE)


def spills(remarks: str) -> tuple[int, int]:
    """``(registers, spilled bytes)`` from ``ptxas -v``: the most registers any function uses, and every spill."""
    registers = [int(n) for n in re.findall(r"Used (\d+) registers", remarks)]
    spilled = sum(int(n) for n in re.findall(r"(\d+) bytes spill stores", remarks))
    return max(registers, default=0), spilled


def hints(ev: Evidence, share: float) -> list[tuple[int, str]]:
    """``(rank, text)`` for every rule ``ev`` triggers; a lower rank is more important."""
    k, verdict = ev.kernel, ev.verdict
    if verdict.error:
        return [
            (
                FAILED,
                f"{k}: failed: {verdict.error.splitlines()[0][:HINT_TEXT]} -> fix the kernel so it builds and runs.",
            )
        ]
    out: list[tuple[int, str]] = []
    if not verdict.ok:
        wrong = f"{k}: wrong result, max rel err {verdict.md_rel:.2g} at {verdict.fp_mode}"
        out.append((WRONG, f"{wrong} -> check index bounds, races and reduction order."))
    if share >= HOT_SHARE:
        out.append((HOT, f"{k} is {share:.0%} of kernel time ({verdict.time_us:.1f} us) -> optimize it first."))
    registers, spilled = spills(ev.remarks)
    if spilled:
        out.append(
            (SPILLS, f"{k}: {spilled} bytes spilled at {registers} registers -> loop body too big, try fission.")
        )
    reasons = vectorizer_reasons(ev.remarks)[:VECTOR_REASONS]
    out += [(NOT_VECTORIZED, f"{k}: loop not vectorized: {r} -> {vector_advice(r)}.") for r in reasons]
    return out


def report(evidence: list[Evidence], limit: int = REPORT_LINES) -> str:
    """The hints of every kernel, most important first, at most ``limit`` lines."""
    total = sum(ev.verdict.time_us for ev in evidence if ev.verdict.ok)
    ranked = [
        hint
        for ev in evidence
        for hint in hints(ev, ev.verdict.time_us / total if ev.verdict.ok and total > 0 and len(evidence) > 1 else 0)
    ]
    lines = [text for _, text in sorted(ranked, key=lambda hint: hint[0])[:limit]]
    times = ", ".join(f"{ev.kernel} {ev.verdict.time_us:.1f} us" for ev in evidence if ev.verdict.ok)
    head = f"kernel times: {times}" if times else "no kernel ran correctly"
    return "\n".join([head, *lines]) if lines else f"{head}\nno hints: every kernel is correct and nothing stands out."
