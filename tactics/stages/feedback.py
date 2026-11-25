# Copyright 2021 ETH Zurich and the Tactics authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage 7: a concise report for the agent. A fixed rule table turns raw evidence (validation, per-kernel times,
compiler remarks) into short hints, most important first."""

from __future__ import annotations

import re
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from tactics.build.isolation import sealed_argv
from tactics.build.toolchain import COMPILE_TIMEOUT_S, compiler_family, openmp_compile_flags
from tactics.stages.kernel import KernelSource, KernelVerdict

#: A kernel above this share of the program's kernel time is named as the one to work on.
HOT_SHARE = 0.5

#: Hints a report keeps.
REPORT_LINES = 8

#: Characters of an error line a hint quotes.
HINT_TEXT = 160

#: Distinct vectorizer reasons a kernel's hints quote.
VECTOR_REASONS = 2

#: Hint ranks, most important first.
FAILED, WRONG, HOT, ORIGIN, SPILLS, NOT_VECTORIZED, UNCHECKED = range(7)

#: The prefix stage 3 puts before a row's label to name the kernel cut from it (``scopes.row_kernel_names``).
KERNEL_PREFIX = "extcall_"

#: Vectorizer reasons, by a phrase they contain, and what to try.
VECTOR_ADVICE: tuple[tuple[str, str], ...] = (
    ("control flow", "move the condition out of the loop (hoist-invariant-guard) or make the body branch-free"),
    ("dependen", "fission the dependent statement into its own loop (for-fission / doall-fission)"),
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


@dataclass(frozen=True, slots=True)
class Origin:
    """The tree row a kernel was cut from, in the agent's words (issue #88): its label, what it is, and the kinds of
    the moves offered on it."""

    label: str
    kind: str
    moves: tuple[str, ...]


def kernel_label(kernel: str, labels: Mapping[str, object]) -> str | None:
    """The label of the row ``kernel`` was cut from, among ``labels``; ``None`` for a kernel no row names. A split
    piece's name carries a ``_<n>`` suffix after the label."""
    if not kernel.startswith(KERNEL_PREFIX):
        return None
    base = kernel.removeprefix(KERNEL_PREFIX)
    return next((label for label in (base, base.rpartition("_")[0]) if label in labels), None)


def origin_line(origin: Origin, share: float) -> str:
    offered = ", ".join(origin.moves) if origin.moves else "none"
    return f"{origin.label} is {share:.0%} of kernel time, {origin.kind}; moves offered on it: {offered}."


def remark_flags(compiler: str) -> list[str]:
    """The flags that make ``compiler`` explain its vectorizer or register decisions on stderr."""
    if Path(compiler).name.startswith("nvcc"):
        return ["-Xptxas", "-v"]
    if compiler_family(compiler) == "llvm":
        return ["-Rpass-missed=loop-vectorize", "-Rpass-analysis=loop-vectorize"]
    return ["-fopt-info-vec-missed"]


def compiler_remarks(src: KernelSource, compiler: str, flags: list[str]) -> str:
    """The remarks of one extra compile of each of the kernel's units, to objects that are thrown away; the build
    error text when one does not compile. A GPU kernel's register use and spills are in its device unit."""
    extra = [] if src.device == "gpu" else openmp_compile_flags(compiler)
    kept = [f for f in flags if f != "-shared"]
    with tempfile.TemporaryDirectory(prefix="nf_remarks_") as scratch:
        cmds = [
            [compiler, *kept, *extra, *remark_flags(compiler), "-c", str(unit), "-o", f"{scratch}/{unit.stem}.o"]
            for unit in src.units
        ]
        try:
            done = [
                subprocess.run(sealed_argv(cmd), capture_output=True, text=True, timeout=COMPILE_TIMEOUT_S)
                for cmd in cmds
            ]
        except subprocess.TimeoutExpired as err:
            return str(err)
    return "".join(d.stderr for d in done)


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
    if verdict.unchecked:
        return [(UNCHECKED, f"{k}: not checked alone: {verdict.unchecked}.")]
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


def failure_attributed(text: str, kernel: str, origin: Origin | None, last: str) -> str:
    """A failed kernel's hint with the row it was cut from and the change that came last (issue #88 item 2), so the
    agent knows which move to take back and which row to look at."""
    where = f" (cut from {origin.label}, {origin.kind})" if origin is not None else ""
    after = f" after {last}" if last else ""
    return text.replace(f"{kernel}: failed:", f"{kernel}{where}: failed{after}:", 1)


def report(
    evidence: list[Evidence],
    origins: Mapping[str, Origin] | None = None,
    limit: int = REPORT_LINES,
    last: str = "",
) -> str:
    """The hints of every kernel, most important first, at most ``limit`` lines. ``origins`` holds the rows of the
    program by label; a kernel cut from one gets a line in its words, and a failed one names its row and ``last``,
    the change applied last."""
    origins = origins if origins is not None else {}
    total = sum(ev.verdict.time_us for ev in evidence if ev.verdict.ok)
    ranked: list[tuple[int, float, str]] = []
    for ev in evidence:
        portion = ev.verdict.time_us / total if ev.verdict.ok and total > 0 else 0.0
        # one kernel is never the hot one among others
        label = kernel_label(ev.kernel, origins)
        origin = origins[label] if label is not None else None
        for rank, text in hints(ev, portion if len(evidence) > 1 else 0):
            shown = failure_attributed(text, ev.kernel, origin, last) if rank == FAILED else text
            ranked.append((rank, -portion, shown))
        if origin is not None and portion > 0:
            ranked.append((ORIGIN, -portion, origin_line(origin, portion)))
    # within a rank the larger share first, so the hottest rows lead
    lines = [text for _, _, text in sorted(ranked, key=lambda hint: hint[:2])[:limit]]
    times = ", ".join(f"{ev.kernel} {ev.verdict.time_us:.1f} us" for ev in evidence if ev.verdict.ok)
    head = f"kernel times: {times}" if times else "no kernel ran correctly"
    return "\n".join([head, *lines]) if lines else f"{head}\nno hints: every kernel is correct and nothing stands out."
