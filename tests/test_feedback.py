# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage 7: the rule table that turns validation, times and compiler remarks into a few
hints, most important first, and the report a session builds from real builds."""

from pathlib import Path
from typing import Any

import dace
import numpy as np
import pytest

from tactics.session import Session
from tactics.stages.feedback import Evidence, Origin, report, spills, vectorizer_reasons
from tactics.stages.kernel import KernelVerdict, computed_symbols_reason, unchecked_verdict
from tactics.stages.performance import kernel_reports

N = dace.symbol("N", dtype=dace.int64)

GCC_REMARKS = """k.cpp:12:23: missed: couldn't vectorize loop
k.cpp:12:23: missed: not vectorized: control flow in loop.
k.cpp:20:5: missed: not vectorized: complicated access pattern.
k.cpp:12:23: missed: not vectorized: control flow in loop.
"""
CLANG_REMARKS = (
    "k.cpp:9:3: remark: loop not vectorized: cannot identify array bounds [-Rpass-analysis=loop-vectorize]\n"
)
PTXAS_REMARKS = """ptxas info    : Used 24 registers, 368 bytes cmem[0]
ptxas info    : Function properties for k
    96 bytes stack frame, 64 bytes spill stores, 64 bytes spill loads
ptxas info    : Used 40 registers, 368 bytes cmem[0]
    0 bytes stack frame, 16 bytes spill stores, 16 bytes spill loads
"""


def verdict(time_us: float = 10.0, md_rel: float = 0.0, error: str = "") -> KernelVerdict:
    return KernelVerdict("strict-ieee", md_rel, md_rel, 2.3e-16, time_us, error)


def evidence(kernel: str = "extcall_0", remarks: str = "", **kw: Any) -> Evidence:
    return Evidence(kernel, verdict(**kw), remarks)


def test_gcc_reasons_are_distinct_and_the_bare_missed_line_is_no_reason() -> None:
    assert vectorizer_reasons(GCC_REMARKS) == ["control flow in loop", "complicated access pattern"]


def test_clang_reasons_stop_before_the_remark_flag() -> None:
    assert vectorizer_reasons(CLANG_REMARKS) == ["cannot identify array bounds"]


def test_ptxas_spills_add_up_over_functions_and_report_the_most_registers() -> None:
    assert spills(PTXAS_REMARKS) == (40, 80)
    assert spills("") == (0, 0)


def test_a_wrong_result_comes_first_and_a_failed_build_before_it() -> None:
    lines = report(
        [
            evidence("extcall_0", remarks=GCC_REMARKS),
            evidence("extcall_1", md_rel=0.25),
            evidence("extcall_2", error="RuntimeError: command failed: g++ -O3\nk.cpp:1: error"),
        ]
    ).splitlines()

    assert (
        lines[1] == "extcall_2: failed: RuntimeError: command failed: g++ -O3 -> fix the kernel so it builds and runs."
    )
    assert lines[2].startswith("extcall_1: wrong result, max rel err 0.25 at strict-ieee ->")


def test_a_kernel_with_most_of_the_time_is_named_hot_and_the_head_lists_every_time() -> None:
    text = report([evidence("extcall_0", time_us=80.0), evidence("extcall_1", time_us=20.0)])

    assert text.splitlines()[0] == "kernel times: extcall_0 80.0 us, extcall_1 20.0 us"
    assert "extcall_0 is 80% of kernel time (80.0 us) -> optimize it first." in text
    assert "extcall_1 is" not in text


def test_the_cost_model_verdict_is_no_vectorizer_hint() -> None:
    """ "Not profitable" is the cost model's call, which stage 6 sweeps; only a real blocker becomes a hint."""
    remarks = "k.cpp:3:5: missed: not vectorized: vectorization is not profitable.\n" + GCC_REMARKS

    assert vectorizer_reasons(remarks) == ["control flow in loop", "complicated access pattern"]


def test_register_spills_suggest_fission_and_vectorizer_reasons_carry_their_advice() -> None:
    text = report([evidence(remarks=PTXAS_REMARKS + GCC_REMARKS)])

    assert "extcall_0: 80 bytes spilled at 40 registers -> loop body too big, try fission." in text
    assert "loop not vectorized: control flow in loop -> move the condition out of the loop" in text


def test_a_report_keeps_at_most_eight_hints_ranked_by_importance() -> None:
    many = [evidence(f"extcall_{i}", md_rel=0.5, remarks=GCC_REMARKS) for i in range(5)]

    lines = report(many).splitlines()

    assert len(lines) == 1 + 8
    assert all("wrong result" in line for line in lines[1:6])


def test_a_kernel_needing_a_value_the_program_computes_is_unchecked_not_wrong() -> None:
    """CloudSC's ``jmelt`` is read from an array the program fills, so no standalone run of its kernel exists."""
    ev = Evidence("extcall_7", unchecked_verdict("strict-ieee", computed_symbols_reason(["jmelt"])), "")

    (kernel,) = kernel_reports([ev], {"extcall_7": "cpu"})

    assert (kernel.correct, kernel.time_us) == (None, None) and "jmelt" in kernel.error
    assert "not checked alone" in report([ev]) and "wrong" not in report([ev]) and "failed" not in report([ev])


def test_a_healthy_kernel_gets_no_hint() -> None:
    assert report([evidence()]).splitlines()[1] == "no hints: every kernel is correct and nothing stands out."


def test_a_kernel_cut_from_a_row_is_named_by_its_label_with_the_moves_offered_on_it() -> None:
    """Issue #88: the agent reads one line per row in its own words, the hottest first; a split piece keeps its
    row's label and a kernel no row names gets no line."""
    gemv = Origin("lib_blas0", "opaque library call (Gemv)", ("inline-library", "set-library-implementation"))
    origins = {"lib_blas0": gemv, "doall1": Origin("doall1", "map", ())}
    kernels = [evidence("extcall_doall1_0", time_us=10.0), evidence("extcall_lib_blas0", time_us=30.0)]

    lines = report([*kernels, evidence("extcall_7", time_us=0.0)], origins).splitlines()

    gemv_line = "lib_blas0 is 75% of kernel time, opaque library call (Gemv); moves offered on it: inline-library, "
    assert lines[2:4] == [
        gemv_line + "set-library-implementation.",
        "doall1 is 25% of kernel time, map; moves offered on it: none.",
    ], lines
    assert len(lines) == 4, "the head, the hot kernel's line and one line per named row"


@dace.program
def square(a: dace.float64[N], b: dace.float64[N]) -> None:
    for i in dace.map[0:N]:
        b[i] = a[i] * a[i]


@pytest.mark.e2e
def test_a_session_reports_the_times_of_its_built_kernels(tmp_path: Path) -> None:
    session = Session(square.to_sdfg(simplify=True), work_dir=str(tmp_path), sizes={"N": 4096})
    (kernel,) = session.default_scopes()

    text = session.feedback(reps=2)

    lines = text.splitlines()
    assert lines[0].startswith(f"kernel times: {kernel.name} ") and lines[0].endswith(" us")
    assert Path(session.kernel(kernel.name).lib_path).exists()
    assert np.isfinite(float(lines[0].split()[-2]))


@pytest.mark.e2e
def test_get_performance_names_the_row_a_kernel_came_from(tmp_path: Path) -> None:
    session = Session(square.to_sdfg(simplify=True), work_dir=str(tmp_path), sizes={"N": 4096})
    (label,) = (label for label in session.row_index() if label.startswith("doall"))
    offered = sorted({move.kind for move in session.list_moves() if label in move.labels})
    a = np.random.default_rng(0).random(4096)

    hints = session.performance({"a": a, "b": np.zeros(4096)}, reps=1).hints

    assert f"{label} is 100% of kernel time, map; moves offered on it: {', '.join(offered) or 'none'}." in hints


def test_a_failed_kernel_names_its_row_and_the_change_that_came_last() -> None:
    """Issue #88 item 2: after expanding only lib_blas0, the agent reads which move broke which row."""
    gemv = Origin("lib_blas0", "opaque library call (Gemv)", ("inline-library",))
    broken = evidence("extcall_lib_blas0", error="KeyError: '_out_gemv'\ntraceback ...")

    text = report([broken, evidence("extcall_doall1")], {"lib_blas0": gemv}, last="inline-library on lib_blas0")

    first = text.splitlines()[1]
    assert first.startswith(
        "extcall_lib_blas0 (cut from lib_blas0, opaque library call (Gemv)): failed after inline-library on lib_blas0: "
        "KeyError: '_out_gemv'"
    ), first
    assert "failed:" in report([broken]).splitlines()[1], "without a row or a change the hint reads as before"
