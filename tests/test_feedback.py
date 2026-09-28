# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage 7: the rule table that turns validation, times, compiler remarks and operational intensity into a few
hints, most important first, and the report a session builds from real builds."""

from pathlib import Path

import numpy as np
import pytest

import dace

from nestforge.session import Session
from nestforge.stages.feedback import Evidence, report, spills, vectorizer_reasons
from nestforge.stages.kernel import KernelVerdict

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


def evidence(kernel: str = "extcall_0", remarks: str = "", oi: float | None = 4.0, **kw) -> Evidence:
    producers, consumers = kw.pop("producers", ()), kw.pop("consumers", ())
    return Evidence(kernel, verdict(**kw), remarks, oi, producers, consumers)


def test_gcc_reasons_are_distinct_and_the_bare_missed_line_is_no_reason():
    assert vectorizer_reasons(GCC_REMARKS) == ["control flow in loop", "complicated access pattern"]


def test_clang_reasons_stop_before_the_remark_flag():
    assert vectorizer_reasons(CLANG_REMARKS) == ["cannot identify array bounds"]


def test_ptxas_spills_add_up_over_functions_and_report_the_most_registers():
    assert spills(PTXAS_REMARKS) == (40, 80)
    assert spills("") == (0, 0)


def test_a_wrong_result_comes_first_and_a_failed_build_before_it():
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


def test_a_kernel_with_most_of_the_time_is_named_hot_and_the_head_lists_every_time():
    text = report([evidence("extcall_0", time_us=80.0), evidence("extcall_1", time_us=20.0)])

    assert text.splitlines()[0] == "kernel times: extcall_0 80.0 us, extcall_1 20.0 us"
    assert "extcall_0 is 80% of kernel time (80.0 us) -> optimize it first." in text
    assert "extcall_1 is" not in text


def test_a_memory_bound_kernel_is_told_which_neighbours_to_fuse_with():
    text = report([evidence("extcall_1", oi=0.125, producers=("extcall_0",), consumers=("extcall_2",))])

    assert "extcall_1: OI 0.12 flop/B, memory-bound -> fuse with producer extcall_0 or consumer extcall_2." in text


def test_a_kernel_that_is_both_producer_and_consumer_is_named_once():
    """In a time loop two kernels feed each other; the hint names the partner once, not as producer and consumer."""
    text = report([evidence("extcall_0", oi=0.19, producers=("extcall_1",), consumers=("extcall_1",))])

    assert "memory-bound -> fuse with extcall_1." in text
    assert "producer" not in text and "consumer" not in text


def test_the_cost_model_verdict_is_no_vectorizer_hint():
    """ "Not profitable" is the cost model's call, which stage 6 sweeps; only a real blocker becomes a hint."""
    remarks = "k.cpp:3:5: missed: not vectorized: vectorization is not profitable.\n" + GCC_REMARKS

    assert vectorizer_reasons(remarks) == ["control flow in loop", "complicated access pattern"]


def test_register_spills_suggest_fission_and_vectorizer_reasons_carry_their_advice():
    text = report([evidence(remarks=PTXAS_REMARKS + GCC_REMARKS)])

    assert "extcall_0: 80 bytes spilled at 40 registers -> loop body too big, try fission." in text
    assert "loop not vectorized: control flow in loop -> move the condition out of the loop" in text


def test_a_report_keeps_at_most_eight_hints_ranked_by_importance():
    many = [evidence(f"extcall_{i}", md_rel=0.5, remarks=GCC_REMARKS, oi=0.1) for i in range(5)]

    lines = report(many).splitlines()

    assert len(lines) == 1 + 8
    assert all("wrong result" in line for line in lines[1:6])


def test_a_healthy_kernel_gets_no_hint():
    assert report([evidence()]).splitlines()[1] == "no hints: every kernel is correct and nothing stands out."


@dace.program
def square(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[i] * a[i]


@pytest.mark.e2e
def test_a_session_reports_the_times_and_the_intensity_of_its_built_kernels(tmp_path):
    session = Session(square.to_sdfg(simplify=True), work_dir=str(tmp_path), sizes={"N": 4096})
    (kernel,) = session.define_scopes()

    text = session.feedback(reps=2)

    lines = text.splitlines()
    assert lines[0].startswith(f"kernel times: {kernel['name']} ") and lines[0].endswith(" us")
    # one multiply per element over 16 bytes: memory-bound, with no neighbour to fuse with
    assert f"{kernel['name']}: OI 0.062 flop/B, memory-bound -> reuse loaded data, drop temporaries." in lines
    assert Path(session.kernel(kernel["name"]).lib_path).exists()
    assert np.isfinite(float(lines[0].split()[-2]))
