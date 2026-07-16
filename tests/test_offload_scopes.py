"""Whole-program offload analysis (:mod:`nestforge.whole_program.offload_scopes`): externalize each nest
into a call, then put each call in its OWN state -- the scope between the call and the host program, where
offload is decided per-call (the externalize-before-offload invariant). Structural checks are unit tests;
one integration test runs the transformed program to prove externalize + isolate stayed value-preserving.
"""
import pathlib

import numpy as np
import pytest

import dace

from nestforge.libnode import ExternalCall
from nestforge.whole_program import OffloadScope, default_offloadable, offload_scopes

N = dace.symbol("N")


@dace.program
def two_nest(a: dace.float64[N], b: dace.float64[N], out: dace.float64[N]):
    tmp = np.empty_like(a)
    for i in dace.map[0:N]:
        tmp[i] = a[i] * 2.0
    for i in dace.map[0:N]:
        out[i] = tmp[i] + b[i]


def _sdfg():
    return two_nest.to_sdfg(simplify=True)


def _calls(sdfg):
    return [n for s in sdfg.states() for n in s.nodes() if isinstance(n, ExternalCall)]


def test_each_nest_becomes_an_offload_scope():
    _, scopes = offload_scopes(_sdfg())
    assert len(scopes) == 2  # two_nest has two compute nests
    assert all(isinstance(s, OffloadScope) for s in scopes)
    assert all(s.offloadable and s.reason == "" for s in scopes)  # default: an externalized nest may offload


def test_each_call_is_isolated_in_its_own_state():
    work, scopes = offload_scopes(_sdfg())
    calls = _calls(work)
    assert len(calls) == 2
    # every externalized call sits ALONE among the nodes-with-compute of its state: the scope between the
    # call and the host program. No two calls share a state.
    call_states = [next(s for s in work.states() if c in s.nodes()) for c in calls]
    assert len(set(id(s) for s in call_states)) == 2, "two calls landed in the same state -- not isolated"
    for state, call in zip(call_states, calls):
        assert sum(isinstance(n, ExternalCall) for n in state.nodes()) == 1


def test_scope_boundary_is_the_host_device_transfer_set():
    _, scopes = offload_scopes(_sdfg())
    # producer nest: reads a, writes the tmp bridge; consumer nest: reads that bridge + b, writes out.
    producer = next(s for s in scopes if "out" not in s.outputs)
    consumer = next(s for s in scopes if "out" in s.outputs)
    assert "a" in producer.inputs
    assert "b" in consumer.inputs and "out" in consumer.outputs


def test_analysis_is_non_destructive():
    src = _sdfg()
    before = (len(list(src.states())), sum(len(s.nodes()) for s in src.states()))
    offload_scopes(src)
    after = (len(list(src.states())), sum(len(s.nodes()) for s in src.states()))
    assert before == after, "offload_scopes mutated its input SDFG"
    assert not _calls(src), "offload_scopes externalized on the caller's SDFG instead of a copy"


def test_offloadable_decision_is_injectable():
    # the per-call verdict is a TOOL's, injected -- the invariant is that each tool decides independently.
    def refuse_all(call, boundary):
        return False, "tool says not GPU-viable"

    _, scopes = offload_scopes(_sdfg(), offloadable=refuse_all)
    assert all(not s.offloadable and s.reason == "tool says not GPU-viable" for s in scopes)


def test_default_offloadable_accepts():
    ok, reason = default_offloadable(None, None)
    assert ok and reason == ""


@pytest.mark.integration  # runs the externalized+isolated program (DaceReference expansion compiles)
def test_externalized_isolated_program_is_value_preserving(tmp_path):
    n = 32
    rng = np.random.default_rng(0)
    a, b = rng.random(n), rng.random(n)

    ref_out = np.zeros(n)
    _sdfg()(a=a.copy(), b=b.copy(), out=ref_out, N=n)  # reference: the un-externalized program

    work, scopes = offload_scopes(_sdfg())
    assert len(scopes) == 2
    got_out = np.zeros(n)
    work(a=a.copy(), b=b.copy(), out=got_out, N=n)  # externalized + isolated, still runnable
    assert np.allclose(got_out, ref_out), "externalize + isolate changed the program value"


def test_value_preserving_run_is_named_by_a_ci_step_that_keeps_integration_tests():
    # The run-and-compare above is the ONLY proof the offload analysis preserves program value, and it is
    # integration-marked -- so it runs in CI only if a step names this file WITHOUT deselecting integration.
    # A CI that never names it lets a value-corrupting change ship green; this unit test is that guarantee.
    ci = pathlib.Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
    naming = [line for line in ci.read_text().splitlines() if "tests/test_offload_scopes.py" in line]
    assert naming, "no CI step names tests/test_offload_scopes.py -- its value-preservation run never executes"
    assert all("not integration" not in line for line in naming), "CI names this file only where integration is off"
