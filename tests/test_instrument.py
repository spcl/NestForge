# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
# Tests for nestforge.instrument: timing tasklets bracketing a nest, ordered by real graph edges.
import numpy as np
import pytest

import dace
from dace.sdfg import SDFGState
from dace.sdfg.state import LoopRegion

from nestforge.instrument import (CLOCK_INCLUDE, NestTimers, instrument_nest, is_instrumented, timer_scalars)

N = dace.symbol('N')


def build_sdfg():
    """A one-nest program built through the FRONTEND, not hand-assembled.

    Hand-built fixtures do not have the shape a frontend emits and hide whole bug classes, so the
    instrumentation is exercised against a real one.
    """

    @dace.program
    def scale(a: dace.float64[N], b: dace.float64[N]):
        for i in dace.map[0:N]:
            b[i] = a[i] * 2.0

    return scale.to_sdfg(simplify=False)


def nest_state(sdfg):
    return next(s for s in sdfg.states() if any(isinstance(n, dace.nodes.MapEntry) for n in s.nodes()))


def test_instrument_adds_exactly_two_scalars():
    sdfg = build_sdfg()
    assert not is_instrumented(sdfg)
    instrument_nest(sdfg, nest_state(sdfg), name='k')
    assert timer_scalars(sdfg) == ('__nf_t0_k', '__nf_t1_k')
    assert is_instrumented(sdfg)


def test_timer_scalars_are_not_transient():
    """A transient nobody reads is dead code; simplify would delete it and we would measure nothing."""
    sdfg = build_sdfg()
    timers = instrument_nest(sdfg, nest_state(sdfg), name='k')
    for scalar in (timers.start, timers.stop):
        assert not sdfg.arrays[scalar].transient


def test_timers_survive_simplify():
    """The regression this guards: instrumentation silently optimized away, reporting zero."""
    sdfg = build_sdfg()
    instrument_nest(sdfg, nest_state(sdfg), name='k')
    sdfg.simplify()
    assert timer_scalars(sdfg) == ('__nf_t0_k', '__nf_t1_k')


def test_start_precedes_nest_precedes_stop():
    """Ordering must be a property of the graph, not of emission order."""
    import networkx as nx

    sdfg = build_sdfg()
    target = nest_state(sdfg)
    timers = instrument_nest(sdfg, target, name='k')
    graph = sdfg.nx
    assert nx.has_path(graph, timers.start_state, target), 'start timer does not reach the nest'
    assert nx.has_path(graph, target, timers.stop_state), 'nest does not reach the stop timer'
    assert not nx.has_path(graph, timers.stop_state, timers.start_state), 'timers are ordered backwards'


def test_timer_states_are_connected_by_edges():
    """Each timer state must be wired into the CFG, not left floating."""
    sdfg = build_sdfg()
    timers = instrument_nest(sdfg, nest_state(sdfg), name='k')
    assert sdfg.out_degree(timers.start_state) > 0
    assert sdfg.in_degree(timers.stop_state) > 0


def test_each_timer_state_writes_its_scalar():
    sdfg = build_sdfg()
    timers = instrument_nest(sdfg, nest_state(sdfg), name='k')
    for state, scalar in ((timers.start_state, timers.start), (timers.stop_state, timers.stop)):
        tasklets = [n for n in state.nodes() if isinstance(n, dace.nodes.Tasklet)]
        writes = [n.data for n in state.nodes() if isinstance(n, dace.nodes.AccessNode)]
        assert len(tasklets) == 1
        assert scalar in writes
        assert state.out_degree(tasklets[0]) == 1, 'the clock read must feed its scalar by an edge'


def test_clock_tasklet_is_cpp():
    """A Python tasklet cannot read a C++ steady_clock."""
    sdfg = build_sdfg()
    timers = instrument_nest(sdfg, nest_state(sdfg), name='k')
    tasklet = next(n for n in timers.start_state.nodes() if isinstance(n, dace.nodes.Tasklet))
    assert tasklet.code.language == dace.dtypes.Language.CPP


def test_chrono_header_is_emitted_once():
    sdfg = build_sdfg()
    instrument_nest(sdfg, nest_state(sdfg), name='a')
    instrument_nest(sdfg, nest_state(sdfg), name='b')
    joined = ''.join(c.as_string for c in sdfg.global_code.values())
    assert joined.count(CLOCK_INCLUDE) == 1


def test_instrumented_sdfg_validates():
    sdfg = build_sdfg()
    instrument_nest(sdfg, nest_state(sdfg), name='k')
    sdfg.validate()


def test_two_nests_get_independent_timers():
    sdfg = build_sdfg()
    state = nest_state(sdfg)
    instrument_nest(sdfg, state, name='one')
    instrument_nest(sdfg, state, name='two')
    assert timer_scalars(sdfg) == ('__nf_t0_one', '__nf_t0_two', '__nf_t1_one', '__nf_t1_two')


def test_elapsed_ns_subtracts():
    timers = NestTimers('t0', 't1', None, None)
    assert timers.elapsed_ns({'t0': np.array([100], np.uint64), 't1': np.array([175], np.uint64)}) == 75


def test_rejects_a_non_nest():
    sdfg = build_sdfg()
    with pytest.raises(TypeError, match='state or a control-flow region'):
        instrument_nest(sdfg, 'not a nest', name='k')


def test_instruments_a_loop_region():
    """The other nest shape: a sequential LoopRegion rather than a map state."""

    @dace.program
    def carried(a: dace.float64[N]):
        for i in range(1, N):
            a[i] = a[i - 1] + 1.0

    sdfg = carried.to_sdfg(simplify=False)
    loop = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, LoopRegion))
    timers = instrument_nest(sdfg, loop, name='loop')
    sdfg.validate()
    assert timer_scalars(sdfg) == ('__nf_t0_loop', '__nf_t1_loop')
    assert isinstance(timers.start_state, SDFGState)


@pytest.mark.e2e
def test_instrumented_kernel_still_computes_the_right_answer():
    """Instrumentation must not perturb the result -- and must report a plausible duration."""
    sdfg = build_sdfg()
    timers = instrument_nest(sdfg, nest_state(sdfg), name='k')
    sdfg.simplify()

    n = 256
    a = np.random.default_rng(0).random(n)
    b = np.zeros(n)
    t0 = np.zeros(1, np.uint64)
    t1 = np.zeros(1, np.uint64)
    sdfg(a=a, b=b, N=n, **{timers.start: t0, timers.stop: t1})

    assert np.allclose(b, a * 2.0), 'timing tasklets changed the computed result'
    assert timers.elapsed_ns({timers.start: t0, timers.stop: t1}) > 0, 'clock did not advance'
