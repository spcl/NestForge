"""Bracket a nest with timing tasklets, so it is timed IN SITU while the whole program runs.

Standalone timing loses the cache state, page mapping and surrounding threads, answering a different
question. Two soundness requirements: timers are wired to the nest by explicit dependency edges (an
unwired clock read may be scheduled past the code it brackets), and they are SDFG arguments rather
than transients (a transient nobody reads is dead code simplify may delete, leaving a build that
measures nothing and reports zero).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import dace
from dace import dtypes
from dace.sdfg import SDFG, SDFGState
from dace.sdfg.state import ControlFlowRegion

#: Monotonic clock, nanoseconds. steady_clock cannot jump backwards the way system_clock can.
CLOCK_READ = ('__out = static_cast<unsigned long long>('
              'std::chrono::steady_clock::now().time_since_epoch().count());')

CLOCK_INCLUDE = '#include <chrono>'


@dataclass(frozen=True)
class NestTimers:
    """Handles to an instrumented nest.

    :param start: scalar holding the clock read taken before the nest.
    :param stop: scalar holding the clock read taken after it.
    :param start_state: state performing the first read.
    :param stop_state: state performing the second.
    """
    start: str
    stop: str
    start_state: SDFGState
    stop_state: SDFGState

    def elapsed_ns(self, results: dict) -> int:
        """Nanoseconds spent in the bracketed nest, from a run's returned arrays."""
        return int(results[self.stop][0]) - int(results[self.start][0])


def clock_tasklet(state: SDFGState, sdfg: SDFG, scalar: str, label: str) -> None:
    """Add a C++ clock read writing into ``scalar`` within ``state``."""
    tasklet = state.add_tasklet(label, {}, {'__out'}, CLOCK_READ, language=dtypes.Language.CPP)
    write = state.add_write(scalar)
    state.add_edge(tasklet, '__out', write, None, dace.Memlet.simple(scalar, '0'))


def instrument_nest(sdfg: SDFG, nest: Union[SDFGState, ControlFlowRegion], name: Optional[str] = None) -> NestTimers:
    """Bracket ``nest`` with a clock read before and after, wired by dependency edges.

    :param sdfg: the SDFG owning ``nest``.
    :param nest: the ``SDFGState`` or ``LoopRegion`` to time.
    :param name: suffix for the generated scalars; defaults to the nest's label.
    :returns: the :class:`NestTimers` handles.
    :raises TypeError: if ``nest`` is not a state or a control-flow region.
    """
    if not isinstance(nest, (SDFGState, ControlFlowRegion)):
        raise TypeError(f'can only instrument a state or a control-flow region, got {type(nest).__name__}')

    parent = nest.parent_graph
    if parent is None:
        raise ValueError('nest has no parent graph to insert timing states into')

    suffix = name or nest.label
    start, stop = f'__nf_t0_{suffix}', f'__nf_t1_{suffix}'
    for scalar in (start, stop):
        # Length-1 array, not Scalar: a Scalar arg is by-value, so the reading would be dropped on the
        # way out. Non-transient so simplify cannot delete it as dead.
        sdfg.add_array(scalar, [1], dace.uint64, transient=False)

    if CLOCK_INCLUDE not in sdfg.global_code.get('frame', dace.properties.CodeBlock('')).as_string:
        sdfg.append_global_code(CLOCK_INCLUDE + '\n')

    start_state = parent.add_state_before(nest, label=f'__nf_start_{suffix}')
    stop_state = parent.add_state_after(nest, label=f'__nf_stop_{suffix}')
    clock_tasklet(start_state, sdfg, start, f'nf_clock_start_{suffix}')
    clock_tasklet(stop_state, sdfg, stop, f'nf_clock_stop_{suffix}')

    return NestTimers(start=start, stop=stop, start_state=start_state, stop_state=stop_state)


def timer_scalars(sdfg: SDFG) -> Tuple[str, ...]:
    """Every timing scalar this module has added to ``sdfg``, in sorted order."""
    return tuple(sorted(n for n in sdfg.arrays if n.startswith(('__nf_t0_', '__nf_t1_'))))


def is_instrumented(sdfg: SDFG) -> bool:
    """Whether ``sdfg`` carries any timing instrumentation."""
    return bool(timer_scalars(sdfg))
