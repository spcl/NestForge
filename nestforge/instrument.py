# Insert clock tasklets around a nest so it can be timed IN SITU, while the whole program runs.
"""Bracket a nest with timing tasklets.

Timing a nest by running it standalone answers a different question than timing it inside the
program it belongs to: standalone loses the cache state, the page mapping and the surrounding
threads. This module measures the nest *where it lives* -- one clock read before it, one after --
so the whole program still runs and only the nest is attributed.

Two things make this sound rather than decorative:

* **Dependency edges.** A tasklet with no data dependence on the nest is free to be scheduled
  anywhere, so a clock read could sink past the code it is supposed to bracket. Every timer is
  wired to the nest with an explicit edge, so the ordering is a property of the graph rather than
  an accident of emission order.
* **The timers are not transients.** A transient scalar nothing reads is dead code, and
  simplification is entitled to delete it -- which would silently produce a build that measures
  nothing and reports zero. Making them SDFG arguments keeps them alive by construction and lets
  the harness read the result.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import dace
from dace import dtypes
from dace.sdfg import SDFG, SDFGState
from dace.sdfg.state import ControlFlowRegion, LoopRegion

#: Monotonic clock, nanoseconds. steady_clock cannot jump backwards the way system_clock can.
CLOCK_READ = ('__out = static_cast<unsigned long long>('
              'std::chrono::steady_clock::now().time_since_epoch().count());')

#: Emitted once per instrumented SDFG so the tasklets above compile.
CLOCK_INCLUDE = '#include <chrono>'


@dataclass(frozen=True)
class NestTimers:
    """Handles to an instrumented nest.

    :param start: name of the scalar holding the clock read taken before the nest.
    :param stop: name of the scalar holding the clock read taken after it.
    :param start_state: the state performing the first read.
    :param stop_state: the state performing the second.
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


def instrument_nest(sdfg: SDFG, nest, name: Optional[str] = None) -> NestTimers:
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
        # A length-1 ARRAY, not a Scalar: a Scalar argument is passed by value, so the clock reading
        # would be written and then dropped on the way out. The repo's ABI convention is Scalar for
        # by-value inputs, length-1 array for anything the caller reads back.
        # NOT transient either: a transient nobody reads is dead code and simplify may delete it,
        # leaving a build that silently measures nothing and reports zero.
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
