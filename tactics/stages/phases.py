# Copyright 2021 ETH Zurich and the Tactics authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tactics' phases, owned by Tactics (``docs/phases.md``). Phase 1, the canonical parallel form, runs once
when the session starts and has no way back. The agent then works through four phases in order:

1. ``structured``: moves on the whole program (fusion, tiling, layouts, precision of arrays, ...);
2. ``scopes``: which parts of the program become kernels, and where they run;
3. ``scoped``: moves inside the kernels (schedules, vectorization, staging, library implementations, ...);
4. ``kernels``: each kernel's code and its compiler configuration.

A packet names the phases the agent runs; every other phase runs its default when the agent passes it.
``next_phase`` leaves a phase; ``prev_phase`` takes the current phase back completely, to the program as the agent
left the phase before; ``goto_phase`` does that until the phase named."""

import enum
from collections.abc import Iterable


class PhaseName(enum.StrEnum):
    """A phase the agent can be in, in order."""

    STRUCTURED = "structured"
    SCOPES = "scopes"
    SCOPED = "scoped"
    KERNELS = "kernels"


#: The phases in order.
PHASE_ORDER: tuple[PhaseName, ...] = tuple(PhaseName)


def checked_phases(names: Iterable[str]) -> tuple[PhaseName, ...]:
    """``names`` as phases in their order; none named: every phase.

    :raises ValueError: A name is no phase."""
    wanted = {str(name) for name in names}
    unknown = sorted(wanted - {phase.value for phase in PHASE_ORDER})
    if unknown:
        raise ValueError(f"no phase {unknown}; the phases are {[p.value for p in PHASE_ORDER]}")
    return tuple(p for p in PHASE_ORDER if p.value in wanted) if wanted else PHASE_ORDER
