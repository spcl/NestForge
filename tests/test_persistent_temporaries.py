# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Session.persist_temporaries: a large temporary of the program is allocated once at init, not on every call, and the
program computes the same (NestForge#63)."""

import copy
import os

import dace
import numpy as np

from nestforge.build.sdfg import make_persistent
from nestforge.records import MoveStatus
from nestforge.session import Session

N = dace.symbol("N", dtype=dace.int64)
SIZE = 4096
TEMP_BYTES = SIZE * 8


@dace.program
def through_temporary(A: dace.float64[N], B: dace.float64[N]):
    tmp = dace.define_local([N], dace.float64)
    for i in dace.map[0:N]:
        tmp[i] = A[i] * 2.0
    for i in dace.map[0:N]:
        B[i] = tmp[i] + 1.0


def fresh_session(sizes: dict[str, int] | None) -> tuple[Session, dace.SDFG]:
    sdfg = through_temporary.to_sdfg(simplify=True)
    sdfg.name = f"{sdfg.name}_{os.getpid()}"  # one build folder per xdist worker
    reference = copy.deepcopy(sdfg)
    reference.name = f"{sdfg.name}_reference"
    return Session(sdfg, sizes=sizes), reference


def lifetimes(sdfg: dace.SDFG) -> dict[str, dace.AllocationLifetime]:
    """The lifetime of each large transient (the program's other transients are scalars)."""
    return {
        name: desc.lifetime
        for name, desc in sdfg.arrays.items()
        if desc.transient and desc.total_size == N and desc.shape == (N,)
    }


def test_large_temporary_becomes_persistent() -> None:
    session, _ = fresh_session({"N": SIZE})
    assert lifetimes(session.sdfg), "the program has no temporary of N doubles"
    assert set(lifetimes(session.sdfg).values()) == {dace.AllocationLifetime.Scope}
    done = session.persist_temporaries(min_bytes=TEMP_BYTES)
    assert done.status is MoveStatus.APPLIED
    assert done.kind == "persist-temporaries"
    assert set(lifetimes(session.sdfg).values()) == {dace.AllocationLifetime.Persistent}
    assert done.reason == ", ".join(f"{session.sdfg.label}.{name}" for name in lifetimes(session.sdfg))
    assert session.applied[-1] is done


def test_below_the_threshold_nothing_changes() -> None:
    session, _ = fresh_session({"N": SIZE})
    done = session.persist_temporaries(min_bytes=TEMP_BYTES + 1)
    assert done.status is MoveStatus.ILLEGAL
    assert set(lifetimes(session.sdfg).values()) == {dace.AllocationLifetime.Scope}
    assert session.applied == []


def test_without_sizes_nothing_changes() -> None:
    session, _ = fresh_session(None)
    done = session.persist_temporaries()
    assert done.status is MoveStatus.ILLEGAL
    assert "sizes" in done.reason
    assert set(lifetimes(session.sdfg).values()) == {dace.AllocationLifetime.Scope}


def test_a_symbol_the_sizes_lack_is_left_alone() -> None:
    session, _ = fresh_session({"M": 1})
    assert make_persistent(session.sdfg, {"M": 1}, 1) == []


def init_section(sdfg: dace.SDFG) -> str:
    """The generated C++ of ``__dace_init_*``, up to ``__dace_exit_*``."""
    code = "\n".join(obj.clean_code for obj in sdfg.generate_code() if obj.language == "cpp")
    init = code[code.index("__dace_init_") :]
    return init[: init.index("__dace_exit_")]


def test_allocated_in_init_and_values_match_across_calls() -> None:
    session, reference = fresh_session({"N": SIZE})
    (name,) = lifetimes(session.sdfg)
    assert name not in init_section(session.sdfg)
    session.persist_temporaries(min_bytes=TEMP_BYTES)
    assert name in init_section(session.sdfg)
    rng = np.random.default_rng(0)
    for _ in range(2):  # the second call runs on the memory the first left
        a = rng.random(SIZE)
        got, want = np.zeros(SIZE), np.zeros(SIZE)
        session.sdfg(A=a, B=got, N=SIZE)
        reference(A=a, B=want, N=SIZE)
        np.testing.assert_allclose(got, want, rtol=1e-14)
