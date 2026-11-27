# Copyright 2021 ETH Zurich and the Tactics authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A kernel whose temporary is sized by a value it reads while it runs (``lavamd``'s ``neighbor_counts[l]``): stage 3
flags it instead of refusing the program, it has no NumPy oracle, and placement keeps it on the CPU."""

import dace
import numpy as np
import pytest

from tactics.corpus.translate import prepare, runtime_sized_scratch
from tactics.ir.emit_python import RuntimeSizedScratch, oracle_sdfg
from tactics.records import MoveStatus
from tactics.session import Session
from tactics.stages.canonicalize import Targets
from tactics.stages.scopes import node_boundary

N = dace.symbol("N", dtype=dace.int64)
M = dace.symbol("M", dtype=dace.int64)


@dace.program
def ragged_rows(counts: dace.int32[N], x: dace.float64[N, M], y: dace.float64[N]) -> None:
    for i in range(N):
        c = int(counts[i])
        row = np.empty(1 + c, dtype=np.float64)
        row[0] = 1.0
        row[1:] = x[i, :c]
        y[i] = np.sum(row)


def scoped(gpu: bool) -> tuple[Session, list[str]]:
    """``ragged_rows`` through stages 1-3; the names of the kernels ``default_scopes`` flagged."""
    session = Session(ragged_rows.to_sdfg(), targets=Targets(gpu=gpu), sizes={"N": 8, "M": 4})
    session.canonicalize()
    session.default_moves()
    flagged = [k.name for k in session.default_scopes() if k.runtime_sized]
    return session, flagged


# Reproducer: default_scopes raised UnsupportedNest on a scratch buffer sized by data read in the kernel (#121, lavamd)
def test_a_temporary_sized_by_data_read_in_the_kernel_is_flagged_not_refused() -> None:
    session, flagged = scoped(gpu=False)
    assert len(flagged) == 1
    ext = session.kernel(flagged[0])
    assert ext.numpy_source == ""
    boundary = node_boundary(ext)
    with pytest.raises(RuntimeSizedScratch) as err:
        oracle_sdfg(boundary)
    assert err.value.arrays == runtime_sized_scratch(boundary)
    assert err.value.arrays
    prep = prepare(boundary, ext.name)
    assert prep.numpy_source == ""
    assert prep.runtime_sized == err.value.arrays
    assert "while it runs" in prep.unspelled
    assert not set(prep.args) & set(err.value.arrays)  # the kernel allocates its own: no scratch argument


def test_a_kernel_sizing_a_temporary_while_it_runs_stays_on_the_cpu() -> None:
    session, flagged = scoped(gpu=True)
    (name,) = flagged
    placed = session.place()
    assert placed.status is MoveStatus.APPLIED
    assert placed.devices[name] == "cpu"
    forced = session.place({name: "gpu"})
    assert forced.status is MoveStatus.ILLEGAL
    assert "while it runs" in forced.reason
