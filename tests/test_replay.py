# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A rebuilt session repeats a written kernel and a sweep as the journal records them: the source and the winner are
built and linked again, nothing is checked or timed, so a check near its tolerance cannot end the episode."""

from pathlib import Path

import dace
import numpy as np
import pytest

from nestforge import session as session_module
from nestforge.records import KernelStatus
from nestforge.server import answer, applied
from nestforge.session import REPEATED, Session
from nestforge.stages.kernel import failed_verdict

N = dace.symbol("N", dtype=dace.int64)
SIZE = 64
NO_DATA: tuple[dict[str, np.ndarray], None] = ({}, None)

pytestmark = pytest.mark.e2e


@dace.program
def scaled_sum(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        c[i] = 2.0 * a[i] + b[i]


def one_kernel_session(tmp_path: Path) -> tuple[Session, str]:
    sut = Session(scaled_sum.to_sdfg(simplify=True), work_dir=str(tmp_path), sizes={"N": SIZE})
    (kernel,) = sut.default_scopes()
    sut.focus_kernel(kernel.name)
    return sut, kernel.name


def never(*args: object, **kwargs: object) -> None:
    raise AssertionError("a repeat checks or times nothing")


def test_a_journaled_kernel_source_is_linked_again_even_where_its_check_would_now_fail(tmp_path, monkeypatch):
    sut, name = one_kernel_session(tmp_path)
    source = "// edited by hand\n" + sut.kernel_source(name)
    monkeypatch.setattr(session_module, "measure_kernel", lambda *args: failed_verdict("strict", "flaky"))
    repeat = {"name": name, "source": source, "language": "cpp", "checked": False}

    reply = answer(sut, "write_kernel", repeat, NO_DATA, timed=False)

    assert applied(reply), reply
    assert (reply.result.status, reply.result.reason) == (KernelStatus.UNCHECKED, REPEATED)
    assert sut.kernel_source(name) == source
    assert [move.kind for move in sut.applied] == ["write-kernel"]


def test_a_journaled_sweep_links_its_winner_again_without_timing_a_variant(tmp_path, monkeypatch):
    live, name = one_kernel_session(tmp_path / "live")
    winner = live.sweep(name).winner
    assert winner is not None
    sut, _ = one_kernel_session(tmp_path / "rebuilt")
    monkeypatch.setattr(session_module, "select_variant", never)
    monkeypatch.setattr(session_module, "measure_kernel", never)

    reply = answer(sut, "tune_kernel", {"name": name, "variant": winner}, NO_DATA, timed=False)

    assert applied(reply), reply
    assert reply.result.winner == winner
    assert sut.builds[name].variant.label == winner
    assert sut.builds[name].verdict.unchecked == REPEATED
