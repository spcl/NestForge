# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Session: the epoch guard on labeled calls, kernels named by name, and plain JSON-able results. The wrapped
stages have their own tests."""

import numpy as np
import dace

from nestforge.session import Session
from nestforge.stages.canonicalize import Targets
from nestforge.stages.scopes import top_level_map_entries

N = dace.symbol("N")


@dace.program
def vertical_pair(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    T = np.empty_like(A)  # transient -> the two maps fuse vertically
    for i in dace.map[0:N]:
        T[i] = A[i] + B[i]
    for i in dace.map[0:N]:
        C[i] = T[i] * 2.0


@dace.program
def live_and_transient(A: dace.float64[N], B: dace.float64[N], live_out: dace.float64[N], C: dace.float64[N]):
    T = np.empty_like(A)  # transient producer output
    for i in dace.map[0:N]:
        T[i] = A[i] + B[i]
        live_out[i] = A[i] * 3.0  # a second, non-transient producer output
    for i in dace.map[0:N]:
        C[i] = T[i] * 2.0 + live_out[i]  # consumer reads both


def make_session(tmp_path=None) -> Session:
    return Session(vertical_pair.to_sdfg(simplify=True), work_dir=str(tmp_path) if tmp_path else None)


def top_level_map_count(sdfg) -> int:
    return sum(len(top_level_map_entries(state)) for state in sdfg.all_states())


def test_a_move_bumps_the_epoch_and_its_labels_go_stale():
    s = make_session()
    (move,) = s.list_moves("map-fusion")
    assert move["epoch"] == 0

    applied = s.apply_move(move["kind"], move["labels"], move["epoch"])

    assert (applied.status, s.epoch) == ("applied", 1)
    assert s.apply_move(move["kind"], move["labels"], move["epoch"]).status == "stale"


def test_define_scopes_bumps_the_epoch_and_names_kernels_with_their_boundary_sets():
    s = make_session()
    kernels = s.define_scopes()
    assert s.epoch == 1
    assert [k["name"] for k in kernels] == ["extcall_0", "extcall_1"]
    producer = next(k for k in kernels if k["writes"] == ["T"])
    assert producer["reads"] == ["A", "B"] and producer["symbols"] == ["N"] and producer["parallel"]


def test_a_no_op_define_scopes_keeps_the_epoch_so_labels_stay_valid():
    sdfg = dace.SDFG("no_maps")
    sdfg.add_state_after(sdfg.add_state())
    session = Session(sdfg)
    assert session.define_scopes() == []
    assert session.epoch == 0


def test_canonicalize_then_default_moves_bump_the_epoch_each_time_and_reduce_top_level_maps():
    sdfg = live_and_transient.to_sdfg(simplify=True)
    frontend_maps = top_level_map_count(sdfg)
    session = Session(sdfg, targets=Targets())
    session.canonicalize()
    assert session.epoch == 1
    session.default_moves()
    assert session.epoch == 2
    assert top_level_map_count(session.sdfg) < frontend_maps


def test_metrics_answers_for_a_map_a_kernel_and_refuses_an_unknown_label():
    s = make_session()
    map_label = next(label for label, (obj, _) in s.row_index().items() if isinstance(obj, dace.nodes.MapEntry))
    assert "work=" in s.metrics(map_label) and "OI=" in s.metrics(map_label)
    producer = next(k["name"] for k in s.define_scopes() if k["writes"] == ["T"])
    assert s.metrics(producer).startswith(f"{producer}: work=N depth=")  # one add per element
    assert s.metrics("nothing_0") == "no tree row is labeled nothing_0."
