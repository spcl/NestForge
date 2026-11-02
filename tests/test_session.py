# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Session: labels that stay on their rows and retire with them, kernels named by name, and plain JSON-able results. The wrapped
stages have their own tests."""

import time

import dace
import numpy as np

from nestforge.ir import introspect
from nestforge.ir.names import top_level_map_entries
from nestforge.records import MoveStatus
from nestforge.session import Session
from nestforge.stages.canonicalize import Targets

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


@dace.program
def three_maps(A: dace.float64[N], C: dace.float64[N], D: dace.float64[N]):
    T = np.empty_like(A)
    for i in dace.map[0:N]:
        T[i] = A[i] + 1.0
    for i in dace.map[0:N]:
        C[i] = T[i] * 2.0
    for i in dace.map[0:N]:
        D[i] = A[i] - 1.0


def map_labels(session: Session) -> list[str]:
    return [label for label, (obj, _) in session.row_index().items() if isinstance(obj, dace.nodes.MapEntry)]


def make_session(tmp_path=None) -> Session:
    return Session(vertical_pair.to_sdfg(simplify=True), work_dir=str(tmp_path) if tmp_path else None)


def top_level_map_count(sdfg) -> int:
    return sum(len(top_level_map_entries(state)) for state in sdfg.states())


def test_a_move_retires_the_labels_it_took_and_names_the_fused_row_afresh():
    s = make_session()
    (move,) = s.list_moves("doall-fusion")

    applied = s.apply_move(move.kind, move.labels)
    again = s.apply_move(move.kind, move.labels)

    assert applied.status is MoveStatus.APPLIED
    assert (again.status, again.reason) == (
        MoveStatus.NOT_FOUND,
        f"no tree row is labeled {', '.join(move.labels)} ({', '.join(move.labels)} changed; read the tree again).",
    )
    (fused,) = [label for label, (obj, _) in s.row_index().items() if isinstance(obj, dace.nodes.MapEntry)]
    assert fused not in move.labels


def test_a_row_no_move_touched_keeps_its_label():
    s = Session(three_maps.to_sdfg(simplify=True))
    first, second, third = map_labels(s)

    s.apply_move("doall-fusion", [first, second])

    assert third in map_labels(s)


def test_default_scopes_names_kernels_with_their_boundary_sets():
    s = make_session()
    kernels = s.default_scopes()
    assert [k.name for k in kernels] == ["extcall_0", "extcall_1"]
    producer = next(k for k in kernels if k.writes == ("T",))
    assert producer.reads == ("A", "B") and producer.symbols == ("N",) and producer.parallel


def test_a_no_op_default_scopes_keeps_every_label():
    sdfg = dace.SDFG("no_maps")
    sdfg.add_state_after(sdfg.add_state())
    session = Session(sdfg)
    before = list(session.row_index())
    assert session.default_scopes() == []
    assert list(session.row_index()) == before


def test_canonicalize_then_default_moves_reduce_top_level_maps():
    sdfg = live_and_transient.to_sdfg(simplify=True)
    frontend_maps = top_level_map_count(sdfg)
    session = Session(sdfg, targets=Targets())
    session.canonicalize()
    session.default_moves()
    assert top_level_map_count(session.sdfg) < frontend_maps


def test_metrics_answers_for_a_map_a_kernel_and_refuses_an_unknown_label():
    s = make_session()
    map_label = next(label for label, (obj, _) in s.row_index().items() if isinstance(obj, dace.nodes.MapEntry))
    assert "work=" in s.metrics(map_label) and "depth=" in s.metrics(map_label)
    producer = next(k.name for k in s.default_scopes() if k.writes == ("T",))
    assert s.metrics(producer).startswith(f"{producer}: work=N depth=")  # one add per element
    assert s.metrics("nothing_0") == "no tree row is labeled nothing_0."


def test_describe_prints_the_tree_without_costs_when_work_depth_fails(monkeypatch):
    # spcl/NestForge#28: DaCe's work-depth pass raises on some programs; the tree, and the move, must still come back
    def fail(*args, **kwargs):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(introspect.work_depth, "analyze_sdfg", fail)
    s = make_session()
    tree = s.describe()
    assert tree.splitlines()[1] == (
        "(no work= or depth=: work-depth analysis failed: RecursionError: maximum recursion depth exceeded)"
    )
    assert "work=" not in "\n".join(tree.splitlines()[2:])
    (move,) = s.list_moves("doall-fusion")
    assert s.apply_move(move.kind, move.labels).status is MoveStatus.APPLIED
    assert "no work= or depth=" in s.describe().splitlines()[1]
    assert s.costs == {} and s.costs_skipped is not None


def test_describe_skips_costs_when_work_depth_outlasts_its_budget(monkeypatch):
    monkeypatch.setattr(introspect, "WORK_DEPTH_BUDGET_S", 1)
    monkeypatch.setattr(introspect.work_depth, "analyze_sdfg", lambda *args, **kwargs: time.sleep(5))
    s = make_session()
    assert "work-depth analysis took over 1 s" in s.describe().splitlines()[1]


def test_costs_come_back_after_the_next_change_once_work_depth_works(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("The argument 'nan' is not comparable.")

    s = make_session()
    with monkeypatch.context() as patch:
        patch.setattr(introspect.work_depth, "analyze_sdfg", fail)
        assert s.describe().splitlines()[1].startswith("(no work= or depth=")
    (move,) = s.list_moves("doall-fusion")
    result = s.apply_move(move.kind, move.labels)
    assert result.status is MoveStatus.APPLIED
    assert s.costs_skipped is None and "no work= or depth=" not in s.describe()
