# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regressions from a review of the stages: the snapshot copy of a slice that starts past zero, the snapshot probe
that must validate what it made, untile-for that must not report a change it did not make, the whole-program timer
that must start every timed call from the same inputs, and a remark compile that times out."""

import functools
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

import dace
import numpy as np
import pytest
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion

from nestforge.records import MoveStatus
from nestforge.session import Session
from nestforge.stages import feedback, performance, snapshot
from nestforge.stages.kernel import KernelSource
from nestforge.stages.moves import Rewrite, plan_move
from nestforge.stages.performance import ProgramCall, run_program

N = dace.symbol("N", dtype=dace.int64)
SIZE = 21


def offset_reader() -> tuple[dace.SDFG, dace.SDFGState, nodes.MapEntry]:
    """A doall writing column 0 of ``A`` beside a doall over ``1:N-1`` reading column 2 of ``A``."""
    sdfg = dace.SDFG("offset_reader")
    sdfg.add_array("A", [N, 4], dace.float64)
    sdfg.add_array("B", [N], dace.float64)
    state = sdfg.add_state()
    a = state.add_access("A")
    state.add_mapped_tasklet(
        "w",
        {"j": "0:N"},
        {"x": Memlet("A[j, 1]")},
        "y = x + 1",
        {"y": Memlet("A[j, 0]")},
        input_nodes={"A": a},
        external_edges=True,
    )
    _, entry, _ = state.add_mapped_tasklet(
        "r",
        {"i": "1:N-1"},
        {"x": Memlet("A[i, 2]")},
        "y = x",
        {"y": Memlet("B[i]")},
        input_nodes={"A": a},
        external_edges=True,
    )
    sdfg.validate()
    return sdfg, state, entry


def test_a_snapshot_of_a_slice_starting_past_zero_stays_in_bounds_and_copies_the_read_values():
    sdfg, state, entry = offset_reader()
    rng = np.random.default_rng(0)
    a = rng.random((SIZE, 4))
    expected = np.zeros(SIZE)
    expected[1:-1] = a[1:-1, 2]

    assert snapshot.snapshot_applies(state, entry)
    assert snapshot.snapshot(state, entry)

    sdfg.validate()
    (copy,) = [n for n in state.nodes() if isinstance(n, nodes.MapEntry) and n.map.label.startswith("snapshot_A")]
    assert copy.map.range.ranges == [(0, N - 3, 1)], "the copy walks the transient from zero"
    got = np.zeros(SIZE)
    sdfg(A=a, B=got, N=SIZE)
    np.testing.assert_array_equal(got, expected)


def test_the_snapshot_probe_refuses_a_result_that_does_not_validate(monkeypatch):
    _, state, entry = offset_reader()
    made = snapshot.snapshot

    def invalid(twin_state, twin_entry, written=None):
        applied = made(twin_state, twin_entry, written)
        twin_state.add_node(nodes.AccessNode("undeclared"))
        return applied

    monkeypatch.setattr(snapshot, "snapshot", invalid)

    assert not snapshot.snapshot_applies(state, entry)


@dace.program
def running_sum(a: dace.float64[N], b: dace.float64[N]):
    for i in range(1, N):
        a[i] = a[i - 1] + b[i]


def test_untile_for_raises_rather_than_report_a_change_it_did_not_make(tmp_path):
    sut = Session(running_sum.to_sdfg(simplify=True), work_dir=str(tmp_path), sizes={"N": SIZE})
    (tile,) = sut.list_moves("tile-for")
    assert sut.apply_move("tile-for", tile.labels, tile_sizes=[8]).status is MoveStatus.APPLIED
    (tiles,) = [
        r
        for r in sut.sdfg.all_control_flow_regions(recursive=True)
        if isinstance(r, LoopRegion) and any(isinstance(b, LoopRegion) for b in r.nodes())
    ]
    move = plan_move("untile-for", [(tiles, None)])
    assert isinstance(move, Rewrite), move
    move.commit()

    with pytest.raises(ValueError, match="UntileLoops refused"):
        move.commit()


class DoublingProgram:
    """Stands in for a built program that doubles ``A`` in place; records what every call starts from."""

    seen: list[np.ndarray] = []

    def __init__(self, *args: object) -> None:
        pass

    def init(self, sizes: object) -> None:
        pass

    def program(self, buffers: dict[str, np.ndarray], sizes: object) -> None:
        DoublingProgram.seen.append(buffers["A"].copy())
        buffers["A"] *= 2

    def bind(self, buffers: dict[str, np.ndarray], sizes: object) -> Callable[[], None]:
        return functools.partial(self.program, buffers, sizes)

    def close(self) -> None:
        pass


def test_every_timed_call_of_the_program_starts_from_the_inputs_the_checked_call_saw(monkeypatch):
    monkeypatch.setattr(performance, "BuiltSDFG", DoublingProgram)
    monkeypatch.setattr(performance.ctypes, "CDLL", lambda path: None)
    monkeypatch.setattr(performance, "signature", lambda source, name: "")
    monkeypatch.setattr(performance, "parse_params", lambda sig: [])
    DoublingProgram.seen = []
    inputs = np.arange(4, dtype=np.float64)

    timed = run_program(ProgramCall("x.so", "", "p", {"A": inputs.copy()}, {}, 3, None))

    assert timed.time_us > 0 and len(DoublingProgram.seen) == 4
    for start in DoublingProgram.seen:
        np.testing.assert_array_equal(start, inputs)


def test_a_remark_compile_that_times_out_is_its_remark(monkeypatch):
    def hang(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])

    monkeypatch.setattr(feedback.subprocess, "run", hang)
    with tempfile.TemporaryDirectory() as scratch:
        unit = Path(scratch) / "k.cpp"
        src = KernelSource("k", unit, [], None, "cpu")  # type: ignore[arg-type]
        remarks = feedback.compiler_remarks(src, "g++", ["-O3"])

    assert "timed out" in remarks and "k.cpp" in remarks


if __name__ == "__main__":
    pytest.main([__file__])
