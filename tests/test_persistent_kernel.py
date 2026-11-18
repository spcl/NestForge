# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``persistent-kernel``: a time loop of doalls inside a kernel runs as one persistent kernel (DaCe's MegakernelLoops). On the
GPU that is one launch instead of one per doall and step; on the CPU, where code generation already runs the loop in
one OpenMP team, its doalls share the work without an implicit barrier each and synchronize explicitly."""

import re

import dace
import numpy as np
from dace.libraries.standard.nodes.external_call import external_calls

from nestforge.records import MoveStatus, Refusal
from nestforge.session import Session
from nestforge.stages.performance import Reference

N = dace.symbol("N", dtype=dace.int64)
T = dace.symbol("T", dtype=dace.int64)
SIZE, STEPS = 64, 5


@dace.program
def steps(a: dace.float64[N], b: dace.float64[N]):
    for _ in range(T):
        for i in dace.map[1 : N - 1]:
            b[i] = (a[i - 1] + a[i] + a[i + 1]) / 3.0
        for i in dace.map[1 : N - 1]:
            a[i] = b[i]


@dace.program
def two_loops(a: dace.float64[N], b: dace.float64[N]):
    for _ in range(T):
        for i in dace.map[1 : N - 1]:
            b[i] = (a[i - 1] + a[i] + a[i + 1]) / 3.0
        for i in dace.map[1 : N - 1]:
            a[i] = b[i]
    for _ in range(N):  # another trip count than the first loop's, so canonicalization keeps the two loops apart
        for i in dace.map[1 : N - 1]:
            b[i] = (a[i - 1] + a[i + 1]) / 2.0
        for i in dace.map[1 : N - 1]:
            a[i] = b[i]


def offered(sut: Session, kernel: str) -> list[str]:
    """The for loops of ``kernel`` that list_moves offers to persistent-kernel, in tree order."""
    return [move.labels[0] for move in sut.list_moves("persistent-kernel", scope=kernel)]


def scoped(tmp_path) -> tuple[Session, str]:
    """The program with its time loop as one kernel."""
    sut = Session(steps.to_sdfg(simplify=True), work_dir=str(tmp_path), sizes={"N": SIZE, "T": STEPS})
    sut.canonicalize()
    (loop,) = [label for label in sut.row_index() if label.startswith("for")]
    assert sut.set_scope([loop]).status is MoveStatus.APPLIED
    (ext,) = external_calls(sut.sdfg)
    return sut, ext.name


def parallel_regions(source: str) -> int:
    return len(re.findall(r"#pragma omp parallel\b", source))


def data() -> tuple[dict[str, np.ndarray], Reference]:
    a = np.linspace(0.0, 1.0, SIZE)
    want = a.copy()
    for _ in range(STEPS):
        b = want.copy()
        b[1:-1] = (want[:-2] + want[1:-1] + want[2:]) / 3.0
        want[1:-1] = b[1:-1]
    return {"a": a, "b": np.zeros(SIZE)}, Reference({"a": want}, 1e-12, 1e-12)


def test_the_time_loop_runs_in_one_parallel_region_and_computes_the_same(tmp_path):
    sut, kernel = scoped(tmp_path)
    before = sut.default_kernel(kernel).unit.read_text()

    (loop,) = offered(sut, kernel)
    done = sut.apply_move("persistent-kernel", [loop], scope=kernel)

    after = sut.default_kernel(kernel).unit.read_text()
    assert done.status is MoveStatus.APPLIED
    assert parallel_regions(before) == parallel_regions(after) == 1
    assert "nowait" not in before and "#pragma omp barrier" not in before
    assert after.count("nowait") == 2 and "#pragma omp barrier" in after, "each doall shares work, then synchronizes"
    assert sut.performance(*data()).structured.correct is True
    sut.expand([label for label in sut.row_index() if label.startswith("state")])
    (row,) = [line for line in sut.describe().splitlines() if f"{kernel}  LIBNODE" in line]
    assert row.endswith(f"persistent: {loop}"), "the tree names the loop that runs persistent"
    assert offered(sut, kernel) == [], "a persistent loop is not offered again"
    sut.apply_move("rollback", [])
    assert sut.persistent == frozenset()


def test_only_the_named_loop_runs_persistent(tmp_path):
    sut = Session(two_loops.to_sdfg(simplify=True), work_dir=str(tmp_path), sizes={"N": SIZE, "T": STEPS})
    sut.canonicalize()
    loops = [label for label in sut.row_index() if label.startswith("for")]
    assert len(loops) == 2 and sut.set_scope(loops).status is MoveStatus.APPLIED
    (ext,) = external_calls(sut.sdfg)
    first, second = offered(sut, ext.name)

    done = sut.apply_move("persistent-kernel", [second], scope=ext.name)

    source = sut.default_kernel(ext.name).unit.read_text()
    assert done.status is MoveStatus.APPLIED
    assert sut.persistent == frozenset({(ext.name, second)})
    assert source.count("nowait") == 2, "only the second loop's two doalls share work without an implicit barrier"
    assert offered(sut, ext.name) == [first]


def test_a_doall_and_a_kernel_without_a_loop_of_kernels_are_refused(tmp_path):
    sut = Session(steps.to_sdfg(simplify=True), work_dir=str(tmp_path), sizes={"N": SIZE, "T": STEPS})
    sut.canonicalize()
    sut.default_scopes()
    kernel = next(ext for ext in external_calls(sut.sdfg) if ext.standalone_sdfg is not None)
    doall = next(
        entry.map.label
        for entry, _ in kernel.standalone_sdfg.all_nodes_recursive()
        if isinstance(entry, dace.nodes.MapEntry)
    )

    refused = sut.apply_move("persistent-kernel", [doall], scope=kernel.name)
    unknown = sut.apply_move("persistent-kernel", [doall], scope="no_such_kernel")

    assert offered(sut, kernel.name) == []
    assert (refused.status, refused.refusal) == (MoveStatus.ILLEGAL, Refusal.WRONG_ROW_KIND)
    assert (unknown.status, unknown.refusal) == (MoveStatus.ILLEGAL, Refusal.ARGUMENT)
