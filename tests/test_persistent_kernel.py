# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``persistent-kernel``: a kernel's time loop of doalls runs as one persistent kernel (DaCe's MegakernelLoops). On the
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

    done = sut.apply_move("persistent-kernel", [], scope=kernel)

    after = sut.default_kernel(kernel).unit.read_text()
    assert done.status is MoveStatus.APPLIED
    assert parallel_regions(before) == parallel_regions(after) == 1
    assert "nowait" not in before and "#pragma omp barrier" not in before
    assert after.count("nowait") == 2 and "#pragma omp barrier" in after, "each doall shares work, then synchronizes"
    assert sut.performance(*data()).structured.correct is True
    sut.apply_move("rollback", [])
    assert sut.persistent == frozenset()


def test_a_kernel_without_a_loop_of_kernels_is_refused(tmp_path):
    sut = Session(steps.to_sdfg(simplify=True), work_dir=str(tmp_path), sizes={"N": SIZE, "T": STEPS})
    sut.canonicalize()
    sut.default_scopes()
    names = [ext.name for ext in external_calls(sut.sdfg)]

    refused = sut.apply_move("persistent-kernel", [], scope=names[0])
    unknown = sut.apply_move("persistent-kernel", [], scope="no_such_kernel")

    assert (refused.status, refused.refusal) == (MoveStatus.ILLEGAL, Refusal.WRONG_ROW_KIND)
    assert (unknown.status, unknown.refusal) == (MoveStatus.ILLEGAL, Refusal.ARGUMENT)
