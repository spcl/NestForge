# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The scopes phase places kernels with moves: ``run-on-cpu`` and ``run-on-gpu`` name the kernels, each one change
``rollback`` takes back; ``run-on-cpu-persistent`` merges consecutive kernels into one and runs it as one persistent
kernel, which computes what the separate kernels did."""

import dace
import numpy as np
from dace.libraries.standard.nodes.external_call import external_calls

from nestforge.records import MoveStatus
from nestforge.session import CheckData, Session
from nestforge.stages.canonicalize import Targets
from nestforge.stages.performance import Reference

N = dace.symbol("N", dtype=dace.int64)
SIZE = 32
STEPS = 3


@dace.program
def two_time_loops(A: dace.float64[N], B: dace.float64[N]):
    for _t in range(STEPS):
        for i in dace.map[0:N]:
            A[i] = A[i] * 0.5 + 1.0
    for _t in range(STEPS):
        for i in dace.map[0:N]:
            B[i] = B[i] * 0.5 + A[i]


@dace.program
def kernels_with_work_between(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for _t in range(STEPS):
        for i in dace.map[0:N]:
            A[i] = A[i] * 0.5 + 1.0
    for i in dace.map[0:N]:
        C[i] = A[i] + 2.0
    for _t in range(STEPS):
        for i in dace.map[0:N]:
            B[i] = B[i] * 0.5 + C[i]


def scoped(program: dace.frontend.python.parser.DaceProgram, tmp_path, gpu: bool = False) -> tuple[Session, list[str]]:
    """A session with each top-level time loop its own kernel, and the kernels' names in program order."""
    sut = Session(program.to_sdfg(simplify=True), Targets(gpu=gpu), work_dir=str(tmp_path), sizes={"N": SIZE})
    sut.canonicalize()
    sut.describe()
    loops = [label for label in sut.row_index() if label.startswith("for")]
    names = [str(sut.set_scope([loop]).reason) for loop in loops if sut.row_index().get(loop)]
    return sut, names


def expected() -> CheckData:
    a, b = np.linspace(0.0, 1.0, SIZE), np.linspace(1.0, 2.0, SIZE)
    want_a, want_b = a.copy(), b.copy()
    for _ in range(STEPS):
        want_a = want_a * 0.5 + 1.0
    for _ in range(STEPS):
        want_b = want_b * 0.5 + want_a
    return CheckData({"A": a, "B": b}, Reference({"A": want_a, "B": want_b}, 1e-12, 1e-12), {"N": SIZE})


def test_run_on_gpu_places_the_kernels_named_and_rollback_takes_the_placement_back(tmp_path):
    sut, (first, second) = scoped(two_time_loops, tmp_path, gpu=True)

    done = sut.apply_move("run-on-gpu", [second])
    placed = sut.devices()
    sut.apply_move("rollback", [])

    assert done.status is MoveStatus.APPLIED
    assert placed == {first: "cpu", second: "gpu"}, "the kernel not named keeps its default device"
    assert sut.unplaced is None, "rollback takes the placement back"


def test_run_on_gpu_is_refused_without_a_gpu_target(tmp_path):
    sut, (first, _) = scoped(two_time_loops, tmp_path)

    done = sut.apply_move("run-on-gpu", [first])

    assert done.status is MoveStatus.ILLEGAL and "no GPU target" in str(done.reason)


def test_consecutive_kernels_merge_into_one_persistent_kernel_that_computes_what_they_did(tmp_path):
    sut, names = scoped(two_time_loops, tmp_path)
    offered = [m.labels for m in sut.list_scope_moves("run-on-cpu-persistent")]

    done = sut.apply_move("run-on-cpu-persistent", names)

    (merged,) = (ext.name for ext in external_calls(sut.sdfg))
    assert tuple(names) in offered, "the run of consecutive kernels is offered"
    assert done.status is MoveStatus.APPLIED, done.reason
    assert len({block for kernel, block in sut.persistent if kernel == merged}) == 2, "both time loops in one run"
    assert sut.check_numerics(expected()) == ""


def test_kernels_with_a_doall_between_them_are_not_merged(tmp_path):
    sut, names = scoped(kernels_with_work_between, tmp_path)
    tree = sut.describe()

    done = sut.apply_move("run-on-cpu-persistent", names)

    assert done.status is MoveStatus.ILLEGAL and "no consecutive kernels" in str(done.reason)
    assert sut.describe() == tree and len(external_calls(sut.sdfg)) == 2
