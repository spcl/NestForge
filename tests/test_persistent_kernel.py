# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``persistent-kernel``: a time loop of doalls inside a kernel runs as one persistent kernel (DaCe's MegakernelLoops). On the
GPU that is one launch instead of one per doall and step; on the CPU, where code generation already runs the loop in
one OpenMP team, its doalls share the work without an implicit barrier each and synchronize explicitly."""

import re
from pathlib import Path

import dace
import numpy as np
from dace.libraries.standard.nodes.external_call import external_calls

from tactics.records import MoveStatus, Refusal
from tactics.server import Setup, started_session
from tactics.session import Session
from tactics.stages.performance import Reference

N = dace.symbol("N", dtype=dace.int64)
T = dace.symbol("T", dtype=dace.int64)
SIZE, STEPS = 64, 5


@dace.program
def steps(a: dace.float64[N], b: dace.float64[N]) -> None:
    for _ in range(T):
        for i in dace.map[1 : N - 1]:
            b[i] = (a[i - 1] + a[i] + a[i + 1]) / 3.0
        for i in dace.map[1 : N - 1]:
            a[i] = b[i]


@dace.program
def two_loops(a: dace.float64[N], b: dace.float64[N]) -> None:
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


@dace.program
def steps_with_if(a: dace.float64[N], b: dace.float64[N]) -> None:
    for t in range(T):
        if t > 0:
            for i in dace.map[0:N]:
                b[i] = a[i] + b[i]
            for i in dace.map[0:N]:
                a[i] = b[i] * 0.5
        else:
            for i in dace.map[0:N]:
                b[i] = a[i]


def offered(sut: Session, kernel: str) -> list[tuple[str, ...]]:
    """The rows of ``kernel`` that list_moves offers to persistent-kernel, one tuple per offer."""
    return [move.labels for move in sut.list_moves("persistent-kernel", scope=kernel)]


def two_loops_scoped(tmp_path: Path) -> tuple[Session, str, str, str]:
    """``two_loops`` with both time loops in one kernel: the session, the kernel, the first and the second loop."""
    sut = Session(two_loops.to_sdfg(simplify=True), work_dir=str(tmp_path), sizes={"N": SIZE, "T": STEPS})
    sut.canonicalize()
    loops = [label for label in sut.row_index() if label.startswith("for")]
    assert len(loops) == 2 and sut.set_scope(loops).status is MoveStatus.APPLIED
    (ext,) = external_calls(sut.sdfg)
    first, second = sorted(run[0] for run in offered(sut, ext.name) if len(run) == 1)
    return sut, ext.name, first, second


def two_loops_data() -> tuple[dict[str, np.ndarray], Reference]:
    a = np.linspace(0.0, 1.0, SIZE)
    want = a.copy()
    for _ in range(STEPS):
        want[1:-1] = (want[:-2] + want[1:-1] + want[2:]) / 3.0
    for _ in range(SIZE):
        want[1:-1] = (want[:-2] + want[2:]) / 2.0
    return {"a": a, "b": np.zeros(SIZE)}, Reference({"a": want}, 1e-12, 1e-12)


def scoped(tmp_path: Path) -> tuple[Session, str]:
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


def test_the_time_loop_runs_in_one_parallel_region_and_computes_the_same(tmp_path: Path) -> None:
    sut, kernel = scoped(tmp_path)
    before = sut.default_kernel(kernel).unit.read_text()

    ((loop,),) = offered(sut, kernel)
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
    assert offered(sut, kernel) == [], "a kernel holds one persistent run"
    sut.apply_move("rollback", [])
    assert sut.persistent == frozenset()


def test_only_the_named_loop_runs_persistent(tmp_path: Path) -> None:
    sut, kernel, first, second = two_loops_scoped(tmp_path)
    assert (first, second) in offered(sut, kernel), "the two consecutive loops are offered as one run"

    done = sut.apply_move("persistent-kernel", [second], scope=kernel)

    source = sut.default_kernel(kernel).unit.read_text()
    assert done.status is MoveStatus.APPLIED
    assert sut.persistent == frozenset({(kernel, second)})
    assert source.count("nowait") == 2, "only the second loop's two doalls share work without an implicit barrier"


def test_two_consecutive_loops_run_as_one_persistent_kernel_and_compute_the_same(tmp_path: Path) -> None:
    sut, kernel, first, second = two_loops_scoped(tmp_path)

    done = sut.apply_move("persistent-kernel", [first, second], scope=kernel)

    source = sut.default_kernel(kernel).unit.read_text()
    assert done.status is MoveStatus.APPLIED
    assert sut.persistent == frozenset({(kernel, first), (kernel, second)})
    assert parallel_regions(source) == 1 and source.count("nowait") == 4, "one team runs both loops' four doalls"
    assert sut.performance(*two_loops_data()).structured.correct is True


def test_rows_of_two_regions_a_second_run_and_an_unknown_kernel_are_refused(tmp_path: Path) -> None:
    sut, kernel, first, second = two_loops_scoped(tmp_path)
    body = sut.kernel(kernel).standalone_sdfg
    inner = next(
        entry.map.label
        for entry, _ in next(
            r for r in body.all_control_flow_regions(recursive=True) if r.label == first
        ).all_nodes_recursive()
        if isinstance(entry, dace.nodes.MapEntry)
    )

    split = sut.apply_move("persistent-kernel", [inner, second], scope=kernel)
    unknown = sut.apply_move("persistent-kernel", [second], scope="no_such_kernel")
    sut.apply_move("persistent-kernel", [second], scope=kernel)
    again = sut.apply_move("persistent-kernel", [first], scope=kernel)

    assert (split.status, split.refusal) == (MoveStatus.ILLEGAL, Refusal.WRONG_ROW_KIND)
    assert (unknown.status, unknown.refusal) == (MoveStatus.ILLEGAL, Refusal.ARGUMENT)
    assert (again.status, again.refusal) == (MoveStatus.ILLEGAL, Refusal.ARGUMENT)


def test_a_persistent_gpu_time_loop_of_2d_doalls_renders_as_one_megakernel(tmp_path: Path) -> None:
    """spcl/tactics#113: ``jacobi_2d``'s time loop on the GPU, persistent: its 2-D doalls take the megakernel's 1-D
    block, so the GPU tiling makes no 16x16 thread blocks that conflict with the megakernel's launch."""
    sut = started_session(Setup("jacobi_2d", "S", "float64", True, str(tmp_path)), codegen="first")
    sut.run_defaults("moves")
    kernel = sut.set_scope(["for0"]).reason
    sut.run_defaults("scopes")

    done = sut.apply_move("run-on-gpu-persistent", [kernel])
    sut.finalize_scope()
    src = sut.default_kernel(kernel)

    assert done.status is MoveStatus.APPLIED, done.reason
    assert src.device == "gpu" and src.device_unit is not None
    assert len(re.findall(r"__global__[^;{]*run_megakernel", src.device_unit.read_text())) == 1


def test_a_kernel_whose_time_loop_holds_an_if_lists_its_moves(tmp_path: Path) -> None:
    """spcl/tactics#110: the walk over the kernel's regions yields the conditional, which has no start block."""
    sut = Session(steps_with_if.to_sdfg(simplify=True), work_dir=str(tmp_path), sizes={"N": SIZE, "T": STEPS})
    sut.canonicalize()
    (loop,) = [label for label in sut.row_index() if label.startswith("for")]
    assert sut.set_scope([loop]).status is MoveStatus.APPLIED
    (ext,) = external_calls(sut.sdfg)

    assert offered(sut, ext.name) == [(loop,)], "the time loop holding the if is offered whole"
    assert sut.list_scope_moves(scope=ext.name), "the kernel's scope moves are listed"
    assert sut.list_scope_moves(), "and so are every kernel's"
