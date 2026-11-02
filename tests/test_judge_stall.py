# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""What keeps a dozen agents moving on one judge: a library call keeps its implementation from the first call to the
oracle, the oracle's cost is bounded, a long call's work survives its stop, and feedback rebuilds only what changed."""

import os
import re
import signal
import time
from pathlib import Path

import dace
import numpy as np
import pytest
from dace.libraries.standard.nodes.external_call import external_calls
from dace.sdfg import nodes

from nestforge.build import arena
from nestforge.build.arena import bounded_oracle, make_inputs, reps_within_budget, run_oracle
from nestforge.build.isolation import Failure, Result, run_isolated
from nestforge.corpus.translate import Prepared, prepare
from nestforge.ir.emit_python import oracle_sdfg
from nestforge.ir.extract import Boundary
from nestforge.records import KernelOutcomeStatus, MoveStatus
from nestforge.session import Session, Stopped, Stopper
from nestforge.stages import kernel, variants
from nestforge.stages.kernel import KernelSource, Probe, Workload, bounded_probe, build_kernel_library, measure_kernel
from nestforge.stages.scopes import node_boundary

N = dace.symbol("N", dtype=dace.int64)
SIZE = 64
SMALL = 8


@dace.program
def factor(a: dace.float64[N, N], b: dace.float64[N, N]):
    b[:] = np.linalg.cholesky(a)


@dace.program
def product_total(a: dace.float64[N, N], c: dace.float64[N, N], total: dace.float64[1]):
    c[:] = a @ a
    total[0] = dace.reduce(lambda x, y: x + y, c)


@dace.program
def scaled(a: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        c[i] = 2.0 * a[i]


@dace.program
def two_loops(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
    for i in dace.map[0:N]:
        c[i] = 2.0 * a[i]
    for i in dace.map[0:N]:
        d[i] = 3.0 * b[i]


def session_of(program: dace.frontend.python.parser.DaceProgram, tmp_path: Path, sizes: int = SIZE) -> Session:
    session = Session(program.to_sdfg(simplify=True), work_dir=str(tmp_path), sizes={"N": sizes})
    session.canonicalize()
    return session


def library_rows(session: Session) -> list[str]:
    session.expand(re.findall(r"\b(state\d+)\b", session.describe()))
    return [line for line in session.describe().splitlines() if "LIBNODE" in line]


def only_kernel(session: Session) -> str:
    (ext,) = external_calls(session.sdfg)
    return ext.name


def test_every_library_call_has_the_fastest_implementation_from_the_start_and_the_tree_prints_it(tmp_path):
    sut = session_of(factor, tmp_path)

    (row,) = library_rows(sut)
    (move,) = list(sut.list_moves("set-library-implementation"))

    assert "LIBNODE Cholesky [OpenBLAS]" in row, row
    assert (move.implementation, move.current) == ("pure", "OpenBLAS"), "the current one is shown and not offered"


def test_pure_comes_only_from_the_agent_and_the_tree_says_so(tmp_path):
    sut = session_of(factor, tmp_path)
    label = sut.list_moves("set-library-implementation")[0].labels

    moved = sut.apply_move("set-library-implementation", label, implementation="pure", current="OpenBLAS")

    assert moved.status is MoveStatus.APPLIED, moved
    assert "[pure]" in library_rows(sut)[0]


def test_a_move_listed_before_the_implementation_changed_is_refused_and_changes_nothing(tmp_path):
    sut = session_of(factor, tmp_path)
    label = sut.list_moves("set-library-implementation")[0].labels
    sut.apply_move("set-library-implementation", label, implementation="pure")

    stale = sut.apply_move("set-library-implementation", label, implementation="OpenBLAS", current="LAPACK")

    assert stale.status is MoveStatus.ILLEGAL and "read the moves again" in stale.reason, stale
    assert "[pure]" in library_rows(sut)[0]


def oracle_of(program: dace.frontend.python.parser.DaceProgram, tmp_path: Path) -> tuple[Boundary, Prepared]:
    """The boundary and oracle of the one kernel of ``program``."""
    session = session_of(program, tmp_path)
    session.default_scopes()
    (ext,) = external_calls(session.sdfg)
    boundary = node_boundary(ext)
    return boundary, prepare(boundary, ext.name)


def test_the_oracle_of_a_library_call_is_that_call_and_never_its_pure_loop(tmp_path):
    boundary, prep = oracle_of(factor, tmp_path)

    assert "np.linalg.cholesky(" in prep.numpy_source and "while" not in prep.numpy_source, prep.numpy_source
    assert [n for n, _ in oracle_sdfg(boundary).all_nodes_recursive() if isinstance(n, nodes.LibraryNode)]


def test_a_library_oracle_computes_what_the_library_computes(tmp_path):
    sizes = {"N": SIZE}
    boundary, prep = oracle_of(factor, tmp_path)

    inputs = make_inputs(boundary, sizes)
    (output,) = boundary.outputs
    outputs = run_oracle(prep, boundary, inputs, sizes)

    (matrix,) = (value for name, value in inputs.items() if name in boundary.inputs)
    np.testing.assert_allclose(outputs[output] @ outputs[output].T, matrix, atol=1e-10)


def test_a_wrong_kernel_of_a_library_call_is_caught_by_its_oracle(tmp_path):
    sut = session_of(factor, tmp_path)
    sut.default_scopes()
    name = only_kernel(sut)
    source = sut.kernel_source(name)
    wrong = re.sub(r"(extern \"C\" void [^{]*\{)", r"\1 return;", source, count=1)

    written = sut.set_kernel_source(name, wrong, "cpp")

    assert written.status is KernelOutcomeStatus.WRONG, written


def test_a_reduction_and_a_product_stay_library_calls_and_match_their_library_results(tmp_path):
    sut = session_of(product_total, tmp_path)
    sut.default_scopes()

    evidence = sut.kernel_evidence(reps=1)

    assert len(evidence) == 2 and all(ev.verdict.ok for ev in evidence), evidence
    sources = "\n".join(sut.prepare_kernel(ext.name).numpy_source for ext in external_calls(sut.sdfg))
    assert "np.matmul(" in sources and "np.add.reduce(" in sources, sources


def test_a_factorized_input_is_positive_definite_so_both_sides_factor_the_same_matrix(tmp_path):
    sizes = {"N": SIZE}
    boundary, _ = oracle_of(factor, tmp_path)

    (matrix,) = (value for name, value in make_inputs(boundary, sizes).items() if name in boundary.inputs)

    assert np.linalg.eigvalsh(matrix).min() > 0.0


def test_an_oracle_out_of_time_is_stopped_and_the_caller_reads_why(tmp_path, monkeypatch):
    sizes = {"N": SIZE}
    boundary, prep = oracle_of(scaled, tmp_path)
    monkeypatch.setattr(arena, "load_emitted", lambda source, name: SleepingOracle(name))
    started = time.monotonic()

    answer = bounded_oracle(prep, boundary, make_inputs(boundary, sizes), sizes, timeout_s=0.5)

    assert isinstance(answer, Failure) and "timeout" in answer.error, answer
    assert time.monotonic() - started < 30.0


class SleepingOracle:
    """An emitted module whose kernel never finishes."""

    def __init__(self, name: str) -> None:
        setattr(self, name, lambda **_: time.sleep(600))


def test_a_bounded_oracle_runs_in_a_child_and_its_outputs_still_reach_the_caller(tmp_path):
    sizes = {"N": SIZE}
    boundary, prep = oracle_of(scaled, tmp_path)
    inputs = make_inputs(boundary, sizes)

    outputs = bounded_oracle(prep, boundary, inputs, sizes, timeout_s=60.0)

    assert not isinstance(outputs, Failure)
    expected = run_oracle(prep, boundary, inputs, sizes)
    for name, value in expected.items():
        np.testing.assert_array_equal(outputs[name], value)


def test_an_oracle_too_slow_for_the_timed_sizes_checks_at_the_small_ones_and_times_at_the_large(tmp_path, monkeypatch):
    sizes, small = {"N": 1 << 16}, {"N": SMALL}
    boundary, prep = oracle_of(scaled, tmp_path)
    monkeypatch.setattr(kernel, "ORACLE_TIMEOUT_S", 1.0)

    probe = bounded_probe(prep, boundary, sizes, small, full=True)

    assert probe.checked is not probe.timed and not probe.unchecked, "an out-of-time oracle compares at the small sizes"
    assert probe.checked.sizes == small and probe.timed.sizes == sizes
    assert all(value.shape[0] == SMALL for value in probe.oracle.values())


def test_a_kernel_wrong_at_the_small_sizes_is_wrong_whatever_it_costs_at_the_large(tmp_path):
    sut = session_of(scaled, tmp_path)
    sut.default_scopes()
    name = only_kernel(sut)
    src = sut.default_kernel(name)
    library = sut.optimize_kernel(name)
    timed = Workload(make_inputs(src.boundary, {"N": SIZE}), {"N": SIZE})
    checked = Workload(make_inputs(src.boundary, {"N": SMALL}), {"N": SMALL})
    expected = run_oracle(sut.prepare_kernel(name), src.boundary, checked.inputs, checked.sizes)
    right = measure_kernel(library.library, src, Probe(timed, checked, expected), 1, "strict-ieee")

    spoiled = Probe(timed, checked, {key: value + 1.0 for key, value in expected.items()})
    wrong = measure_kernel(library.library, src, spoiled, 1, "strict-ieee")

    assert right.ok and right.time_us > 0.0, right
    assert not wrong.ok


def test_a_kernel_whose_oracle_never_finishes_is_unchecked_not_refused(tmp_path):
    sut = session_of(scaled, tmp_path)
    sut.default_scopes()
    name = only_kernel(sut)
    src = sut.default_kernel(name)
    library = sut.optimize_kernel(name)
    timed = Workload({}, {"N": SIZE})

    verdict = measure_kernel(
        library.library, src, Probe(timed, timed, {}, "its oracle did not finish"), 1, "strict-ieee"
    )

    assert verdict.unchecked == "its oracle did not finish" and not verdict.error and not verdict.ok


def test_a_long_call_gets_fewer_timed_runs_never_none():
    assert reps_within_budget(10, 0.001) == 10
    assert reps_within_budget(10, arena.TIMED_BUDGET_S / 2) == 2
    assert reps_within_budget(10, arena.TIMED_BUDGET_S * 100) == 1
    assert reps_within_budget(10, 0.0) == 10


class Slept(Result):
    pid: int


def test_stopping_a_caller_kills_the_child_it_waits_on(tmp_path):
    marker = tmp_path / "pid"
    stopper = Stopper()

    def work() -> Slept:
        marker.write_text(str(os.getpid()))
        time.sleep(600)
        return Slept(pid=0)

    def ask(signum: int, frame: object) -> None:
        stopper.ask()

    previous = signal.signal(signal.SIGALRM, ask)
    signal.setitimer(signal.ITIMER_REAL, 1.0)
    try:
        with pytest.raises(Stopped), stopper.armed():
            run_isolated(work, Slept)
    finally:
        signal.signal(signal.SIGALRM, previous)

    child = int(marker.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)


def test_a_stop_asked_outside_an_armed_section_waits_for_the_next_one():
    stopper = Stopper()

    stopper.ask()

    with pytest.raises(Stopped), stopper.armed():
        pytest.fail("an armed section must not start after a stop was asked")


def test_a_stopped_sweep_goes_on_from_the_cells_it_finished(tmp_path, monkeypatch):
    sut = Session(scaled.to_sdfg(simplify=True), work_dir=str(tmp_path), sizes={"N": SIZE}, codegen="sweep")
    (scoped,) = sut.default_scopes()
    measured: list[str] = []
    real = variants.measure_kernel

    def stopping(*args: object, **kwargs: object) -> object:
        if len(measured) == 2:
            raise Stopped
        measured.append("cell")
        return real(*args, **kwargs)

    monkeypatch.setattr(variants, "measure_kernel", stopping)
    with pytest.raises(Stopped):
        sut.sweep(scoped.name)
    assert len(measured) == 2

    def counted(*args: object, **kwargs: object) -> object:
        measured.append("cell")
        return real(*args, **kwargs)

    monkeypatch.setattr(variants, "measure_kernel", counted)
    measured.clear()
    swept = sut.sweep(scoped.name)

    assert swept.winner is not None
    assert len(measured) == swept.cells - 2, "the two cells of the stopped sweep were not measured again"


def test_feedback_builds_each_kernel_once_and_a_move_rebuilds_only_the_kernels_it_changed(tmp_path, monkeypatch):
    sut = Session(two_loops.to_sdfg(simplify=True), work_dir=str(tmp_path), sizes={"N": SIZE}, codegen="sweep")
    sut.canonicalize()
    inputs = {name: np.random.default_rng(0).random(SIZE) for name in ("a", "b", "c", "d")}
    built: list[str] = []
    real = build_kernel_library

    def building(src: KernelSource, *args: object) -> Path:
        built.append(src.name)
        return real(src, *args)

    monkeypatch.setattr("nestforge.session.build_kernel_library", building)

    first = sut.performance(inputs, codegen="first")
    assert first.structured.error == "" and len(built) == 2, (first, built)
    builds = len(built)
    move = sut.list_moves("tile-doall")[0]
    sut.apply_move(move.kind, move.labels, (4,))
    second = sut.performance(inputs, codegen="first")

    assert second.structured.error == "" and len(built) == builds + 1, built


def test_the_feedback_basis_never_sweeps_and_the_session_keeps_sweeping_for_finalize(tmp_path, monkeypatch):
    sut = Session(scaled.to_sdfg(simplify=True), work_dir=str(tmp_path), sizes={"N": SIZE}, codegen="sweep")
    sut.canonicalize()
    monkeypatch.setattr(Session, "sweep", lambda *args, **kwargs: pytest.fail("feedback swept"))
    inputs = {name: np.ones(SIZE) for name in ("a", "c")}

    report = sut.performance(inputs, codegen="first")

    assert report.structured.error == "" and sut.codegen == "sweep"


def test_a_session_with_no_library_call_is_untouched_by_the_default_step(tmp_path):
    sut = session_of(scaled, tmp_path)

    assert library_rows(sut) == []
