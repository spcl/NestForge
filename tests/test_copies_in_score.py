# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The offloading feedback names the host/device copies of the timed program with their bytes, and the GPU kernels
that still copy inside their own call in the session's program. Transformations only: no kernel is built."""

from pathlib import Path

import dace
import numpy as np
import pytest
from dace.libraries.standard.helper import GPU_RESIDENT_STORAGES
from test_placement import chain, kernel_session

import nestforge.session as session_mod
from nestforge.session import Session
from nestforge.stages.canonicalize import Targets
from nestforge.stages.performance import (
    CopyReport,
    OffloadReport,
    ProgramReport,
    array_bytes,
    copy_reports,
    offload_report,
)
from nestforge.stages.placement import copies_in_call, has_device_memory

N = dace.symbol("N", dtype=dace.int64)

#: The size the reports are evaluated at.
SIZE = 100
#: Bytes of one float64 vector of ``SIZE`` elements.
VECTOR_BYTES = SIZE * 8
#: ``chain``'s copies once inserted: its inputs in, its output out; ``t`` stays in device memory.
CHAIN_COPIES = [("a", "a_gpu"), ("b", "b_gpu"), ("c_gpu", "c")]


def program(session: Session) -> dace.SDFG:
    assert session.sdfg is not None
    return session.sdfg


def report_of(session: Session) -> OffloadReport:
    devices = session.devices()
    sdfg = program(session)
    return offload_report(sdfg, devices, session.transfers(devices), {"N": SIZE}, copies_in_call(sdfg))


def copy_pairs(report: OffloadReport) -> list[tuple[str, str]]:
    return sorted((copy.source, copy.destination) for copy in report.copies)


def test_a_placed_gpu_program_copies_inside_its_calls_and_none_in_the_program(tmp_path: Path) -> None:
    session = kernel_session(True, tmp_path, chain)
    session.place()

    report = report_of(session)

    assert report.copies_in_call == ["extcall_doall0", "extcall_doall1"]
    assert report.copies == [] and report.copy_bytes == 0
    assert report.transfer_bytes == 3 * VECTOR_BYTES, "the transfers the kernel DAG implies do not depend on it"


@dace.program
def scaled(a: dace.float64[N], c: dace.float64[N], s: dace.float64):
    for i in dace.map[0:N]:
        c[i] = a[i] * s


def test_a_scalar_passed_by_value_is_no_copy_in_the_call(tmp_path: Path) -> None:
    session = kernel_session(True, tmp_path, scaled)
    session.place()
    (kernel,) = session.devices()
    assert copies_in_call(program(session)) == [kernel]

    session.insert_copies()

    # s stays a host scalar the call passes by value; the arrays are device memory, so nothing is copied in the call
    assert program(session).arrays["s"].storage not in GPU_RESIDENT_STORAGES
    assert copies_in_call(program(session)) == []


def test_a_transfer_of_an_array_the_program_sizes_at_run_time_has_no_bytes() -> None:
    """``gmres`` moves an array of ``m*(m + 1)`` elements, ``m`` computed by the program: the report must not raise."""
    sdfg = dace.SDFG("sized_at_run_time")
    sdfg.add_symbol("m", dace.int64)
    sdfg.add_array("h", [dace.symbol("m") * (dace.symbol("m") + 1)], dace.float64)
    sdfg.add_array("a", [N], dace.float64)

    assert array_bytes(sdfg, "h", {"N": SIZE}) is None
    assert array_bytes(sdfg, "a", {"N": SIZE}) == VECTOR_BYTES
    assert array_bytes(sdfg, "h", {"N": SIZE, "m": 3}) == 12 * 8
    assert array_bytes(sdfg, "no_such_array", {"N": SIZE}) is None


def test_inserting_the_copies_moves_them_into_the_program_with_their_bytes(tmp_path: Path) -> None:
    session = kernel_session(True, tmp_path, chain)
    session.place()
    session.insert_copies()

    report = report_of(session)

    assert report.copies_in_call == []
    assert copy_pairs(report) == CHAIN_COPIES
    assert all(copy.bytes == VECTOR_BYTES for copy in report.copies)
    assert report.copy_bytes == len(CHAIN_COPIES) * VECTOR_BYTES
    wire = report.model_dump(mode="json")
    assert wire["copy_bytes"] == report.copy_bytes and wire["copies_in_call"] == []
    assert {"source": "a", "destination": "a_gpu", "bytes": VECTOR_BYTES} in wire["copies"]


def test_a_cpu_program_has_no_copies(tmp_path: Path) -> None:
    session = kernel_session(False, tmp_path, chain)
    session.place()

    report = report_of(session)

    assert report.copies == [] and report.copies_in_call == [] and report.copy_bytes == 0


def test_a_copy_sized_by_a_symbol_the_session_does_not_fix_has_no_bytes(tmp_path: Path) -> None:
    session = kernel_session(True, tmp_path, chain)
    session.place()
    session.insert_copies()

    copies = copy_reports(program(session), {})

    assert sorted((copy.source, copy.destination) for copy in copies) == CHAIN_COPIES
    assert all(copy.bytes is None for copy in copies)


def test_performance_reads_the_in_call_copies_off_the_session_not_the_timed_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The trial ``get_performance`` times runs the default placement, which inserts the copies; the kernels that copy
    inside their call are those of the session's own program, which the measurement leaves unchanged."""
    session = Session(
        chain.to_sdfg(simplify=True), targets=Targets(gpu=True), work_dir=str(tmp_path), sizes={"N": SIZE}
    )
    assert session.default_scopes()
    session.place()
    monkeypatch.setattr(Session, "kernel_evidence", lambda self, reps: [])
    monkeypatch.setattr(session_mod, "time_program", lambda *args: ProgramReport(time_us=1.0))
    inputs = {name: np.zeros(SIZE) for name in ("a", "b", "c")}

    offloading = session.performance(inputs).offloading

    assert offloading.copies_in_call == ["extcall_doall0", "extcall_doall1"]
    assert copy_pairs(offloading) == CHAIN_COPIES, "the timed program had its copies inserted"
    assert all(isinstance(copy, CopyReport) and copy.bytes == VECTOR_BYTES for copy in offloading.copies)
    assert not has_device_memory(program(session)), "measuring does not insert the session's copies"
