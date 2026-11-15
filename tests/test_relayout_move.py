# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``relayout``: the agent lays an argument out with one operation of the bench's requests; the program works on it
laid out, is timed on data laid out as the harness lays it out, and hands the operations to the bench."""

import dace
import numpy as np

from nestforge.records import MoveStatus
from nestforge.session import Session
from nestforge.stages.layout import RELAYOUT, layout_refusal, requests_of
from nestforge.stages.performance import Reference

N = dace.symbol("N", dace.int64, positive=True)
M = dace.symbol("M", dace.int64, positive=True)
SIZES = {"N": 6, "M": 5}
PERMUTE_A = {"action": "permute", "array": "A", "order": ["i1", "i0"]}


@dace.program
def scaled_transpose(A: dace.float64[N, M], B: dace.float64[M, N]):  # noqa: N803
    for i, j in dace.map[0:M, 0:N]:
        B[i, j] = 2.0 * A[j, i]


def session(tmp_path) -> Session:
    sut = Session(scaled_transpose.to_sdfg(simplify=True), work_dir=str(tmp_path), sizes=SIZES)
    sut.canonicalize()
    return sut


def extents(sut: Session, name: str) -> list[int]:
    return [int(dace.symbolic.evaluate(e, SIZES)) for e in sut.sdfg.arrays[name].shape]


def correct(sut: Session) -> bool:
    """Whether the program, fed and read back as the harness does, computes ``B = 2 A^T``."""
    a = np.arange(30.0).reshape(6, 5)
    report = sut.performance({"A": a, "B": np.zeros((5, 6))}, Reference({"B": 2.0 * a.T}, 0.0, 0.0), codegen="first")
    assert report.structured.error == "", report.structured
    return report.structured.correct is True


def test_a_permuted_argument_is_taken_laid_out_and_the_program_stays_correct(tmp_path):
    sut = session(tmp_path)

    result = sut.apply_move(RELAYOUT, [], operation=PERMUTE_A)

    assert (result.status, result.reason) == (MoveStatus.APPLIED, "relayout permute"), result
    assert extents(sut, "A") == [5, 6]
    assert requests_of(sut.layout) == {"operations": [PERMUTE_A]}
    assert correct(sut)


def test_a_laid_out_output_is_read_back_into_its_logical_shape(tmp_path):
    sut = session(tmp_path)
    pad = {"action": "pad", "array": "B", "digit": "i1", "amount": 3}

    result = sut.apply_move(RELAYOUT, [], operation=pad)

    assert result.status is MoveStatus.APPLIED, result
    assert extents(sut, "B") == [5, 9]
    assert correct(sut)


def test_a_block_after_the_permute_names_the_digits_the_permute_left(tmp_path):
    sut = session(tmp_path)
    block = {"action": "block", "array": "A", "digit": "i0", "factor": 4, "into": ["ib", "ii"]}

    sut.apply_move(RELAYOUT, [], operation=PERMUTE_A)
    result = sut.apply_move(RELAYOUT, [], operation=block)

    assert result.status is MoveStatus.APPLIED, result
    # i0 (N = 6, cut into 2 blocks of 4 with dead cells) is the inner axis after the permute: M, then the blocks
    assert extents(sut, "A") == [5, 2, 4]
    assert requests_of(sut.layout) == {"operations": [PERMUTE_A, block]}
    assert correct(sut)


def test_a_cast_argument_is_taken_at_its_new_precision_and_refused_at_finalize(tmp_path):
    sut = session(tmp_path)
    cast = {"action": "cast", "array": "A", "to": "fp32"}

    result = sut.apply_move(RELAYOUT, [], operation=cast)

    assert result.status is MoveStatus.APPLIED, result
    assert sut.sdfg.arrays["A"].dtype == dace.float32
    a = np.arange(30.0).reshape(6, 5)
    report = sut.performance({"A": a, "B": np.zeros((5, 6))}, Reference({"B": 2.0 * a.T}, 1e-6, 0.0), codegen="first")
    assert report.structured.error == "" and report.structured.correct is True, report.structured
    assert layout_refusal(sut.layout) is not None and "cast" in layout_refusal(sut.layout)


def test_relayout_is_listed_until_another_move_and_refused_after_it(tmp_path):
    sut = session(tmp_path)
    assert [m.labels for m in sut.list_moves(RELAYOUT)] == [()]
    (other, *_) = sut.list_moves("expand-doall")
    sut.apply_move(other.kind, other.labels)

    refused = sut.apply_move(RELAYOUT, [], operation=PERMUTE_A)

    assert sut.list_moves(RELAYOUT) == []
    assert refused.status is MoveStatus.ILLEGAL and "before any other change" in refused.reason, refused
    assert sut.layout == ()


def test_an_operation_on_no_argument_is_refused_with_the_arguments(tmp_path):
    sut = session(tmp_path)

    refused = sut.apply_move(RELAYOUT, [], operation={"action": "permute", "array": "T", "order": ["i1", "i0"]})

    assert refused.status is MoveStatus.ILLEGAL and "['A', 'B']" in refused.reason, refused
    assert sut.layout == () and sut.unlaid is None


def test_rollback_takes_the_layout_back_with_the_program(tmp_path):
    sut = session(tmp_path)
    sut.apply_move(RELAYOUT, [], operation=PERMUTE_A)

    sut.apply_move("rollback", [])

    assert sut.layout == ()
    assert extents(sut, "A") == [6, 5]
    assert correct(sut)
