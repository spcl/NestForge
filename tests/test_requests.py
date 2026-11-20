# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``request_transformation``: an agent names a transformation no move offers, with pairs of small programs before
and after it. The request is checked, kept, and changes nothing in the program."""

from pathlib import Path

import dace

from nestforge.records import Refusal
from nestforge.session import Session
from nestforge.stages.requests import MIN_EXAMPLES, Example

N = dace.symbol("N", dtype=dace.int64)


@dace.program
def doubled(a: dace.float64[N], b: dace.float64[N]) -> None:
    for i in dace.map[0:N]:
        b[i] = 2.0 * a[i] + 1.0


def session(tmp_path: Path) -> Session:
    sut = Session(doubled.to_sdfg(simplify=True), work_dir=str(tmp_path), sizes={"N": 64})
    sut.canonicalize()
    return sut


def scaled(body: str) -> str:
    return (
        "N = dace.symbol('N')\n\n"
        "@dace.program\n"
        "def scaled(a: dace.float64[N], b: dace.float64[N]):\n"
        "    for i in dace.map[0:N]:\n"
        f"        b[i] = {body}\n"
    )


def division_by_reciprocal(divisors: tuple[float, ...] = (2.0, 4.0, 8.0, 0.5, 0.25)) -> list[Example]:
    """Division by a power of two as multiplication by its reciprocal: exact, and canonicalization keeps the division."""
    return [Example(before=scaled(f"a[i] / {d}"), after=scaled(f"a[i] * {1.0 / d}")) for d in divisors]


def a_row(sut: Session) -> str:
    return sut.list_moves("tile-doall")[0].labels[0]


def test_a_request_whose_examples_pass_is_recorded_and_the_program_stays(tmp_path: Path) -> None:
    sut = session(tmp_path)
    tree, row = sut.describe(), a_row(sut)

    result = sut.request_transformation("reciprocal-division", "x / c -> x * (1/c)", [row], division_by_reciprocal())

    assert (result.status, result.reason) == ("recorded", "")
    assert [request.name for request in sut.requests] == ["reciprocal-division"]
    assert sut.describe() == tree


def test_too_few_examples_or_an_unknown_row_are_refused_with_their_class(tmp_path: Path) -> None:
    sut = session(tmp_path)
    row = a_row(sut)

    few = sut.request_transformation("r", "d", [row], division_by_reciprocal()[: MIN_EXAMPLES - 1])
    unknown = sut.request_transformation("r", "d", ["doall99"], division_by_reciprocal())

    assert (few.status, few.refusal) == ("refused", Refusal.ARGUMENT)
    assert (unknown.status, unknown.refusal) == ("refused", Refusal.UNKNOWN_LABEL)
    assert sut.requests == []


def test_a_pair_that_computes_something_else_is_refused(tmp_path: Path) -> None:
    sut = session(tmp_path)
    wrong = [*division_by_reciprocal()[:4], Example(before=scaled("a[i] / 3.0"), after=scaled("a[i] * 0.5"))]

    result = sut.request_transformation("r", "d", [a_row(sut)], wrong)

    assert result.status == "refused" and "different values in b" in result.reason


def test_a_pair_that_is_one_program_after_canonicalization_asks_for_nothing(tmp_path: Path) -> None:
    sut = session(tmp_path)
    same = [*division_by_reciprocal()[:4], Example(before=scaled("a[i] / 2.0"), after=scaled("a[i] / 2.0"))]

    result = sut.request_transformation("r", "d", [a_row(sut)], same)

    assert result.status == "refused" and "asks for nothing" in result.reason


def test_an_example_that_copies_the_reference_source_is_refused(tmp_path: Path) -> None:
    sut = session(tmp_path)
    examples = division_by_reciprocal()
    reference = "\n".join(examples[2].before.splitlines()[2:5])

    result = sut.request_transformation("r", "d", [a_row(sut)], examples, reference=reference)

    assert result.status == "refused" and "copies the task's reference" in result.reason
