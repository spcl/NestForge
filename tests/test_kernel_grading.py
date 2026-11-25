# Copyright 2021 ETH Zurich and the Tactics authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A kernel graded against its oracle by HPCAgent-Bench's rule (``tactics.build.grading``): a correct kernel that
reorders a long reduction passes, as the bench would pass it; a wrong one still fails. Tactics' old gate (rtol
1e-14 per FP mode) failed the first: a reordered fp64 sum of 1e6 terms drifts past 1e-14."""

import numpy as np
import pytest
from hpcagent_bench.precision import Precision

from tactics.build.arena import diff_stats
from tactics.build.grading import Expected, grade_kernel, output_precision

#: A long reduction, the length a reordered fp64 sum drifts past 1e-14 at.
LONG = 1_000_000

#: The parts a parallel reduction splits its sum into.
CHUNKS = 64

#: Tactics' old strict-ieee gate, which a correct reordered reduction failed.
OLD_STRICT_RTOL = 1e-14

#: A relative error far outside the fp64 band (rtol 1e-9): a wrong result, not rounding.
WRONG = 1e-6

#: An fp32 reduction long enough that ``eps32 * sqrt(l)`` exceeds the fp32 rtol of 1e-3: the bench will not grade it.
UNGRADEABLE_FP32 = 10**8


def sequential_and_chunked(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """One sum of ``LONG`` terms, in the oracle's order and in the order of a ``CHUNKS``-way parallel reduction."""
    terms = np.random.default_rng(seed).random(LONG)
    sequential = np.array([np.cumsum(terms)[-1]])
    chunked = np.array([sum(float(np.sum(part)) for part in np.array_split(terms, CHUNKS))])
    return sequential, chunked


def test_a_reordered_long_sum_passes_where_the_old_gate_failed_it() -> None:
    sequential, chunked = sequential_and_chunked(0)
    _, old_scaled = diff_stats({"s": sequential}, {"s": chunked})
    assert old_scaled > OLD_STRICT_RTOL, "the reordering drifts past the old gate"

    grade = grade_kernel(Expected({"s": sequential}, {"s": LONG}), {"s": chunked})

    assert grade.ok and not grade.detail and not grade.ungradeable, grade
    assert grade.error > 0.0, "the two orders really differ"


def test_a_wrong_reduction_fails() -> None:
    sequential, _ = sequential_and_chunked(1)
    grade = grade_kernel(Expected({"s": sequential}, {"s": LONG}), {"s": sequential * (1 + WRONG)})
    assert not grade.ok and grade.detail.startswith("s: "), grade


def test_one_wrong_element_among_many_right_ones_fails() -> None:
    want = np.ones(10_000)
    got = want.copy()
    got[4_242] = 2.0
    grade = grade_kernel(Expected({"a": want}, {"a": 1}), {"a": got})
    assert not grade.ok and grade.error == pytest.approx(1.0), grade


def test_a_missing_output_and_a_comparison_of_nothing_fail() -> None:
    assert not grade_kernel(Expected({"a": np.ones(3)}), {}).ok
    empty = grade_kernel(Expected({"a": np.zeros(0)}), {"a": np.zeros(0)})
    assert not empty.ok and empty.detail == "no output element to compare", empty


def test_an_integer_output_compares_exactly() -> None:
    """Integers are right or wrong: no band admits an off-by-one index."""
    want = np.arange(8)
    assert grade_kernel(Expected({"i": want}), {"i": want.copy()}).ok
    assert not grade_kernel(Expected({"i": want}), {"i": want + np.eye(1, 8, 3, dtype=int)[0]}).ok


def test_an_ungradeable_length_is_reported_not_failed_as_wrong() -> None:
    """At fp32 ``eps_acc * sqrt(l)`` reaches the band at ``l`` near 7e7; the bench refuses rather than widen it."""
    want = np.ones(4, dtype=np.float32)
    grade = grade_kernel(Expected({"a": want}, {"a": UNGRADEABLE_FP32}), {"a": want.copy()})
    assert not grade.ok and grade.ungradeable.startswith("a: "), grade


@pytest.mark.parametrize(
    ("dtype", "precision"),
    [
        (np.float64, Precision.FP64),
        (np.float32, Precision.FP32),
        (np.complex64, Precision.FP32),
        (np.complex128, Precision.FP64),
        (np.int32, Precision.FP64),
    ],
)
def test_each_output_is_graded_at_its_own_precision(dtype: type, precision: Precision) -> None:
    """A complex output takes its component's band; an integer one compares exactly, whatever band it is given."""
    assert output_precision(np.zeros(2, dtype=dtype)) is precision
