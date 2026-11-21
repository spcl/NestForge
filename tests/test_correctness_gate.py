# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The difference a measured kernel reports against its oracle: ``arena.diff_stats``, the ``maxdiff`` of a verdict (the
grade itself is the bench's, ``tests/test_kernel_grading.py``).

Too tight, it drops a whole kernel class from the sweep; too loose, it admits a miscompile as a speed-up. Neither
shows up anywhere but here.
"""

import numpy as np
import pytest

from nestforge.build import arena


def test_a_nan_anywhere_fails_even_when_it_is_not_the_first_element() -> None:
    """Builtin ``max`` keeps its left operand when the right is NaN (``nan > x`` is False), so a NaN that is not
    the first thing compared would leave ``worst`` at 0.0 and a NaN-poisoned kernel could win the arena."""
    good = np.ones(8)
    for position in (0, 3, 7):
        poisoned = good.copy()
        poisoned[position] = np.nan
        assert arena.diff_stats({"a": good}, {"a": poisoned}) == (float("inf"), float("inf"))


def test_a_nan_in_a_later_ARRAY_fails_too() -> None:
    """Same hazard one level up: the first array can match perfectly and the second be poison."""
    clean = {"first": np.ones(4), "second": np.ones(4)}
    poisoned = {"first": np.ones(4), "second": np.array([1.0, 1.0, np.nan, 1.0])}
    assert arena.diff_stats(clean, poisoned) == (float("inf"), float("inf"))


def test_an_infinity_fails_rather_than_dominating_the_maximum() -> None:
    """``inf - 1`` is ``inf`` and ``inf/inf`` is NaN; neither is a tolerance a gate can compare."""
    assert arena.diff_stats({"a": np.ones(4)}, {"a": np.array([1.0, np.inf, 1.0, 1.0])}) == (float("inf"), float("inf"))


def test_a_comparison_that_touched_no_element_fails_instead_of_passing() -> None:
    """The gate is ``<= atol``, so a verdict of 0.0 read off zero elements passes -- a kernel whose
    outputs are all zero-size would be admitted as bit-exact having computed nothing."""
    empty = {"a": np.zeros(0), "b": np.zeros((0, 4))}
    assert arena.diff_stats(empty, empty) == (float("inf"), float("inf"))


@pytest.mark.parametrize(
    ("want", "got", "expected"),
    [
        (np.array([True, False]), np.array([True, False]), (0.0, 0.0)),
        (np.array([True, False]), np.array([True, True]), (1.0, 1.0)),
        (np.array([1, 2], np.uint32), np.array([2, 2], np.uint32), (1.0, 0.5)),
    ],
    ids=["equal-bool", "one-bool-differs", "unsigned-below"],
)
def test_bool_and_unsigned_outputs_compare_as_numbers(
    want: np.ndarray, got: np.ndarray, expected: tuple[float, float]
) -> None:
    """NumPy cannot subtract booleans, and an unsigned ``1 - 2`` wraps to ``2**32 - 1``."""
    assert arena.diff_stats({"a": want}, {"a": got}) == expected


def test_complex_outputs_compare_by_magnitude() -> None:
    """A complex kernel output must not crash the gate; its difference is the modulus."""
    a = np.ones(3, np.complex128)
    worst_abs, _ = arena.diff_stats({"z": a}, {"z": a + 1e-3j})
    assert worst_abs == pytest.approx(1e-3)


def test_one_empty_array_is_skipped_but_a_real_one_beside_it_still_decides() -> None:
    """Skipping an individual empty output is right (a kernel may legitimately declare one); letting
    it suppress the array that does have elements is not."""
    a = {"empty": np.zeros(0), "real": np.ones(4)}
    b = {"empty": np.zeros(0), "real": np.array([1.0, 1.0, 1.25, 1.0])}
    worst_abs, _worst_rel = arena.diff_stats(a, b)
    assert worst_abs == pytest.approx(0.25)


def test_the_gate_is_elementwise_and_a_single_bad_element_survives_averaging() -> None:
    """not a norm: one wrong element among ten thousand right ones must show at full size, because a
    miscompile that touches one index is still a miscompile."""
    a = {"a": np.ones(10_000)}
    b = np.ones(10_000)
    b[4_242] = 2.0
    worst_abs, worst_rel = arena.diff_stats(a, {"a": b})
    assert worst_abs == pytest.approx(1.0)
    assert worst_rel == pytest.approx(0.5)  # scaled by max(|1|, |2|, 1.0) = 2
