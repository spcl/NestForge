# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The gate that admits a measured kernel as correct: ``arena.diff_stats`` against ``arena.rung_rtol``.

Too tight, it drops a whole kernel class from the sweep; too loose, it admits a miscompile as a speed-up. Neither
shows up anywhere but here.
"""

import numpy as np
import pytest

from nestforge.build import arena
from nestforge.build import flags


def test_a_nan_anywhere_fails_even_when_it_is_not_the_first_element():
    """Builtin ``max`` keeps its left operand when the right is NaN (``nan > x`` is False), so a NaN that is not
    the first thing compared would leave ``worst`` at 0.0 and a NaN-poisoned kernel could win the arena."""
    good = np.ones(8)
    for position in (0, 3, 7):
        poisoned = good.copy()
        poisoned[position] = np.nan
        assert arena.diff_stats({"a": good}, {"a": poisoned}) == (float("inf"), float("inf"))


def test_a_nan_in_a_later_ARRAY_fails_too():
    """Same hazard one level up: the first array can match perfectly and the second be poison."""
    clean = {"first": np.ones(4), "second": np.ones(4)}
    poisoned = {"first": np.ones(4), "second": np.array([1.0, 1.0, np.nan, 1.0])}
    assert arena.diff_stats(clean, poisoned) == (float("inf"), float("inf"))


def test_an_infinity_fails_rather_than_dominating_the_maximum():
    """``inf - 1`` is ``inf`` and ``inf/inf`` is NaN; neither is a tolerance a gate can compare."""
    assert arena.diff_stats({"a": np.ones(4)}, {"a": np.array([1.0, np.inf, 1.0, 1.0])}) == (float("inf"), float("inf"))


def test_a_comparison_that_touched_no_element_fails_instead_of_passing():
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
def test_bool_and_unsigned_outputs_compare_as_numbers(want, got, expected):
    """NumPy cannot subtract booleans, and an unsigned ``1 - 2`` wraps to ``2**32 - 1``."""
    assert arena.diff_stats({"a": want}, {"a": got}) == expected


def test_complex_outputs_compare_by_magnitude():
    """A complex kernel output must not crash the gate; its difference is the modulus."""
    a = np.ones(3, np.complex128)
    worst_abs, _ = arena.diff_stats({"z": a}, {"z": a + 1e-3j})
    assert worst_abs == pytest.approx(1e-3)


def test_one_empty_array_is_skipped_but_a_real_one_beside_it_still_decides():
    """Skipping an individual empty output is right (a kernel may legitimately declare one); letting
    it suppress the array that does have elements is not."""
    a = {"empty": np.zeros(0), "real": np.ones(4)}
    b = {"empty": np.zeros(0), "real": np.array([1.0, 1.0, 1.25, 1.0])}
    worst_abs, _worst_rel = arena.diff_stats(a, b)
    assert worst_abs == pytest.approx(0.25)


def test_small_magnitudes_keep_the_strict_absolute_reading():
    """The relative denominator floors at 1.0, so the gate is never loosened where fp64 can deliver: at magnitude
    ~0.5 the scaled difference must equal the absolute one."""
    a = {"a": np.array([0.5, 0.25, 0.125])}
    b = {"a": np.array([0.5 + 1e-13, 0.25, 0.125])}
    worst_abs, worst_rel = arena.diff_stats(a, b)
    assert worst_rel == pytest.approx(worst_abs)
    assert worst_rel > arena.rung_rtol("strict-ieee", arena.dtype_floor(b)), "the default rung refuses it"


def test_a_reduction_sized_result_is_judged_relatively_not_absolutely():
    """Summing 32000 values near 1 lands near 1.6e4, where one ULP is 180x an absolute 1e-14 gate: an absolute
    gate would reject every correctly vectorized reduction."""
    total = 1.6e4
    a = {"sum": np.array([total])}
    b = {"sum": np.array([total + 14 * np.spacing(total)])}
    worst_abs, worst_rel = arena.diff_stats(a, b)
    assert worst_abs > 1e-14, "the absolute difference is larger than an absolute 1e-14 gate"
    assert worst_rel <= arena.rung_rtol("contract-fma", arena.dtype_floor(b)), "scaled, the contract-fma rung admits it"


def test_the_gate_is_elementwise_and_a_single_bad_element_survives_averaging():
    """not a norm: one wrong element among ten thousand right ones must show at full size, because a
    miscompile that touches one index is still a miscompile."""
    a = {"a": np.ones(10_000)}
    b = np.ones(10_000)
    b[4_242] = 2.0
    worst_abs, worst_rel = arena.diff_stats(a, {"a": b})
    assert worst_abs == pytest.approx(1.0)
    assert worst_rel == pytest.approx(0.5)  # scaled by max(|1|, |2|, 1.0) = 2


def test_the_gate_is_never_tighter_than_the_output_dtype_can_express():
    """A bit-exact rung asks for 0.0, which fp32 cannot deliver on any real arithmetic; the dtype
    floor is what keeps a correct fp32 kernel from being recorded as a miscompile."""
    fp32 = {"a": np.ones(4, dtype=np.float32)}
    fp64 = {"a": np.ones(4, dtype=np.float64)}

    def gate(mode: str, outputs: dict[str, np.ndarray]) -> float:
        return arena.rung_rtol(mode, arena.dtype_floor(outputs))

    assert gate("strict-ieee", fp32) >= flags.DTYPE_RTOL["float32"]
    assert gate("strict-ieee", fp32) > gate("strict-ieee", fp64)


def test_an_integer_output_contributes_no_tolerance_floor():
    """Integer and boolean results are exact or they are wrong; giving them a floor would admit an
    off-by-one index as a rounding difference."""
    assert arena.dtype_floor({"i": np.arange(4)}) == 0.0
    assert arena.dtype_floor({"b": np.ones(4, dtype=bool)}) == 0.0
    mixed = {"i": np.arange(4), "f": np.ones(4, dtype=np.float32)}
    assert arena.dtype_floor(mixed) == flags.DTYPE_RTOL["float32"]
