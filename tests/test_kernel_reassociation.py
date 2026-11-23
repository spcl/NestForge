# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""#106: a kernel's check allows the drift a reordered accumulation carries, as the bench grades it (eps * sqrt(l),
l the length the output sums over), so a correct parallel or library reduction is not called wrong."""

import math

import dace
import numpy as np
import pytest
from dace.frontend.python.parser import DaceProgram

from nestforge.stages.kernel import KernelVerdict, accumulation_length, reassociation_growth

N = dace.symbol("N", dace.int64)
K = dace.symbol("K", dace.int64)
EPS64 = float(np.finfo(np.float64).eps)


@dace.program
def summed(a: dace.float64[N], out: dace.float64[1]) -> None:
    for i in dace.map[0:N]:
        out[0] += a[i]


@dace.program
def row_sums(a: dace.float64[N, K], out: dace.float64[N]) -> None:
    for i, k in dace.map[0:N, 0:K]:
        out[i] += a[i, k]


@dace.program
def product(a: dace.float64[N, K], b: dace.float64[K, N], c: dace.float64[N, N]) -> None:
    c[:] = a @ b


@dace.program
def scaled(a: dace.float64[N], b: dace.float64[N]) -> None:
    for i in dace.map[0:N]:
        b[i] = 2.0 * a[i]


@pytest.mark.parametrize(
    "program, length",
    [(summed, 1000), (row_sums, 70), (product, 70), (scaled, 1)],
    ids=["full-sum", "row-sum", "matmul", "elementwise"],
)
def test_the_accumulation_length_is_the_extent_the_output_does_not_keep(program: DaceProgram, length: int) -> None:
    sizes = {"N": 1000, "K": 70} if program is not row_sums else {"N": 9, "K": 70}
    assert accumulation_length(program.to_sdfg(simplify=True), sizes) == length


def test_the_growth_is_eps_times_the_root_of_the_length_and_nothing_without_an_accumulation() -> None:
    sdfg = summed.to_sdfg(simplify=True)
    assert reassociation_growth(sdfg, ["out"], {"N": 10**6}) == pytest.approx(EPS64 * 1000)
    assert reassociation_growth(scaled.to_sdfg(simplify=True), ["b"], {"N": 10**6}) == 0.0
    assert reassociation_growth(sdfg, ["out"], {}) == 0.0, "an unresolved length adds nothing (the stricter side)"


def test_a_narrow_format_takes_the_tree_bound() -> None:
    sdfg = summed.to_sdfg(simplify=True)
    sdfg.arrays["out"].dtype = dace.float16
    eps16 = float(np.finfo(np.float16).eps)
    assert reassociation_growth(sdfg, ["out"], {"N": 1024}) == pytest.approx(
        eps16 / 2 * (math.ceil(math.log2(1024)) + 1)
    )


def test_a_correct_reordered_sum_of_a_million_terms_passes_and_a_wrong_one_still_fails() -> None:
    """2e-13 is the drift a correct fp64 sum of 10^6 terms shows in another order: wrong at the flat 1e-14 gate,
    right once the growth is added; a real error stays wrong."""
    growth = EPS64 * math.sqrt(10**6)
    drifted = KernelVerdict("strict-ieee", 2e-13, 2e-13, 2.3e-16, 1.0)
    assert not drifted.ok
    assert KernelVerdict("strict-ieee", 2e-13, 2e-13, 2.3e-16, 1.0, growth=growth).ok
    assert not KernelVerdict("strict-ieee", 1e-6, 1e-6, 2.3e-16, 1.0, growth=growth).ok
