# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The accumulation length ``l`` of a kernel's outputs, the ``l`` of the bench's reassociation floor
``eps_acc * sqrt(l)``: too short, a correct reordered reduction fails the kernel check; too long, the floor forgives
a wrong small element."""

import dace
import numpy as np
from dace.frontend.python.parser import DaceProgram

from nestforge.ir.accumulation import Interval, accumulation_lengths, corners

M, N, K = (dace.symbol(name) for name in "MNK")

#: Sizes the programs below are counted at.
ROWS, COLS, DEPTH = 8, 6, 1000

#: A long reduction, the length a reordered fp64 sum drifts past 1e-14 at.
LONG = 1_000_000


@dace.program
def matmul_in_place(a: dace.float64[M, K], b: dace.float64[K, N], c: dace.float64[M, N]) -> None:
    for i, j in dace.map[0:M, 0:N]:
        c[i, j] = 0.0
        for k in range(K):
            c[i, j] = c[i, j] + a[i, k] * b[k, j]


@dace.program
def matmul_wcr(a: dace.float64[M, K], b: dace.float64[K, N], c: dace.float64[M, N]) -> None:
    for i, j, k in dace.map[0:M, 0:N, 0:K]:
        c[i, j] += a[i, k] * b[k, j]


@dace.program
def matmul_scalar(a: dace.float64[M, K], b: dace.float64[K, N], c: dace.float64[M, N]) -> None:
    for i, j in dace.map[0:M, 0:N]:
        acc = 0.0
        for k in range(K):
            acc = acc + a[i, k] * b[k, j]
        c[i, j] = acc


@dace.program
def total(x: dace.float64[N], s: dace.float64[1]) -> None:
    s[0] = 0.0
    for i in range(N):
        s[0] = s[0] + x[i]


@dace.program
def vadd(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]) -> None:
    for i in dace.map[0:N]:
        c[i] = a[i] + b[i]


@dace.program
def lower_row_sums(a: dace.float64[N, N], x: dace.float64[N]) -> None:
    for i in range(N):
        x[i] = 0.0
        for j in range(i):
            x[i] = x[i] + a[i, j]


@dace.program
def through_scratch(x: dace.float64[N], y: dace.float64[1]) -> None:
    t = np.zeros((1,), dtype=np.float64)
    for i in range(N):
        t[0] = t[0] + x[i]
    y[0] = t[0] * 2.0


@dace.program
def library_matmul(a: dace.float64[M, K], b: dace.float64[K, N], c: dace.float64[M, N]) -> None:
    c[:] = a @ b


def lengths(program: DaceProgram, sizes: dict[str, int], outputs: list[str]) -> dict[str, int]:
    """``l`` of ``outputs`` of ``program`` at ``sizes``, its arrays allocated at those sizes."""
    sdfg = program.to_sdfg(simplify=True)
    arrays = {
        name: np.zeros(tuple(int(dace.symbolic.evaluate(dim, sizes)) for dim in desc.shape))
        for name, desc in sdfg.arrays.items()
        if not desc.transient
    }
    return accumulation_lengths(sdfg, sizes, arrays, outputs)


MATMUL_SIZES = {"M": ROWS, "N": COLS, "K": DEPTH}


def test_a_matmul_accumulates_over_its_contracted_extent_however_it_is_written() -> None:
    """In place, as a WCR, or in a scalar kept per output element: each element sums ``K`` products."""
    for program in (matmul_in_place, matmul_wcr, matmul_scalar):
        assert lengths(program, MATMUL_SIZES, ["c"]) == {"c": DEPTH}, program.name


def test_a_full_reduction_has_the_length_of_its_input() -> None:
    assert lengths(total, {"N": LONG}, ["s"]) == {"s": LONG}


def test_an_elementwise_kernel_accumulates_nothing() -> None:
    """Writes through a per-iteration scalar are no accumulation: each element is computed once."""
    assert lengths(vadd, {"N": COLS * DEPTH}, ["c"]) == {"c": 1}


def test_a_triangular_loop_takes_its_longest_row() -> None:
    """The inner bound reads the outer iterator; the count takes its larger end, at most the longest row."""
    got = lengths(lower_row_sums, {"N": DEPTH}, ["x"])["x"]
    assert DEPTH // 2 <= got <= DEPTH, got


def test_an_output_computed_from_scratch_inherits_its_length() -> None:
    """The sum runs in a transient and is copied out scaled: the output still carries the sum's rounding."""
    assert lengths(through_scratch, {"N": LONG}, ["y"]) == {"y": LONG}


def test_a_library_node_falls_back_to_the_largest_input() -> None:
    """A library node is not counted; its output takes the bench's own fallback, the largest input's size."""
    assert lengths(library_matmul, MATMUL_SIZES, ["c"]) == {"c": ROWS * DEPTH}


def test_corners_bound_a_linear_expression_over_its_symbols() -> None:
    env: dict[str, Interval] = {"i": (0, 9), "N": (100, 100)}
    assert corners("N - i", env) == (91, 100)
    assert corners("N", env) == (100, 100)
    assert corners("Q + 1", env) is None, "a symbol with no interval has no value"
