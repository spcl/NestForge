# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``nestforge.run``: the executing entry point (``docs/PLAN_optimize_contract.md`` item 1).

``parse_sizes``/``load_python_program``/``load_sdfg`` are pure input-handling and run with no
compiler; ``optimize_program`` end-to-end needs the arena, so that one test is ``integration``.
"""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import pytest

import dace
from nestforge.__main__ import parse_sizes
from nestforge.run import load_python_program, load_sdfg, optimize_program

#: No ``@dace.program`` anywhere -- ``load_python_program`` must refuse rather than measure nothing.
NO_PROGRAM_SRC = textwrap.dedent("""
    def not_a_dace_program(a):
        return a + 1
    """)

#: Two ``@dace.program``s at module level -- ``load_python_program`` must never GUESS which one to
#: measure, since a silently-picked function yields a plausible-looking but wrong report.
TWO_PROGRAM_SRC = textwrap.dedent("""
    import dace

    N = dace.symbol("N")


    @dace.program
    def add_one(A: dace.float64[N], B: dace.float64[N]):
        for i in dace.map[0:N]:
            B[i] = A[i] + 1.0


    @dace.program
    def add_two(A: dace.float64[N], B: dace.float64[N]):
        for i in dace.map[0:N]:
            B[i] = A[i] + 2.0
    """)

#: A single, plain elementwise add -- no FMA-sensitive expression, so strict-ieee is bit-exact by
#: construction and the integration test measures the pipeline, not floating-point rounding.
VADD_SRC = textwrap.dedent("""
    import dace

    N = dace.symbol("N")


    @dace.program
    def vadd(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
        for i in dace.map[0:N]:
            C[i] = A[i] + B[i]
    """)

# ---------------------------------------------------------------- parse_sizes


def test_parse_sizes_happy_path() -> None:
    """Prevents a regression that drops a symbol or mangles its value on the CLI's only size input."""
    assert parse_sizes(["N=1024", "M=512"]) == {"N": 1024, "M": 512}


def test_parse_sizes_rejects_missing_equals() -> None:
    """Prevents a bare ``--size N`` (missing ``=INT``) from being swallowed instead of failing loudly."""
    with pytest.raises(argparse.ArgumentTypeError, match="SYMBOL=INT"):
        parse_sizes(["N"])


def test_parse_sizes_rejects_non_integer_value() -> None:
    """Prevents a typo'd size (``N=abc``) from reaching ``optimize_program`` as a string, which would
    fail deep inside input generation instead of at the CLI boundary."""
    with pytest.raises(argparse.ArgumentTypeError, match="not an integer"):
        parse_sizes(["N=abc"])


# ---------------------------------------------------------------- load_python_program


def test_load_python_program_requires_at_least_one_program(tmp_path: Path) -> None:
    """Prevents silently returning nothing to measure when a ``.py`` defines no ``@dace.program``."""
    path = tmp_path / "no_program.py"
    path.write_text(NO_PROGRAM_SRC)
    with pytest.raises(ValueError, match="defines no @dace.program"):
        load_python_program(path)


def test_load_python_program_refuses_to_guess_among_several(tmp_path: Path) -> None:
    """Prevents measuring the wrong function: two ``@dace.program``s and no ``func_name`` must raise,
    never silently pick one -- a wrong pick still produces a plausible-looking report."""
    path = tmp_path / "two_programs.py"
    path.write_text(TWO_PROGRAM_SRC)
    with pytest.raises(ValueError, match="pass func_name"):
        load_python_program(path)


def test_load_python_program_uses_func_name_to_disambiguate(tmp_path: Path) -> None:
    """Prevents ``func_name`` from being ignored: naming one of several programs must succeed."""
    path = tmp_path / "two_programs.py"
    path.write_text(TWO_PROGRAM_SRC)
    sdfg = load_python_program(path, func_name="add_two")
    assert isinstance(sdfg, dace.SDFG)


# ---------------------------------------------------------------- load_sdfg


def test_load_sdfg_rejects_unsupported_suffix(tmp_path: Path) -> None:
    """Prevents a provided C/C++/Fortran source from being read as if it had a numpy oracle to
    validate against -- the contract gap tracked as ``docs/PLAN_optimize_contract.md`` item 2."""
    path = tmp_path / "kernel.c"
    path.write_text("void kernel(void) {}\n")
    with pytest.raises(ValueError, match="PLAN_optimize_contract"):
        load_sdfg(path)


def test_load_sdfg_missing_path_raises_file_not_found(tmp_path: Path) -> None:
    """Prevents a typo'd path from surfacing as an opaque error deep inside SDFG deserialization."""
    with pytest.raises(FileNotFoundError):
        load_sdfg(tmp_path / "missing.sdfg")


# ---------------------------------------------------------------- optimize_program end-to-end


@pytest.mark.integration
def test_optimize_program_reaches_arena_bit_exact_strict_ieee(tmp_path: Path) -> None:
    """Prevents a regression anywhere in load -> lower -> translate -> arena: every nest must reach
    the arena, and its strict-ieee winner must be bit-exact vs the numpy oracle -- this repo's
    non-negotiable correctness gate."""
    path = tmp_path / "vadd.py"
    path.write_text(VADD_SRC)

    report = optimize_program(path, sizes={"N": 1 << 14}, out_dir=tmp_path / "build", reps=3)

    assert report.nests, "expected at least one nest to be extracted"
    assert all(nest.arena is not None for nest in report.nests), [(n.name, n.error) for n in report.nests]

    winners = report.winners()
    nest_name = report.nests[0].name
    assert nest_name in winners
    assert winners[nest_name].maxdiff == 0.0
