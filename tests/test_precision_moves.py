# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The ``precision`` family: ``set-fp-mode`` fixes the floating-point flags of every kernel build, and
``change-precision`` stores an internal array in another floating-point type. Both are undone by ``rollback``."""

import dace
import numpy as np

from nestforge.records import MoveStatus, Refusal
from nestforge.session import Session

N = dace.symbol("N", dtype=dace.int64)
SIZE = 64


@dace.program
def staged(a: dace.float64[N], b: dace.float64[N]):
    t = np.empty_like(a)
    for i in dace.map[0:N]:
        t[i] = a[i] * 3.0
    for i in dace.map[0:N]:
        b[i] = t[i] + 1.0


def session(tmp_path) -> Session:
    sut = Session(staged.to_sdfg(simplify=True), work_dir=str(tmp_path), sizes={"N": SIZE})
    sut.canonicalize()
    return sut


def internal(sut: Session) -> str:
    (name,) = [name for name, desc in sut.sdfg.arrays.items() if desc.transient and isinstance(desc, dace.data.Array)]
    return name


def test_set_fp_mode_builds_every_kernel_in_that_mode_until_rollback(tmp_path):
    sut = session(tmp_path)
    offered = [move.fp_mode for move in sut.list_moves("set-fp-mode")]

    done = sut.apply_move("set-fp-mode", [], fp_mode="fast-math")

    assert offered == ["strict-ieee", "contract-fma", "fast-math"]
    assert done.status is MoveStatus.APPLIED and done.warning
    assert sut.first_variant("cpu").fp_mode == "fast-math"
    assert [move.fp_mode for move in sut.list_moves("set-fp-mode")] == ["strict-ieee", "contract-fma"]
    sut.apply_move("rollback", [])
    assert sut.fp_mode is None


def test_a_mode_the_target_does_not_build_is_refused(tmp_path):
    sut = session(tmp_path)

    refused = sut.apply_move("set-fp-mode", [], fp_mode="loose")

    assert (refused.status, refused.refusal) == (MoveStatus.ILLEGAL, Refusal.ARGUMENT)
    assert sut.fp_mode is None


def test_change_precision_stores_an_internal_array_in_float32_and_the_program_computes_in_it(tmp_path):
    sut = session(tmp_path)
    name = internal(sut)
    listed = [(move.container, move.dtype) for move in sut.list_moves("change-precision")]

    done = sut.apply_move("change-precision", [], container=name, dtype="float32")

    assert listed == [(name, "float32")]
    assert done.status is MoveStatus.APPLIED
    assert sut.sdfg.arrays[name].dtype == dace.float32
    a = np.linspace(0.1, 1.7, SIZE)
    b = np.zeros(SIZE)
    sut.sdfg.compile()(a=a, b=b, N=SIZE)
    assert np.array_equal(b, np.float32(a * 3.0).astype(np.float64) + 1.0), "t holds float32 values"
    sut.apply_move("rollback", [])
    assert sut.sdfg.arrays[name].dtype == dace.float64


def test_change_precision_refuses_an_argument_an_unknown_array_and_a_second_change(tmp_path):
    sut = session(tmp_path)
    name = internal(sut)
    sut.apply_move("change-precision", [], container=name, dtype="float32")

    argument = sut.apply_move("change-precision", [], container="a", dtype="float32")
    unknown = sut.apply_move("change-precision", [], container="zz", dtype="float32")
    again = sut.apply_move("change-precision", [], container=name, dtype="float32")

    assert all(r.status is MoveStatus.ILLEGAL and r.refusal is Refusal.ARGUMENT for r in (argument, unknown, again))
    assert "relayout cast" in argument.reason


@dace.program
def copied(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    t = np.empty_like(a)
    for i in dace.map[0:N]:
        t[i] = a[i] * 3.0
    b[:] = t
    for i in dace.map[0:N]:
        c[i] = t[i] + 1.0


def test_an_array_copied_to_another_array_keeps_its_type(tmp_path):
    sut = Session(copied.to_sdfg(simplify=True), work_dir=str(tmp_path), sizes={"N": SIZE})
    sut.canonicalize()
    (name,) = [n for n, d in sut.sdfg.arrays.items() if d.transient and isinstance(d, dace.data.Array)]

    refused = sut.apply_move("change-precision", [], container=name, dtype="float32")

    assert refused.status is MoveStatus.ILLEGAL and "copied to or read by" in refused.reason
    assert sut.list_moves("change-precision") == []
