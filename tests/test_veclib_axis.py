# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The veclib axis and the fp-level vocabulary the arena compiles against.

Every rule here was measured with ``nm`` on a ``sin()`` loop before it was encoded; the docstrings name
the measurement, so a future toolchain that behaves differently fails loudly instead of silently
re-enabling a cell that produces a duplicate binary.
"""
import shutil
import subprocess

import pytest

from nestforge.arena import compile_object
from nestforge.optimizers import ExternalOptimizer, deterministic_optimizers
from nestforge.perf import flags

KERNEL = "#include <math.h>\nvoid k(double *restrict a, const double *restrict b, int n){\n" \
         "  for (int i = 0; i < n; ++i) a[i] = sin(b[i]);\n}\n"


def external(veclib, fp_mode="fast-math", family="gnu", compiler="gcc"):
    return ExternalOptimizer("c", family, compiler, fp_mode=fp_mode, cost_model="default", veclib=veclib)


def test_svml_is_declined_on_gnu_and_accepted_on_llvm():
    """gcc emits ``_ZGV*`` even under ``-mveclibabi=svml`` (measured), never ``__svml_*``, so an SVML link
    resolves nothing. clang emits ``__svml_*`` when told via ``-fveclib=SVML``."""
    gnu = external("svml")
    assert gnu.propose() is None
    assert "svml" in gnu.skip_reason and "gcc" in gnu.skip_reason
    assert external("svml", family="llvm", compiler="clang").propose() is not None


@pytest.mark.parametrize("family,compiler", [("gnu", "gcc"), ("llvm", "clang")])
@pytest.mark.parametrize("fp_mode", [m for m in flags.FP_LEVELS if m != "fast-math"])
def test_a_veclib_below_fast_math_is_declined_as_a_duplicate(family, compiler, fp_mode):
    """Measured by ``nm -u`` on a sin() loop: BOTH gcc and clang emit zero packed-math symbols at
    strict-ieee/contract-fma/assume-finite -- clang too, even with an explicit ``-fveclib``. Such a cell is
    byte-identical to ``veclib=none``, so measuring it would report a duplicate as a distinct variant."""
    opt = external("libmvec", fp_mode=fp_mode, family=family, compiler=compiler)
    assert opt.propose() is None
    assert "fast-math" in opt.skip_reason


@pytest.mark.parametrize("veclib", ["sleef", "libmvec"])
def test_the_veclib_axis_is_live_at_fast_math_on_both_families(veclib):
    """Where the calls really are emitted, the cell must survive -- otherwise the axis is dead."""
    assert external(veclib, family="gnu", compiler="gcc").propose() is not None
    assert external(veclib, family="llvm", compiler="clang").propose() is not None


def test_the_duplicate_gate_is_scoped_to_the_families_it_was_measured_on():
    """gnu and llvm were measured; intel was not, so it is left alone rather than gated on a guess. This
    also pins that the decline above comes from the gate, not from some unrelated composition failure."""
    gated = external("libmvec", fp_mode="strict-ieee", family="gnu", compiler="gcc")
    assert gated.propose() is None
    ungated = external("libmvec", fp_mode="strict-ieee", family="intel", compiler="icx")
    assert ungated.skip_reason is None or "fast-math" not in ungated.skip_reason


def test_veclib_none_survives_every_fp_level():
    """``none`` is the scalar floor and never depends on the fp rung; it must never be gated."""
    for fp_mode in flags.FP_LEVELS:
        assert external("none", fp_mode=fp_mode).propose() is not None


def test_a_declined_cell_always_carries_a_reason():
    """A dropped variant with no reason reads as "never tried"; every decline must say why."""
    for opt in (external("svml"), external("libmvec", fp_mode="strict-ieee")):
        assert opt.propose() is None
        assert opt.skip_reason


def test_deterministic_optimizers_enumerates_the_veclib_axis():
    """The axis has to reach the sweep, not merely exist on the class."""
    opts = deterministic_optimizers(external=(("c", "gnu", "gcc"), ),
                                    fp_modes=("fast-math", ),
                                    cost_models=("default", ),
                                    veclibs=flags.VECLIBS)
    ext = [o for o in opts if isinstance(o, ExternalOptimizer)]
    assert {o.veclib for o in ext} == set(flags.VECLIBS)
    assert [o.propose() is not None for o in ext].count(True) == 3  # all but svml


@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not on PATH")
@pytest.mark.parametrize("fp_mode", flags.FP_LEVELS)
def test_compile_object_accepts_every_fp_level(tmp_path, fp_mode):
    """``arena.FP_MODES`` had three gnu-only spellings and no ``contract-fma``; the external lane already
    emitted :data:`flags.FP_LEVELS` names, so those raised KeyError at the seam."""
    src = tmp_path / "k.c"
    src.write_text(KERNEL)
    assert compile_object("gcc", fp_mode, src, "k", tmp_path / fp_mode).exists()


@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not on PATH")
def test_compile_object_refuses_an_unsupported_combination(tmp_path):
    """An unsupported axis must fail loudly: compiling it anyway would measure something that is not what
    was asked for and label it with what was."""
    src = tmp_path / "k.c"
    src.write_text(KERNEL)
    with pytest.raises(ValueError, match="svml"):
        compile_object("gcc", "fast-math", src, "k", tmp_path / "svml", veclib="svml")
    with pytest.raises(ValueError, match="nope"):
        compile_object("gcc", "fast-math", src, "k", tmp_path / "nope", veclib="nope")


@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not on PATH")
def test_compile_object_emits_position_independent_code(tmp_path):
    """The object is linked into the parent ``.so``; ``-fPIC`` moved out of the fp-mode table, so nothing
    else carries it. Linking is the real check -- x86-64 rejects a non-PIC object in a shared library."""
    src = tmp_path / "k.c"
    src.write_text(KERNEL)
    obj = compile_object("gcc", "strict-ieee", src, "k", tmp_path / "pic")
    so = tmp_path / "libk.so"
    done = subprocess.run(["gcc", "-shared", str(obj), "-o", str(so)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-400:]
    assert so.exists()
