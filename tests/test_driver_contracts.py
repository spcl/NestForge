"""CI contract tests for the perf drivers -- the cross-driver invariants that keep DRIFTING.

nest-forge runs four near-copy drivers (tsvc_arena / tsvc_full / crosslang_xl / calloverhead) plus two
scripts. A fix landed in one has repeatedly failed to land in the others: the ``baseline`` ->
``simplify-parallel`` opt-mode rename, the removed ``--select`` flag, the index-array fill. Every one of
those was invisible until a real run produced nothing (or, worse, wrong numbers that still validated).

Each test here encodes a contract a copy-paste driver must satisfy, so the NEXT drift fails CI instead of
silently publishing a wrong measurement. They are deliberately generic -- they scan every driver rather
than pinning one known bug -- because the bug class, not the instance, is what recurs.
"""
import argparse
import ast
import ctypes
import importlib
from pathlib import Path

import numpy as np
import pytest

from nestforge.perf import harness

REPO = Path(__file__).resolve().parents[1]

#: Every module that exposes a ``main(argv)`` CLI. A new driver belongs here.
DRIVERS = [
    "nestforge.perf.tsvc_arena",
    "nestforge.perf.tsvc_full",
    "nestforge.perf.crosslang_xl",
    "nestforge.perf.calloverhead",
    "nestforge.perf.staticlib_overhead",
]


def captured_parser(module_name):
    """The driver's REAL argparse parser, built by its own ``main`` and intercepted at ``parse_args``.

    Built at runtime rather than AST-scanned so ``choices=list(tsvc.OPT_MODES)`` resolves to its actual
    values. Every driver constructs the parser and then calls ``parse_args`` before doing any work, so
    raising there stops before the job starts.
    """
    module = importlib.import_module(module_name)
    grabbed = {}

    class Stop(Exception):
        pass

    def intercept(self, args=None, namespace=None):
        grabbed["parser"] = self
        raise Stop

    original = argparse.ArgumentParser.parse_args
    argparse.ArgumentParser.parse_args = intercept
    try:
        module.main([])
    except Stop:
        pass
    finally:
        argparse.ArgumentParser.parse_args = original
    assert "parser" in grabbed, f"{module_name}.main([]) never reached parse_args"
    return grabbed["parser"]


@pytest.mark.parametrize("module_name", DRIVERS)
def test_every_argparse_default_is_a_valid_choice(module_name):
    """A flag's default must be a member of its own ``choices``.

    argparse validates a value against ``choices`` only when the flag is PASSED -- never when the default
    falls through. So a stale default is invisible at parse time and only surfaces deep in the consumer,
    where the driver's per-kernel ``except`` swallows it: the documented default invocation then skips
    EVERY kernel and reports zero measurements. That is exactly how the ``baseline`` -> ``simplify-parallel``
    rename escaped.
    """
    parser = captured_parser(module_name)
    bad = []
    for action in parser._actions:
        if action.choices is None or action.default is None:
            continue
        # An ``nargs='+'``/``'*'`` flag defaults to a LIST whose ELEMENTS are drawn from choices; a scalar
        # flag's default is itself a choice. Check the values either way.
        values = action.default if isinstance(action.default, (list, tuple)) else [action.default]
        for value in values:
            if value not in action.choices:
                bad.append(f"{action.option_strings}: default value {value!r} not in {list(action.choices)}")
    assert not bad, f"{module_name} has a default outside its own choices -> the default run skips everything:\n" + \
                    "\n".join(bad)


def make_inputs_calls_without_given():
    """Every ``make_inputs(...)`` call site in the package/scripts that omits ``given=``.

    A driver building inputs for a CORPUS kernel cannot know whether that kernel declares an integer index
    array, so it must always pass the manifest fills. Omitting them fills ``ip`` with
    ``(rng.random()*0.25).astype(int32)`` == all zeros: a gather ``a[i] = b[ip[i]]`` degenerates to one
    cached read of ``b[0]`` and a scatter ``a[ip[i]] = ...`` collapses onto ``a[0]``. The oracle degenerates
    identically, so validation passes VACUOUSLY while every timing number measures the wrong memory
    behaviour -- there is no failure to notice.
    """
    offenders = []
    for path in sorted([*(REPO / "nestforge").rglob("*.py"), *(REPO / "scripts").rglob("*.py")]):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if isinstance(fn, ast.Attribute):
                name = fn.attr
            elif isinstance(fn, ast.Name):
                name = fn.id
            else:
                continue  # a call on an expression (subscript, lambda, ...) names no function here
            if name != "make_inputs":
                continue
            if not any(k.arg == "given" for k in node.keywords):
                offenders.append(f"{path.relative_to(REPO)}:{node.lineno}")
    return offenders


def test_every_driver_passes_index_fills_when_building_inputs():
    assert not make_inputs_calls_without_given(), (
        "make_inputs called without given=index_fills -- integer index arrays fill to ALL-ZEROS, so gather/"
        "scatter kernels are measured degenerate and validate vacuously:\n  " +
        "\n  ".join(make_inputs_calls_without_given()))


def test_rank_and_size_fails_loud_on_size_without_rank(monkeypatch):
    """An asymmetric launcher env must fail loud in BOTH directions.

    The rank-set/size-unset direction already raises. The mirror -- size set, rank unset -- silently yields
    ``(0, size)``, so ``my_slice`` hands rank 0 only 1/size of the corpus and the run publishes a geomean
    over a fraction of the kernels with no error. ``sbatch --ntasks=4`` running the module directly (no
    ``srun``) sets SLURM_NTASKS without SLURM_PROCID and does exactly this.
    """
    for var in [*harness.RANK_VARS, *harness.SIZE_VARS]:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(harness.SIZE_VARS[0], "4")  # a size, no rank
    with pytest.raises(RuntimeError, match="size|rank"):
        harness.rank_and_size()


def test_rank_and_size_plain_run_is_single_process(monkeypatch):
    for var in [*harness.RANK_VARS, *harness.SIZE_VARS]:
        monkeypatch.delenv(var, raising=False)
    assert harness.rank_and_size() == (0, 1)


def test_c_call_args_honors_the_declared_ctype():
    """A by-value arg must be built with the ctype the signature declared, not a hardcoded int64.

    ``c_argtypes`` types a leaked FLOAT value-scalar as ``c_double``; passing it a ``c_int64`` makes the
    ctypes call raise once ``fn.argtypes`` is set, so every validated timing cell for that nest is dropped
    while the sibling driver (which uses ``t(sizes[a])``) times it fine.
    """
    from nestforge.perf.tsvc_full import c_call_args

    order = ["arr", "n", "x"]
    argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int64, ctypes.c_double]
    work = {"arr": np.zeros(4, dtype=np.float64)}
    sizes = {"n": 8, "x": 0.5}
    args = c_call_args(order, argtypes, work, sizes)
    assert isinstance(args[1], ctypes.c_int64)
    assert isinstance(args[2], ctypes.c_double), (
        f"float value-scalar built as {type(args[2]).__name__}, not c_double -- c_call_args ignored the "
        "declared argtype and hardcoded c_int64")
