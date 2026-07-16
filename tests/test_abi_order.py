"""The ABI-order contract: whatever order nest-forge binds ctypes arguments in MUST be the order the
emitted kernel actually declares.

numpyto emits the C signature in ``param_order()`` -- arrays sorted alphabetically, then scalars sorted --
and its IR docstring says it "deliberately ignores input_args for ordering". nest-forge's manifest
``input_args`` is a ROLE order (inputs, then outputs, then symbols). The two coincide only by luck of the
alphabet, and every in-tree demo kernel (vadd, fma) happens to be lucky: ``[A, B, C, N]`` is already
sorted. A kernel whose OUTPUT sorts before its INPUT breaks the tie -- and since both are ``double*``,
ctypes cannot catch it: the kernel writes through the wrong pointer, silently.

`harness.signature_order` exists precisely to parse the real emitted signature. These tests pin that the
binding follows it.
"""
import numpy as np
import pytest

import dace

from nestforge.extract import extract_nest_to_sdfg
from nestforge.perf.harness import signature_order
from nestforge.strategies import get_strategy
from nestforge.translate import emit_sources, prepare

N = dace.symbol("N")


@dace.program
def writes_a_reads_b(a: dace.float64[N], b: dace.float64[N]):
    """Role order is [b, a, N] (input, output, symbol); sorted order is [a, b, N]. The output `a` sorts
    BEFORE the input `b`, so the two orders disagree -- the exact shape the lucky demo kernels dodge."""
    for i in dace.map[0:N]:
        a[i] = b[i] + 1.0


def prepared_nest(tmp_path):
    sdfg = writes_a_reads_b.to_sdfg(simplify=True)
    parent, node = get_strategy("outer")(sdfg)[0]
    boundary = extract_nest_to_sdfg(parent, node, name="wab")
    prep = prepare(boundary, "wab", tmp_path, sizes={"N": 32})
    return prep, boundary


@pytest.mark.integration  # runs the numpyto emitter
def test_emitted_signature_disagrees_with_manifest_role_order(tmp_path):
    """Pins the HAZARD itself, so nobody "simplifies" a binder back onto manifest order.

    This is not a bug -- numpyto is entitled to its own parameter order -- it is the reason every binder
    must parse the emitted signature. The day these coincide for this kernel, the guard below stops
    guarding anything.
    """
    prep, boundary = prepared_nest(tmp_path)
    csrc = next(s for s in emit_sources(prep, tmp_path, target="c") if s.suffix == ".c" and "pluto" not in s.name)
    emitted = signature_order(csrc.read_text(), "wab_fp64")
    assert emitted == ["a", "b", "N"]  # sorted arrays, then scalars
    assert list(prep.manifest["input_args"]) == ["b", "a", "N"]  # role order: input, output, symbol
    assert emitted != list(prep.manifest["input_args"]), \
        "the two orders now coincide for this kernel -- pick one whose output still sorts before its input"


@pytest.mark.integration  # compiles + runs the nest
def test_arena_binds_by_the_emitted_signature_not_the_manifest(tmp_path):
    """End-to-end: the arena must compute a[i] = b[i] + 1 correctly for a kernel whose OUTPUT sorts before
    its INPUT.

    Binding by manifest order hands the kernel arena's `b` where it expects `a` and vice versa: it writes
    into the input buffer and reads the output buffer, so `a` comes back as the untouched zeros the arena
    allocated for it (and `b`, which the test does not read, silently holds the answer). Both are double*,
    so ctypes raises nothing -- only the value is wrong.
    """
    from nestforge.arena import run_arena

    prep, boundary = prepared_nest(tmp_path)
    csrc = next(s for s in emit_sources(prep, tmp_path, target="c") if s.suffix == ".c" and "pluto" not in s.name)
    result = run_arena(prep, boundary, csrc, tmp_path / "build", sizes={"N": 32}, reps=2)
    cells = [c for c in result.cells if c.ok]
    assert cells, f"no arena cell validated -- the ABI bind order is wrong: {[c.error for c in result.cells]}"
