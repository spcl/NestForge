"""M0 end-to-end: lower a map-nest to ExternalCall, then run it two ways --
(1) DaceReference (numpy->dace fallback/competitor), (2) ExternCall linking the arena winner --
and check both reproduce the original SDFG."""
import numpy as np
import dace

from nestforge.pass_lower import lower_nests_to_external_call
from nestforge.libnode import ExternalCall
from nestforge.translate import prepare, emit_sources
from nestforge.arena import run_arena

N = dace.symbol('N')


@dace.program
def vadd(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] + B[i]


def _ref(n):
    A = np.random.default_rng(0).random(n)
    B = np.random.default_rng(1).random(n)
    return A, B, A + B


def test_lower_inserts_external_call():
    sdfg = vadd.to_sdfg(simplify=True)
    lowered = lower_nests_to_external_call(sdfg, strategy="outer")
    assert len(lowered) == 1
    ext, boundary = lowered[0]
    assert isinstance(ext, ExternalCall)
    assert set(ext.in_connectors) == {"_in_A", "_in_B"}
    assert set(ext.out_connectors) == {"_out_C"}
    assert "def " in ext.numpy_source


def test_dace_reference_runs_correctly():
    sdfg = vadd.to_sdfg(simplify=True)
    lower_nests_to_external_call(sdfg, strategy="outer")   # default impl = DaceReference
    sdfg.expand_library_nodes()
    sdfg.validate()
    n = 1 << 12
    A, B, ref = _ref(n)
    C = np.zeros(n)
    sdfg(A=A, B=B, C=C, N=n)
    np.testing.assert_allclose(C, ref)


def test_extern_call_links_winner_and_runs(tmp_path):
    # Build + lower.
    sdfg = vadd.to_sdfg(simplify=True)
    lowered = lower_nests_to_external_call(sdfg, strategy="outer")
    ext, boundary = lowered[0]

    # Translate + arena to get a compiled winner.
    prep = prepare(boundary, ext.name, tmp_path / "kern")
    c_source = next(p for p in emit_sources(prep, tmp_path / "gen") if p.suffix == ".c")
    sizes = {"N": 1 << 14}
    res = run_arena(prep, boundary, c_source, tmp_path / "build", sizes=sizes, reps=25)
    win = res.winners["ieee-strict"]
    assert win.maxdiff == 0.0

    # Point the node at the winning lib + expand the extern call.
    ext.implementation = "ExternCall"
    ext.lib_path = win.so_path
    ext.symbol = win.symbol
    sdfg.expand_library_nodes()
    sdfg.validate()

    n = 1 << 14
    A, B, ref = _ref(n)
    C = np.zeros(n)
    sdfg(A=A, B=B, C=C, N=n)
    np.testing.assert_allclose(C, ref)


if __name__ == "__main__":
    import tempfile, pathlib
    test_lower_inserts_external_call()
    test_dace_reference_runs_correctly()
    test_extern_call_links_winner_and_runs(pathlib.Path(tempfile.mkdtemp()))
    print("M0 end-to-end OK")
