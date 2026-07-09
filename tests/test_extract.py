import numpy as np
import dace

from nestforge.strategies import outer
from nestforge.extract import extract_nest_to_sdfg

N = dace.symbol('N')


@dace.program
def vadd(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] + B[i]


def test_outer_finds_the_map():
    sdfg = vadd.to_sdfg(simplify=True)
    refs = outer(sdfg)
    assert len(refs) == 1
    _, node = refs[0]
    assert isinstance(node, dace.sdfg.nodes.MapEntry)


def test_extract_map_nest_boundary_and_correctness():
    sdfg = vadd.to_sdfg(simplify=True)
    psdfg, node = outer(sdfg)[0]
    b = extract_nest_to_sdfg(psdfg, node, name="vadd_nest")

    assert set(b.inputs) == {"A", "B"}
    assert set(b.outputs) == {"C"}
    assert "N" in b.symbols

    standalone = b.standalone_sdfg
    for name in ("A", "B", "C"):
        assert name in standalone.arrays

    # The parent SDFG (now holding the NestedSDFG) still computes vadd.
    A = np.random.default_rng(0).random(16)
    B = np.random.default_rng(1).random(16)
    C = np.zeros(16)
    sdfg(A=A, B=B, C=C, N=16)
    np.testing.assert_allclose(C, A + B)

    # The standalone SDFG computes vadd on its own.
    A2 = np.random.default_rng(2).random(16)
    B2 = np.random.default_rng(3).random(16)
    C2 = np.zeros(16)
    standalone(A=A2, B=B2, C=C2, N=16)
    np.testing.assert_allclose(C2, A2 + B2)


if __name__ == "__main__":
    test_outer_finds_the_map()
    test_extract_map_nest_boundary_and_correctness()
    print("extract OK")
