import numpy as np
import dace

from nestforge.strategies import outer
from nestforge.extract import extract_nest_to_sdfg
from nestforge.emit_numpy import nest_to_numpy
from nestforge.translate import prepare, emit_sources

N = dace.symbol('N')


@dace.program
def vadd(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] + B[i]


def boundary():
    sdfg = vadd.to_sdfg(simplify=True)
    psdfg, node = outer(sdfg)[0]
    return extract_nest_to_sdfg(psdfg, node, name="vadd_nest")


def test_numpy_emit_runs():
    b = boundary()
    src = nest_to_numpy(b, fn_name="vadd")
    ns = {}
    exec(src, ns)
    A = np.random.default_rng(0).random(32)
    B = np.random.default_rng(1).random(32)
    C = np.zeros(32)
    ns["vadd"](A=A, B=B, C=C, N=32)
    np.testing.assert_allclose(C, A + B)


def test_translate_to_c(tmp_path):
    b = boundary()
    prep = prepare(b, "vadd", tmp_path / "kern")
    assert prep.numpy_path.exists() and prep.yaml_path.exists()
    srcs = emit_sources(prep, tmp_path / "gen", target="c")
    c_files = [p for p in srcs if p.suffix == ".c"]
    assert c_files, f"no C emitted; got {srcs}"
    text = c_files[0].read_text()
    # correct ABI: three double* arrays + an int64 size (order = input_args)
    assert "double *restrict A" in text
    assert "double *restrict C" in text
    assert "int64_t N" in text
    assert "(A[i] + B[i])" in text
    assert "C[i] = " in text


if __name__ == "__main__":
    test_numpy_emit_runs()
    import tempfile
    import pathlib
    test_translate_to_c(pathlib.Path(tempfile.mkdtemp()))
    print("translate OK")
