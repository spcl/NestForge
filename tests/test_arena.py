import numpy as np
import dace

from nestforge.strategies import outer
from nestforge.extract import extract_nest_to_sdfg
from nestforge.translate import prepare, emit_sources
from nestforge.arena import run_arena, discover_compilers, FP_MODES

N = dace.symbol('N')


@dace.program
def vadd(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] + B[i]


def test_arena_vadd(tmp_path):
    sdfg = vadd.to_sdfg(simplify=True)
    psdfg, node = outer(sdfg)[0]
    b = extract_nest_to_sdfg(psdfg, node, name="vadd")

    prep = prepare(b, "vadd", tmp_path / "kern")
    srcs = emit_sources(prep, tmp_path / "gen", target="c")
    c_source = next(p for p in srcs if p.suffix == ".c")

    sizes = {"N": 1 << 15}
    res = run_arena(prep, b, c_source, tmp_path / "build", sizes=sizes, reps=50)

    assert discover_compilers(), "no compilers on PATH"
    # every FP mode has at least one correct build, and a winner
    for mode in FP_MODES:
        assert mode in res.winners, f"no correct build for {mode}: {[(c.compiler,c.fp_mode,c.ok,c.maxdiff) for c in res.cells if c.fp_mode==mode]}"
    # ieee-strict must be bit-exact vs the numpy oracle
    assert res.winners["ieee-strict"].maxdiff == 0.0
    # all vadd cells are correct (a pure add reassociates trivially)
    assert all(c.ok for c in res.cells), [(c.compiler, c.fp_mode, c.maxdiff, c.error) for c in res.cells if not c.ok]


if __name__ == "__main__":
    import tempfile, pathlib
    test_arena_vadd(pathlib.Path(tempfile.mkdtemp()))
    print("arena OK")
