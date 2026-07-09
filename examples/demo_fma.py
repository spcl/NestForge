"""Demo: lower an FMA-sensitive elementwise nest and print the arena report.

``E[i] = A[i]*B[i] + D[i]`` is a fused-multiply-add candidate, so the FP modes diverge:
``ieee-strict`` (``-ffp-contract=off``) must be bit-exact vs the numpy oracle, while ``fast-math``
(FMA on) may round differently. Run: ``python examples/demo_fma.py``.
"""
import tempfile
from pathlib import Path

import dace

from nestforge.pass_lower import lower_nests_to_external_call
from nestforge.translate import prepare, emit_sources
from nestforge.arena import run_arena
from nestforge.report import render_markdown

N = dace.symbol('N')


@dace.program
def fma(A: dace.float64[N], B: dace.float64[N], D: dace.float64[N], E: dace.float64[N]):
    for i in dace.map[0:N]:
        E[i] = A[i] * B[i] + D[i]


def main():
    work = Path(tempfile.mkdtemp(prefix="nf_demo_"))
    sdfg = fma.to_sdfg(simplify=True)
    (ext, boundary), = lower_nests_to_external_call(sdfg, strategy="outer")

    prep = prepare(boundary, ext.name, work / "kern")
    c_source = next(p for p in emit_sources(prep, work / "gen") if p.suffix == ".c")

    result = run_arena(prep, boundary, c_source, work / "build", sizes={"N": 1 << 20}, reps=50)
    print(render_markdown(result))


if __name__ == "__main__":
    main()
