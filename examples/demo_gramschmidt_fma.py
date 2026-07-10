"""Demo: gramschmidt is numerically stable under fast-math *only when well-conditioned*.

The kernel's two ``np.dot`` reductions lower to ``s += a[i]*b[i]``. ``-ffast-math`` reassociates that
sum (vector partial sums), changing the accumulation order. This prints, for a well-conditioned and an
ill-conditioned input, each compile mode's **relative error vs the ieee-strict sequential baseline**
(the arena's stability metric). The lesson: FMA contraction alone is bit-exact; it is *reassociation*
that diverges, and only when a near-zero pivot amplifies it (cond ~ 1e14). Run:
``python examples/demo_gramschmidt_fma.py``.
"""
import ctypes
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from dace import symbolic

from nestforge.corpus import iter_dace_kernels
from nestforge.pass_lower import lower_nests_to_external_call
from nestforge.translate import prepare, emit_sources

_CT = {"float64": ctypes.c_double, "int64": ctypes.c_int64}
_BASE = ["-O3", "-march=native", "-fPIC", "-shared"]
_MODES = {
    "ieee-strict-seq": ["-ffp-contract=off", "-fno-tree-vectorize"],
    "contract-fast (FMA)": ["-ffp-contract=fast"],
    "O3 (auto-vec)": [],
    "fast-math (reassoc)": ["-ffast-math"],
}


def _A(M, N, conditioning):
    rng = np.random.default_rng(0)
    if conditioning == "well":
        return rng.standard_normal((M, N))
    U, _ = np.linalg.qr(rng.standard_normal((M, N)))
    V, _ = np.linalg.qr(rng.standard_normal((N, N)))
    return U @ np.diag(np.logspace(0, -14, N)) @ V  # cond ~ 1e14


def main():
    gcc = shutil.which("gcc")
    if gcc is None:
        raise SystemExit("gcc not on PATH")
    work = Path(tempfile.mkdtemp(prefix="nf_gs_demo_"))
    sdfg = {k.short_name: k for k in iter_dace_kernels()}[
        "hpc/dense_linear_algebra/gramschmidt/gramschmidt"].to_sdfg(simplify=True)
    _, boundary = lower_nests_to_external_call(sdfg, strategy="outer")[2]
    prep = prepare(boundary, "gs_compute", work / "kern")
    csrc = next(p for p in emit_sources(prep, work / "gen") if p.suffix == ".c" and "pluto" not in p.name)
    order = [p.strip().split()[-1].lstrip("*") for p in
             re.search(r"void\s+gs_compute_fp64\s*\((.*?)\)\s*\{", csrc.read_text(), re.S).group(1).split(",")]
    bsdfg = boundary.standalone_sdfg
    M, N = 256, 48
    sizes = {"M": M, "N": N, "j": 0, "k": 0}
    env = {symbolic.symbol("M"): M, symbolic.symbol("N"): N}
    argt = [ctypes.POINTER(_CT[np.dtype(bsdfg.arrays[a].dtype.type).name]) if a in bsdfg.arrays
            else ctypes.c_int64 for a in order]

    print("gramschmidt compute nest -- rel-err vs ieee-strict-seq baseline")
    print(f"{'mode':22s} | {'well-conditioned':>18s} | {'ill-conditioned (1e14)':>22s}")
    print("-" * 68)
    results = {}
    for conditioning in ("well", "ill"):
        A = _A(M, N, conditioning)
        base = {a: (A.copy() if a == "A" else
                    np.zeros(tuple(int(symbolic.evaluate(x, env)) for x in bsdfg.arrays[a].shape),
                             np.dtype(bsdfg.arrays[a].dtype.type)))
                for a in order if a in bsdfg.arrays}
        for mode, flags in _MODES.items():
            so = work / f"lib_{conditioning}_{mode.split()[0]}.so"
            subprocess.run([gcc, *_BASE, *flags, str(csrc), "-o", str(so)], check=True, capture_output=True)
            fn = ctypes.CDLL(str(so)).gs_compute_fp64
            fn.argtypes, fn.restype = argt, None
            buf = {a: v.copy() for a, v in base.items()}
            fn(*[buf[a].ctypes.data_as(t) if a in buf else ctypes.c_int64(sizes[a])
                 for a, t in zip(order, argt)])
            results[(conditioning, mode)] = {o: buf[o].copy() for o in ("__return_0", "__return_1")}

    for mode in _MODES:
        row = []
        for conditioning in ("well", "ill"):
            got = results[(conditioning, mode)]
            ref = results[(conditioning, "ieee-strict-seq")]
            num = max(float(np.max(np.abs(got[k] - ref[k]))) for k in got)
            den = max(float(np.max(np.abs(ref[k]))) for k in got) or 1.0
            row.append(num / den)
        print(f"{mode:22s} | {row[0]:18.2e} | {row[1]:22.2e}")
    print("\nfast-math is stable when well-conditioned, dangerous when ill-conditioned (reassociation")
    print("of the dot-product reductions, amplified by the near-zero pivot). See docs/FP_RISK.md.")


if __name__ == "__main__":
    main()
