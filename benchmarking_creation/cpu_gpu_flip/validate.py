"""Check the three C schedules against the numpy oracle; optionally time them (single thread).

    python validate.py            # correctness only, preset-S-like and an odd-sized case
    python validate.py --time     # also time each schedule at a larger size (CPU, one thread)
"""

import argparse
import ctypes
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from cpu_gpu_flip_numpy import cpu_gpu_flip

HERE = Path(__file__).parent
VARIANTS = ("v0_naive", "v1_cpu", "v2_gpu")


def build() -> ctypes.CDLL:
    out = Path(tempfile.mkdtemp()) / "variants.so"
    # No fast-math and no FMA contraction, so the C result stays within rounding of numpy.
    subprocess.run(["cc", "-O3", "-ffp-contract=off", "-shared", "-fPIC", str(HERE / "cpu_gpu_flip_variants.c"),
                    "-o", str(out), "-lm"], check=True)
    return ctypes.CDLL(str(out))


def inputs(nlev: int, ncol: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    shape = (nlev, ncol)
    return (rng.uniform(200.0, 300.0, shape), rng.uniform(0.0, 0.02, shape), rng.uniform(1e4, 1e5, shape))


def run_c(lib, name: str, t, q, p):
    nlev, ncol = t.shape
    x = np.zeros_like(t)
    dp = ctypes.POINTER(ctypes.c_double)
    getattr(lib, name)(*(a.ctypes.data_as(dp) for a in (t, q, p, x)), ctypes.c_long(nlev), ctypes.c_long(ncol))
    return x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", action="store_true")
    args = parser.parse_args()
    lib = build()
    for nlev, ncol in ((32, 512), (7, 1000), (137, 777)):  # 777 is not a multiple of the block size
        t, q, p = inputs(nlev, ncol)
        x = np.zeros_like(t)
        cpu_gpu_flip(t.copy(), q.copy(), p.copy(), x, nlev, ncol)
        for name in VARIANTS:
            got = run_c(lib, name, t, q, p)
            err = float(np.max(np.abs(got - x) / (np.abs(x) + 1e-300)))
            print(f"nlev={nlev:4d} ncol={ncol:5d} {name:9s} max rel err {err:.2e}  {'ok' if err < 1e-9 else 'FAIL'}")
    if args.time:
        nlev, ncol = 137, 32768
        t, q, p = inputs(nlev, ncol)
        print(f"\ntiming nlev={nlev} ncol={ncol}, one thread, best of 3 (this machine's CPU only)")
        for name in VARIANTS:
            best = min(_timed(lambda: run_c(lib, name, t, q, p)) for _ in range(3))
            print(f"  {name:9s} {best * 1e3:9.1f} ms")


def _timed(fn) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


if __name__ == "__main__":
    main()
