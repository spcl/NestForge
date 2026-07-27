# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Offload correctness + overhead baseline over the optarena corpus (single compiler, single flag set).

For each corpus nest this: (1) emits nest-forge's offload path (numpy -> translator C), builds it with
one fixed compiler+flags (nest-forge OWNS this build -- gcc on the emitted C), and times a bare ctypes
call; (2) runs the SAME nest through DaCe's own codegen as a cross-check; (3) validates the offload
output against the numpy oracle AND against DaCe's result. Reports per nest: offload time, offload==oracle,
offload==dace.

Scope note: the *fair head-to-head offload-vs-DaCe TIME* is deliberately NOT reported here -- DaCe's
Python __call__ marshaling and instrumentation-report semantics confound it. Per the design, nest-forge
should own the DaCe build too (own codegen+compile+link, a C++ <chrono> timing-transformation, and LTO
static-lib inlining -- see BUILD.md); the overhead ratio lands once that subsystem exists. What this
script establishes now is that the offload path is CORRECT ("check if correct nicely") and how fast the
offloaded C itself runs.

Usage:
    python scripts/overhead_baseline.py [--track hpc|ml|foundation] [--limit N] [--reps 50]
"""
from __future__ import annotations

import argparse
import copy
import ctypes
import re
import subprocess
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

import dace  # noqa: E402
from dace import symbolic  # noqa: E402

from nestforge.corpus import iter_dace_kernels  # noqa: E402
from nestforge.strategies import get_strategy  # noqa: E402
from nestforge.extract import extract_nest_to_sdfg  # noqa: E402
from nestforge.translate import prepare, emit_sources  # noqa: E402
from nestforge.arena import (
    CTYPE,
    discover_compilers,
    make_inputs,
    rewind,
    rewind_snapshot,  # noqa: E402
    run_oracle)
from nestforge.tsvc import index_fills_for_manifest  # noqa: E402

# one compiler, one flag set -- the fixed operating point for the overhead comparison.
FLAGS = ["-O3", "-march=native", "-fPIC", "-shared"]
DEFAULT_SIZE = 96


def sizes_for(boundary) -> dict:
    """A size for each boundary symbol: a size symbol (appears in an array shape) -> DEFAULT_SIZE;
    a loop-carried index (appears in no array shape) -> 0 (the nest resets it before use)."""
    sdfg = boundary.standalone_sdfg
    shape_syms = {
        str(s)
        for d in sdfg.arrays.values()
        for dim in d.shape
        for s in symbolic.pystr_to_symbolic(dim).free_symbols
    }
    return {s: (DEFAULT_SIZE if s in shape_syms else 0) for s in boundary.symbols}


def abi_order(csrc: Path, symbol: str) -> list:
    sig = re.search(rf"void\s+{symbol}\s*\((.*?)\)\s*\{{", csrc.read_text(), re.S).group(1)
    return [p.strip().split()[-1].lstrip("*") for p in sig.split(",")]


def time_offload(prep, boundary, sizes, inputs, reps):
    """Compile the emitted C once (gcc, FLAGS), call via ctypes; return (outputs, time_us)."""
    work = Path(tempfile.mkdtemp(prefix="nf_ovh_c_"))
    csrc = next(p for p in emit_sources(prep, work) if p.suffix == ".c" and "pluto" not in p.name)
    symbol = f"{prep.name}_fp64"
    order = abi_order(csrc, symbol)
    bsdfg = boundary.standalone_sdfg
    gcc = discover_compilers()["gcc"]
    so = work / f"lib_{prep.name}.so"
    subprocess.run([gcc, *FLAGS, str(csrc), "-o", str(so)], check=True, capture_output=True)
    argt = [
        ctypes.POINTER(CTYPE[np.dtype(bsdfg.arrays[a].dtype.type).name]) if a in bsdfg.arrays else ctypes.c_int64
        for a in order
    ]
    fn = ctypes.CDLL(str(so))[symbol]
    fn.argtypes, fn.restype = argt, None

    def args_for(buf):
        return [buf[a].ctypes.data_as(t) if a in buf else ctypes.c_int64(int(sizes[a])) for a, t in zip(order, argt)]

    fresh = {k: v.copy() for k, v in inputs.items()}  # correctness on a fresh buffer
    fn(*args_for(fresh))
    outputs = {o: fresh[o].copy() for o in boundary.outputs}
    tbuf = {k: v.copy() for k, v in inputs.items()}
    cargs = args_for(tbuf)  # built ONCE; the reused buffers' pointers stay valid -> bare C call in loop
    # Snapshot BEFORE the warm call: an in-place nest mutates its own inputs, so a snapshot taken after it
    # would restore every rep to `a*b` rather than to `a`. Timing 50 reps without any restore measured
    # `a * b**50` -- denormal arithmetic, not the kernel -- and this script's whole output is the offload
    # time. Both the warm call and every rep start from the same state; the restore is outside the timing.
    snapshot = rewind_snapshot(boundary, tbuf)
    fn(*cargs)  # warm
    total = 0.0
    for _ in range(reps):
        rewind(snapshot)
        t0 = time.perf_counter()
        fn(*cargs)
        total += time.perf_counter() - t0
    return outputs, total / reps * 1e6


def dace_reference(boundary, sizes, inputs):
    """Run the standalone nest through DaCe's OWN codegen as a semantic cross-check (does nest-forge's
    offloaded C match what DaCe itself computes for the same nest?). Correctness only -- a *fair*
    head-to-head TIME needs nest-forge to own the DaCe build too (own compile + a C++ <chrono> timer +
    LTO, not sdfg.compile / sdfg.instrument, whose Python wrapper and report semantics confound timing);
    that owned-build subsystem is specified in BUILD.md and is the next step for the overhead number."""
    sdfg = copy.deepcopy(boundary.standalone_sdfg)
    csdfg = sdfg.compile()
    symvals = {s: int(sizes[s]) for s in boundary.symbols}
    # DaCe passes a Scalar descriptor BY VALUE (nest-forge's C-style emission treats the same argument
    # as a size-1 array pointer) -- so hand DaCe the scalar value, not the array make_inputs built.
    scalars = {a for a, d in sdfg.arrays.items() if isinstance(d, dace.data.Scalar)}
    fresh = {k: v.copy() for k, v in inputs.items()}
    csdfg(**{**{k: (v.item() if k in scalars else v) for k, v in fresh.items()}, **symvals})
    return {o: fresh[o].copy() for o in boundary.outputs if o not in scalars}


def matches(outs, oracle) -> bool:
    return all(np.allclose(outs[o], oracle[o], rtol=1e-9, atol=1e-9, equal_nan=True) for o in oracle)


def run(track=None, limit=None, reps=50):
    rows = []
    strat = get_strategy("skip-taskloops")
    for k in iter_dace_kernels(track):
        try:
            sdfg = k.to_sdfg(simplify=True)
        except Exception:
            continue
        for idx, (parent, node) in enumerate(strat(sdfg)):
            name = f"{k.short_name.split('/')[-1]}_{idx}"
            try:
                boundary = extract_nest_to_sdfg(parent, node, name=name)
                sizes = sizes_for(boundary)
                # The manifest's index arrays, keyed by the kernel's manifest name (these are OptArena
                # kernels, not TsvcKernels). Without them a declared integer index array fills to all-zeros
                # and the gather being measured degenerates to one cached read.
                fills = index_fills_for_manifest(k.short_name.split("/")[-1], boundary, sizes)
                inputs = make_inputs(boundary, sizes, seed=0, given=fills)
                prep = prepare(boundary, name, Path(tempfile.mkdtemp(prefix="nf_ovh_p_")))
                oracle = run_oracle(prep, boundary, inputs, sizes)
                off_out, off_us = time_offload(prep, boundary, sizes, inputs, reps)
                dace_out = dace_reference(boundary, sizes, inputs)
                rows.append((name, off_us, matches(off_out, oracle), matches(dace_out, off_out)))
            except Exception as e:
                rows.append((name, f"skip:{type(e).__name__}", None, None))
        if limit and len(rows) >= limit:
            break

    # nest-forge OWNS the offload build (gcc on the emitted C, raw ctypes) -> offload_us is a real,
    # marshaling-free measurement. The DaCe column is a correctness cross-check (does the offload match
    # DaCe's own codegen for the same nest?); the fair head-to-head TIME is the owned-build + C++ <chrono>
    # timing-transformation + LTO work specified in BUILD.md.
    print(f"{'nest':40s} {'offload_us':>11s}  {'off==oracle':>11s} {'off==dace':>10s}")
    print("-" * 78)
    for name, off_us, off_ok, dace_ok in rows:
        if off_ok is None:
            print(f"{name:40s} {str(off_us):>34s}")
        else:
            print(f"{name:40s} {off_us:11.2f}  {str(off_ok):>11s} {str(dace_ok):>10s}")
    ran = [r for r in rows if r[2] is not None]
    good = [r for r in ran if r[2] is True and r[3] is True]
    if ran:
        print(f"\n{len(good)}/{len(ran)} nests: offload matches BOTH the numpy oracle and DaCe's own "
              f"codegen. Fair offload-vs-dace overhead ratio pending the owned-build subsystem (BUILD.md).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--reps", type=int, default=50)
    args = ap.parse_args()
    run(track=args.track, limit=args.limit, reps=args.reps)
