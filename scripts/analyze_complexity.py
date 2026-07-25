# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Classify each corpus kernel's COMPUTE COMPLEXITY, so presets can be sized by work rather than by the
name of the size symbol.

The symbol does not decide the class: several ``LEN_1D`` kernels carry a nested loop and run O(n^2), and
several ``LEN_2D`` kernels sweep one loop and run O(n).

``k`` is read STATICALLY from the kernel's own ``_reference.cpp``: a loop multiplies the work when its
CONDITION (never its init or increment) names something that SCALES -- a scalar parameter, or the index
of an enclosing loop that itself scales -- and does not name the index of an enclosing STRIDED loop.

Each clause earns its place against a kernel the simpler rule got wrong:

  * "names a scalar parameter" rather than "spells ``LEN_1D``": the baselines rebind the size to a local
    (``s471``: ``int m = len_1d;`` then ``for (i = 0; i < m; ++i)``), so name-matching the manifest
    symbol reported 7 kernels as O(1). Local rebindings are followed to a fixed point.
  * "or an enclosing scaling index": a TRIANGULAR inner bound names NO parameter at all, only the outer
    index (``s118``: ``for (j = 0; j <= i - 1; ++j)``) -- yet it is genuinely O(n^2).
  * "not an enclosing STRIDED index": this is what separates work from nesting depth.
    ``jacobi2d_double_tiled_sym`` nests six loops and does O(n^2) work, because the inner ones run
    ``iii < ii + t1_v`` inside ``ii += t1_v``. The stride is the tell: the same ``bound names the outer
    index`` shape is tile-local under ``ii += t`` and triangular under ``++i``.

Known limit, reported not hidden: a loop whose body is a single unbraced statement opens no scope for
the brace walk, so it is not counted (``s31111``, which lands in the skip list rather than in a class).

Timing was tried first and REJECTED: fitting ``log(time) ~ k*log(n)`` reports 1.54 for the provably O(n)
``s000`` because the size ladder crosses L2 into L3, and -0.07 for ``vbor``. Cache transitions swamp the
exponent at any ladder short enough to run, so the static rule is the reliable one -- and this script does
no timing at all, so it needs no compiler and no run.

Usage: ``python scripts/analyze_complexity.py [--out FILE] [--corpus tsvc2 tsvc2_5] [--kernels K ...]``
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from nestforge import tsvc
from nestforge.perf.harness import native_symbol

#: Identifiers in a loop condition / an ``extern "C"`` parameter list.
NAME = re.compile(r"\b[A-Za-z_]\w*\b")

#: A local integer binding, ``int m = len_1d;`` -- the baselines rebind the size before looping on it.
LOCAL = re.compile(r"\b(?:int|long|unsigned|size_t|std::size_t)\s+(\w+)\s*=\s*([^;]+);")


def strip_comments(src: str) -> str:
    """Comments carry prose like ``// for i: if a[i] > x`` and the attribution header's parentheses, which
    the brace/paren walk below would otherwise read as code."""
    return re.sub(r"//[^\n]*", " ", re.sub(r"/\*.*?\*/", " ", src, flags=re.S))


def scalar_params(src: str, symbol: str) -> set:
    """The kernel's by-value scalar parameter names -- the only bounds that can make a loop scale.

    Pointer parameters are the data, so they are excluded: a loop bounded by one would be a bug, not a
    size. Returns an empty set when the entry is not found, which makes every loop uncountable and the
    kernel a recorded skip rather than a silent O(1)."""
    match = re.search(rf"\b{re.escape(symbol)}\s*\(([^)]*)\)", src)
    if match is None:
        return set()
    names = set()
    for param in match.group(1).split(","):
        if "*" in param:
            continue  # a pointer: data, not a bound
        found = NAME.findall(param)
        if found:
            names.add(found[-1])  # the declarator, after the type keywords
    return with_derived_locals(src, names)


def with_derived_locals(src: str, names: set) -> set:
    """``names`` plus every local bound from one of them, to a fixed point.

    ``s471`` does ``int m = len_1d;`` and then loops on ``m``. Matching parameter names alone reads that
    loop as unbounded-by-size and reports the kernel as O(1) -- so the rebinding has to be followed."""
    out = set(names)
    while True:
        grown = {n for n, rhs in LOCAL.findall(src) if set(NAME.findall(rhs)) & out} - out
        if not grown:
            return out
        out |= grown


def work_exponent(src: str, params: set) -> int:
    """The deepest SIMULTANEOUS nesting of work-carrying loops = the work exponent k.

    Walks braces to track which loops are still open, so two sequential nests count as the deeper of the
    two rather than their sum -- sequential work adds, only nesting multiplies. A loop carries work per
    the condition rule in the module docstring."""
    best = cur = depth = 0
    open_loops: List[tuple] = []  # (induction var, closing brace depth, whether it counted, strided)
    i = 0
    while i < len(src):
        c = src[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            while open_loops and open_loops[-1][1] > depth:
                cur -= int(open_loops.pop()[2])
        elif src.startswith("for", i) and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == "_")):
            j = src.find("(", i)
            if j != -1:
                d, end = 1, j + 1
                while end < len(src) and d:
                    d += (src[end] == "(") - (src[end] == ")")
                    end += 1
                # The brace must follow the header across whitespace ONLY. A `for` whose body is a single
                # statement opens no scope, so the brace walk would never close it and every later loop
                # would be counted as nested inside it.
                body = end
                while body < len(src) and src[body].isspace():
                    body += 1
                if body < len(src) and src[body] == "{":
                    parts = src[j + 1:end - 1].split(";")
                    init, cond, step = (parts + ["", "", ""])[:3]
                    iv = NAME.findall(init.split("=")[0])  # the DECLARATOR, not the initializer
                    names = set(NAME.findall(cond))
                    # A bound naming an enclosing induction variable is tile-local only when that
                    # enclosing loop STRIDES (``ii += t``). When it steps by one and itself scales, the
                    # same shape is a TRIANGULAR nest (``for j <= i - 1``) -- which really does multiply
                    # the work, and whose bound names no parameter at all, only the outer index.
                    tiled = {var for var, _close, _counted, strided in open_loops if strided}
                    scaling = params | {var for var, _close, counted, strided in open_loops if counted and not strided}
                    counts = bool(names & scaling) and not (names & tiled)
                    cur += int(counts)
                    best = max(best, cur)
                    unit = bool(re.search(r"(\+\+|--)|[-+]=\s*1\b", step))
                    open_loops.append((iv[-1] if iv else "", depth + 1, counts, not unit))
        i += 1
    return best


def size_symbol(kernel: tsvc.TsvcKernel) -> Optional[str]:
    """The manifest size symbol this kernel scales with, or ``None`` when it declares none we classify.
    ``yaml_presets`` is keyed by SHAPE SYMBOL (``{LEN_1D: [512, 32768, ...]}``), not by preset tier."""
    presets = tsvc.yaml_presets(kernel)
    return next((s for s in ("LEN_3D", "LEN_2D", "LEN_1D") if s in presets), None)


def analyze(kernel: tsvc.TsvcKernel) -> Dict[str, object]:
    """One kernel's work exponent, or a recorded reason it has none."""
    cpp = kernel.native_cpp
    if cpp is None:
        return {"key": kernel.key, "error": "no native baseline"}
    text = strip_comments(cpp.read_text())
    try:
        symbol = native_symbol(text, kernel.native_symbol)
    except LookupError as e:
        return {"key": kernel.key, "error": str(e)}
    params = scalar_params(text, symbol)
    if not params:
        return {"key": kernel.key, "symbol": size_symbol(kernel), "error": f"no scalar param on {symbol}"}
    k = work_exponent(text, params)
    if k == 0:
        # No size-bounded loop at all: the kernel is O(1) in the manifest symbol, so a preset that scales
        # that symbol does not scale its runtime. Worth reporting, never worth sizing by.
        return {"key": kernel.key, "symbol": size_symbol(kernel), "error": "no size-bounded loop"}
    return {"key": kernel.key, "symbol": size_symbol(kernel), "k": k}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("complexity.json"))
    ap.add_argument("--corpus", nargs="*", default=["tsvc2", "tsvc2_5"])
    ap.add_argument("--kernels", nargs="*", default=None, help="kernel keys (default: every corpus kernel)")
    args = ap.parse_args(argv)

    kernels = [k for corpus in args.corpus for k in tsvc.iter_tsvc_kernels(corpus=corpus)]
    if args.kernels:
        want = set(args.kernels)
        kernels = [k for k in kernels if k.key in want]

    rows = [analyze(kernel) for kernel in kernels]
    for row in rows:
        print(f"{row['key']:>28}  " +
              (f"O(n^{row['k']})  symbol={row['symbol']}" if "k" in row else f"SKIP {row['error']}"),
              flush=True)
    args.out.write_text(json.dumps(rows, indent=1))

    classified = [r for r in rows if "k" in r]
    by_class = Counter((r["k"], r["symbol"]) for r in classified)
    print(f"\n{len(classified)}/{len(rows)} classified")
    for (k, symbol), n in sorted(by_class.items()):
        print(f"  {n:>4}  O(n^{k})  symbol={symbol}")
    print("->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
