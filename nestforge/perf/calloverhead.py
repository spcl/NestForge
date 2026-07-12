"""Runtime call-overhead job: what does it cost to call an emitted loop-nest from an external static
``.a`` instead of inlining it, and does LTO recover that cost?

The emitted numpyto kernels are STATELESS (``extern "C" void <key>_fp64(double* a, ...)`` -- pure over
their buffer arguments, no static/global/TLS state, no init/exit), so they can always be inlined back.
For each kernel we build a tiny compiled trampoline ``run_<key>(<kernel args>, nreps)`` that calls the
kernel ``nreps`` times in a loop -- so the rep loop is COMPILED C where inlining is visible (a Python/
ctypes rep loop would call across the boundary every iteration regardless) -- three ways:

  * **inline**       -- ``#include`` the kernel source into the trampoline TU; the compiler inlines it;
  * **external-lto** -- kernel compiled to a FAT-LTO object, ``ar``'d into a ``.a``, linked ``-flto``;
    the LTO plugin inlines the kernel out of the archive across the TU boundary;
  * **external**     -- kernel in a plain ``.a`` (no LTO); an ordinary out-of-line call each iteration.

We time ONE ctypes call to the trampoline (amortizing the single Python->C crossing over ``nreps`` inner
calls) and report per-call microseconds. Ratios ``external / inline`` (the call overhead) and
``external-lto / inline`` (how much LTO recovers) are the result. Kernels self-partition across ranks;
``--tables-only`` merges the per-kernel JSON into markdown.

Usage::

    python -m nestforge.perf.calloverhead --compiler gcc --reps 7 --inner 2000 --out perf_results/calloverhead
    python -m nestforge.perf.calloverhead --tables-only --out perf_results/calloverhead
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import re
import shutil
import socket
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dace  # noqa: F401 -- ensure the real DaCe package is importable (not a cwd stub)

from nestforge import tsvc
from nestforge.arena import make_inputs
from nestforge.build import ar_for, compiler_family, fat_lto_flags
from nestforge.extract import extract_nest_to_sdfg
from nestforge.isolation import run_isolated
from nestforge.perf import flags
from nestforge.perf.tsvc_arena import (abi_order, c_argtypes, discover_toolchains, my_slice, rank_and_size)
from nestforge.perf.tsvc_full import c_call_args
from nestforge.strategies import get_strategy
from nestforge.translate import emit_sources, prepare


def median(xs: List[float]) -> float:
    return float(statistics.median(xs))


def signature_params(csrc: str, symbol: str) -> str:
    """The raw parameter declaration list (types + names) of ``void <symbol>(...)`` in the emitted C."""
    m = re.search(rf"void\s+{re.escape(symbol)}\s*\((.*?)\)\s*\{{", csrc, re.S)
    if not m:
        raise LookupError(f"entry point {symbol} not found in the emitted C")
    return " ".join(m.group(1).split())


def runner_source(symbol: str, params: str, argnames: List[str], kernel_c: Optional[Path]) -> str:
    """A trampoline TU: ``run_<symbol>(<params>, nreps)`` looping ``nreps`` calls to the kernel. When
    ``kernel_c`` is given the kernel is ``#include``d (inline build); otherwise it is declared ``extern``
    (external build, resolved from the linked ``.a``)."""
    forward = ", ".join(argnames)
    head = f'#include "{kernel_c.resolve()}"\n' if kernel_c is not None else f"extern void {symbol}({params});\n"
    return (f"#include <stdint.h>\n{head}"
            f"void run_{symbol}({params}, int64_t nreps) {{\n"
            f"    for (int64_t r = 0; r < nreps; ++r) {symbol}({forward});\n"
            f"}}\n")


def run_compile(cmd: List[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{r.stderr[-800:]}")


def build_inline(cc: str, cflags: List[str], kernel_c: Path, symbol: str, params: str, argnames: List[str],
                 workdir: Path) -> Path:
    """Kernel ``#include``d into the trampoline TU -> the compiler inlines it. One compile."""
    runner = workdir / f"inline_{symbol}.c"
    runner.write_text(runner_source(symbol, params, argnames, kernel_c))
    so = workdir / f"inline_{symbol}.so"
    run_compile([cc, *cflags, "-fPIC", "-shared", str(runner), "-o", str(so)])
    return so


def build_external(cc: str, cflags: List[str], kernel_c: Path, symbol: str, params: str, argnames: List[str],
                   workdir: Path, lto: bool) -> Path:
    """Kernel compiled to an object, ``ar``'d into a ``.a``, linked into the trampoline ``.so``. With
    ``lto`` the object is a FAT-LTO object and the link is ``-flto`` (the linker inlines from the archive);
    without it the call stays out-of-line."""
    tag = "extlto" if lto else "ext"
    lto_c = fat_lto_flags(cc) if lto else []
    if lto and not lto_c:  # compiler cannot fat-LTO -> this variant is not measurable for it
        raise RuntimeError(f"{Path(cc).name} has no fat-LTO support; external-lto not measurable")
    obj = workdir / f"{tag}_{symbol}.o"
    run_compile([cc, *cflags, *lto_c, "-fPIC", "-c", str(kernel_c), "-o", str(obj)])
    archive = workdir / f"lib{tag}_{symbol}.a"
    archive.unlink(missing_ok=True)
    run_compile([ar_for(cc), "rcs", str(archive), str(obj)])
    runner = workdir / f"{tag}_{symbol}.c"
    runner.write_text(runner_source(symbol, params, argnames, kernel_c=None))
    so = workdir / f"{tag}_{symbol}.so"
    link_lto = ["-flto"] if lto else []
    run_compile([cc, *cflags, *link_lto, "-fPIC", "-shared", str(runner), str(archive), "-o", str(so)])
    return so


def time_work(so: Path, run_symbol: str, order: List[str], argtypes: list, work: Dict, sizes: Dict[str, int],
              inner: int, reps: int) -> Dict:
    """Per-call microseconds: warm, then time ``reps`` single calls to the trampoline (each running the
    kernel ``inner`` times) and divide the median by ``inner``. Runs in a forked child (segfault-safe)."""
    fn = ctypes.CDLL(str(so))[run_symbol]
    fn.argtypes, fn.restype = [*argtypes, ctypes.c_int64], None
    cargs = c_call_args(order, argtypes, work, sizes) + [ctypes.c_int64(inner)]
    fn(*cargs)  # warm
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*cargs)
        samples.append((time.perf_counter() - t0) * 1e6)
    return {"per_call_us": median(samples) / inner}


def build_and_time(cc: str, family: str, kernel_c: Path, symbol: str, params: str, argnames: List[str],
                   order: List[str], argtypes: list, boundary, sizes: Dict[str, int], inner: int, reps: int,
                   workdir: Path) -> Dict:
    """Build the three variants and time each in isolation. A variant that fails to build/run is recorded
    as ``None`` (never aborts the others)."""
    cflags = flags.base_flags(family)
    builders = {
        "inline": lambda d: build_inline(cc, cflags, kernel_c, symbol, params, argnames, d),
        "external_lto": lambda d: build_external(cc, cflags, kernel_c, symbol, params, argnames, d, lto=True),
        "external": lambda d: build_external(cc, cflags, kernel_c, symbol, params, argnames, d, lto=False),
    }
    out: Dict[str, Optional[float]] = {}
    for name, build in builders.items():
        try:
            so = build(workdir)
            work = make_inputs(boundary, sizes, seed=0)
            res = run_isolated(lambda so=so, work=work: time_work(so, f"run_{symbol}", order, argtypes, work, sizes,
                                                                  inner, reps))
            out[name] = None if "error" in res else res["per_call_us"]
        except Exception as e:  # noqa: BLE001 -- one variant failing must not sink the others
            out[name] = None
            out[f"{name}_error"] = f"{type(e).__name__}: {str(e)[:120]}"
    return out


def run_kernel(kernel: "tsvc.TsvcKernel", cc: str, family: str, opt_mode: str, preset: str, inner: int, reps: int,
               workdir: Path) -> Dict:
    """Emit + build + time one kernel; return per-variant per-call times and the overhead ratios."""
    result = {"key": kernel.key, "corpus": kernel.corpus, "compiler": family, "host": socket.gethostname()}
    try:
        sdfg = tsvc.build_sdfg(kernel, opt_mode=opt_mode)
        refs = get_strategy("skip-taskloops")(sdfg)
        if len(refs) != 1:
            return {**result, "skipped": f"{len(refs)} compute nests (need 1)"}
        boundary = extract_nest_to_sdfg(refs[0][0], refs[0][1], name=kernel.key)
        sizes = tsvc.sample_sizes(kernel, boundary, preset=preset)
        prep = prepare(boundary, kernel.key, workdir, sizes=sizes)
        symbol = f"{kernel.key}_fp64"
        src = next(s for s in emit_sources(prep, workdir, target="c") if s.suffix == ".c" and "pluto" not in s.name)
        text = src.read_text()
        order = abi_order(text, symbol)
        argtypes = c_argtypes(order, boundary)
        params = signature_params(text, symbol)
    except Exception as e:  # noqa: BLE001
        return {**result, "skipped": f"emit: {type(e).__name__}: {str(e)[:150]}"}

    # the trampoline forwards the kernel's parameters by name; abi_order gives exactly those names.
    times = build_and_time(cc, family, src, symbol, params, order, order, argtypes, boundary, sizes, inner, reps,
                           workdir)
    inline, ext, extlto = times.get("inline"), times.get("external"), times.get("external_lto")
    result.update({
        "inline_us": inline,
        "external_us": ext,
        "external_lto_us": extlto,
        "call_overhead": (ext / inline if inline and ext else None),  # external / inline (>1 = real cost)
        "lto_overhead": (extlto / inline if inline and extlto else None),  # external-lto / inline (~1 = recovered)
    })
    for k in ("inline_error", "external_error", "external_lto_error"):
        if k in times:
            result[k] = times[k]
    return result


def render_tables(out: Path) -> str:
    files = sorted(p for p in out.glob("*.json") if p.name != "tables.md")
    rows = [json.loads(p.read_text()) for p in files]
    done = [r for r in rows if r.get("inline_us") is not None]
    skipped = [r for r in rows if "skipped" in r]
    lines = [
        "# TSVC external-`.a` call overhead (inline vs external-lto vs external)", "",
        f"{len(done)} kernels timed, {len(skipped)} skipped. Per-call microseconds; overhead = variant / inline "
        "(>1 = the external call costs more; external-lto ~1 means LTO inlined it back).", "",
        "| kernel | compiler | inline (us) | external-lto (us) | external (us) | call overhead x | lto overhead x |",
        "|" + "---|" * 7,
    ]
    def us(x):
        return "—" if x is None else f"{x:.4f}"

    def ratio(x):
        return "—" if x is None else f"{x:.3f}"

    call_ratios, lto_ratios = [], []
    for r in sorted(done, key=lambda x: x["key"]):
        co, lo = r.get("call_overhead"), r.get("lto_overhead")
        if co:
            call_ratios.append(co)
        if lo:
            lto_ratios.append(lo)
        lines.append(f"| {r['key']} | {r['compiler']} | {us(r['inline_us'])} | {us(r.get('external_lto_us'))} "
                     f"| {us(r.get('external_us'))} | {ratio(co)} | {ratio(lo)} |")

    def geomean(xs):
        return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None

    gco, glo = geomean(call_ratios), geomean(lto_ratios)
    if gco:
        lines += ["", f"**Geomean call overhead (external / inline):** {gco:.4f}x over {len(call_ratios)} kernels."]
    if glo:
        lines += [f"**Geomean LTO overhead (external-lto / inline):** {glo:.4f}x over {len(lto_ratios)} kernels "
                  "(closer to 1.0 = LTO recovers more of the inlining)."]
    if skipped:
        lines += ["", "## skipped", ""] + [f"- `{r['key']}` — {r['skipped']}" for r in sorted(skipped, key=lambda x: x['key'])]
    report = "\n".join(lines) + "\n"
    (out / "tables.md").write_text(report)
    return report


def resolve_cc(compiler: str) -> Tuple[str, str]:
    """Map a compiler token to (C-compiler path, family label) via the toolchain discovery (PATH + spack)."""
    tcs = discover_toolchains(compiler)
    if not tcs:
        raise SystemExit(f"no toolchain discovered for {compiler!r}")
    tc = tcs[0]
    return tc.cc, tc.fp_family


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="TSVC external-.a runtime call-overhead job")
    ap.add_argument("--compiler", default="gcc", help="compiler family for the C build (gcc/clang/nvc/icx)")
    ap.add_argument("--opt-mode", default="baseline", choices=list(tsvc.OPT_MODES))
    ap.add_argument("--preset", default="M", help="problem size (small enough that call cost is visible)")
    ap.add_argument("--inner", type=int, default=2000, help="kernel calls per timed trampoline invocation")
    ap.add_argument("--reps", type=int, default=7, help="timed trampoline invocations (median)")
    ap.add_argument("--corpora", nargs="*", default=["tsvc2", "tsvc2_5"])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="perf_results/calloverhead")
    ap.add_argument("--tables-only", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.tables_only:
        print(render_tables(out))
        return 0

    cc, family = resolve_cc(args.compiler)
    procid, ntasks = rank_and_size()
    kernels = [k for c in args.corpora for k in tsvc.iter_tsvc_kernels(only=args.only, corpus=c)]
    mine = my_slice(kernels, procid, ntasks)
    if args.limit:
        mine = mine[:args.limit]
    print(f"[calloverhead] rank {procid}/{ntasks}: {len(mine)} of {len(kernels)} kernels, cc={cc} ({family}) -> {out}")
    for i, kernel in enumerate(mine):
        workdir = Path(tempfile.mkdtemp(prefix=f"nf_co_{kernel.key}_"))
        try:
            res = run_kernel(kernel, cc, family, args.opt_mode, args.preset, args.inner, args.reps, workdir)
        except Exception as e:  # pragma: no cover -- a kernel must never crash the rank
            res = {"key": kernel.key, "skipped": f"crash: {type(e).__name__}: {str(e)[:150]}"}
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        (out / f"{kernel.corpus}_{kernel.key}.json").write_text(json.dumps(res, indent=1))
        print(f"[calloverhead] ({i + 1}/{len(mine)}) {kernel.key}: {res.get('skipped', 'ok')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
