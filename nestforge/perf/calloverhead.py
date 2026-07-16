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
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dace  # noqa: F401 -- ensure the real DaCe package is importable (not a cwd stub)

from nestforge import tsvc
from nestforge.arena import make_inputs
from nestforge.build import ar_for, fat_lto_flags
from nestforge.isolation import run_isolated
from nestforge.multinest import extract_all_nests
from nestforge.perf import flags
from nestforge.perf.tsvc_arena import discover_toolchains
from nestforge.perf.harness import c_argtypes, median, my_slice, rank_and_size, signature_order
from nestforge.perf.tsvc_full import c_call_args
from nestforge.translate import emit_sources, prepare


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
    # Bound each compile so a pathological build can't hang the rank forever (see build.run /
    # NF_COMPILE_TIMEOUT); a timeout surfaces as a normal build failure for this cell.
    try:
        r = subprocess.run(cmd,
                           capture_output=True,
                           text=True,
                           timeout=float(os.environ.get("NF_COMPILE_TIMEOUT", "900")))
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"command timed out ({float(os.environ.get('NF_COMPILE_TIMEOUT', '900')):.0f}s): "
                           f"{' '.join(cmd[:2])} ... (ceiling is NF_COMPILE_TIMEOUT)")
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


def build_and_time(cc: str,
                   family: str,
                   kernel_c: Path,
                   symbol: str,
                   params: str,
                   argnames: List[str],
                   order: List[str],
                   argtypes: list,
                   boundary,
                   sizes: Dict[str, int],
                   inner: int,
                   reps: int,
                   workdir: Path,
                   given=None) -> Dict:
    """Build the three variants and time each in isolation. A variant that fails to build/run is recorded
    as ``None`` (never aborts the others).

    ``given`` is forwarded to :func:`make_inputs` -- the manifest's index-array fills. Without them a
    gather/scatter kernel is timed with an all-zero index array, i.e. against the wrong memory behaviour,
    which is precisely what this job claims to measure."""
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
            work = make_inputs(boundary, sizes, seed=0, given=given)
            res = run_isolated(
                lambda so=so, work=work: time_work(so, f"run_{symbol}", order, argtypes, work, sizes, inner, reps))
            out[name] = None if "error" in res else res["per_call_us"]
        except Exception as e:  # noqa: BLE001 -- one variant failing must not sink the others
            out[name] = None
            out[f"{name}_error"] = f"{type(e).__name__}: {str(e)[:120]}"
    return out


@dataclass
class CoNest:
    """One extracted nest of a kernel plus its emitted C source and parsed C signature. A single-nest
    kernel has one (named ``<key>`` / symbol ``<key>_fp64``); a multi-nest kernel one per ``<key>_n<idx>``."""
    idx: int
    name: str
    symbol: str
    boundary: object
    sizes: Dict[str, int]
    src: Path
    order: List[str]
    argtypes: list
    params: str
    nest_dir: Path
    given: Dict[str, object] = field(default_factory=dict)  # manifest index fills for this nest


def run_kernel(kernel: "tsvc.TsvcKernel", cc: str, family: str, opt_mode: str, preset: str, inner: int, reps: int,
               workdir: Path) -> Dict:
    """Emit + build + time one kernel; return per-variant per-call times and the overhead ratios.

    A kernel may split into several compute nests; the call cost of the kernel is the SUM of its nests'
    calls, so each variant's per-call time (inline / external / external-lto) is summed over nests before
    the overhead ratios are taken. A single-nest kernel is exactly the old single measurement."""
    result = {"key": kernel.key, "corpus": kernel.corpus, "compiler": family, "host": socket.gethostname()}
    try:
        nests = extract_all_nests(lambda: tsvc.build_sdfg(kernel, opt_mode=opt_mode), "skip-taskloops", kernel.key)
        if not nests:
            return {**result, "skipped": "no compute nest"}
        units: List[CoNest] = []
        for idx, name, symbol, boundary in nests:
            sizes = tsvc.sample_sizes(kernel, boundary, preset=preset)
            nest_dir = workdir / f"n{idx}"
            prep = prepare(boundary, name, nest_dir, sizes=sizes)
            src = next(s for s in emit_sources(prep, nest_dir, target="c")
                       if s.suffix == ".c" and "pluto" not in s.name)
            text = src.read_text()
            order = signature_order(text, symbol)
            units.append(
                CoNest(idx, name, symbol, boundary, sizes, src, order, c_argtypes(order, boundary),
                       signature_params(text, symbol), nest_dir, tsvc.index_fills(kernel, boundary, sizes)))
    except Exception as e:  # noqa: BLE001
        return {**result, "skipped": f"emit: {type(e).__name__}: {str(e)[:150]}"}

    # Time each variant per nest; the kernel's per-variant call cost is the SUM over its nests. A variant
    # is None for the kernel if ANY nest could not build/time it (a missing part cannot be summed).
    per_variant: Dict[str, List[Optional[float]]] = {"inline": [], "external": [], "external_lto": []}
    multi = len(units) > 1
    for u in units:
        # the trampoline forwards the kernel's parameters by name; abi_order gives exactly those names.
        times = build_and_time(cc, family, u.src, u.symbol, u.params, u.order, u.order, u.argtypes, u.boundary, u.sizes,
                               inner, reps, u.nest_dir, u.given)
        for v in per_variant:
            per_variant[v].append(times.get(v))
        for k in ("inline_error", "external_error", "external_lto_error"):
            if k in times:
                result[f"n{u.idx}_{k}" if multi else k] = times[k]

    def total(variant: str) -> Optional[float]:
        vals = per_variant[variant]
        return sum(vals) if vals and all(x is not None for x in vals) else None

    inline, ext, extlto = total("inline"), total("external"), total("external_lto")
    result.update({
        "inline_us": inline,
        "external_us": ext,
        "external_lto_us": extlto,
        "call_overhead": (ext / inline if inline and ext else None),  # external / inline (>1 = real cost)
        "lto_overhead": (extlto / inline if inline and extlto else None),  # external-lto / inline (~1 = recovered)
    })
    return result


def render_tables(out: Path) -> str:
    files = sorted(p for p in out.glob("*.json") if p.name != "tables.md")
    rows = [json.loads(p.read_text()) for p in files]
    done = [r for r in rows if r.get("inline_us") is not None]
    skipped = [r for r in rows if "skipped" in r]
    lines = [
        "# TSVC external-`.a` call overhead (inline vs external-lto vs external)",
        "",
        f"{len(done)} kernels timed, {len(skipped)} skipped. Per-call microseconds; overhead = variant / inline "
        "(>1 = the external call costs more; external-lto ~1 means LTO inlined it back).",
        "",
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
        lines += [
            f"**Geomean LTO overhead (external-lto / inline):** {glo:.4f}x over {len(lto_ratios)} kernels "
            "(closer to 1.0 = LTO recovers more of the inlining)."
        ]
    if skipped:
        lines += ["", "## skipped", ""
                  ] + [f"- `{r['key']}` — {r['skipped']}" for r in sorted(skipped, key=lambda x: x['key'])]
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
    ap.add_argument("--opt-mode", default="simplify-parallel", choices=list(tsvc.OPT_MODES))
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
