"""Static-library overhead job.

For each TSVC kernel, own-build the DaCe SDFG (compile DaCe's C++ ourselves -- no CMake) BOTH ways and
time only the post-codegen toolchain work:

  * **monolithic** -- one translation unit (the compiler inlines freely);
  * **external**   -- compile to an object, ``ar`` it into a static ``.a``, link the ``.so`` from that
    archive -- the assembly path the arena uses to link per-kernel winners into a whole program.

The ratio ``external / monolithic`` is the per-kernel static-lib assembly overhead. Codegen is done once
and shared, so only compile/link differs. Kernels are self-partitioned across ranks (SLURM or MPI);
``--tables-only`` merges the per-kernel JSON into markdown.

Usage::

    python -m nestforge.perf.staticlib_overhead --compiler g++ --reps 5 --out perf_results/staticlib
    python -m nestforge.perf.staticlib_overhead --tables-only --out perf_results/staticlib
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import socket
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import dace  # noqa: F401 -- ensure the real DaCe package is importable (not a cwd stub)

from nestforge import tsvc
from nestforge.arena import discover_blas_libraries
from nestforge.build import BuildOptions, compare_link_modes
from nestforge.perf.tsvc_arena import my_slice, rank_and_size


def median(xs: List[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def run_kernel(kernel: "tsvc.TsvcKernel", compiler: str, reps: int, opt_mode: str, fast_libnodes: bool) -> Dict:
    """Own-build one kernel monolithic vs external ``reps`` times; return the median compile timings."""
    result = {"key": kernel.key, "compiler": compiler, "host": socket.gethostname()}
    try:
        sdfg = tsvc.build_sdfg(kernel, opt_mode=opt_mode)
    except Exception as e:
        return {**result, "skipped": f"{type(e).__name__}: {str(e)[:160]}"}
    blas = discover_blas_libraries().get("openblas") if fast_libnodes else None
    opts = BuildOptions(compiler=compiler, fast_libnodes=fast_libnodes, blas_link=(blas.link_flags if blas else None))
    mono, ext, codegen = [], [], []
    for _ in range(reps):
        workdir = Path(tempfile.mkdtemp(prefix=f"nf_slo_{kernel.key}_"))
        try:
            lt = compare_link_modes(sdfg, workdir, opts)
            mono.append(lt.compile_seconds_monolithic)
            ext.append(lt.compile_seconds_external)
            codegen.append(lt.codegen_seconds)
        except Exception as e:
            return {**result, "skipped": f"build: {type(e).__name__}: {str(e)[:160]}"}
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
    mono_ms, ext_ms = median(mono) * 1e3, median(ext) * 1e3
    return {
        **result, "codegen_ms": median(codegen) * 1e3,
        "monolithic_ms": mono_ms,
        "external_ms": ext_ms,
        "overhead_ratio": (ext_ms / mono_ms if mono_ms else None)
    }


def seed_dir(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    return out


def render_tables(out: Path) -> str:
    files = sorted(p for p in out.glob("*.json") if p.name != "tables.md")
    rows = [json.loads(p.read_text()) for p in files]
    done = [r for r in rows if "monolithic_ms" in r]
    skipped = [r for r in rows if "skipped" in r]
    lines = [
        "# TSVC static-lib overhead (own-build, direct compile)", "",
        f"{len(done)} kernels built, {len(skipped)} skipped.", "",
        "| kernel | compiler | codegen (ms) | monolithic (ms) | external .a (ms) | overhead x |", "|" + "---|" * 6
    ]
    ratios = []
    for r in sorted(done, key=lambda x: x["key"]):
        ov = r["overhead_ratio"]
        if ov:
            ratios.append(ov)
        lines.append(f"| {r['key']} | {r['compiler']} | {r['codegen_ms']:.1f} | {r['monolithic_ms']:.1f} "
                     f"| {r['external_ms']:.1f} | {'—' if ov is None else f'{ov:.2f}'} |")
    if ratios:
        geo = math.exp(sum(math.log(x) for x in ratios) / len(ratios))
        lines += [
            "", f"**Geomean static-lib compile overhead (external / monolithic):** {geo:.3f}x "
            f"over {len(ratios)} kernels."
        ]
    if skipped:
        lines += ["", "## skipped", ""
                  ] + [f"- `{r['key']}` — {r['skipped']}" for r in sorted(skipped, key=lambda x: x['key'])]
    report = "\n".join(lines) + "\n"
    (out / "tables.md").write_text(report)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="TSVC static-library overhead job")
    ap.add_argument("--compiler", default="g++", help="C++ compiler for the owned DaCe build")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--opt-mode",
                    default="baseline",
                    choices=list(tsvc.OPT_MODES),
                    help="pre-split optimization mode (baseline / canonicalize)")
    ap.add_argument("--fast-libnodes", action="store_true", help="pick fast BLAS impl + link it (set-fast-impl)")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="perf_results/staticlib")
    ap.add_argument("--tables-only", action="store_true")
    args = ap.parse_args(argv)
    out = seed_dir(Path(args.out))

    if args.tables_only:
        print(render_tables(out))
        return 0

    procid, ntasks = rank_and_size()
    kernels = tsvc.iter_tsvc_kernels(only=args.only)
    mine = my_slice(kernels, procid, ntasks)
    if args.limit:
        mine = mine[:args.limit]
    print(f"[staticlib-overhead] rank {procid}/{ntasks}: {len(mine)} of {len(kernels)} kernels -> {out}")
    for i, kernel in enumerate(mine):
        try:
            res = run_kernel(kernel, args.compiler, args.reps, args.opt_mode, args.fast_libnodes)
        except Exception as e:  # pragma: no cover - a kernel must never crash the rank
            res = {"key": kernel.key, "skipped": f"crash: {type(e).__name__}: {str(e)[:160]}"}
        (out / f"{kernel.key}.json").write_text(json.dumps(res, indent=1))
        print(f"[staticlib-overhead] ({i + 1}/{len(mine)}) {kernel.key}: {res.get('skipped', 'ok')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
