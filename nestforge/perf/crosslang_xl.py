"""Cross-compiler x cross-language TSVC job at a fixed preset (XL by default).

For every kernel of the ``tsvc2`` + ``tsvc2_5`` corpora, extract the compute nest, translate it to C
AND Fortran (numpyto's two AOT-compiled targets), and for each language x each discovered compiler
compile it, run it, and validate it against the numpy oracle -- then time it. Both languages emit the
same C-ABI ``<key>_fp64`` symbol (C plainly, Fortran via ``bind(c)``), so the run is uniform ctypes
across languages (only the signature syntax and Fortran's leading-underscore name munge differ). Sizes
come from the fixed ``--preset`` scale so every language/compiler sees the same problem.

Kernels self-partition across ranks (SLURM or MPI); each cell runs in a forked child so an OOB / runaway
in freshly-compiled code cannot take down the rank. ``--tables-only`` merges the per-kernel JSON.

Usage::

    python -m nestforge.perf.crosslang_xl --corpora tsvc2 tsvc2_5 --languages c fortran \\
        --preset XL --compilers auto --reps 20 --out perf_results/crosslang
    python -m nestforge.perf.crosslang_xl --tables-only --out perf_results/crosslang
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import socket
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import dace  # noqa: F401 -- ensure the real DaCe package is importable

from nestforge import tsvc
from nestforge.arena import maxdiff, make_inputs, run_oracle
from nestforge.extract import extract_nest_to_sdfg
from nestforge.isolation import run_isolated
from nestforge.perf import flags
from nestforge.perf.tsvc_arena import (Toolchain, c_argtypes, call_c, discover_toolchains, rank_and_size, my_slice,
                                       run_compile)
from nestforge.strategies import get_strategy
from nestforge.translate import emit_sources, prepare

#: language -> (numpyto target, source suffix, per-family compiler-exe candidates). numpyto has no
#: distinct C++ AOT target; C and Fortran both emit the SAME C-ABI ``<key>_fp64`` symbol (Fortran via
#: ``bind(c)``), so the ctypes run is uniform across languages.
_LANGS = {
    "c": {
        "target": "c",
        "suffix": ".c",
        "exes": {
            "gcc": ["gcc"],
            "clang": ["clang"],
            "nvhpc": ["nvc"]
        }
    },
    "fortran": {
        "target": "fortran",
        "suffix": ".f90",
        "exes": {
            "gcc": ["gfortran"],
            "clang": ["flang-new", "flang"],
            "nvhpc": ["nvfortran"],
            "intel": ["ifx"]
        }
    },
}
# C-language compiler exes gain the Intel entry too (the family label -> exe map above is per language).
_LANGS["c"]["exes"]["intel"] = ["icx"]


def lang_compilers(languages: List[str], toolchains: List[Toolchain]) -> Dict[str, Dict[str, str]]:
    """``{language: {family: compiler_path}}`` -- the compiler for each discovered family in each
    requested language (e.g. the gcc family compiles C with ``gcc`` and Fortran with ``gfortran``; the
    clang family uses ``flang``/``flang-new``). A family with no compiler for a language is absent."""
    out: Dict[str, Dict[str, str]] = {}
    for lang in languages:
        spec = _LANGS[lang]
        per_family: Dict[str, str] = {}
        for tc in toolchains:
            for exe in spec["exes"].get(tc.name, []):  # exes keyed by the family LABEL (gcc/clang/nvhpc)
                path = shutil.which(exe)
                if path:
                    per_family[tc.name] = path
                    break
        out[lang] = per_family
    return out


def signature_order(text: str, symbol: str, lang: str) -> List[str]:
    """Parameter names of the kernel entry, in declaration order, for the language's syntax.

    A long Fortran argument list wraps across lines with ``&`` free-form continuations
    (``arg1, &\\n  & arg2``); those markers are stripped before splitting, or an arg name would carry a
    stray ``&``/newline (``&\\n&aa_slice`` -> a bogus name -> KeyError at the call)."""
    if lang == "fortran":
        m = re.search(rf"subroutine\s+{re.escape(symbol)}\s*\((.*?)\)", text, re.S | re.I)
        if not m:
            raise LookupError(f"entry {symbol} not found in the emitted {lang} source")
        return [a.strip() for a in m.group(1).replace("&", " ").split(",") if a.strip()]
    m = re.search(rf"void\s+{re.escape(symbol)}\s*\((.*?)\)\s*\{{", text, re.S)
    if not m:
        raise LookupError(f"entry {symbol} not found in the emitted {lang} source")
    return [p.strip().split()[-1].lstrip("*") for p in m.group(1).split(",") if p.strip() and p.strip() != "void"]


def fortran_unmunge(order: List[str], names: List[str]) -> List[str]:
    """Map Fortran arg names back to the SDFG/size names. Fortran forbids a leading underscore, so the
    translator rewrites a leading ``_`` to ``x`` (``__sym_out_i`` -> ``x_sym_out_i``); reverse that by
    matching each Fortran name to the original whose munge equals it."""
    munge = {("x" + n[1:] if n.startswith("_") else n): n for n in names}
    return [munge.get(a, a) for a in order]


#: preset scale order, so validation can pick the smaller of the requested size and a cheap cap.
_PRESET_ORDER = ["S", "M", "L", "XL"]
#: correctness is validated at most at this preset -- the numpy oracle is a pure-Python O(N) loop, so a
#: 268M-element XL oracle would take minutes/kernel. The .so is size-agnostic (LEN is a runtime arg), so
#: validating small and timing at the requested (XL) preset measures the same code correctly and fast.
_VALIDATE_CAP = "M"


def validate_preset(preset: str) -> str:
    return preset if _PRESET_ORDER.index(preset) <= _PRESET_ORDER.index(_VALIDATE_CAP) else _VALIDATE_CAP


def cell_work(so: Path, symbol: str, order: List[str], argtypes, boundary, validate_sizes, time_inputs, time_sizes,
              oracle, reps: int, atol: float) -> Dict:
    """Runs inside the forked child. Validates correctness at the SMALL ``validate_sizes`` (fresh buffers
    built here, fast oracle) and times at the large ``time_sizes``. ``time_inputs`` is the pre-built
    XL buffer set inherited COW from the parent -- timing does not check output, so it needs no per-cell
    freshness; ``call_c`` runs the kernel in place on that COW copy, whose page-out OOM-kills only this child.
    ``atol`` is the FP-level tolerance (:data:`flags.FP_ATOL`): ``strict-ieee`` ~bit-exact, ``fast-math``
    admits reassociation drift."""
    vin = make_inputs(boundary, validate_sizes, seed=0)
    vout, _ = call_c(so, symbol, order, argtypes, boundary, vin, validate_sizes, reps=1)
    md = float(maxdiff(oracle, vout))
    del vin, vout
    _, us = call_c(so, symbol, order, argtypes, boundary, time_inputs, time_sizes, reps)
    return {"ok": bool(md <= atol), "maxdiff": md, "time_us": float(us)}


@dataclass
class Cell:
    language: str
    compiler: str
    fp_level: str  # FP-precision rung (flags.FP_LEVELS): strict-ieee | contract-fma | assume-finite | fast-math
    cost_model: str  # vectorizer cost-model (flags.COST_MODELS): default | cheap | no-vec
    ok: bool
    maxdiff: float
    time_us: float
    compile_us: float
    error: Optional[str] = None


def run_kernel(kernel: "tsvc.TsvcKernel", languages: List[str], compilers: Dict[str, Dict[str, str]], strategy: str,
               preset: str, reps: int, workdir: Path, opt_mode: str = "baseline") -> Dict:
    """Emit + compile + run + time one kernel across every requested language x compiler."""
    result = {"key": kernel.key, "corpus": kernel.corpus, "preset": preset, "host": socket.gethostname()}
    try:
        sdfg = tsvc.build_sdfg(kernel, opt_mode=opt_mode)
        refs = get_strategy(strategy)(sdfg)
        if not refs:
            return {**result, "skipped": "no compute nest"}
        if len(refs) > 1:
            return {**result, "skipped": f"{len(refs)} compute nests; multi-nest not supported"}
        parent, node = refs[0]
        boundary = extract_nest_to_sdfg(parent, node, name=kernel.key)
        # Validate correctness at a SMALL preset (fast pure-Python oracle in the parent); time at the
        # requested preset. The compiled .so is size-agnostic (LEN is a runtime arg), so both use it.
        time_sizes = tsvc.sample_sizes(kernel, boundary, preset=preset)
        validate_sizes = tsvc.sample_sizes(kernel, boundary, preset=validate_preset(preset))
        prep = prepare(boundary, kernel.key, workdir, sizes=validate_sizes)
        oracle = run_oracle(prep, boundary, make_inputs(boundary, validate_sizes, seed=0), validate_sizes)
        # Build the large XL timing buffers ONCE per kernel; every cell's fork COW-inherits them (timing
        # does not validate output, so the same buffers are reused across cells -- no per-cell re-fill).
        time_inputs = make_inputs(boundary, time_sizes, seed=0)
    except Exception as e:
        return {**result, "skipped": f"{type(e).__name__}: {str(e)[:160]}"}

    symbol = f"{kernel.key}_fp64"
    names = list(boundary.standalone_sdfg.arrays) + list(validate_sizes)
    rows: List[Dict] = []
    for lang in languages:
        spec = _LANGS[lang]
        try:
            src = next(s for s in emit_sources(prep, workdir, target=spec["target"])
                       if s.suffix == spec["suffix"] and "pluto" not in s.name)
            order = signature_order(src.read_text(), symbol, lang)
            order = fortran_unmunge(order, names) if lang == "fortran" else order
            argtypes = c_argtypes(order, boundary)
        except Exception as e:
            rows.append(
                asdict(Cell(lang, "-", "-", "-", False, float("inf"), float("inf"), 0.0, error=f"emit: {str(e)[:150]}")))
            continue
        for fam_label, cc in compilers.get(lang, {}).items():
            fam = family_of(fam_label)
            # Sweep the FP-precision level x vectorizer cost-model matrix for this compiler+language.
            for fp_level, cost_model, cflags in flags.flag_matrix(fam, lang):
                tag = f"{fam_label}_{fp_level}_{cost_model}"
                so = workdir / f"{kernel.key}_{lang}_{tag}.so"
                ok, compile_us, err = run_compile([cc, *cflags, str(src), "-o", str(so)])
                if not ok:
                    rows.append(
                        asdict(Cell(lang, fam_label, fp_level, cost_model, False, float("inf"), float("inf"),
                                    compile_us, error=err)))
                    continue
                # Generous timeout: an XL timing run (268M elements x reps) is legitimately long; the fork
                # makes an OOM/segfault kill only the child, and the timeout only catches a genuine runaway.
                atol = flags.FP_ATOL[fp_level]
                res = run_isolated(
                    lambda so=so, atol=atol: cell_work(so, symbol, order, argtypes, boundary, validate_sizes,
                                                       time_inputs, time_sizes, oracle, reps, atol),
                    timeout=3600.0)
                if "error" in res:
                    rows.append(
                        asdict(Cell(lang, fam_label, fp_level, cost_model, False, float("inf"), float("inf"),
                                    compile_us, error=res["error"])))
                else:
                    rows.append(
                        asdict(Cell(lang, fam_label, fp_level, cost_model, res["ok"], res["maxdiff"], res["time_us"],
                                    compile_us)))
    result["cells"] = rows
    result["sizes"] = {
        "validate": {
            k: int(v)
            for k, v in validate_sizes.items()
        },
        "time": {
            k: int(v)
            for k, v in time_sizes.items()
        }
    }
    return result


def family_of(name: str) -> str:
    """Toolchain family label (gcc/clang/nvhpc/intel) -> the flag-matrix FP family (gnu/llvm/nvidia/intel)."""
    return {"gcc": "gnu", "clang": "llvm", "nvhpc": "nvidia", "intel": "intel"}.get(name, "gnu")


def cells_winner(cells: List[Dict]) -> Optional[Dict]:
    """The fastest cell that validated at its FP-level tolerance (min ``time_us`` among ``ok``), or None."""
    ok = [c for c in cells if c["ok"] and c["time_us"] != float("inf")]
    return min(ok, key=lambda c: c["time_us"]) if ok else None


def render_tables(out: Path) -> str:
    files = sorted(p for p in out.glob("*.json") if p.name != "tables.md")
    kernels = [json.loads(p.read_text()) for p in files]
    done = [k for k in kernels if "cells" in k]
    skipped = [k for k in kernels if "skipped" in k]
    lines = [
        "# TSVC cross-compiler x cross-language (FP-level x cost-model matrix)", "",
        f"{len(done)} kernels measured, {len(skipped)} skipped. Each cell is a compiler x language x "
        "FP-precision-level x vectorizer-cost-model point; the winner is the fastest cell that validates at "
        "its level's tolerance. `fp speedup` = strict-ieee time / winner time.", "",
        "| kernel | corpus | preset | language | compiler | winner fp/cost | strict maxdiff | winner (us) | fp speedup |",
        "|" + "---|" * 9
    ]
    ok_by_lang: Dict[str, int] = {}
    tot_by_lang: Dict[str, int] = {}
    for k in sorted(done, key=lambda x: (x["corpus"], x["key"])):
        groups: Dict[tuple, List[Dict]] = {}
        for c in k["cells"]:
            groups.setdefault((c["language"], c["compiler"]), []).append(c)
        for (lang, comp), cells in sorted(groups.items()):
            tot_by_lang[lang] = tot_by_lang.get(lang, 0) + 1
            win = cells_winner(cells)
            strict = next((c for c in cells if c["fp_level"] == "strict-ieee"), None)
            if win:
                ok_by_lang[lang] = ok_by_lang.get(lang, 0) + 1
            wtxt = f"{win['fp_level']}/{win['cost_model']}" if win else "—"
            md = "—" if not strict or strict["maxdiff"] == float("inf") else f"{strict['maxdiff']:g}"
            wt = "—" if not win else f"{win['time_us']:.2f}"
            sp = "—"
            if win and strict and strict["ok"] and strict["time_us"] not in (0.0, float("inf")):
                sp = f"{strict['time_us'] / win['time_us']:.2f}x"
            lines.append(f"| {k['key']} | {k['corpus']} | {k['preset']} | {lang} | {comp} | {wtxt} | {md} | {wt} "
                         f"| {sp} |")
    lines += ["", "## compiler+language pairs with a validating cell, per language"]
    for lang in sorted(tot_by_lang):
        lines.append(f"- **{lang}**: {ok_by_lang.get(lang, 0)}/{tot_by_lang[lang]}")
    if skipped:
        lines += ["", "## skipped", ""] + [
            f"- `{k['key']}` ({k['corpus']}) — {k['skipped']}" for k in sorted(skipped, key=lambda x: x['key'])
        ]
    report = "\n".join(lines) + "\n"
    (out / "tables.md").write_text(report)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="TSVC cross-compiler x cross-language job")
    ap.add_argument("--corpora", nargs="*", default=["tsvc2", "tsvc2_5"], choices=["tsvc2", "tsvc2_5"])
    ap.add_argument("--languages", nargs="*", default=["c", "fortran"], choices=list(_LANGS))
    ap.add_argument("--preset", default="XL", choices=["S", "M", "L", "XL"])
    ap.add_argument("--compilers", default="auto")
    ap.add_argument("--strategy", default="skip-taskloops")
    ap.add_argument("--opt-mode", default="baseline", choices=list(tsvc.OPT_MODES),
                    help="pre-split optimization mode (baseline / canonicalize)")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="perf_results/crosslang")
    ap.add_argument("--tables-only", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.tables_only:
        print(render_tables(out))
        return 0

    toolchains = discover_toolchains(args.compilers)
    compilers = lang_compilers(args.languages, toolchains)
    print("[crosslang] compilers per language: " + "; ".join(f"{lang}={sorted(fam)}"
                                                             for lang, fam in compilers.items()))
    if not any(compilers.values()):
        print("[crosslang] no compilers found for any requested language")
        return 1

    # kernels of every corpus, tagged, then self-partitioned across ranks as one combined list.
    kernels = [k for corpus in args.corpora for k in tsvc.iter_tsvc_kernels(only=args.only, corpus=corpus)]
    procid, ntasks = rank_and_size()
    mine = my_slice(kernels, procid, ntasks)
    if args.limit:
        mine = mine[:args.limit]
    print(f"[crosslang] rank {procid}/{ntasks}: {len(mine)} of {len(kernels)} kernels (preset {args.preset}) -> {out}")

    for i, kernel in enumerate(mine):
        workdir = Path(tempfile.mkdtemp(prefix=f"nf_xl_{kernel.corpus}_{kernel.key}_"))
        try:
            res = run_kernel(kernel, args.languages, compilers, args.strategy, args.preset, args.reps, workdir,
                             opt_mode=args.opt_mode)
        except Exception as e:  # pragma: no cover
            res = {"key": kernel.key, "corpus": kernel.corpus, "skipped": f"crash: {type(e).__name__}: {str(e)[:160]}"}
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        (out / f"{kernel.corpus}_{kernel.key}.json").write_text(json.dumps(res, indent=1))
        print(f"[crosslang] ({i + 1}/{len(mine)}) {kernel.corpus}/{kernel.key}: {res.get('skipped', 'ok')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
