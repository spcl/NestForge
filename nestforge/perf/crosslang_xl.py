"""Cross-compiler x cross-language TSVC job at a fixed preset (XL by default).

For every ``tsvc2``/``tsvc2_5`` kernel: extract the compute nest, translate to C and Fortran, compile
with each discovered compiler, run, validate against the numpy oracle, and time. Both languages emit
the same C-ABI ``<key>_fp64`` symbol, so the run is uniform ctypes across languages.

Kernels self-partition across ranks; each cell runs in a forked child so a crash can't take down the
rank. ``--tables-only`` merges the per-kernel JSON.

Usage::

    python -m nestforge.perf.crosslang_xl --corpora tsvc2 tsvc2_5 --languages c fortran \\
        --preset XL --compilers auto --reps 20 --out perf_results/crosslang
    python -m nestforge.perf.crosslang_xl --tables-only --out perf_results/crosslang
"""
from __future__ import annotations

import argparse
import json
import shutil
import socket
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import dace  # noqa: F401 -- ensures real dace importable

from nestforge import tsvc
from nestforge.arena import maxdiff, make_inputs, run_oracle
from nestforge.isolation import run_isolated
from nestforge.multinest import extract_all_nests
from nestforge.perf import flags
from nestforge.perf.tsvc_arena import Toolchain, discover_toolchains
from nestforge.perf.harness import c_argtypes, call_c, my_slice, rank_and_size, run_compile, signature_order
from nestforge.translate import emit_sources, prepare

#: language -> (numpyto target, suffix, compiler-exe candidates per family). C and Fortran both emit
#: the same C-ABI ``<key>_fp64`` symbol (Fortran via ``bind(c)``), so ctypes calls are uniform.
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
# add the Intel entry for C (map above is per language).
_LANGS["c"]["exes"]["intel"] = ["icx"]


def lang_compilers(languages: List[str], toolchains: List[Toolchain]) -> Dict[str, Dict[str, str]]:
    """``{language: {family: compiler_path}}`` for each discovered family x requested language (e.g.
    gcc compiles C with ``gcc``, Fortran with ``gfortran``). A family missing a compiler is absent."""
    out: Dict[str, Dict[str, str]] = {}
    for lang in languages:
        spec = _LANGS[lang]
        per_family: Dict[str, str] = {}
        for tc in toolchains:
            for exe in spec["exes"].get(tc.name, []):  # keyed by family label (gcc/clang/nvhpc)
                path = shutil.which(exe)
                if path:
                    per_family[tc.name] = path
                    break
        out[lang] = per_family
    return out


def fortran_unmunge(order: List[str], names: List[str]) -> List[str]:
    """Map Fortran arg names back to SDFG/size names. Fortran forbids a leading underscore, so the
    translator rewrites it to ``x`` (``__sym_out_i`` -> ``x_sym_out_i``); reverse via the munge map."""
    munge = {("x" + n[1:] if n.startswith("_") else n): n for n in names}
    return [munge.get(a, a) for a in order]


#: preset scale order, for picking min(requested, cheap cap).
_PRESET_ORDER = ["S", "M", "L", "XL"]
#: max validation preset -- the numpy oracle is O(N) pure Python, so an XL oracle takes minutes/kernel.
#: The .so is size-agnostic, so validating small + timing at XL measures the same code, fast.
_VALIDATE_CAP = "M"


def validate_preset(preset: str) -> str:
    return preset if _PRESET_ORDER.index(preset) <= _PRESET_ORDER.index(_VALIDATE_CAP) else _VALIDATE_CAP


def cell_work(so: Path,
              symbol: str,
              order: List[str],
              argtypes,
              boundary,
              validate_sizes,
              time_inputs,
              time_sizes,
              oracle,
              reps: int,
              atol: float,
              given=None) -> Dict:
    """Runs inside the forked child. Validates at ``validate_sizes`` (fresh buffers, fast oracle) and
    times at ``time_sizes``. ``time_inputs`` is COW-inherited from the parent, mutated in place by
    ``call_c``, so only this child's page-out can OOM. ``atol`` is the FP tolerance
    (:data:`flags.FP_ATOL`): ``strict-ieee`` ~bit-exact, ``fast-math`` admits reassociation drift."""
    vin = make_inputs(boundary, validate_sizes, seed=0, given=given)
    vout, _ = call_c(so, symbol, order, argtypes, boundary, vin, validate_sizes, reps=1)
    md = float(maxdiff(oracle, vout))
    del vin, vout
    # copy_outputs=False: no output check needed here, and at XL snapshotting would double peak RSS.
    _, us = call_c(so, symbol, order, argtypes, boundary, time_inputs, time_sizes, reps, copy_outputs=False)
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


@dataclass
class XlNest:
    """One extracted nest plus its oracle / sizes / timing buffers. Per-language source is parsed
    lazily in :func:`run_kernel`."""
    idx: int
    name: str
    symbol: str
    boundary: object
    prep: object
    nest_dir: Path
    time_sizes: Dict[str, int]
    validate_sizes: Dict[str, int]
    oracle: Dict[str, object]
    time_inputs: Dict[str, object]
    names: List[str]  # SDFG array + size names, for the Fortran un-munge
    validate_fills: Dict[str, object] = field(default_factory=dict)


def measure_xl_cell(cc: str, lang: str, fam_label: str, fp_level: str, cost_model: str, cflags: List[str],
                    units: List[XlNest], per_nest_src: List, reps: int, workdir: Path) -> Cell:
    """One (language, compiler, fp-level, cost-model) cell summed over every nest: compile + validate +
    time each nest and aggregate. ``time_us``/``compile_us`` are sums, ``ok`` iff every nest validated,
    ``maxdiff`` the max. A failure on any nest makes the whole cell not-ok with infinite time."""
    atol = flags.FP_ATOL[fp_level]
    tag = f"{fam_label}_{fp_level}_{cost_model}"
    ok_all, md_max, time_sum, compile_sum, err = True, 0.0, 0.0, 0.0, None
    for u, (src, order, argtypes) in zip(units, per_nest_src):
        so = workdir / f"{u.name}_{lang}_{tag}.so"
        cok, compile_us, cerr = run_compile([cc, *cflags, str(src), "-o", str(so)])
        compile_sum += compile_us
        if not cok:
            ok_all, md_max, time_sum, err = False, float("inf"), float("inf"), cerr
            break
        # generous timeout: an XL run is legitimately long; the fork isolates OOM/segfault to the child.
        res = run_isolated(lambda u=u, order=order, argtypes=argtypes, so=so: cell_work(
            so, u.symbol, order, argtypes, u.boundary, u.validate_sizes, u.time_inputs, u.time_sizes, u.oracle, reps,
            atol, u.validate_fills),
                           timeout=3600.0)
        if "error" in res:
            ok_all, md_max, time_sum, err = False, float("inf"), float("inf"), res["error"]
            break
        ok_all = ok_all and res["ok"]
        md_max = max(md_max, res["maxdiff"])
        time_sum += res["time_us"]
    return Cell(lang, fam_label, fp_level, cost_model, ok_all, md_max, time_sum, compile_sum, error=err)


def run_kernel(kernel: "tsvc.TsvcKernel",
               languages: List[str],
               compilers: Dict[str, Dict[str, str]],
               strategy: str,
               preset: str,
               reps: int,
               workdir: Path,
               opt_mode: str = "simplify-parallel") -> Dict:
    """Emit + compile + run + time one kernel across every requested language x compiler.

    A kernel may split into several nests; each cell sums per-nest compile/time, so the result schema
    stays unchanged."""
    result = {"key": kernel.key, "corpus": kernel.corpus, "preset": preset, "host": socket.gethostname()}
    try:
        nests = extract_all_nests(lambda: tsvc.build_sdfg(kernel, opt_mode=opt_mode), strategy, kernel.key)
        if not nests:
            return {**result, "skipped": "no compute nest"}
        units: List[XlNest] = []
        for idx, name, symbol, boundary in nests:
            # validate at a small preset (fast oracle); time at the requested one; same size-agnostic .so.
            time_sizes = tsvc.sample_sizes(kernel, boundary, preset=preset)
            validate_sizes = tsvc.sample_sizes(kernel, boundary, preset=validate_preset(preset))
            nest_dir = workdir / f"n{idx}"
            prep = prepare(boundary, name, nest_dir, sizes=validate_sizes)
            # seeded once per nest so the oracle and every validating cell see the same subscripts.
            validate_fills = tsvc.index_fills(kernel, boundary, validate_sizes)
            oracle = run_oracle(prep, boundary, make_inputs(boundary, validate_sizes, seed=0, given=validate_fills),
                                validate_sizes)
            # build timing buffers once per nest; every cell's fork COW-inherits them, no per-cell re-fill.
            time_inputs = make_inputs(boundary,
                                      time_sizes,
                                      seed=0,
                                      given=tsvc.index_fills(kernel, boundary, time_sizes))
            names = list(boundary.standalone_sdfg.arrays) + list(validate_sizes)
            units.append(
                XlNest(idx, name, symbol, boundary, prep, nest_dir, time_sizes, validate_sizes, oracle, time_inputs,
                       names, validate_fills))
    except Exception as e:
        return {**result, "skipped": f"{type(e).__name__}: {str(e)[:160]}"}

    rows: List[Dict] = []
    for lang in languages:
        spec = _LANGS[lang]
        # emit + parse each nest's source up front; any nest failing drops the whole language.
        try:
            per_nest_src = []
            for u in units:
                src = next(s for s in emit_sources(u.prep, u.nest_dir, target=spec["target"])
                           if s.suffix == spec["suffix"] and "pluto" not in s.name)
                order = signature_order(src.read_text(), u.symbol, lang)
                order = fortran_unmunge(order, u.names) if lang == "fortran" else order
                per_nest_src.append((src, order, c_argtypes(order, u.boundary)))
        except Exception as e:
            rows.append(
                asdict(Cell(lang, "-", "-", "-", False, float("inf"), float("inf"), 0.0,
                            error=f"emit: {str(e)[:150]}")))
            continue
        for fam_label, cc in compilers.get(lang, {}).items():
            fam = family_of(fam_label)
            # sweep the fp-precision level x vectorizer cost-model matrix.
            for fp_level, cost_model, cflags in flags.flag_matrix(fam, lang):
                rows.append(
                    asdict(
                        measure_xl_cell(cc, lang, fam_label, fp_level, cost_model, cflags, units, per_nest_src, reps,
                                        workdir)))
    result["cells"] = rows
    # union of per-nest sizes.
    merged_validate: Dict[str, int] = {}
    merged_time: Dict[str, int] = {}
    for u in units:
        merged_validate.update(u.validate_sizes)
        merged_time.update(u.time_sizes)
    result["sizes"] = {
        "validate": {
            k: int(v)
            for k, v in merged_validate.items()
        },
        "time": {
            k: int(v)
            for k, v in merged_time.items()
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
        "| kernel | corpus | preset | language | compiler | winner fp/cost | strict maxdiff | winner (us) "
        "| fp speedup |", "|" + "---|" * 9
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
    ap.add_argument("--opt-mode",
                    default="simplify-parallel",
                    choices=list(tsvc.OPT_MODES),
                    help="pre-split optimization mode (simplify-parallel / canonicalize / auto-opt)")
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

    # kernels of every corpus, self-partitioned across ranks.
    kernels = [k for corpus in args.corpora for k in tsvc.iter_tsvc_kernels(only=args.only, corpus=corpus)]
    procid, ntasks = rank_and_size()
    mine = my_slice(kernels, procid, ntasks)
    if args.limit:
        mine = mine[:args.limit]
    print(f"[crosslang] rank {procid}/{ntasks}: {len(mine)} of {len(kernels)} kernels (preset {args.preset}) -> {out}")

    for i, kernel in enumerate(mine):
        workdir = Path(tempfile.mkdtemp(prefix=f"nf_xl_{kernel.corpus}_{kernel.key}_"))
        try:
            res = run_kernel(kernel,
                             args.languages,
                             compilers,
                             args.strategy,
                             args.preset,
                             args.reps,
                             workdir,
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
