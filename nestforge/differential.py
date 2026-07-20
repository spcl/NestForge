"""Per-nest full-program DIFFERENTIAL measurement (paper method): optimize one nest, but always measure
the WHOLE program with only that nest swapped.

A per-nest micro-time lies -- it ignores the offload boundary and the nest's real share of whole-program
time. The honest measurement swaps ONLY the nest under test to a candidate variant, keeps every other
nest at its current implementation (the numpy-reference fallback by default), builds the whole lowered
program, validates it bit-exact against the whole-program oracle, and times it. Comparing two variants for
a nest = two :func:`measure_in_context` calls that differ only in that nest's swap.

Reuses the whole-program lane's build/validate/time path (:mod:`nestforge.whole_program`), pointed at the
LOWERED parent (``ExternalCall`` nests) instead of a single extracted nest. The compiled program runs
FORKED (:func:`~nestforge.isolation.run_isolated`) so fresh code crashing never takes down the caller.
"""
from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from nestforge import build, tsvc
from nestforge.arena import make_inputs, maxdiff, run_oracle
from nestforge.extract import whole_program_boundary
from nestforge.isolation import run_isolated
from nestforge.libnode import ExternalCall, ExternLibEnv
from nestforge.pass_lower import lower_nests_to_external_call
from nestforge.perf import flags
from nestforge.perf.harness import median
from nestforge.translate import prepare_whole_program


@dataclass
class NestVariant:
    """A compiled kernel to swap into one nest: a static/shared library, its ``extern "C"`` entry, and the
    argument order that symbol expects. ``abi_order`` is the silent-break field -- it must match the
    compiled signature, not the manifest/role order."""
    lib_path: str
    symbol: str
    abi_order: List[str]


@dataclass
class ContextResult:
    """One whole-program measurement with a given set of nests swapped to variants (the rest at reference).
    ``median_us`` is ``inf`` and ``error`` is set when the build or forked run failed; ``ok`` is the
    bit-exact verdict vs the whole-program oracle."""
    swapped: List[str]
    ok: bool
    maxdiff: float
    median_us: float
    reps: int
    error: Optional[str] = None


def variant_link_args(variants: Dict[str, NestVariant]) -> List[str]:
    """Link args for the extern variant libs to swap in: each UNIQUE ``lib_path`` once, with its ``-rpath``
    so the built program finds it at load. nest-forge compiles the frame directly (not via DaCe's CMake), so
    the variant libs must be passed to the linker explicitly (:attr:`build.BuildOptions.extra_link`);
    multiple nests sharing one backend ``.so`` collapse to a single ``-l`` entry."""
    args: List[str] = []
    for lib in dict.fromkeys(v.lib_path for v in variants.values()):  # unique, order-preserving
        args += [lib, f"-Wl,-rpath,{os.path.dirname(os.path.abspath(lib))}"]
    return args


def set_nest_variant(ext: ExternalCall, variant: NestVariant) -> None:
    """Point one ``ExternalCall`` at a compiled variant AND select the extern-call expansion, so
    ``expand_library_nodes`` routes this nest to the linked ``.a`` symbol instead of the ``DaceReference``
    numpy fallback. Every field of every other nest is left untouched -- an un-swapped nest keeps the
    default ``DaceReference`` implementation, so the rest of the program stays at its current form."""
    ext.lib_path = variant.lib_path
    ext.symbol = variant.symbol
    ext.abi_order = list(variant.abi_order)
    ext.implementation = "ExternCall"  # without this the swap is inert: expand picks DaceReference


def measure_in_context(kernel: tsvc.TsvcKernel,
                       out_dir,
                       variants: Optional[Dict[str, NestVariant]] = None,
                       granularity: str = "skip-taskloops",
                       opt_mode: str = "canonicalize",
                       apply_granularity: Optional[Callable[["object"], None]] = None,
                       sdfg: Optional["object"] = None,
                       preset: str = "S",
                       reps: int = 7,
                       seed: int = 0,
                       atol: Optional[float] = None,
                       timeout: float = 900.0) -> ContextResult:
    """Build + validate + time the whole program with ``variants`` swapped in (nest name -> variant), every
    other nest at the numpy-reference fallback.

    ``apply_granularity`` (a :meth:`granularity.GranularityPoint.apply`) mutates the canonical program to
    one rung of the fusion-granularity ladder (Axis 1) before measuring -- this is how E1 sweeps granularity.
    ``granularity`` is the P2 offloading unit; ``opt_mode`` builds the canonical P0 base.
    Default is ``"canonicalize"`` -- the canonical P0 start (the same start point for every kernel). Its
    soundness-guard state (``check_assumption`` CPP traps) is inert and skipped by the numpy oracle emitter
    (:func:`nestforge.emit_numpy.is_assumption_guard_block`). With ``variants`` empty this measures the
    all-reference lowered program -- the differential baseline point every swapped variant is compared to.
    """
    variants = variants or {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    atol = flags.FP_ATOL["strict-ieee"] if atol is None else atol

    if sdfg is None:
        sdfg = tsvc.build_sdfg(kernel, opt_mode)  # the canonical P0 program
        if apply_granularity is not None:
            apply_granularity(sdfg)  # P1: mutate to a chosen partition (a granularity.GranularityPoint.apply);
            sdfg.validate()  # value-preserving, so the oracle emitted below still matches
    # else: the caller already built the canonical program AND applied its granularity -- reuse it rather
    # than repeat a canonicalize (~0.5s warm, ~2.9s cold) that is pure duplicated work per sweep cell. The
    # SDFG is only READ here (the lowering runs on a deepcopy), so the caller's copy stays pristine.
    boundary = whole_program_boundary(sdfg)
    sizes = tsvc.sample_sizes(kernel, boundary, preset=preset)
    inputs = make_inputs(boundary, sizes, seed=seed, given=tsvc.index_fills(kernel, boundary, sizes, seed=seed))
    prep = prepare_whole_program(sdfg, kernel.key, out_dir, sizes=sizes)
    oracle = run_oracle(prep, boundary, inputs, sizes)

    lowered = copy.deepcopy(sdfg)  # externalize on a copy; the oracle SDFG stays intact
    calls = lower_nests_to_external_call(lowered, granularity)
    swapped: List[str] = []
    for ext, _boundary in calls:
        if ext.name in variants:
            set_nest_variant(ext, variants[ext.name])
            swapped.append(ext.name)
    missing = set(variants) - set(swapped)
    if missing:
        raise KeyError(f"variants name nests not in this granularity: {sorted(missing)} "
                       f"(present: {sorted(ext.name for ext, _ in calls)})")
    ExternLibEnv.reset()  # drop libraries accumulated by a previous build (their temp dirs may be gone)
    lowered.expand_library_nodes()
    built = build.build_sdfg(lowered,
                             out_dir / "build",
                             opts=build.BuildOptions(extra_link=variant_link_args(variants)))

    def work():
        vbuf = {k: v.copy() for k, v in inputs.items()}
        built.run(vbuf, sizes)
        outs = {o: vbuf[o] for o in boundary.outputs if o in vbuf}
        if outs:
            md = float(maxdiff({o: oracle[o] for o in outs}, outs))
            verdict = {"ok": bool(md <= atol), "maxdiff": md}
        else:
            verdict = {"ok": False, "maxdiff": float("inf")}
        tbuf = {k: v.copy() for k, v in inputs.items()}
        built.init(sizes)
        try:
            fn, cargs = built.bind_program(tbuf, sizes)
            fn(*cargs)  # warm
            # Restore every buffer the program WRITES before each timed rep. Without this an in-place nest
            # (a[:] = a[:] * b) feeds on its own previous output -- rep k computes a * b**k, which reaches
            # Inf/denormals in a few reps, and denormal arithmetic rather than the kernel dominates the
            # median. That silently biases the exact quantity E1 compares across granularity rungs. The
            # restore writes in place (cargs holds these buffers) and sits OUTSIDE the timed region.
            mutated = [o for o in boundary.outputs if o in tbuf]
            samples: List[float] = []
            for _ in range(reps):
                for name in mutated:
                    tbuf[name][...] = inputs[name]
                t0 = time.perf_counter()
                fn(*cargs)
                samples.append((time.perf_counter() - t0) * 1e6)
        finally:
            built.close()
        return {**verdict, "median_us": median(samples)}

    res = run_isolated(work, timeout=timeout)
    if "error" in res:
        return ContextResult(swapped, False, float("inf"), float("inf"), reps, res["error"])
    return ContextResult(swapped, bool(res["ok"]), float(res["maxdiff"]), float(res["median_us"]), reps)
