"""E1 driver (paper C1): the fusion-granularity x backend heatmap.

C1 says the performance-optimal fusion partition is *backend-dependent* -- the winning granularity moves
when the compiler/ISA changes. To show it, the whole program is measured at each granularity rung with
EVERY nest compiled by one backend, for each backend in turn, and the argmin-granularity is read off per
backend. The variability across the backend column IS the finding.

This stitches the three built pieces together:
  * Axis 1 (:mod:`nestforge.granularity`) picks the partition (a :class:`~granularity.GranularityPoint`);
  * the arena's per-backend compile (:func:`nestforge.arena.compile_object` /
    :func:`~nestforge.arena.link_shared`) builds one shared lib per (kernel, backend, granularity) holding
    every nest's object -- one lib, so it resolves at load regardless of the parent's (dace-sorted) link
    order and there is a single library to swap in;
  * the differential harness (:func:`nestforge.differential.measure_in_context`) links those variant libs
    into the full program, swaps every nest to them, times it, and validates bit-exact against the
    whole-program oracle first.

"Backend" here is a compiler discovered on PATH (gcc/clang); the same driver extends to an ISA axis on the
cluster by adding cross-compilers to :func:`nestforge.arena.discover_compilers`. Every variant is built at
``ieee-strict`` so the swapped program stays bit-exact -- E1 measures where granularity moves time, not
where fast-math moves accuracy (that is the arena's FP axis, orthogonal here).
"""
from __future__ import annotations

import copy
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from nestforge import tsvc
from nestforge.arena import compile_object, discover_compilers, link_shared
from nestforge.differential import NestVariant, measure_in_context
from nestforge.emit_numpy import UnsupportedNest
from nestforge.granularity import GranularityPoint, granularity_ladder
from nestforge.pass_lower import lower_nests_to_external_call
from nestforge.perf.harness import signature_order
from nestforge.translate import emit_sources, prepare


@dataclass(frozen=True)
class E1Cell:
    """One heatmap cell: a kernel measured at one granularity rung with every nest built by one backend.
    ``median_us`` is ``inf`` (and ``ok`` False) when the build failed or the swapped program did not match
    the oracle; ``error`` carries the reason."""
    kernel: str
    backend: str
    granularity: str
    unit: str
    median_us: float
    ok: bool
    error: Optional[str] = None


def build_backend_variants(calls,
                           backend_name: str,
                           backend_path: str,
                           out_dir: Path,
                           fp_mode: str = "ieee-strict") -> Dict[str, NestVariant]:
    """Compile every nest of a lowered program with ONE backend and link them into a single shared lib.

    ``calls`` is the ``[(ExternalCall, Boundary)]`` from :func:`lower_nests_to_external_call`. Each nest is
    emitted to C, compiled to an object with ``backend_path`` at ``fp_mode``, then all objects are linked
    into one shared lib (:func:`~nestforge.arena.link_shared`) -- one ``.so`` per backend so
    ``ExternLibEnv`` links a single library (M0), and a ``.so`` resolves at load regardless of the parent's
    link order. Returns nest-name -> :class:`NestVariant`, all pointing at that one lib with their own
    extern-C symbol + ABI order."""
    out_dir.mkdir(parents=True, exist_ok=True)
    objs: List[Path] = []
    metas: List[Tuple[str, str, List[str]]] = []
    for ext, boundary in calls:
        vdir = out_dir / ext.name
        prep = prepare(boundary, ext.name, vdir)
        c_source = next(p for p in emit_sources(prep, vdir, target="c") if p.suffix == ".c")
        symbol = f"{ext.name}_fp64"
        order = signature_order(c_source.read_text(), symbol)
        objs.append(compile_object(backend_path, fp_mode, c_source, ext.name, vdir))
        metas.append((ext.name, symbol, order))
    lib = link_shared(objs, backend_name, out_dir, backend_path)
    return {name: NestVariant(str(lib), symbol, order) for name, symbol, order in metas}


def run_e1_cell(kernel: tsvc.TsvcKernel,
                backend_name: str,
                backend_path: str,
                point: GranularityPoint,
                out_dir,
                unit: str = "map",
                opt_mode: str = "canonicalize",
                reps: int = 7) -> E1Cell:
    """Measure one (kernel, backend, granularity) heatmap cell.

    Lowers the granularity-applied canonical program to find its nests (deterministic, so the nest names
    match the copy :func:`measure_in_context` lowers internally), builds a per-backend archive for them, and
    swaps every nest to that backend before timing the whole program in context."""
    out_dir = Path(out_dir)
    sdfg = tsvc.build_sdfg(kernel, opt_mode)  # the canonical P0 program
    point.apply(sdfg)  # P1: mutate to this partition
    sdfg.validate()
    calls = lower_nests_to_external_call(copy.deepcopy(sdfg), unit)
    variants = build_backend_variants(calls, backend_name, backend_path, out_dir / "variants")
    res = measure_in_context(kernel,
                             out_dir / "ctx",
                             variants=variants,
                             granularity=unit,
                             opt_mode=opt_mode,
                             apply_granularity=point.apply,
                             reps=reps)
    return E1Cell(kernel.key, backend_name, point.name, unit, res.median_us, res.ok, res.error)


def run_e1(kernels: Sequence[tsvc.TsvcKernel],
           out_dir,
           unit: str = "map",
           max_granularity_points: int = 3,
           opt_mode: str = "canonicalize",
           backends: Optional[Dict[str, str]] = None,
           reps: int = 7) -> List[E1Cell]:
    """The bounded E1 sweep: every (kernel, backend, granularity rung) at a fixed offloading ``unit``.

    ``backends`` defaults to the compilers on PATH; the granularity ladder is subsampled to
    ``max_granularity_points`` per kernel (:func:`granularity.granularity_ladder`), so the cell count is
    ``kernels x backends x <=max_granularity_points`` -- bounded like the rest of the sweep.

    A cell whose variant fails to emit/compile (a corpus kernel the translator cannot render, e.g. a
    non-integer array subscript) is recorded as a failed :class:`E1Cell` with the reason -- a
    skip-with-reason, never a silent drop and never a crash that loses the rest of the sweep."""
    out_dir = Path(out_dir)
    backends = backends or discover_compilers()
    cells: List[E1Cell] = []
    caught = (subprocess.CalledProcessError, OSError, UnsupportedNest, ValueError, KeyError)
    for kernel in kernels:
        try:  # a kernel that cannot even canonicalize/build its ladder is a skip, not a sweep-ending crash
            ladder = granularity_ladder(tsvc.build_sdfg(kernel, opt_mode), max_points=max_granularity_points)
        except caught as e:
            for backend_name in backends:
                cells.append(E1Cell(kernel.key, backend_name, "-", unit, float("inf"), False, repr(e)))
            continue
        for backend_name, backend_path in backends.items():
            for point in ladder:
                cell_dir = out_dir / kernel.key / backend_name / point.name
                try:
                    cells.append(run_e1_cell(kernel, backend_name, backend_path, point, cell_dir, unit, opt_mode, reps))
                except caught as e:
                    cells.append(E1Cell(kernel.key, backend_name, point.name, unit, float("inf"), False, repr(e)))
    return cells


def best_granularity_per_backend(cells: Sequence[E1Cell]) -> Dict[Tuple[str, str], str]:
    """The C1 read-off: for each (kernel, backend), the granularity rung with the fastest valid time. When
    the optimum differs across the backend column for a kernel, that is the backend-dependence C1 claims."""
    best: Dict[Tuple[str, str], Tuple[float, str]] = {}
    for c in cells:
        if not c.ok:
            continue
        key = (c.kernel, c.backend)
        if key not in best or c.median_us < best[key][0]:
            best[key] = (c.median_us, c.granularity)
    return {key: gran for key, (_us, gran) in best.items()}
