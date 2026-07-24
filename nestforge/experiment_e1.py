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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

from nestforge import tsvc
from nestforge.arena import compile_object, discover_compilers, link_shared
from nestforge.differential import NestVariant, measure_in_context
from nestforge.granularity import GranularityPoint, granularity_ladder
from nestforge.pass_lower import lower_nests_to_external_call
from nestforge.perf.harness import signature_order
from nestforge.translate import emit_sources, prepare

if TYPE_CHECKING:
    from nestforge.extract import Boundary
    from nestforge.libnode import ExternalCall


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


def build_backend_variants(calls: List[Tuple[ExternalCall, Boundary]],
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
        if not order:
            # An empty order reaches the extern-call expansion as "has no abi_order", which reads as
            # "the arena never recorded it" and sends the reader to the wrong place -- the order WAS
            # recorded, the emitted entry simply takes no parameters. A nest that crosses no data
            # computes nothing observable, so say that here, with what the boundary thinks it carries.
            raise ValueError(f"nest {ext.name!r} emitted the zero-argument entry 'void {symbol}(void)': it "
                             f"crosses no data, so an extern call to it is a no-op. Boundary inputs="
                             f"{sorted(boundary.inputs)}, outputs={sorted(boundary.outputs)}, "
                             f"symbols={sorted(boundary.symbols)}; source={c_source}")
        objs.append(compile_object(backend_path, fp_mode, c_source, ext.name, vdir))
        metas.append((ext.name, symbol, order))
    lib = link_shared(objs, backend_name, out_dir, backend_path)
    return {name: NestVariant(str(lib), symbol, order) for name, symbol, order in metas}


def run_e1_cell(kernel: tsvc.TsvcKernel,
                backend_name: str,
                backend_path: str,
                point: GranularityPoint,
                out_dir: Union[str, Path],
                unit: str = "map",
                opt_mode: str = "canonicalize",
                reps: int = 7,
                canonical: Optional["object"] = None,
                preset: str = "S",
                seed: int = 0) -> E1Cell:
    """Measure one (kernel, backend, granularity) heatmap cell.

    Lowers the granularity-applied canonical program to find its nests (deterministic, so the nest names
    match the copy :func:`measure_in_context` lowers internally), builds a per-backend archive for them, and
    swaps every nest to that backend before timing the whole program in context."""
    out_dir = Path(out_dir)
    # ``canonical`` is the kernel's P0 program, built ONCE by the sweep and shared across its cells. Each
    # cell mutates its own copy (point.apply), and a deepcopy is ~195x cheaper than re-canonicalizing
    # (1.5 ms vs 299 ms measured), so rebuilding per cell was almost pure waste.
    sdfg = copy.deepcopy(canonical) if canonical is not None else tsvc.build_sdfg(kernel, opt_mode)
    point.apply(sdfg)  # P1: mutate to this partition
    sdfg.validate()
    calls = lower_nests_to_external_call(copy.deepcopy(sdfg), unit)
    if not calls:
        # No nest at this unit (e.g. unit='cfg' on a kernel with no LoopRegion). Measuring anyway would time
        # the all-reference program under a backend label -- identical for every backend, because no backend
        # compiled anything -- fabricating "backend-independent" heatmap data. Record it as a skip instead.
        return E1Cell(kernel.key, backend_name, point.name, unit, float("inf"), False,
                      f"no {unit!r} nest to offload at granularity {point.name!r}")
    variants = build_backend_variants(calls, backend_name, backend_path, out_dir / "variants")
    res = measure_in_context(
        kernel,
        out_dir / "ctx",
        variants=variants,
        granularity=unit,
        opt_mode=opt_mode,
        sdfg=sdfg,  # already canonical + granularity-applied above; do not rebuild it
        reps=reps,
        # Pinned, not defaulted: E2 divides these times by a baseline measured through a DIFFERENT entry
        # point (measure_whole_program), and two independent defaults drifting apart would silently
        # produce a speedup ratio between runs at different problem sizes.
        preset=preset,
        seed=seed)
    return E1Cell(kernel.key, backend_name, point.name, unit, res.median_us, res.ok, res.error)


def run_e1(kernels: Sequence[tsvc.TsvcKernel],
           out_dir: Union[str, Path],
           unit: str = "map",
           max_granularity_points: int = 3,
           opt_mode: str = "canonicalize",
           backends: Optional[Dict[str, str]] = None,
           reps: int = 7,
           preset: str = "S",
           seed: int = 0) -> List[E1Cell]:
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
    # Every per-cell failure is RECORDED (repr(e) on the cell), never silenced -- so the catch is broad on
    # purpose: a corpus sweep meets dace InvalidSDFGError from validate(), TypeError from extract,
    # ZeroDivisionError from a degenerate ladder, and more. Enumerating them invites the next unlisted
    # exception to discard every remaining kernel. KeyboardInterrupt/SystemExit are not Exception, so an
    # operator can still stop the run.
    caught = Exception
    for kernel in kernels:
        try:  # a kernel that cannot even canonicalize/build its ladder is a skip, not a sweep-ending crash
            canonical = tsvc.build_sdfg(kernel, opt_mode)  # built ONCE per kernel, shared by every cell
            ladder = granularity_ladder(canonical, max_points=max_granularity_points)
        except caught as e:
            for backend_name in backends:
                cells.append(E1Cell(kernel.key, backend_name, "-", unit, float("inf"), False, repr(e)))
            continue
        for backend_name, backend_path in backends.items():
            for point in ladder:
                cell_dir = out_dir / kernel.key / backend_name / point.name
                try:
                    cells.append(
                        run_e1_cell(kernel,
                                    backend_name,
                                    backend_path,
                                    point,
                                    cell_dir,
                                    unit,
                                    opt_mode,
                                    reps,
                                    canonical=canonical,
                                    preset=preset,
                                    seed=seed))
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
