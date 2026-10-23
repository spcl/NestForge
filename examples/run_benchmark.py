# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Replay one stage 2 benchmark of ``examples/benchmarks.md``: the default (``default_moves()``) against the reference
route in ``examples/benchmarks.json``, both built through stages 3 to 5, checked against the NumPy oracle at preset S
and timed at preset M (median of the timed calls after one warm-up).

On CPU every kernel is rebuilt with OpenMP, with ``collapse(N)`` where the benchmark asks for it. The thread count is
``OMP_NUM_THREADS``: when it differs from ``--threads`` the script restarts itself with it set, before any OpenMP
runtime starts."""

import argparse
import copy
import dataclasses
import importlib.util
import json
import os
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import dace
import numpy as np
from dace.codegen.compiled_sdfg import CompiledSDFG

from nestforge.build.sdfg import compile_linked_program
from nestforge.build.toolchain import openmp_compile_flags
from nestforge.corpus.bench import CorpusKernel, iter_dace_kernels, preset_sizes
from nestforge.session import KernelBuild, Session
from nestforge.stages.canonicalize import Targets
from nestforge.stages.kernel import KernelSource, build_kernel_library

#: Where the reference routes live, next to this script.
ROUTES = Path(__file__).with_name("benchmarks.json")
#: Calls timed after the warm-up; the median is reported.
TIMED_REPS = 5
#: The pragma DaCe emits for a parallel CPU map, rewritten to ``collapse(N)`` when a benchmark asks for it.
PARALLEL_PRAGMA = "#pragma omp parallel for\n"
#: Denominator guard of the relative error.
TINY = 1e-30
#: Largest relative error against the oracle that counts as a match.
ORACLE_RTOL = 1e-8
#: Milliseconds per second.
MS = 1e3
#: Seed of the uniform inputs of a kernel without ``initialize()``.
SEED = 0

#: One stage 2 step: "finish", [kind, labels or None], or ["set_scope", labels].
Step = str | list[str | list[str] | None]


@dataclass(frozen=True, slots=True)
class Benchmark:
    """One entry of ``benchmarks.json``."""

    name: str
    moves: int
    device: str
    threads: int
    collapse: int
    baseline_collapse: int
    ratio: float
    reference: list[Step]


@dataclass(frozen=True, slots=True)
class Case:
    """The sizes, inputs and preset parameters of one preset."""

    sizes: dict[str, int]
    inputs: dict[str, np.ndarray | int | float]
    params: dict[str, int | float]


def load_benchmarks() -> dict[str, Benchmark]:
    raw: dict[str, dict] = json.loads(ROUTES.read_text())
    return {
        name: Benchmark(
            name=name,
            moves=entry["moves"],
            device=entry["device"],
            threads=entry["threads"],
            collapse=entry["collapse"],
            baseline_collapse=entry.get("baseline_collapse", 0),
            ratio=entry["ratio"],
            reference=entry["reference"],
        )
        for name, entry in raw.items()
        if not name.startswith("_")
    }


def module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def kernel_of(name: str) -> CorpusKernel:
    return next(k for k in iter_dace_kernels("benchmarking_creation") if k.short_name.endswith(f"/{name}/{name}"))


def uniform_inputs(kernel: CorpusKernel, sizes: dict[str, int]) -> dict[str, np.ndarray | int | float]:
    """Uniform draws for every argument of a kernel without ``initialize()`` (the TSVC kernels)."""
    sdfg, rng = kernel.to_sdfg(simplify=False), np.random.default_rng(SEED)
    inputs: dict[str, np.ndarray | int | float] = {}
    for name, desc in sdfg.arrays.items():
        if desc.transient:
            continue
        shape = tuple(int(dace.symbolic.evaluate(dim, sizes)) for dim in desc.shape)
        inputs[name] = rng.random(shape).astype(desc.dtype.as_numpy_dtype())
    return inputs


def case(kernel: CorpusKernel, preset: str) -> Case:
    """The inputs of ``preset``: from the kernel's ``initialize()`` when it has one, else uniform draws."""
    spec, folder, sizes = kernel.spec, kernel.dace_file.parent, preset_sizes(kernel, preset)
    generator = folder / f"{folder.name}.py"
    if not generator.exists():
        return Case(sizes, uniform_inputs(kernel, sizes), dict(sizes))
    knobs = {c: v.get("value") if isinstance(v, dict) else v for c, v in (spec.config or {}).items()}
    params = {**knobs, **dict(spec.parameters.get(preset, {}))}
    made = module(generator).initialize(*[params[a] for a in spec.init.input_args])
    made = made if isinstance(made, tuple) else (made,)
    inputs = {
        a: np.asarray(v, dtype=spec.init.dtypes[a]) if a in spec.init.dtypes else np.array(v, copy=True)
        for a, v in zip(spec.init.output_args, made, strict=True)
    }
    return Case(sizes, inputs, {**params, **sizes})


def oracle(kernel: CorpusKernel, c: Case) -> dict[str, np.ndarray]:
    folder = kernel.dace_file.parent
    entry: Callable[..., object] = vars(module(folder / f"{folder.name}_numpy.py"))[kernel.spec.func_name]
    arrays = {a: np.array(v, copy=True) for a, v in c.inputs.items()}
    entry(*[arrays[a] if a in arrays else c.params[a] for a in kernel.spec.input_args])
    return {o: arrays[o] for o in kernel.spec.output_args}


def apply_step(session: Session, step: Step) -> None:
    if step == "finish":
        session.finish_moves()
        return
    kind, labels = step[0], step[1]
    if kind == "set_scope":
        result = session.set_scope(list(labels), session.epoch)
    else:
        rows = session.list_moves(kind)
        hit = next((r for r in rows if labels is None or list(r.labels) == labels), None)
        if hit is None:
            raise RuntimeError(f"{kind} {labels} is not offered; offered {[list(r.labels) for r in rows]}")
        result = session.apply_move(kind, list(hit.labels), hit.epoch)
    if result.status.name != "APPLIED":
        raise RuntimeError(f"{kind} {labels}: {result.status.name}: {result.reason}")


def collapsed(src: KernelSource, depth: int, out_dir: Path) -> KernelSource:
    """A copy of ``src`` whose parallel CPU loops carry ``collapse(depth)``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    unit = out_dir / Path(src.unit).name
    text = Path(src.unit).read_text().replace(PARALLEL_PRAGMA, f"#pragma omp parallel for collapse({depth})\n")
    unit.write_text(text)
    return dataclasses.replace(src, unit=unit)


def build_kernels(session: Session, depth: int) -> None:
    """Stage 5 with the default builds; every CPU kernel rebuilt with OpenMP (and ``collapse(depth)`` when > 0)."""
    for item in session.list_kernels():
        session.optimize_kernel(item.name)
        built = session.builds[item.name]
        if built.source.device != "cpu":
            continue
        src = collapsed(built.source, depth, session.work_dir / item.name / "collapse") if depth else built.source
        flags = list(built.variant.flags) + openmp_compile_flags(built.variant.compiler)
        archive = build_kernel_library(src, built.variant.compiler, flags, session.work_dir / item.name / "omp")
        session.bind(item.name, KernelBuild(src, built.variant, archive))


def build(
    kernel: CorpusKernel, steps: list[Step], device: str, depth: int, sizes: dict[str, int], work: Path
) -> CompiledSDFG:
    """The linked program of ``steps`` (``None``: ``default_moves()``) after stages 3 to 5."""
    work.mkdir(parents=True, exist_ok=True)
    dace.Config.set("default_build_folder", value=str(work / "dacecache"))
    session = Session(
        kernel.to_sdfg(simplify=True), targets=Targets(gpu=device == "gpu"), work_dir=str(work), sizes=sizes
    )
    session.canonicalize()
    if not steps:
        session.default_moves()
    for step in steps:
        apply_step(session, step)
    if not session.list_kernels():
        session.default_scopes()
    session.place()
    build_kernels(session, depth)
    program = copy.deepcopy(session.sdfg)
    program.expand_library_nodes()
    return compile_linked_program(program, work / "parent")


def call(program: CompiledSDFG, kernel: CorpusKernel, c: Case) -> tuple[dict[str, np.ndarray], float]:
    """One call on fresh copies of the inputs: its outputs and its time in ms."""
    arrays = {n: v.copy() if isinstance(v, np.ndarray) else v for n, v in c.inputs.items()}
    names = set(map(str, program.sdfg.free_symbols)) | set(program.sdfg.arg_names)
    start = time.perf_counter()
    program(**arrays, **{n: v for n, v in c.sizes.items() if n in names})
    elapsed = MS * (time.perf_counter() - start)
    return {o: arrays[o] for o in kernel.spec.output_args}, elapsed


def rel_error(got: dict[str, np.ndarray], want: dict[str, np.ndarray]) -> float:
    return max(float(np.max(np.abs(got[o] - want[o]) / (np.abs(want[o]) + TINY))) for o in want)


def measure(kernel: CorpusKernel, steps: list[Step], device: str, depth: int, work: Path) -> tuple[float, float]:
    """The relative error against the oracle at S and the median time at M of one schedule."""
    small, medium = case(kernel, "S"), case(kernel, "M")
    got, _ = call(build(kernel, steps, device, depth, small.sizes, work / "S"), kernel, small)
    error = rel_error(got, oracle(kernel, small))
    program = build(kernel, steps, device, depth, medium.sizes, work / "M")
    call(program, kernel, medium)
    times = [call(program, kernel, medium)[1] for _ in range(TIMED_REPS)]
    return error, statistics.median(times)


def run(bench: Benchmark, device: str, work: Path) -> int:
    kernel = kernel_of(bench.name)
    depth = bench.collapse if device == "cpu" else 0
    base_depth = bench.baseline_collapse if device == "cpu" else 0
    base_err, base_ms = measure(kernel, [], device, base_depth, work / "baseline")
    ref_err, ref_ms = measure(kernel, bench.reference, device, depth, work / "reference")
    threads = os.environ.get("OMP_NUM_THREADS", "?")
    print(f"{bench.name} on {device}" + (f", {threads} threads" if device == "cpu" else ""))
    print(f"  default    {base_ms:10.2f} ms  oracle error at S {base_err:.1e}")
    print(f"  reference  {ref_ms:10.2f} ms  oracle error at S {ref_err:.1e}")
    print(f"  ratio      {base_ms / ref_ms:10.2f}x  (recorded {bench.ratio}x on its target)")
    return 0 if max(base_err, ref_err) <= ORACLE_RTOL else 1


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("name", nargs="?", help="benchmark name (see --list)")
    ap.add_argument("--list", action="store_true", help="list the benchmarks and exit")
    ap.add_argument("--device", choices=("cpu", "gpu"), help="default: the benchmark's target")
    ap.add_argument("--threads", type=int, help="CPU threads; default: the benchmark's target")
    ap.add_argument("--work", type=Path, default=Path("experiments") / "benchmarks", help="build folder")
    return ap.parse_args()


def main() -> int:
    args = parse()
    benchmarks = load_benchmarks()
    if args.list or args.name is None:
        for b in benchmarks.values():
            target = b.device + (f", {b.threads} threads" if b.device == "cpu" else "")
            print(f"{b.name:30s} {b.moves} moves  {target:18s} recorded {b.ratio}x")
        return 0
    bench = benchmarks[args.name]
    device = args.device or bench.device
    threads = str(args.threads or bench.threads or 1)
    if os.environ.get("OMP_NUM_THREADS") != threads:  # host code of a GPU program can run OpenMP too
        env = {**os.environ, "OMP_NUM_THREADS": threads, "OMP_PROC_BIND": "close", "OMP_PLACES": "cores"}
        os.execve(sys.executable, [sys.executable, *sys.argv], env)
    return run(bench, device, args.work / bench.name / device)


if __name__ == "__main__":
    sys.exit(main())
