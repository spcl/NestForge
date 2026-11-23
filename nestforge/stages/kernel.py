# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage 5, kernel optimization: render one kernel as a standalone CPF unit with one C entry, then build and validate
``lib<kernel>.a``."""

from __future__ import annotations

import copy
import dataclasses
import math
import re
from collections.abc import Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path

import dace
import numpy as np
from dace import dtypes
from dace.codegen import cpf
from dace.codegen.common import get_gpu_backend
from dace.libraries.standard.nodes.external_call import ExternalCall
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize.finalize import finalize_for_target, offload_to_gpu
from dace.transformation.passes.megakernel.loops import MegakernelLoops
from dace.transformation.passes.megakernel.megakernel import MegaKernelOptimization

from nestforge.build.arena import (
    TIMED_REPS,
    DeviceCall,
    accumulating_outputs,
    bounded_oracle,
    call_device,
    call_native,
    diff_stats,
    dtype_floor,
    make_inputs,
    run_oracle,
    rung_rtol,
)
from nestforge.build.flags import cuda_base_flags, hip_base_flags
from nestforge.build.isolation import Failure, Result, run_isolated, run_spawned
from nestforge.build.sdfg import (
    BuildOptions,
    build_archive,
    build_cuda_archive,
    build_hip_archive,
    force_register_arrays,
    new_gpu_codegen,
    program_compiler,
)
from nestforge.build.toolchain import (
    SymbolValue,
    cudart_dir,
    cudart_link_flags,
    hip_link_flags,
    hip_runtime_dir,
    openmp_link_flags,
    parse_params,
    raw_signature,
)
from nestforge.corpus.translate import Prepared
from nestforge.ir.extract import INT64, Boundary
from nestforge.stages.expand import pin_pure_without_library
from nestforge.stages.placement import kernel_device

# Suffix of a GPU kernel's device unit: ``<kernel>_device.cu`` (``.hip``) beside the host unit ``<kernel>.cu``.
DEVICE_UNIT_SUFFIX = "_device"
# Line that separates the host and device units in the one text an agent reads and writes for a GPU kernel.
DEVICE_UNIT_MARKER = "// ---- device unit ----"

#: Seconds a kernel's NumPy oracle may run per size: its loops are interpreted, about 1000x slower than the kernel's,
#: and a program has dozens of kernels (harris_corner has 48, each about 55 s at XL).
ORACLE_TIMEOUT_S = 10.0


@dataclass(slots=True)
class KernelSource:
    """The kernel's CPF translation unit, defining ``extern "C" <name>`` with parameters in ``abi_order``. A GPU
    kernel CPF renders in two has the host launcher in ``unit`` and the ``__global__`` kernels and the launchers
    it calls in ``device_unit``."""

    name: str
    unit: Path
    abi_order: list[str]
    boundary: Boundary
    device: str
    device_unit: Path | None = None
    link: list[str] = dataclasses.field(default_factory=list)  # the vendor libraries the units call

    @property
    def units(self) -> list[Path]:
        """Every file the kernel's library is built from, host unit first."""
        return [self.unit] if self.device_unit is None else [self.unit, self.device_unit]

    @property
    def symbol(self) -> str:
        return self.name


@dataclass(slots=True)
class KernelVerdict:
    """A build compared to the NumPy oracle and timed; ``ok`` gates at ``fp_mode``. ``unchecked`` says why the
    kernel cannot run standalone (a symbol only the program computes), so it was neither compared nor timed."""

    fp_mode: str
    maxdiff: float
    md_rel: float
    dtype_floor: float
    time_us: float
    error: str = ""
    unchecked: str = ""
    #: The drift a reordered accumulation may carry (:func:`reassociation_growth`), added to the gate as the bench
    #: grades (#106): a parallel or library reduction sums in another order than NumPy.
    growth: float = 0.0

    @property
    def ok(self) -> bool:
        gate = rung_rtol(self.fp_mode, self.dtype_floor) + self.growth
        return not self.error and not self.unchecked and self.md_rel <= gate


def extent(expr: object, sizes: Mapping[str, object]) -> int:
    """``expr`` at ``sizes`` as a count; 1 when it does not resolve, which adds no drift (the stricter side)."""
    try:
        value = dace.symbolic.evaluate(expr, dict(sizes))
    except (TypeError, ValueError, KeyError, AttributeError):
        return 1
    try:
        return max(int(value), 1)
    except (TypeError, ValueError):
        return 1


def accumulation_length(sdfg: dace.SDFG, sizes: Mapping[str, object]) -> int:
    """The longest accumulation in ``sdfg`` at ``sizes``, the bench's contracted extent ``l``: for a reduction (a
    write-conflict edge out of a map) the product of the map's ranges its written subset does not mention, for a
    library call the product of its inputs' dimensions no output keeps; 1 when there is none."""
    longest = 1
    for node, parent in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.MapExit) and isinstance(parent, dace.SDFGState):
            entry = parent.entry_node(node)
            for edge in parent.in_edges(node):
                if edge.data.wcr is None:
                    continue
                written = {str(symbol) for symbol in edge.data.subset.free_symbols}
                lengths = [
                    extent(size, sizes)
                    for param, size in zip(entry.map.params, entry.map.range.size(), strict=True)
                    if param not in written
                ]
                longest = max(longest, math.prod(lengths))
        elif isinstance(node, nodes.LibraryNode) and isinstance(parent, dace.SDFGState):
            graph = parent.sdfg
            kept = {
                str(dim)
                for edge in parent.out_edges(node)
                if edge.data.data
                for dim in graph.arrays[edge.data.data].shape
            }
            for edge in parent.in_edges(node):
                if not edge.data.data:
                    continue
                dropped = [extent(dim, sizes) for dim in graph.arrays[edge.data.data].shape if str(dim) not in kept]
                longest = max(longest, math.prod(dropped))
    return longest


def reassociation_growth(sdfg: dace.SDFG, outputs: Sequence[str], sizes: Mapping[str, object]) -> float:
    """The relative drift an accumulation of :func:`accumulation_length` may carry in the narrowest float the
    outputs hold, as the bench's ``precision.accumulation_growth`` gives it: ``eps * sqrt(l)`` from fp32 up (two
    summation orders), ``eps / 2 * (ceil(log2 l) + 1)`` for a narrower format (a tree)."""
    floats = [
        np.finfo(sdfg.arrays[name].dtype.type)
        for name in outputs
        if name in sdfg.arrays and np.issubdtype(sdfg.arrays[name].dtype.type, np.inexact)
    ]
    length = accumulation_length(sdfg, sizes)
    if not floats or length <= 1:
        return 0.0
    eps = float(max(info.eps for info in floats))
    if eps <= float(np.finfo(np.float32).eps):
        return eps * math.sqrt(length)
    return eps / 2 * (math.ceil(math.log2(length)) + 1)


def failed_verdict(fp_mode: str, error: str) -> KernelVerdict:
    return KernelVerdict(fp_mode, float("inf"), float("inf"), 0.0, float("inf"), error)


def unchecked_verdict(fp_mode: str, reason: str) -> KernelVerdict:
    return KernelVerdict(fp_mode, float("inf"), float("inf"), 0.0, float("inf"), unchecked=reason)


def computed_symbols_reason(symbols: list[str]) -> str:
    return f"takes {', '.join(symbols)}, which only the running program computes; the whole program checks it"


def abi_ready_copy(boundary: Boundary) -> dace.SDFG:
    """A copy of the kernel whose entry takes the integer symbols as the ``int64_t`` ``ExternalCall`` passes."""
    sdfg = copy.deepcopy(boundary.standalone_sdfg)
    for name in boundary.symbols:
        if sdfg.symbols[name] in dtypes.INTEGER_TYPES:
            sdfg.symbol_repo.set_type(name, INT64)
    return sdfg


def cpu_schedule(boundary: Boundary) -> dace.SDFG:
    """The kernel copy the C++ unit renders: :func:`abi_ready_copy` finalized for the CPU."""
    sdfg = abi_ready_copy(boundary)
    pin_pure_without_library(sdfg)
    sdfg = finalize_for_target(sdfg, "cpu")
    force_register_arrays(sdfg)
    return sdfg


def gpu_schedule(boundary: Boundary) -> dace.SDFG:
    """The kernel copy the CUDA or HIP unit renders: offloaded whole, so its arrays arrive as device pointers, then
    finalized for the GPU."""
    sdfg = abi_ready_copy(boundary)
    # placement offloaded the body already; DaCe's offloading takes only Default or CPU scopes and derives them again
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, (nodes.MapEntry, nodes.LibraryNode)) and node.schedule == dtypes.ScheduleType.GPU_Device:
            node.schedule = dtypes.ScheduleType.Default
    offload_to_gpu(sdfg)
    return finalize_for_target(sdfg, "gpu")


def build_cpu_library(
    units: Sequence[Path], compiler: str, flags: list[str] | None, archive: Path, link: Sequence[str]
) -> None:
    opts = BuildOptions(compiler=compiler, flags=flags, libraries=list(link))
    build_archive(units, None, archive, archive.with_suffix(".so"), opts)


def build_cuda_library(
    units: Sequence[Path], compiler: str, flags: list[str] | None, archive: Path, link: Sequence[str]
) -> None:
    chosen = flags if flags is not None else cuda_base_flags(cpf.CUDA_BUILD_FLAGS)
    build_cuda_archive(units, archive, archive.with_suffix(".so"), compiler, chosen, link)


def build_hip_library(
    units: Sequence[Path], compiler: str, flags: list[str] | None, archive: Path, link: Sequence[str]
) -> None:
    chosen = flags if flags is not None else hip_base_flags()
    build_hip_archive(units, archive, archive.with_suffix(".so"), compiler, chosen, link)


def process_runtime_libraries() -> list[str]:
    """libomp, the process's one OpenMP runtime, spelled for the program's linker."""
    return openmp_link_flags(program_compiler())


def cpu_runtime_libraries(compiler: str) -> list[str]:
    return process_runtime_libraries()


def cuda_runtime_libraries(compiler: str) -> list[str]:
    """libomp, and the ``libcudart`` the kernel's own nvcc links."""
    return [*process_runtime_libraries(), *cudart_link_flags(cudart_dir(compiler))]


def hip_runtime_libraries(compiler: str) -> list[str]:
    """libomp, and the ``libamdhip64`` of the kernel's own hipcc."""
    return [*process_runtime_libraries(), *hip_link_flags(hip_runtime_dir(compiler))]


class KernelNumbers(Result):
    """One measured kernel: its distance to the oracle, the dtype's floor on it, and its time per call."""

    maxdiff: float
    md_rel: float
    dtype_floor: float
    time_us: float


def kernel_numbers(
    oracle: dict[str, np.ndarray] | None, outputs: dict[str, np.ndarray], time_us: float
) -> KernelNumbers:
    """``oracle`` is ``None`` for a kernel timed at sizes it was not checked at: there is nothing to compare."""
    md, md_rel = (0.0, 0.0) if oracle is None else diff_stats(oracle, outputs)
    return KernelNumbers(maxdiff=md, md_rel=md_rel, dtype_floor=dtype_floor(outputs), time_us=time_us)


def measure_on_cpu(
    shared: Path,
    src: KernelSource,
    inputs: dict[str, np.ndarray],
    oracle: dict[str, np.ndarray] | None,
    sizes: Mapping[str, SymbolValue],
    reps: int,
) -> KernelNumbers | Failure:
    """The CPU twin, called in a forked child."""
    argtypes = [p.ctype for p in parse_params(raw_signature(src.unit.read_text(), src.symbol))]

    def work() -> KernelNumbers:
        outs, us = call_native(shared, src.symbol, src.abi_order, argtypes, src.boundary, inputs, sizes, reps)
        assert outs is not None, "call_native snapshots outputs unless told not to"
        return kernel_numbers(oracle, outs, us)

    return run_isolated(work, KernelNumbers)


def measure_on_gpu(
    shared: Path,
    src: KernelSource,
    inputs: dict[str, np.ndarray],
    oracle: dict[str, np.ndarray] | None,
    sizes: Mapping[str, SymbolValue],
    reps: int,
) -> KernelNumbers | Failure:
    """The GPU twin, called in a freshly spawned interpreter: a child forked from a process that already holds a
    device context cannot use the device. The comparison runs here, so this process never touches the device."""
    call = DeviceCall(
        str(shared),
        src.symbol,
        list(src.abi_order),
        raw_signature(src.unit.read_text(), src.symbol),
        list(src.boundary.outputs),
        accumulating_outputs(src.boundary, inputs),
        inputs,
        sizes,
        reps,
    )
    res = run_spawned(call_device, call)
    if isinstance(res, Failure):
        return res
    return kernel_numbers(oracle, res.outputs, res.time_us)


@dataclass(frozen=True, slots=True)
class KernelForm:
    """How the kernels of one device are scheduled, rendered, built and called."""

    language: str
    suffix: str
    schedule: Callable[[Boundary], dace.SDFG]
    build: Callable[[Sequence[Path], str, list[str] | None, Path, Sequence[str]], None]
    measure: Callable[
        [Path, KernelSource, dict[str, np.ndarray], dict[str, np.ndarray] | None, Mapping[str, SymbolValue], int],
        KernelNumbers | Failure,
    ]
    runtime: Callable[[str], list[str]]


CPU_FORM = KernelForm("c++", ".cpp", cpu_schedule, build_cpu_library, measure_on_cpu, cpu_runtime_libraries)

#: The GPU form per DaCe GPU backend (``compiler.cuda.backend``, else probed from the machine).
GPU_FORMS: dict[str, KernelForm] = {
    "cuda": KernelForm("cuda", ".cu", gpu_schedule, build_cuda_library, measure_on_gpu, cuda_runtime_libraries),
    "hip": KernelForm("hip", ".hip", gpu_schedule, build_hip_library, measure_on_gpu, hip_runtime_libraries),
}


def kernel_form(device: str) -> KernelForm:
    """How a kernel on ``device`` (``cpu`` or ``gpu``) is scheduled, rendered, built and called; a GPU kernel in
    the dialect of this machine's GPU backend."""
    return CPU_FORM if device == "cpu" else GPU_FORMS[get_gpu_backend()]


def persistent_loops(sdfg: dace.SDFG, device: str, loops: AbstractSet[str]) -> int:
    """Run the sequential loops of kernels labeled ``loops`` in the scheduled kernel ``sdfg`` as one persistent kernel
    each (DaCe's ``MegakernelLoops``: on the GPU behind a runtime test that every kernel's work fits the resident threads, with the
    per-launch loop as the fallback; on the CPU as one OpenMP team). The number of loops it fused, 0 for none."""
    device_type = dtypes.DeviceType.GPU if device == "gpu" else dtypes.DeviceType.CPU
    return MegakernelLoops(device_type, set(loops)).apply_pass(sdfg, {}) or 0


def launches(sdfg: dace.SDFG) -> int:
    """The GPU kernels ``sdfg`` launches: its ``GPU_Device`` doalls."""
    return sum(
        isinstance(node, nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.GPU_Device
        for node, _ in sdfg.all_nodes_recursive()
    )


def thread_block_maps(sdfg: dace.SDFG) -> bool:
    """Whether a doall of ``sdfg`` runs on a GPU kernel's thread block."""
    return any(
        isinstance(node, nodes.MapEntry)
        and node.map.schedule in (dtypes.ScheduleType.GPU_ThreadBlock, dtypes.ScheduleType.GPU_ThreadBlock_Dynamic)
        for node, _ in sdfg.all_nodes_recursive()
    )


def schedule_kernel(
    ext: ExternalCall, boundary: Boundary, out_dir: Path, persistent: AbstractSet[str] = frozenset()
) -> KernelSource:
    """The kernel for the device stage 4 placed ``ext`` on, rendered by CPF into ``<out_dir>/<kernel>.cpp``,
    ``.cu`` or ``.hip``, whose one entry is ``ext``'s symbol; the loops labeled ``persistent`` run as one persistent
    kernel each (:func:`persistent_loops`)."""
    device = kernel_device(ext)
    form = kernel_form(device)
    sdfg = form.schedule(boundary)
    if persistent:
        persistent_loops(sdfg, device, persistent)
    elif device == "gpu" and launches(sdfg) > 1 and not thread_block_maps(sdfg):
        # one scope is one kernel: its doalls (a branch of each, a run of them) launch as one persistent kernel.
        # shortcut: DaCe's megakernel takes flat kernels only, so a body with thread-block maps (a block reduction)
        # launches per doall; upgrade when MegaKernelOptimization keeps an inner block's size
        MegaKernelOptimization().apply_pass(sdfg, {})
    sdfg.name = ext.name  # pyright: ignore[reportAttributeAccessIssue]  # a DaCe Property, not read-only
    with new_gpu_codegen():
        rendering = cpf.render(sdfg, language=form.language)
    out_dir.mkdir(parents=True, exist_ok=True)
    unit = out_dir / f"{ext.name}{form.suffix}"
    unit.write_text(rendering.code)
    device_unit = out_dir / f"{ext.name}{DEVICE_UNIT_SUFFIX}{form.suffix}" if rendering.device_code else None
    if device_unit is not None:
        device_unit.write_text(rendering.device_code)
    return KernelSource(ext.name, unit, list(rendering.arguments), boundary, device, device_unit, list(rendering.link))


def kernel_text(src: KernelSource) -> str:
    """The kernel's source as one text: the host unit, then, for a GPU kernel CPF rendered in two, the marker line
    and the device unit."""
    if src.device_unit is None:
        return src.unit.read_text()
    return f"{src.unit.read_text()}\n{DEVICE_UNIT_MARKER}\n{src.device_unit.read_text()}"


def split_kernel_text(text: str) -> tuple[str, str | None]:
    """The inverse of :func:`kernel_text`: the host unit and the device unit, ``None`` when there is no marker."""
    host, marker, device = text.partition(f"\n{DEVICE_UNIT_MARKER}\n")
    return (host, device) if marker else (text, None)


def build_kernel_library(src: KernelSource, compiler: str, flags: list[str] | None, out_dir: Path) -> Path:
    """Build ``<out_dir>/lib<kernel>.a`` from the kernel's unit, plus its shared twin for validation."""
    archive = out_dir / f"lib{src.name}.a"
    kernel_form(src.device).build(src.units, compiler, flags, archive, src.link)
    return archive


def kernel_runtime_libraries(src: KernelSource, compiler: str) -> list[str]:
    """What a program linking ``src``'s library, built by ``compiler``, must link after its objects."""
    return [*src.link, *kernel_form(src.device).runtime(compiler)]


def use_kernel_library(ext: ExternalCall, lib_path: Path, src: KernelSource, runtime_libraries: Sequence[str]) -> None:
    """Point ``ext`` at a built library and the runtimes it needs, and select the extern-call expansion. The
    prototype is the unit's own parameter list: the parent's symbol types may be narrower than the entry's."""
    ext.lib_path, ext.symbol, ext.abi_order = str(lib_path), src.symbol, list(src.abi_order)
    ext.signature = raw_signature(src.unit.read_text(), src.symbol)
    ext.link_flags = list(runtime_libraries)
    ext.implementation = "ExternCall"


@dataclass(frozen=True, slots=True)
class Workload:
    """Inputs and the sizes they were drawn at."""

    inputs: dict[str, np.ndarray]
    sizes: Mapping[str, SymbolValue]


@dataclass(frozen=True, slots=True)
class Probe:
    """What a build of a kernel is measured on: timed on ``timed``, compared on ``checked`` to the ``oracle`` outputs
    of those inputs. ``checked`` is ``timed`` unless the oracle cannot finish at the timed sizes; a kernel checked at
    a smaller size is timed without a comparison. ``unchecked`` says why the oracle finished at no size: such a kernel
    goes unchecked, and the whole program checks it."""

    timed: Workload
    checked: Workload
    oracle: dict[str, np.ndarray]
    unchecked: str = ""


def full_probe(prep: Prepared, boundary: Boundary, sizes: Mapping[str, SymbolValue]) -> Probe:
    """A probe at ``sizes``, however long the oracle takes."""
    timed = Workload(make_inputs(boundary, sizes), sizes)
    return Probe(timed, timed, run_oracle(prep, boundary, timed.inputs, sizes))


def bounded_probe(
    prep: Prepared,
    boundary: Boundary,
    sizes: Mapping[str, SymbolValue],
    check_sizes: Mapping[str, SymbolValue],
    full: bool,
) -> Probe:
    """A probe whose oracle takes at most :data:`ORACLE_TIMEOUT_S` per size. The oracle runs at ``check_sizes``, and at
    ``sizes`` too (the probe then compares there) when ``full`` and it should finish in time: a loop scales from the
    first run; a library call's time at small sizes says nothing, and one call costs little, so it is always tried.

    :param full: Whether ``sizes`` may be tried; an oracle known to be out of time there is not tried again."""
    timed = Workload(make_inputs(boundary, sizes), sizes)
    checked = Workload(make_inputs(boundary, check_sizes), check_sizes)
    small = bounded_oracle(prep, boundary, checked.inputs, check_sizes, ORACLE_TIMEOUT_S)
    if isinstance(small, Failure):
        return Probe(timed, checked, {}, f"its oracle did not finish ({small.error}); the whole program checks it")
    outputs, seconds = small
    scale = math.prod(sizes[s] / check_sizes[s] for s in boundary.symbols if sizes.get(s) and check_sizes.get(s))
    interpreted = re.search(r"^\s*(for|while) ", prep.numpy_source, re.MULTILINE)
    if full and (not interpreted or seconds * scale <= ORACLE_TIMEOUT_S):
        at_size = bounded_oracle(prep, boundary, timed.inputs, sizes, ORACLE_TIMEOUT_S)
        if not isinstance(at_size, Failure):
            return Probe(timed, timed, at_size[0])
    return Probe(timed, checked, outputs)


def measure_kernel(archive: Path, src: KernelSource, probe: Probe, reps: int, fp_mode: str) -> KernelVerdict:
    """Call the twin's C entry in an isolated child (forked for a CPU kernel, spawned for a GPU kernel): one run
    compared to the oracle, then ``reps`` timed runs. A crash or timeout comes back as a verdict with ``error`` set."""
    if probe.unchecked:
        return unchecked_verdict(fp_mode, probe.unchecked)
    measure, shared = kernel_form(src.device).measure, archive.with_suffix(".so")
    checked = measure(shared, src, probe.checked.inputs, probe.oracle, probe.checked.sizes, reps)
    if isinstance(checked, Failure):
        return failed_verdict(fp_mode, checked.error)
    growth = reassociation_growth(src.boundary.standalone_sdfg, src.boundary.outputs, probe.checked.sizes)
    verdict = KernelVerdict(
        fp_mode, checked.maxdiff, checked.md_rel, checked.dtype_floor, checked.time_us, growth=growth
    )
    if probe.checked is probe.timed or not verdict.ok:
        return verdict
    timed = measure(shared, src, probe.timed.inputs, None, probe.timed.sizes, reps)
    if isinstance(timed, Failure):
        return failed_verdict(fp_mode, timed.error)
    return dataclasses.replace(verdict, time_us=timed.time_us)


def validate_kernel(
    archive: Path,
    src: KernelSource,
    prep: Prepared,
    sizes: Mapping[str, SymbolValue],
    reps: int = TIMED_REPS,
    fp_mode: str = "strict-ieee",
) -> KernelVerdict:
    """:func:`measure_kernel` on seeded inputs against the kernel's NumPy oracle."""
    return measure_kernel(archive, src, full_probe(prep, src.boundary, sizes), reps, fp_mode)
