# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage 5, kernel optimization: render one kernel as a standalone CPF unit with one C entry, then build and validate
``lib<kernel>.a``."""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import dace
import numpy as np
from dace import dtypes
from dace.codegen import cpf
from dace.codegen.common import get_gpu_backend
from dace.libraries.standard.nodes.external_call import ExternalCall
from dace.transformation.passes.canonicalize.finalize import finalize_for_target, offload_to_gpu

from nestforge.build.arena import (
    TIMED_REPS,
    DeviceCall,
    accumulating_outputs,
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

    @property
    def ok(self) -> bool:
        return not self.error and not self.unchecked and self.md_rel <= rung_rtol(self.fp_mode, self.dtype_floor)


def failed_verdict(fp_mode: str, error: str) -> KernelVerdict:
    return KernelVerdict(fp_mode, float("inf"), float("inf"), 0.0, float("inf"), error)


def unchecked_verdict(fp_mode: str, symbols: list[str]) -> KernelVerdict:
    reason = f"takes {', '.join(symbols)}, which only the running program computes; the whole program checks it"
    return KernelVerdict(fp_mode, float("inf"), float("inf"), 0.0, float("inf"), unchecked=reason)


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


def kernel_numbers(oracle: dict[str, np.ndarray], outputs: dict[str, np.ndarray], time_us: float) -> KernelNumbers:
    md, md_rel = diff_stats(oracle, outputs)
    return KernelNumbers(maxdiff=md, md_rel=md_rel, dtype_floor=dtype_floor(outputs), time_us=time_us)


def measure_on_cpu(
    shared: Path,
    src: KernelSource,
    inputs: dict[str, np.ndarray],
    oracle: dict[str, np.ndarray],
    sizes: dict[str, int],
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
    oracle: dict[str, np.ndarray],
    sizes: dict[str, int],
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
        [Path, KernelSource, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int], int], KernelNumbers | Failure
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


def schedule_kernel(ext: ExternalCall, boundary: Boundary, out_dir: Path) -> KernelSource:
    """The kernel for the device stage 4 placed ``ext`` on, rendered by CPF into ``<out_dir>/<kernel>.cpp``,
    ``.cu`` or ``.hip``, whose one entry is ``ext``'s symbol."""
    device = kernel_device(ext)
    form = kernel_form(device)
    sdfg = form.schedule(boundary)
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


def measure_kernel(
    archive: Path,
    src: KernelSource,
    inputs: dict[str, np.ndarray],
    oracle: dict[str, np.ndarray],
    sizes: dict[str, int],
    reps: int,
    fp_mode: str,
) -> KernelVerdict:
    """Call the twin's C entry in an isolated child (forked for a CPU kernel, spawned for a GPU kernel): one run
    compared to ``oracle``, then ``reps`` timed runs. A crash or timeout comes back as a verdict with ``error`` set."""
    res = kernel_form(src.device).measure(archive.with_suffix(".so"), src, inputs, oracle, sizes, reps)
    if isinstance(res, Failure):
        return failed_verdict(fp_mode, res.error)
    return KernelVerdict(fp_mode, res.maxdiff, res.md_rel, res.dtype_floor, res.time_us)


def validate_kernel(
    archive: Path,
    src: KernelSource,
    prep: Prepared,
    sizes: dict[str, int],
    reps: int = TIMED_REPS,
    fp_mode: str = "strict-ieee",
) -> KernelVerdict:
    """:func:`measure_kernel` on seeded inputs against the kernel's NumPy oracle."""
    inputs = make_inputs(src.boundary, sizes)
    oracle = run_oracle(prep, src.boundary, inputs, sizes)
    return measure_kernel(archive, src, inputs, oracle, sizes, reps, fp_mode)
