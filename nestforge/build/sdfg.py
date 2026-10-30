# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compile DaCe-generated code with a chosen compiler and flags and call it through ctypes, instead of
``dace.compile()``, whose ``__call__`` re-marshals arguments on every call."""

from __future__ import annotations

import copy
import ctypes
import functools
import time
import warnings
from _ctypes import dlclose  # release a built .so mapping (BuiltSDFG.unload)
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import dace
import numpy as np
from dace.codegen import codegen
from dace.codegen import compiler as dace_compiler
from dace.data import Array
from dace.libraries.standard.nodes.external_call import external_calls
from dace.sdfg.scope import is_devicelevel_gpu
from dace.transformation.passes.canonicalize.finalize import force_map_scoped_registers

from nestforge.build.toolchain import (
    AR,
    AS_NEEDED,
    CXX_STD,
    DEFAULT_COMPILER,
    DEFAULT_FLAGS,
    LIBOMP,
    Param,
    bind_argument,
    cudart_dir,
    cudart_link_flags,
    hip_link_flags,
    hip_runtime_dir,
    lib_linkable,
    openmp_compile_flags,
    openmp_link_flags,
    run,
    runtime_library,
    support_rpath_flags,
)


@functools.lru_cache(maxsize=None, typed=True)
def dace_runtime_include() -> Path:
    """The ``-I`` directory holding DaCe's runtime headers."""
    inc = Path(dace.__file__).parent / "runtime" / "include"
    if not inc.is_dir():
        raise FileNotFoundError(f"DaCe runtime include not found at {inc}")
    return inc


@dataclass(slots=True)
class BuiltSDFG:
    """A built DaCe program: its ``.so`` and the parameters of its init and program entries. ``lib`` is ``None``
    after :meth:`unload`."""

    name: str
    so_path: Path
    lib: ctypes.CDLL | None
    init_params: list[Param]
    prog_params: list[Param]
    #: wall time of DaCe code generation
    codegen_seconds: float = 0.0
    #: wall time of compiling and linking the .so
    compile_seconds: float = 0.0
    handle: ctypes.c_void_p | None = field(default=None, repr=False)

    def function(self, symbol: str, restype: Any, argtypes: list[Any]) -> Any:
        """The C function ``symbol`` of the mapped library, typed."""
        if self.lib is None:
            raise RuntimeError(f"{self.name}: {symbol} called after unload(); the compiled library is not mapped")
        fn = self.lib[symbol]  # ctypes CDLL indexing (not getattr) binds the entry point
        fn.restype = restype
        fn.argtypes = argtypes
        return fn

    def init(self, sizes: dict[str, int]) -> None:
        fn = self.function(f"__dace_init_{self.name}", ctypes.c_void_p, [p.ctype for p in self.init_params])
        # each parameter's own ctype: symbol widths differ between programs
        self.handle = ctypes.c_void_p(fn(*[p.ctype(int(sizes[p.name])) for p in self.init_params]))

    def program(self, buffers: dict[str, np.ndarray], sizes: dict[str, int]) -> None:
        """Call ``__program_N(handle, args...)`` once, in place (init must have run)."""
        fn = self.function(f"__program_{self.name}", None, [ctypes.c_void_p] + [p.ctype for p in self.prog_params])
        args = [bind_argument(p.name, p.ctype, buffers, sizes) for p in self.prog_params]
        fn(self.handle, *args)

    def unload(self) -> None:
        """Close the library, so a long sweep does not keep one mapping per kernel."""
        if self.lib is not None:
            dlclose(self.lib._handle)
            self.lib = None

    def close(self) -> None:
        """Run ``__dace_exit`` on the open handle, if any; call it before :meth:`unload`."""
        if self.handle is None:
            return
        self.function(f"__dace_exit_{self.name}", ctypes.c_int, [ctypes.c_void_p])(self.handle)
        self.handle = None

    def run(self, buffers: dict[str, np.ndarray], sizes: dict[str, int]) -> None:
        """Init, one program call, exit."""
        self.init(sizes)
        try:
            self.program(buffers, sizes)
        finally:
            self.close()


#: DaCe's CUDA code generator for every GPU build: the new one, which hoists a per-thread array of symbolic size
#: out of its kernel into global memory (``MoveArrayOutOfKernel``) instead of refusing it.
CUDA_CODEGEN = "experimental"


def new_gpu_codegen() -> AbstractContextManager[object]:
    """Pin :data:`CUDA_CODEGEN`, whatever the environment or the config file chose."""
    return dace.config.set_temporary("compiler", "cuda", "implementation", value=CUDA_CODEGEN)


def kernel_sdfgs(sdfg: dace.SDFG) -> Iterator[dace.SDFG]:
    """Each host kernel's standalone SDFG, nested ones included: an ``ExternalCall`` expands it at build time. A GPU
    kernel's arrays are device memory, and a host stack pin would survive its offloading."""
    for node in external_calls(sdfg):
        if node.standalone_sdfg is not None and node.schedule not in dace.dtypes.GPU_SCHEDULES:
            yield from node.standalone_sdfg.all_sdfgs_recursive()


def stack_candidates(sdfg: dace.SDFG, *, scope_default: bool) -> Iterator[Array]:
    """The transient arrays of ``sdfg`` itself in unforced registers, plus, with ``scope_default``, the
    scope-lived ones of default storage (a kernel's, which codegen puts in registers once expanded)."""
    for desc in sdfg.arrays.values():
        if type(desc) is not Array or not desc.transient:
            continue
        storage = desc.storage
        unforced = storage == dace.StorageType.Register and not storage.force
        scoped = (
            scope_default and storage == dace.StorageType.Default and desc.lifetime == dace.AllocationLifetime.Scope
        )
        if unforced or scoped:
            yield desc


def on_device(sdfg: dace.SDFG) -> bool:
    """Whether ``sdfg`` runs inside a GPU kernel, where a register array is per-thread memory."""
    node = sdfg.parent_nsdfg_node
    return node is not None and is_devicelevel_gpu(sdfg.parent.sdfg, sdfg.parent, node)


def force_register_arrays(sdfg: dace.SDFG, kernels: bool = True) -> int:
    """Settle where every array headed for registers lives: unforced register arrays, scope-lived arrays inside a map
    (DaCe's rule), and, with ``kernels``, a host kernel's scope-lived arrays. Map-scoped arrays and arrays of a
    constant size are pinned to ``Register(force=True)``, so they stay on the stack; a host array whose size names a
    symbol is a heap allocation, never a variable-length array of the problem's size (CloudSC's kernels held
    megabytes of them). Returns how many."""
    pinned = force_map_scoped_registers(sdfg)
    for owned, scope_default in [
        *((s, False) for s in sdfg.all_sdfgs_recursive()),
        *((s, True) for s in (kernel_sdfgs(sdfg) if kernels else ())),
    ]:
        arrays = list(stack_candidates(owned, scope_default=scope_default))
        heap = not on_device(owned)
        for desc in arrays:
            if heap and dace.symbolic.issymbolic(desc.total_size, owned.constants):
                desc.storage = dace.StorageType.CPU_Heap
            else:
                desc.storage = dace.StorageType.Register(force=True)
        pinned += len(arrays)
    return pinned


def generate_program_folder(sdfg: dace.SDFG, out_dir: Path) -> Path:
    """Write DaCe's source tree (``src/cpu/<name>.cpp`` and ``include/``) without compiling it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    force_register_arrays(sdfg)
    with dace.config.temporary_config(), new_gpu_codegen():
        dace.config.Config.set("compiler", "emit_tree_reductions", value=True)
        code_objects = codegen.generate_code(sdfg)
    folder = Path(dace_compiler.generate_program_folder(sdfg, code_objects, str(out_dir)))
    frame = folder / "src" / "cpu" / f"{sdfg.name}.cpp"
    if not frame.exists():
        frame = next(folder.glob("src/cpu/*.cpp"))
    return frame


@dataclass(slots=True)
class BuildOptions:
    """Compiler, flags and whether the build links OpenMP (libomp) of a build."""

    compiler: str = DEFAULT_COMPILER
    flags: list[str] | None = None  # None -> DEFAULT_FLAGS
    openmp: bool = True
    link_external: bool = False  # link the nest as a separate static .a (else a monolithic single TU)
    libraries: list[str] = field(default_factory=list)  # linked ahead of the runtimes, e.g. CPF's vendor libraries

    def resolved_flags(self) -> list[str]:
        """``flags`` (or :data:`DEFAULT_FLAGS`), with the C++ standard and ``-Wall`` guaranteed."""
        # DaCe's runtime headers need C++20; add -std= only when the caller's flags set none
        flags = list(self.flags if self.flags is not None else DEFAULT_FLAGS)
        if not any(f.startswith("-std=") for f in flags):
            flags.append(f"-std={CXX_STD}")
        if "-Wall" not in flags and "-w" not in flags:
            flags.append("-Wall")
        return flags


@dataclass(slots=True)
class BuildCommands:
    compiler: str
    cflags: list[str]
    compile_extra: list[str]
    link_libs: list[str]  # after the object: the linker resolves left to right


def build_commands(folder: Path | None, opts: BuildOptions) -> BuildCommands:
    compiler = opts.compiler
    # without OpenMP every multicore map's pragma is ignored
    omp = opts.openmp and lib_linkable(LIBOMP, compiler)
    if opts.openmp and not omp:
        warnings.warn(
            f"{Path(compiler).name} cannot link libomp; building serial, so every parallel map runs on one thread"
        )
    omp_c = openmp_compile_flags(compiler) if omp else []
    omp_l = openmp_link_flags(compiler) if omp else []
    includes = [f"-I{folder / 'include'}", f"-I{dace_runtime_include()}"] if folder is not None else []
    return BuildCommands(
        compiler=compiler,
        cflags=[f for f in opts.resolved_flags() if f != "-shared"],
        compile_extra=[*omp_c, *includes],
        # icx links libsvml/libimf from off the loader path without a RUNPATH
        link_libs=[*opts.libraries, *omp_l, *support_rpath_flags(compiler)],
    )


def build_archive(
    sources: Sequence[Path], folder: Path | None, archive: Path, shared: Path, opts: BuildOptions
) -> float:
    """Compile ``sources`` (against ``folder``'s headers, if given), archive them, and link ``shared`` from the
    whole archive."""
    cmds = build_commands(folder, opts)
    objs = [archive.parent / f"{src.stem}.o" for src in sources]
    archive.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for src, obj in zip(sources, objs):
        run([cmds.compiler, *cmds.cflags, "-c", *cmds.compile_extra, str(src), "-o", str(obj)])
    archive_and_link(objs, archive, shared, opts.compiler, cmds.link_libs)
    return time.perf_counter() - t0


def archive_and_link(
    objects: Sequence[Path], archive: Path, shared: Path, linker: str, link_libs: Sequence[str]
) -> None:
    """Archive ``objects`` into ``archive`` and link ``shared`` from the whole archive."""
    if archive.exists():
        archive.unlink()  # ar appends; a rebuild must not keep stale members
    run([AR, "rcs", str(archive), *[str(obj) for obj in objects]])
    whole = ["-Wl,--export-dynamic", "-Wl,--whole-archive", str(archive), "-Wl,--no-whole-archive"]
    link_shared(linker, whole, link_libs, shared)


def link_shared(linker: str, inputs: Sequence[str], link_libs: Sequence[str], shared: Path) -> None:
    """The one shared-library link: :data:`AS_NEEDED` ahead of ``inputs``, then the libraries."""
    run([linker, "-shared", AS_NEEDED, *inputs, *link_libs, "-o", str(shared)])


def build_device_archive(
    sources: Sequence[Path],
    archive: Path,
    shared: Path,
    compiler: str,
    flags: Sequence[str],
    libraries: Sequence[str],
    runtime: Sequence[str],
) -> float:
    """Compile each device unit with ``compiler`` (nvcc or hipcc), archive the objects, and link ``shared`` with the
    host compiler against ``libraries`` and the device ``runtime`` the compiler itself links."""
    archive.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    objs = [archive.parent / f"{source.stem}.o" for source in sources]
    for source, obj in zip(sources, objs):
        run([compiler, *[f for f in flags if f != "-shared"], "-c", str(source), "-o", str(obj)])
    archive_and_link(objs, archive, shared, DEFAULT_COMPILER, [*libraries, *runtime])
    return time.perf_counter() - t0


def build_cuda_archive(
    sources: Sequence[Path], archive: Path, shared: Path, nvcc: str, flags: Sequence[str], libraries: Sequence[str]
) -> float:
    """:func:`build_device_archive` with ``nvcc`` and the ``libcudart`` it links."""
    runtime = cudart_link_flags(cudart_dir(nvcc))
    return build_device_archive(sources, archive, shared, nvcc, flags, libraries, runtime)


def build_hip_archive(
    sources: Sequence[Path], archive: Path, shared: Path, hipcc: str, flags: Sequence[str], libraries: Sequence[str]
) -> float:
    """:func:`build_device_archive` with ``hipcc`` and the ``libamdhip64`` of its ROCm."""
    runtime = hip_link_flags(hip_runtime_dir(hipcc))
    return build_device_archive(sources, archive, shared, hipcc, flags, libraries, runtime)


def program_compiler() -> str:
    """The C++ compiler DaCe's program build runs: ``compiler.cpu.executable``, else CMake's ``c++``."""
    return str(dace.config.Config.get("compiler", "cpu", "executable") or "c++")


def libomp_cmake_args(compiler: str) -> list[str]:
    """CMake cache values under which DaCe's ``find_package(OpenMP)`` resolves LLVM libomp for ``compiler``."""
    library = runtime_library(LIBOMP, compiler)
    if library is None:
        raise LookupError(
            f"no lib{LIBOMP} for {compiler}: neither it, an LLVM driver, llvm-config nor the library search path "
            "names one, and the process's one OpenMP runtime is libomp; install libomp (scripts/setup_apt.sh) or put "
            "its directory on LIBRARY_PATH and LD_LIBRARY_PATH"
        )
    return [f"-DOpenMP_CXX_LIB_NAMES={LIBOMP}", f"-DOpenMP_{LIBOMP}_LIBRARY={library}"]


def compile_linked_program(sdfg: dace.SDFG, build_folder: Path) -> Any:
    """Compile a program that links kernel libraries, with libomp as its one OpenMP runtime; the CMake
    setting holds for this compile only."""
    extra = [dace.config.Config.get("compiler", "extra_cmake_args"), *libomp_cmake_args(program_compiler())]
    # DaCe folds compiler.linker.args into the one CMAKE_SHARED_LINKER_FLAGS it passes, after extra_cmake_args
    linker_args = [dace.config.Config.get("compiler", "linker", "args"), AS_NEEDED]
    sdfg.build_folder = str(build_folder)
    force_register_arrays(sdfg)
    with (
        new_gpu_codegen(),
        dace.config.set_temporary("compiler", "extra_cmake_args", value=" ".join(arg for arg in extra if arg)),
    ):
        with dace.config.set_temporary("compiler", "linker", "args", value=" ".join(arg for arg in linker_args if arg)):
            return sdfg.compile()


@dataclass(slots=True)
class GeneratedProgram:
    """Generated program source, not yet compiled."""

    frame: Path
    name: str
    source: str
    codegen_seconds: float

    @property
    def folder(self) -> Path:
        return self.frame.parent.parent.parent  # <out>/src/cpu/x.cpp -> <out>


def generate_program(sdfg: dace.SDFG, out_dir: Path) -> GeneratedProgram:
    """Emit the program folder of a copy of ``sdfg``, without compiling it."""
    t_opt = time.perf_counter()
    sdfg = copy.deepcopy(sdfg)
    frame = generate_program_folder(sdfg, out_dir)
    return GeneratedProgram(
        frame=frame, name=sdfg.name, source=frame.read_text(), codegen_seconds=time.perf_counter() - t_opt
    )
