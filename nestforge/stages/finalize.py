# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Assembly: the program in the bench C ABI, its kernels linked from their own archives, built into the ``.so`` a
library submission (``source_mode any``) hands the judge. Each kernel keeps the compiler and flags it was built with."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import dace
from dace import dtypes
from dace.codegen.common import get_gpu_backend
from dace.libraries.standard.helper import GPU_RESIDENT_STORAGES
from dace.libraries.standard.nodes.external_call import ExternalCall
from dace.sdfg import nodes

from nestforge.records import Record

__all__ = ["BenchTarget", "Final", "assemble"]


@dataclass(frozen=True, slots=True)
class BenchTarget:
    """The bench task a program is assembled for: its kernel, the graded datatype and the device."""

    kernel: str
    datatype: str
    gpu: bool


class Final(Record):
    """An assembled program: the library to submit with the scratch bytes it needs (``"0"``: none), the entry source
    (and device unit) it was built from and what it links."""

    library: Path
    workspace_bytes: str
    source: str
    #: The HIP device unit on a GPU target with device code; empty otherwise.
    device_source: str = ""
    link: tuple[str, ...]
    kernels: dict[str, str]
    #: The submission's ``requests``: the argument layouts the program works on (``{}``: none).
    requests: dict[str, object] = {}


def host_touched(sdfg: dace.SDFG, name: str) -> bool:
    """Whether code that runs on the host reads or writes argument ``name``: a transition, or a node other than a copy
    to or from device memory (a CPU kernel, a host statement)."""
    if any(name in edge.data.free_symbols for edge in sdfg.all_interstate_edges()):
        return True
    for state in sdfg.states():
        for node in state.data_nodes():
            if node.data != name:
                continue
            for edge in state.all_edges(node):
                other = edge.dst if edge.src is node else edge.src
                if not (
                    isinstance(other, nodes.AccessNode) and other.desc(state.sdfg).storage in GPU_RESIDENT_STORAGES
                ):
                    return True
    return False


def device_arguments(sdfg: dace.SDFG) -> None:
    """The judge grades a GPU task with every argument in device memory (#98): each array argument of ``sdfg`` becomes
    device memory, and one the host touches is read through a host copy, copied in first and back last when written."""
    names = [
        name
        for name, desc in sdfg.arrays.items()
        if not desc.transient and isinstance(desc, dace.data.Array) and desc.storage not in GPU_RESIDENT_STORAGES
    ]
    host = [name for name in names if host_touched(sdfg, name)]
    for name in names:
        if name not in host:
            sdfg.arrays[name].storage = dtypes.StorageType.GPU_Global
    if not host:
        return
    order = list(sdfg.arg_names)
    # abi_order names the nest's own data, which a rename of the outer data must not touch
    calls = [node for node, _ in sdfg.all_nodes_recursive() if isinstance(node, ExternalCall)]
    abi = [list(call.abi_order) for call in calls]
    first = sdfg.add_state_before(sdfg.start_block, "device_arguments_in", is_start_block=True)
    lasts = [sdfg.add_state_after(sink, f"device_arguments_out_{i}") for i, sink in enumerate(sdfg.sink_nodes())]
    for name in host:
        desc = sdfg.arrays[name]
        twin = sdfg.find_new_symbol(f"{name}_host")
        sdfg.replace(name, twin)
        sdfg.arrays[twin].transient = True
        written = any(
            state.in_degree(node) > 0 for state in sdfg.states() for node in state.data_nodes() if node.data == twin
        )
        device = desc.clone()
        device.transient, device.storage = False, dtypes.StorageType.GPU_Global
        sdfg.add_datadesc(name, device)
        first.add_nedge(first.add_read(name), first.add_write(twin), dace.Memlet.from_array(name, device))
        for last in lasts if written else ():
            last.add_nedge(last.add_read(twin), last.add_write(name), dace.Memlet.from_array(twin, sdfg.arrays[twin]))
    sdfg.arg_names = order
    for call, names in zip(calls, abi, strict=True):
        call.abi_order = names


def assemble(
    sdfg: dace.SDFG,
    bench: BenchTarget,
    kernels: dict[str, str],
    out_dir: Path,
    requests: dict[str, object] | None = None,
) -> Final:
    """Render ``sdfg`` (every kernel already bound to its archive) by CPF in the bench C ABI and build it into
    ``out_dir``. On a GPU target with device code the form is a C++ host unit and a device unit in the GPU backend's
    language (CUDA or HIP), built into one library as the judge builds a two-unit GPU submission; without device code it is the CPU form. ``kernels`` maps each kernel to the variant that built it,
    for the answer only; ``requests`` lays out the arguments the program works on laid out."""
    from hpcagent_bench import languages
    from hpcagent_bench.categories import Language
    from hpcagent_bench.cpf.bridge import render_canonical
    from hpcagent_bench.cpf.cache import DEFAULT_HINTS, Target
    from hpcagent_bench.spec import BenchSpec

    spec = BenchSpec.load(bench.kernel)
    if bench.gpu:
        device_arguments(sdfg)
    # a GPU form is asked for in DaCe's backend, the kernels' one; the bench would pick its own
    dialect = Language[get_gpu_backend().upper()] if bench.gpu else Language.CPP
    form = render_canonical(
        spec,
        spec.short_name,
        sdfg,
        dialect.text,
        bench.datatype,
        Target.GPU if bench.gpu else Target.CPU,
        DEFAULT_HINTS,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    host = out_dir / f"{spec.module_name}.{languages.LANG_EXT[Language.CPP]}"
    host.write_text(form.code)
    lib = out_dir / f"lib{spec.module_name}.so"
    if bench.gpu:
        # a program with every kernel on the CPU has no device unit, but copies its arguments with the GPU runtime
        device = out_dir / f"{spec.module_name}_device.{languages.LANG_EXT[dialect]}"
        device.write_text(form.device_code)
        # nvcc rejects -Wl, and takes the same comma list as -Xlinker=; hipcc is clang and takes -Wl,
        nvcc = dialect is Language.CUDA
        link = [f"-Xlinker={f.removeprefix('-Wl,')}" if nvcc and f.startswith("-Wl,") else f for f in form.link]
        commands = languages.build_shared_lib_commands(dialect, device, lib, extra_link=link, extra_sources=[host])
    else:
        commands = languages.build_shared_lib_commands(Language.CPP, host, lib, extra_link=list(form.link))
    failed, log = languages.run_build_commands(commands, out_dir)
    if failed or not lib.exists():
        raise RuntimeError(f"the assembled program did not build:\n{log}")
    return Final(
        library=lib,
        workspace_bytes=form.workspace_bytes,
        source=form.code,
        device_source=form.device_code,
        link=tuple(form.link),
        kernels=kernels,
        requests=requests or {},
    )
