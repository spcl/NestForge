# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Assembly: the program in the bench C ABI, its kernels linked from their own archives, built into the ``.so`` a
library submission (``source_mode any``) hands the judge. Each kernel keeps the compiler and flags it was built with."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import dace
from dace.codegen.common import get_gpu_backend

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


def assemble(sdfg: dace.SDFG, bench: BenchTarget, kernels: dict[str, str], out_dir: Path) -> Final:
    """Render ``sdfg`` (every kernel already bound to its archive) by CPF in the bench C ABI and build it into
    ``out_dir``. On a GPU target with device code the form is a C++ host unit and a device unit in the GPU backend's
    language (CUDA or HIP), built into one library as the judge builds a two-unit GPU submission; without device code it is the CPU form. ``kernels`` maps each kernel to the variant that built it,
    for the answer only."""
    from hpcagent_bench import languages
    from hpcagent_bench.categories import Language
    from hpcagent_bench.cpf.bridge import render_canonical
    from hpcagent_bench.cpf.cache import DEFAULT_HINTS, Target
    from hpcagent_bench.spec import BenchSpec

    spec = BenchSpec.load(bench.kernel)
    form = render_canonical(
        spec, spec.short_name, sdfg, "c++", bench.datatype, Target.GPU if bench.gpu else Target.CPU, DEFAULT_HINTS
    )
    if bench.gpu and not form.device_code:
        # nothing was placed on the device: the host-only program is plain C++
        form = render_canonical(spec, spec.short_name, sdfg, "c++", bench.datatype, Target.CPU, DEFAULT_HINTS)
    out_dir.mkdir(parents=True, exist_ok=True)
    host = out_dir / f"{spec.module_name}.{languages.LANG_EXT[Language.CPP]}"
    host.write_text(form.code)
    lib = out_dir / f"lib{spec.module_name}.so"
    if form.device_code:
        dialect = Language[get_gpu_backend().upper()]
        device = out_dir / f"{spec.module_name}_device.{languages.LANG_EXT[dialect]}"
        device.write_text(form.device_code)
        commands = languages.build_shared_lib_commands(
            dialect, device, lib, extra_link=list(form.link), extra_sources=[host]
        )
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
    )
