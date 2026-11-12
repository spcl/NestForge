# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The commands ``finalize`` builds a GPU program with, read without a GPU or a GPU toolkit: nvcc takes every kernel
link flag as ``-Xlinker=``, hipcc keeps ``-Wl,``, and the device unit is in the dialect it is named for."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from helpers import dace_gpu_backend
from hpcagent_bench import languages

from nestforge.corpus.bench import corpus_kernel, preset_sizes
from nestforge.session import Session, external_calls
from nestforge.stages import kernel as kernel_stage
from nestforge.stages.canonicalize import Targets
from nestforge.stages.finalize import BenchTarget, Final, assemble

KERNEL = "jacobi_1d"
#: Where the device runtime is said to live; nothing is linked, so no such directory is needed.
RUNTIME_DIR = "/opt/device-runtime/lib64"
#: Per backend: the kernel compiler, the device unit's suffix and the runtime header it includes.
BACKENDS = {"cuda": ("nvcc", ".cu", "<cuda_runtime.h>"), "hip": ("hipcc", ".hip", "<hip/hip_runtime.h>")}


@contextmanager
def gpu_backend(api: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """DaCe and the bench on backend ``api``, the device arch given, so nothing probes a GPU or a toolkit."""
    monkeypatch.setenv("DACE_compiler_cuda_backend", api)  # the bench reads the variable, DaCe its config
    monkeypatch.setenv("HPCAGENT_BENCH_SM", "sm_80")
    monkeypatch.setenv("HPCAGENT_BENCH_GFX", "gfx90a")
    monkeypatch.setattr(kernel_stage, "cudart_dir", lambda nvcc: RUNTIME_DIR)
    monkeypatch.setattr(kernel_stage, "hip_runtime_dir", lambda hipcc: RUNTIME_DIR)
    languages.gpu_backend.cache_clear()  # the bench decides its backend once per process
    try:
        with dace_gpu_backend(api):
            yield
    finally:
        languages.gpu_backend.cache_clear()


def assembled(api: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Final, list[list[str]], list[str]]:
    """``jacobi_1d`` placed on the GPU, each kernel bound to an archive with the runtime libraries its compiler would
    link, then assembled with the build commands recorded instead of run: the result, the commands, the kernels."""
    commands: list[list[str]] = []

    def record(cmds: list[list[str]], out_dir: Path) -> tuple[bool, str]:
        commands.extend(cmds)
        (out_dir / f"lib{KERNEL}.so").touch()
        return False, ""

    with gpu_backend(api, monkeypatch):
        kernel = corpus_kernel(KERNEL)
        assert kernel is not None, f"{KERNEL} is in the corpus"
        session = Session(kernel.to_sdfg(), Targets(gpu=True), sizes=preset_sizes(kernel, "S"), work_dir=str(tmp_path))
        session.canonicalize()
        session.default_scopes()
        devices = session.place().devices
        assert devices and set(devices.values()) == {"gpu"}, devices
        compiler = BACKENDS[api][0]
        for ext in external_calls(session.sdfg):
            src = kernel_stage.schedule_kernel(ext, session.default_kernel(ext.name).boundary, tmp_path / "gen")
            archive = tmp_path / "lib" / f"lib{ext.name}.a"
            kernel_stage.use_kernel_library(ext, archive, src, kernel_stage.kernel_runtime_libraries(src, compiler))
        monkeypatch.setattr(languages, "run_build_commands", record)
        final = assemble(session.sdfg, BenchTarget(kernel.short_name, "float64", True), {}, tmp_path / "final")
    return final, commands, sorted(devices)


def library_link(final: Final, commands: list[list[str]]) -> list[str]:
    """The one command that writes the library."""
    (link,) = [cmd for cmd in commands if str(final.library) in cmd and "-c" not in cmd]
    return link


def test_a_cuda_finalize_spells_every_kernel_link_flag_for_nvcc(tmp_path, monkeypatch):
    """nvcc rejects ``-Wl,``: every CUDA finalize failed to link while the kernels' flags reached it so spelled."""
    final, commands, _ = assembled("cuda", tmp_path, monkeypatch)

    link = library_link(final, commands)
    assert Path(link[0]).name == "nvcc"
    assert [arg for cmd in commands for arg in cmd if arg.startswith("-Wl,")] == []
    assert f"-Xlinker=-rpath,{RUNTIME_DIR}" in link and "-Xlinker=--enable-new-dtags" in link
    assert link.index(f"-L{RUNTIME_DIR}") < link.index(f"-Xlinker=-rpath,{RUNTIME_DIR}")


def test_a_hip_finalize_hands_hipcc_the_kernel_link_flags_as_wl(tmp_path, monkeypatch):
    """hipcc is clang: it takes ``-Wl,`` and has no ``-Xlinker=`` form."""
    final, commands, _ = assembled("hip", tmp_path, monkeypatch)

    link = library_link(final, commands)
    assert Path(link[0]).name == "hipcc"
    assert f"-Wl,-rpath,{RUNTIME_DIR}" in link and "-Wl,--enable-new-dtags" in link
    assert [arg for arg in link if arg.startswith("-Xlinker")] == []


@pytest.mark.parametrize("api", sorted(BACKENDS))
def test_a_gpu_finalize_is_a_host_unit_and_a_device_unit_in_the_backends_dialect_linking_every_kernel(
    tmp_path, monkeypatch, api
):
    final, commands, kernels = assembled(api, tmp_path, monkeypatch)

    _, suffix, header = BACKENDS[api]
    sources = sorted(Path(cmd[cmd.index("-c") + 1]).name for cmd in commands if "-c" in cmd)
    assert sources == sorted([f"{KERNEL}.cpp", f"{KERNEL}_device{suffix}"])
    assert f"#include {header}" in final.device_source and f"#include {header}" in final.source
    assert "__global__" not in final.source, "the host unit only launches"
    assert all(f"{name}(" in final.source for name in kernels), "the host unit calls every kernel"
    link = library_link(final, commands)
    assert sorted(Path(arg).name for arg in link if arg.endswith(".a")) == [f"lib{name}.a" for name in kernels]


if __name__ == "__main__":
    import tempfile

    with pytest.MonkeyPatch.context() as patch:
        test_a_cuda_finalize_spells_every_kernel_link_flag_for_nvcc(Path(tempfile.mkdtemp()), patch)
    with pytest.MonkeyPatch.context() as patch:
        test_a_hip_finalize_hands_hipcc_the_kernel_link_flags_as_wl(Path(tempfile.mkdtemp()), patch)
    for backend in sorted(BACKENDS):
        with pytest.MonkeyPatch.context() as patch:
            test_a_gpu_finalize_is_a_host_unit_and_a_device_unit_in_the_backends_dialect_linking_every_kernel(
                Path(tempfile.mkdtemp()), patch, backend
            )
