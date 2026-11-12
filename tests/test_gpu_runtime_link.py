# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Where a GPU kernel finds its device runtime, as strings and with a stand-in nvcc, so without a GPU toolkit: the
``-L`` directory holding the unversioned ``lib<runtime>.so``, found again at load time through a DT_RUNPATH."""

from pathlib import Path

import pytest
from helpers import dace_gpu_backend
from test_build_review_regressions import fake_tool
from test_kernel_stage import gpu_lowered_kernel, vadd

from nestforge.build import toolchain
from nestforge.stages import kernel as kernel_stage

RUNTIME_DIR = "/opt/device-runtime/lib64"
#: Per backend: the kernel compiler and the device runtime's link flag.
RUNTIMES = {"cuda": ("nvcc", "-lcudart"), "hip": ("hipcc", "-lamdhip64")}
#: A stand-in nvcc: it refuses ``-Wl,`` as nvcc does, else prints the link line ``ld`` would run.
NVCC = """for arg in "$@"; do case "$arg" in -Wl,*) echo "nvcc fatal   : Unknown option '$arg'" >&2; exit 1;; esac; done
echo "ld -L{versioned} -L{runtime} -lcudart" >&2"""


def test_cudart_is_found_past_a_versioned_soname_by_a_probe_spelled_for_nvcc(tmp_path: Path) -> None:
    versioned = tmp_path / "usr" / "lib"  # first on the link line, but -lcudart cannot resolve libcudart.so.12
    runtime = tmp_path / "cuda" / "lib64"
    for directory, name in ((versioned, "libcudart.so.12"), (runtime, "libcudart.so")):
        directory.mkdir(parents=True)
        (directory / name).touch()
    nvcc = fake_tool(tmp_path / "bin", "nvcc", NVCC.format(versioned=versioned, runtime=runtime))

    assert toolchain.cudart_dir(nvcc) == str(runtime.resolve())


@pytest.mark.parametrize("api", sorted(RUNTIMES))
def test_a_gpu_kernel_links_its_device_runtime_from_one_directory_it_finds_again_as_a_runpath(
    tmp_path, monkeypatch, api
):
    monkeypatch.setattr(kernel_stage, "cudart_dir", lambda nvcc: RUNTIME_DIR)
    monkeypatch.setattr(kernel_stage, "hip_runtime_dir", lambda hipcc: RUNTIME_DIR)
    compiler, runtime = RUNTIMES[api]
    with dace_gpu_backend(api):
        _, ext, boundary = gpu_lowered_kernel(vadd)
        src = kernel_stage.schedule_kernel(ext, boundary, tmp_path)
        libraries = kernel_stage.kernel_runtime_libraries(src, compiler)

    at = libraries.index(runtime)
    assert libraries[at - 1 : at + 3] == [f"-L{RUNTIME_DIR}", runtime, f"-Wl,-rpath,{RUNTIME_DIR}", toolchain.NEW_DTAGS]
    other = next(flag for other_api, (_, flag) in RUNTIMES.items() if other_api != api)
    assert other not in libraries
    rpaths = [i for i, flag in enumerate(libraries) if flag.startswith("-Wl,-rpath,")]
    assert all(libraries[i + 1] == toolchain.NEW_DTAGS for i in rpaths), "every rpath is a DT_RUNPATH"


if __name__ == "__main__":
    import tempfile

    test_cudart_is_found_past_a_versioned_soname_by_a_probe_spelled_for_nvcc(Path(tempfile.mkdtemp()))
    for backend in sorted(RUNTIMES):
        with pytest.MonkeyPatch.context() as patch:
            test_a_gpu_kernel_links_its_device_runtime_from_one_directory_it_finds_again_as_a_runpath(
                Path(tempfile.mkdtemp()), patch, backend
            )
