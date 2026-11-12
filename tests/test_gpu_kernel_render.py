# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The kernel stage's GPU source on each DaCe GPU backend, chosen and not probed, rendered without a GPU or a GPU
toolkit: a device unit with the ``__global__`` kernel and a host unit that launches it; a CPU kernel has neither."""

import re
from pathlib import Path

import pytest
from helpers import dace_gpu_backend
from test_kernel_stage import gpu_lowered_kernel, lowered_kernel, vadd

from nestforge.stages.kernel import kernel_text, schedule_kernel

#: Per backend: the unit suffix, the runtime header and the launch call of its dialect.
DIALECTS = {
    "cuda": (".cu", "<cuda_runtime.h>", "cudaLaunchKernel("),
    "hip": (".hip", "<hip/hip_runtime.h>", "hipLaunchKernel("),
}


@pytest.mark.parametrize("api", sorted(DIALECTS))
def test_a_gpu_kernel_renders_a_global_kernel_and_its_launch_in_the_backends_dialect(tmp_path, api):
    suffix, header, launch = DIALECTS[api]
    with dace_gpu_backend(api):
        _, ext, boundary = gpu_lowered_kernel(vadd)
        src = schedule_kernel(ext, boundary, tmp_path)

    assert src.device == "gpu" and src.unit.suffix == suffix and src.device_unit is not None
    host, device = src.unit.read_text(), src.device_unit.read_text()
    assert src.device_unit.suffix == suffix and f"#include {header}" in device
    assert re.search(r"__global__\s+void", device) and launch in device
    assert "__global__" not in host and launch not in host
    launchers = set(re.findall(r"\b(__cpf_runkernel_\w+)\(", host))
    assert launchers and all(re.search(rf"\bvoid {name}\(", device) for name in launchers), "the host calls a launcher"
    other_launch = next(call for other, (_, _, call) in DIALECTS.items() if other != api)
    assert other_launch not in device


def test_a_cpu_kernel_renders_no_device_code(tmp_path):
    _, ext, boundary = lowered_kernel(vadd)

    src = schedule_kernel(ext, boundary, tmp_path)

    text = kernel_text(src)
    assert src.device == "cpu" and src.device_unit is None and src.unit.suffix == ".cpp"
    assert "__global__" not in text and "LaunchKernel(" not in text and "__cpf_runkernel_" not in text


if __name__ == "__main__":
    import tempfile

    for backend in sorted(DIALECTS):
        test_a_gpu_kernel_renders_a_global_kernel_and_its_launch_in_the_backends_dialect(
            Path(tempfile.mkdtemp()), backend
        )
    test_a_cpu_kernel_renders_no_device_code(Path(tempfile.mkdtemp()))
