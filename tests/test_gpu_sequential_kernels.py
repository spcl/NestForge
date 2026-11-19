# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A sequential kernel the agent runs on the GPU stays a valid program and renders its CUDA source (#93), without a
GPU: a scalar the kernel reads on the host and the device writes, and a library fill placement already offloaded."""

import pytest
from helpers import dace_gpu_backend

from nestforge.corpus.bench import corpus_kernel, preset_sizes
from nestforge.records import MoveStatus
from nestforge.session import Session
from nestforge.stages.canonicalize import Targets

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("name", ["poisson_cg_3d", "cloudsc_cover_carry"])
def test_a_sequential_kernel_on_the_gpu_validates_and_renders(tmp_path, name):
    kernel = corpus_kernel(name)
    assert kernel is not None, f"{name} is in the corpus"
    session = Session(kernel.to_sdfg(), Targets(gpu=True), sizes=preset_sizes(kernel, "S"), work_dir=str(tmp_path))
    session.canonicalize()
    session.run_defaults("moves")
    session.run_defaults("scopes")
    sequential = [k.name for k in session.list_kernels() if not k.parallel]
    assert sequential, "the fixture must have a sequential kernel, else it tests nothing"

    assert session.apply_move("run-on-gpu", sequential).status is MoveStatus.APPLIED
    session.finalize_scope()

    with dace_gpu_backend("cuda"):
        for placed in sequential:
            body = session.kernel(placed).standalone_sdfg
            assert body is not None
            body.validate()
            assert session.default_kernel(placed).device_unit is not None
