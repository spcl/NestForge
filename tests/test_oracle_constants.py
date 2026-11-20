# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A kernel reading a constant of the program (cloudsc's ``zvqx``, read in its loop over the species) is checked
against an oracle that knows the constant's value: the kernel's C source fixes it, so it is no argument of either."""

from pathlib import Path

import dace
import numpy as np

from nestforge.records import KernelStatus
from nestforge.session import Session
from nestforge.stages.canonicalize import Targets

THRESHOLDS = np.array([0.0, 0.13, 4.0, 1.0, 0.0])
N = dace.symbol("N", dtype=dace.int64)


def falling_species() -> dace.SDFG:
    """``out[i] = THRESHOLDS[i] > 0`` over a doall, ``THRESHOLDS`` a transient array that is a constant of the
    program, as cloudsc declares ``zvqx``."""
    sdfg = dace.SDFG("falling_species")
    sdfg.add_array("out", [N], dace.int32)
    sdfg.add_array("thresholds", [5], dace.float64, transient=True)
    sdfg.add_constant("thresholds", THRESHOLDS, sdfg.arrays["thresholds"])
    state = sdfg.add_state()
    state.add_mapped_tasklet(
        "falling",
        {"i": "0:N"},
        {"t": dace.Memlet("thresholds[i]")},
        "o = 1 if t > 0.0 else 0",
        {"o": dace.Memlet("out[i]")},
        external_edges=True,
    )
    return sdfg


def test_a_kernel_reading_a_program_constant_matches_its_oracle(tmp_path: Path) -> None:
    sut = Session(falling_species(), Targets(gpu=False), work_dir=str(tmp_path), sizes={"N": 5})
    sut.canonicalize()
    sut.default_scopes()
    (kernel,) = sut.list_kernels()

    done = sut.set_kernel_source(kernel.name, sut.kernel_source(kernel.name), "cpp")

    assert done.status is KernelStatus.OK, done
