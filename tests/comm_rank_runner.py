# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""One MPI rank of ``tests/test_comm_ranks.py``: load a program the parent built, run it on this rank's inputs, and
save its outputs. ``python comm_rank_runner.py <build folder> <fixture> <output prefix>`` under ``mpirun``."""

import sys
from typing import Any

import numpy as np
from comm_rank_programs import FIXTURES
from dace.sdfg.utils import load_precompiled_sdfg
from mpi4py import MPI


def main(build: str, name: str, prefix: str) -> None:
    fixture = FIXTURES[name]
    comm = MPI.COMM_WORLD
    arguments = fixture.inputs(comm.rank, comm.size)
    load_precompiled_sdfg(build)(**arguments)
    outputs: dict[str, Any] = {key: np.asarray(arguments[key]) for key in fixture.outputs}
    np.savez(f"{prefix}.{comm.rank}.npz", **outputs)


if __name__ == "__main__":
    main(*sys.argv[1:4])
