# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Random scopes over random programs, placed on random devices and built, compute what the program computed.

Each seed canonicalizes one program of the fusion fuzzer's grammar, groups a random straight run of its top-level
blocks into one kernel (refused when the run is no kernel), outlines the parallel maps left with the default scopes,
places every kernel on a random device, and builds and links the CPF units (stage 5) of a random half of the kernels
beside the rest, which expand from their SDFG, so the seeds link programs that mix both kinds of kernel.
"""

import copy
import os

import numpy as np
import pytest
from dace.libraries.standard.nodes.external_call import external_calls
from test_fusion_fuzz import inputs_for, load_program

from nestforge.ir.names import in_order
from nestforge.records import MoveStatus
from nestforge.session import Session
from nestforge.stages.canonicalize import Targets

NCASES = 8
SIZE = 16


def block_labels(session: Session) -> dict[int, str]:
    """Tree label of every row object, keyed by the object's id."""
    session.describe()
    return {id(obj): label for label, (obj, _) in session.row_index().items()}


def random_block_run(session: Session, rng: np.random.Generator) -> list[str]:
    """Labels of a random straight run of one to three top-level blocks, following single unconditional edges."""
    sdfg, labels = session.sdfg, block_labels(session)
    blocks = [block for block in in_order(sdfg) if id(block) in labels]
    run = [blocks[int(rng.integers(len(blocks)))]]
    for _ in range(int(rng.integers(3))):
        out = sdfg.out_edges(run[-1])
        if len(out) != 1 or not out[0].data.is_unconditional() or sdfg.in_degree(out[0].dst) != 1:
            break
        run.append(out[0].dst)
    return [labels[id(block)] for block in run]


class Case:
    """One seed's session after its random scope, the program as written, its source and the scope's status."""

    __slots__ = ("reference", "rng", "session", "source", "status")

    def __init__(self, seed: int, tmp_path, gpu: bool) -> None:
        program, self.source = load_program(tmp_path, seed + 900)
        sdfg = program.to_sdfg(simplify=True)
        sdfg.name = f"scope_fuzz_{seed}_{int(gpu)}_{os.getpid()}"  # one build folder per xdist worker
        self.reference = copy.deepcopy(sdfg)
        self.reference.name = f"{sdfg.name}_reference"
        self.rng = np.random.default_rng(seed)
        self.session = Session(sdfg, targets=Targets(gpu=gpu), work_dir=str(tmp_path))
        self.session.canonicalize()
        self.status = self.session.set_scope(random_block_run(self.session, self.rng)).status

    def place_and_build(self) -> dict[str, str]:
        """Default scopes, every kernel on a random device when the GPU is a target, and a random half of the kernels
        built and linked (``ExternCall``) beside the rest expanded from their SDFG (``DaceReference``); returns the
        implementation each kernel should have."""
        session = self.session
        session.default_scopes()
        names = [ext.name for ext in external_calls(session.sdfg)]
        devices = {name: ("cpu", "gpu")[int(self.rng.integers(2))] for name in names} if session.targets.gpu else {}
        session.place(devices or None)
        implementations = {name: ("DaceReference", "ExternCall")[int(self.rng.integers(2))] for name in names}
        for name, implementation in implementations.items():
            if implementation == "ExternCall":
                session.optimize_kernel(name)
        return implementations


def assert_same_values(case: Case, seed: int) -> None:
    inputs = inputs_for(SIZE, seed)
    expected = {name: value.copy() for name, value in inputs.items()}
    actual = {name: value.copy() for name, value in inputs.items()}
    case.reference(**expected, N=SIZE)
    case.session.sdfg(**actual, N=SIZE)
    for name in inputs:
        np.testing.assert_allclose(
            actual[name], expected[name], rtol=1e-14, atol=0, err_msg=f"seed={seed} diverged on {name!r}\n{case.source}"
        )


@pytest.mark.parametrize("seed", range(NCASES))
def test_a_random_scope_built_on_the_cpu_computes_what_the_program_computes(seed, tmp_path):
    case = Case(seed, tmp_path, gpu=False)

    implementations = case.place_and_build()

    assert case.status in (MoveStatus.APPLIED, MoveStatus.ILLEGAL)
    expanded = {ext.name: ext.implementation or ext.default_implementation for ext in external_calls(case.session.sdfg)}
    assert expanded == implementations
    assert_same_values(case, seed)


def test_random_scopes_group_blocks_on_most_seeds(tmp_path):
    """Half the seeds at least must apply their scope, else the fuzzer only probes refusals."""
    statuses = [Case(seed, tmp_path, gpu=False).status for seed in range(NCASES)]

    assert statuses.count(MoveStatus.APPLIED) >= NCASES // 2, statuses


@pytest.mark.gpu
@pytest.mark.parametrize("seed", range(NCASES))
def test_a_random_scope_placed_on_random_devices_computes_what_the_program_computes(seed, tmp_path):
    case = Case(seed, tmp_path, gpu=True)

    case.place_and_build()

    assert_same_values(case, seed)
