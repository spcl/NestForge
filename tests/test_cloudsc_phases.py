# Copyright 2021 ETH Zurich and the Tactics authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Every phase an agent drives, end to end on a big program: the bench's cloudsc_monolith, canonicalized once.

Structured (moves, scopes, placement) keeps the whole program within the bench's grading tolerance; the
kernels phase takes back a kernel's own source and refuses a wrong entry; the codegen phase sweeps a kernel and
links a winner; the performance feedback reports a correct program, its offloading and every kernel."""

from collections import Counter

import numpy as np
import pytest
from dace.libraries.standard.nodes.external_call import external_calls
from dace.sdfg import nodes
from helpers import program_mismatch

from tactics.ir.names import in_order
from tactics.records import KernelStatus, MoveStatus
from tactics.session import Move, Session
from tactics.stages.moves import MOVE_SHAPES, PROPAGATE_CONFIG, config_names
from tactics.stages.performance import Reference

pytestmark = pytest.mark.cloudsc

Program = tuple[Session, dict[str, np.ndarray], Reference]


def straight_pair(session: Session) -> tuple[str, str]:
    """Labels of the first two top-level blocks that follow each other on one unconditional edge."""
    sdfg = session.sdfg
    for first in in_order(sdfg):
        out = sdfg.out_edges(first)
        if len(out) == 1 and out[0].data.is_unconditional() and sdfg.in_degree(out[0].dst) == 1:
            return first.label, out[0].dst.label
    raise AssertionError("cloudsc_monolith has no two top-level blocks in a straight run")


def applied_and_valid(base: Session, move: Move) -> MoveStatus:
    """``move`` applied on a copy of ``base`` with every field the listing set, whose program must still validate.
    ``propagate-config`` is listed without values, so it gets one preset size, as an agent passes it."""
    sut = base.trial()
    values = move.config_values
    if move.kind == PROPAGATE_CONFIG and not values:
        name = next(name for name in config_names(sut.sdfg) if name in sut.sizes)
        values = {name: sut.sizes[name]}
    # every field the listing set, as an agent passes them back; the notes for the agent are no arguments
    arguments = {k: v for k, v in move.model_dump().items() if k not in ("kind", "labels", "current")}
    status = sut.apply_move(move.kind, move.labels, **{**arguments, "config_values": values}).status
    sut.sdfg.validate()
    return status


@pytest.fixture(scope="module")
def scoped(cloudsc_monolith: Program) -> Session:
    """The default scopes and placement on a copy, as the unstructured and codegen phases start from them."""
    session = cloudsc_monolith[0].trial()
    session.default_scopes()
    session.place()
    return session


@pytest.fixture(scope="module")
def listed(cloudsc_monolith: Program) -> list[Move]:
    """Every move listed on the canonical program, listed once: a listing of cloudsc takes minutes."""
    return cloudsc_monolith[0].list_moves()


def test_listed_moves_grow_with_the_maps_not_with_their_pairs(cloudsc_monolith: Program, listed: list[Move]) -> None:
    session = cloudsc_monolith[0]
    maps = sum(isinstance(n, nodes.MapEntry) for n, _ in session.sdfg.all_nodes_recursive())

    kinds = Counter(move.kind for move in listed)

    assert set(kinds) <= set(MOVE_SHAPES)
    assert 0 < kinds["doall-fusion"] <= 2 * maps, kinds


def test_the_first_listed_move_of_every_kind_applies_and_leaves_a_valid_program(
    cloudsc_monolith: Program, listed: list[Move]
) -> None:
    base = cloudsc_monolith[0]
    firsts = {move.kind: move for move in reversed(listed)}

    statuses = {kind: applied_and_valid(base, move) for kind, move in firsts.items()}

    assert statuses == dict.fromkeys(firsts, MoveStatus.APPLIED)


def test_a_straight_run_of_blocks_becomes_one_kernel_and_unsetting_it_restores_a_correct_program(
    cloudsc_monolith: Program,
) -> None:
    base, inputs, reference = cloudsc_monolith
    sut = base.trial()
    labels = straight_pair(sut)

    scoped = sut.set_scope(labels)
    stale = sut.set_scope(labels)
    unset = sut.unset_scope(scoped.reason)

    assert scoped.status is MoveStatus.APPLIED, scoped
    assert stale.status is MoveStatus.NOT_FOUND
    assert unset.status is MoveStatus.APPLIED, unset
    assert not external_calls(sut.sdfg)
    assert not (why := program_mismatch(sut, inputs, reference)), why


def test_performance_reports_a_correct_program_its_offloading_and_every_kernel(
    cloudsc_monolith: Program, scoped: Session
) -> None:
    _, inputs, reference = cloudsc_monolith

    # the first build per kernel: what is reported is under test here, the stage 6 sweep in the sweep test below
    report = scoped.performance(inputs, reference, codegen="first")

    kernels = {ext.name for ext in external_calls(scoped.sdfg)}
    assert report.structured.correct is True, report.structured
    assert set(report.offloading.devices) == kernels and set(report.offloading.devices.values()) == {"cpu"}
    assert report.offloading.transfer_bytes == 0
    assert {kernel.name for kernel in report.kernels} == kernels
    assert any(kernel.correct for kernel in report.kernels)
    assert all(kernel.correct is not False for kernel in report.kernels), [k for k in report.kernels if not k.correct]
    assert scoped.performance(inputs, reference, codegen="first") is report, "an unchanged program is measured once"


def test_a_kernel_takes_back_its_own_source_and_refuses_a_wrong_entry(scoped: Session) -> None:
    name = scoped.list_kernels()[0].name
    source = scoped.kernel_source(name)

    same = scoped.set_kernel_source(name, source, "cpp")
    renamed = scoped.set_kernel_source(name, source.replace(f'"C" void {name}(', f'"C" void {name}_x('), "cpp")
    wrong_language = scoped.set_kernel_source(name, source, "cuda")

    assert same.status is KernelStatus.OK, same
    assert renamed.status is KernelStatus.REFUSED
    assert wrong_language.status is KernelStatus.REFUSED


def test_a_kernel_sweep_links_a_measured_winner(scoped: Session) -> None:
    name = scoped.list_kernels()[-1].name

    sweep = scoped.sweep(name, compilers=["gcc"])

    assert sweep.winner is not None and sweep.config is not None, sweep
    assert sweep.config.compiler == "g++" and sweep.config.time_us > 0
    assert scoped.kernel(name).implementation == "ExternCall"
