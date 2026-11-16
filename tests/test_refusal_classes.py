# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Every refusal carries its class, so the plugin record counts refusals by class (docs/plugin-record.md):
each candidate a move kind lists on corpus programs is planned, and every refusal it answers is classed."""

import dace
import pytest
from helpers import corpus_kernel

from nestforge.records import CRASHES, MoveStatus, Refusal, refusal_of
from nestforge.session import Session
from nestforge.stages.moves import (
    AUTO_VECTORIZE,
    CANDIDATES,
    PROPAGATE_CONFIG,
    Rewrite,
    plan_move,
    probe_implementation,
    probe_sizes,
    tree_label,
)

#: Corpus programs whose canonical forms offer loops, doalls, guards, and library calls to every kind.
KERNELS = ("jacobi_1d", "tsvc_2_s231", "loop_level_reasoning/fuse_diamond/fuse_diamond", "gramschmidt")
#: Kinds listed without a probe: no values to plan with, or too costly to plan per candidate.
UNPROBED = frozenset({PROPAGATE_CONFIG, AUTO_VECTORIZE})


def canonical_program(name: str) -> dace.SDFG:
    sut = Session(corpus_kernel(name).to_sdfg())
    sut.canonicalize()
    assert sut.sdfg is not None, "a canonicalized session holds its program"
    return sut.sdfg


@pytest.mark.parametrize("name", KERNELS)
def test_every_refusal_a_listed_candidate_meets_carries_a_class(name: str) -> None:
    sdfg = canonical_program(name)
    unclassed: list[tuple[str, tuple[str, ...], str]] = []
    refused = 0

    for kind, candidates in CANDIDATES.items():
        if kind in UNPROBED:
            continue
        for rows in candidates(sdfg):
            try:
                plan = plan_move(kind, rows, probe_sizes(kind, rows), probe_implementation(kind, rows))
            except CRASHES:  # a check that raises is not a refusal; listing skips it the same way
                continue
            if isinstance(plan, Rewrite):
                continue
            refused += 1
            if refusal_of(plan) is Refusal.OTHER:
                unclassed.append((kind, tuple(tree_label(obj) for obj, _ in rows), plan))

    assert refused > 0, f"{name} offered no refused candidate; the test checks nothing"
    assert unclassed == []


def test_a_move_on_rows_of_the_wrong_kind_is_refused_as_such() -> None:
    sdfg = canonical_program("jacobi_1d")
    loops = next(CANDIDATES["tile-for"](sdfg))

    plan = plan_move("doall-fission", loops)

    assert isinstance(plan, str) and refusal_of(plan) is Refusal.WRONG_ROW_KIND


def test_a_label_no_row_holds_is_refused_as_unknown() -> None:
    sut = Session(corpus_kernel("jacobi_1d").to_sdfg())
    sut.canonicalize()

    result = sut.apply_move("doall-fission", ["doall999"])

    assert (result.status, result.refusal) == (MoveStatus.NOT_FOUND, Refusal.UNKNOWN_LABEL)
