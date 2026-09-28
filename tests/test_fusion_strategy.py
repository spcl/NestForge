# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage 2 deterministic default (:mod:`nestforge.stages.canonicalize`): ``fuse_and_finish`` on a ``canonicalize``d SDFG
reaches the same fixed point as draining the listed fusion moves -- so the deterministic default and the
agent's move-by-move policy agree.
"""

import numpy as np

import dace
from dace.sdfg import nodes

from nestforge.stages.canonicalize import Targets, canonicalize
from nestforge.stages.canonicalize import fuse_and_finish
from helpers import fission_to_fixpoint, fusion_moves

N = dace.symbol("N")
f64 = dace.float64


@dace.program
def producer_consumer_maps(a: f64[N], b: f64[N]):
    tmp = np.empty_like(a)
    for i in dace.map[0:N]:
        tmp[i] = a[i] * 2.0
    for i in dace.map[0:N]:
        b[i] = tmp[i] + 1.0


@dace.program
def sibling_maps(a: f64[N], b: f64[N], c: f64[N]):
    for i in dace.map[0:N]:
        b[i] = a[i] * 2.0
    for i in dace.map[0:N]:
        c[i] = a[i] + 1.0


def map_count(sdfg):
    return sum(
        isinstance(n, nodes.MapEntry) for sd in sdfg.all_sdfgs_recursive() for st in sd.all_states() for n in st.nodes()
    )


def test_fuse_and_finish_is_idempotent_on_an_already_fused_pair():
    """canonicalize already folds a vertical producer/consumer pair to one map; fuse_and_finish must leave it one map with
    no legal fuse move remaining."""
    raw = producer_consumer_maps.to_sdfg(simplify=False)
    assert map_count(raw) == 2, "fixture must start with two separate maps"

    sdfg = producer_consumer_maps.to_sdfg(simplify=False)
    targets = Targets()
    canonicalize(sdfg, targets)
    assert map_count(sdfg) == 1, "canonicalize already fused the pair"
    fuse_and_finish(sdfg, targets)
    assert map_count(sdfg) == 1
    assert fusion_moves(sdfg) == []


def test_fuse_and_finish_redrains_legal_fusions_after_fission():
    """Fissioning a fused horizontal pair back apart reopens a legal move that fuse_and_finish re-drains."""
    sdfg = sibling_maps.to_sdfg(simplify=False)
    targets = Targets()
    canonicalize(sdfg, targets)
    fuse_and_finish(sdfg, targets)
    assert map_count(sdfg) == 1, "canonicalize+fuse_and_finish must reach one map before fission reopens anything"

    assert fission_to_fixpoint(sdfg) >= 1
    assert map_count(sdfg) == 2
    reopened = fusion_moves(sdfg)
    assert [kind for kind, _ in reopened] == ["map-fusion"]

    fuse_and_finish(sdfg, targets)
    assert map_count(sdfg) == 1
    assert fusion_moves(sdfg) == []


def test_fuse_and_finish_is_value_preserving():
    """Fusing must not change the vertical pair's numeric result."""
    rng = np.random.default_rng(0)
    inputs = {k: rng.random(48) for k in ("a", "b")}
    ref = {k: v.copy() for k, v in inputs.items()}
    producer_consumer_maps.to_sdfg(simplify=True)(**ref, N=48)

    sdfg = producer_consumer_maps.to_sdfg(simplify=False)
    targets = Targets()
    canonicalize(sdfg, targets)
    fuse_and_finish(sdfg, targets)
    got = {k: v.copy() for k, v in inputs.items()}
    sdfg(**got, N=48)
    for k in inputs:
        np.testing.assert_array_equal(got[k], ref[k], err_msg=k)
