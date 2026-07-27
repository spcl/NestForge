# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Staged screening for the vectorization axis: static pruning / dedup, stable naming, and the
coordinate-descent search. The pruning + descent logic is pure (no compile); one test builds a vectorized
kernel through the owned build to confirm a VectorizeConfig actually round-trips to a correct .so."""
import dataclasses

import pytest

import dace

from nestforge import vectorize_variants as vv
from dace.transformation.passes.vectorization.config import VectorizeConfig

pytest.importorskip("hpcagent_bench")

N = dace.symbol("N")


@dace.program
def axpy(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    c[:] = a + b


def branch_free_sdfg():
    return axpy.to_sdfg(simplify=True)


def test_a_remainder_move_clears_the_previous_arms_fields():
    """Each arm owns two fields. A descent that moved only `remainder_strategy` would carry the previous
    arm's `scalar_remainder_emit` along -- landing on a config that is not a point of the axis, and one the
    orchestrator rejects outright (`tile_k1` is only valid under `scalar_postamble`)."""
    axes = dict(vv.descent_axes(widths=(8, )))
    k1 = dataclasses.replace(vv.base_config(8, "contract-fma"),
                             remainder_strategy="scalar_postamble",
                             scalar_remainder_emit="tile_k1")
    moved = [dataclasses.replace(k1, **delta) for delta in axes["remainder"]]
    assert {vv.remainder_arm(c) for c in moved} == {arm for arm, _ in vv.REMAINDER_ARMS}
    for cfg in moved:
        assert cfg.scalar_remainder_emit == "scalar" or cfg.remainder_strategy.value == "scalar_postamble"


def test_conditional_blocks_sees_into_a_nested_sdfg():
    """The shallow `all_control_flow_regions()` misses a conditional inside a map body's NestedSDFG -- i.e.
    every nest the tile path can actually vectorize."""
    outer = dace.SDFG("outer")
    inner = dace.SDFG("inner")
    from dace.sdfg.state import ConditionalBlock
    inner.add_node(ConditionalBlock("cond", sdfg=inner))
    state = outer.add_state()
    outer.add_array("A", [1], dace.float64)
    state.add_nested_sdfg(inner, {}, {"A"})
    assert vv.conditional_blocks(outer), "a nested conditional was invisible to the walk"


def test_resolved_key_collapses_an_unread_field():
    """`scalar_remainder_emit` is only read under `scalar_postamble`, so two masked_tail configs spelling it
    differently are ONE build -- keyed apart, the sweep would compile and time it twice."""
    base = vv.base_config(8, "contract-fma")
    assert vv.resolved_key(base) == vv.resolved_key(dataclasses.replace(base, scalar_remainder_emit="tile_k1"))
    post = dataclasses.replace(base, remainder_strategy="scalar_postamble")
    assert vv.resolved_key(post) != vv.resolved_key(dataclasses.replace(post, scalar_remainder_emit="tile_k1"))


def test_variant_name_is_greppable_and_carries_no_constant_isa_token():
    assert vv.variant_name(vv.base_config(16, "contract-fma")) == "cpu-w16-masked-fma"
    assert vv.variant_name(vv.base_config(16, "strict-ieee")) == "cpu-w16-masked"


def test_coordinate_descent_finds_the_synthetic_optimum():
    """Against a synthetic cost table whose optimum is (AVX512, width 16, assume_even), multi-start descent
    from the scalar floor reaches it -- validating the search, not a compiler."""

    def cost(cfg: VectorizeConfig) -> float:
        c = 100.0
        c -= 15.0 if cfg.widths == (16, ) else 0.0
        c -= 8.0 if vv.remainder_arm(cfg) == "fullmask" else 0.0
        return c

    axes = vv.descent_axes(widths=(8, 16, 32))
    seeds = vv.default_seeds(widths=(8, 16, 32))
    best, best_t = vv.multistart_descent(seeds, axes, cost)
    assert best.widths == (16, ) and vv.remainder_arm(best) == "fullmask"
    assert best_t == pytest.approx(100.0 - 15.0 - 8.0)


def test_coordinate_descent_skips_unbuildable_cells():
    """A measure returning None (an unbuildable cell) never becomes the winner."""

    def measure(cfg: VectorizeConfig):
        return None if vv.remainder_arm(cfg) == "fullmask" else float(sum(cfg.widths))

    axes = vv.descent_axes(widths=(8, 16))
    seed = vv.base_config(16, "contract-fma")
    best, best_t = vv.coordinate_descent(seed, axes, measure)
    assert vv.remainder_arm(best) != "fullmask" and best_t is not None


def test_the_descent_offers_every_remainder_arm():
    """The remainder axis has four distinct emissions and the search must reach all of them. It reached
    two, and the module's two entry points disagreed about WHICH two, so `full_mask` was searched by
    neither. One REMAINDER_ARMS table is why that cannot recur."""
    axes = dict(vv.descent_axes(widths=(8, )))
    seed = vv.base_config(8, "contract-fma")
    reached = {vv.remainder_arm(dataclasses.replace(seed, **delta)) for delta in axes["remainder"]}
    assert reached == {arm for arm, _ in vv.REMAINDER_ARMS} == {"masked", "fullmask", "posttail", "k1tail"}


def test_fma_follows_the_fp_rung_and_is_not_a_search_axis():
    """Contraction is what every rung above strict-ieee already grants the compiler, so the tile path fuses
    exactly when the requested numerics allow -- denying it would make the vectorized lane stricter than
    the scalar baseline it is divided by. Not a cell to measure: no axis offers it."""
    assert not vv.fma_allowed("strict-ieee")
    assert all(vv.fma_allowed(m) for m in ("contract-fma", "assume-finite", "fast-math"))
    assert not vv.base_config(8, "strict-ieee").fuse_multiply_add
    assert vv.base_config(8, "contract-fma").fuse_multiply_add
    assert "fuse_multiply_add" not in dict(vv.descent_axes())


def test_every_seed_is_k1_on_the_auto_isa():
    """CPU SIMD is a one-dimensional lane count: K>=2 needs a map with >=2 params (MarkTileDims raises
    otherwise) and makes target_isa dead on top. AUTO resolves to the host's best ISA at expansion, so the
    ISA is detected rather than searched -- and `assume_even` is not a way of emitting a remainder."""
    for cfg in vv.default_seeds():
        assert len(cfg.widths) == 1
        assert cfg.target_isa.value == "AUTO"
        assert not cfg.assume_even
    assert "target_isa" not in dict(vv.descent_axes())
