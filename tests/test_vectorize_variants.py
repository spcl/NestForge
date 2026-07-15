"""Staged screening for the vectorization axis: static pruning / dedup, stable naming, and the
coordinate-descent search. The pruning + descent logic is pure (no compile); one test builds a vectorized
kernel through the owned build to confirm a VectorizeConfig actually round-trips to a correct .so."""
import dataclasses

import pytest

import dace

from nestforge import vectorize_variants as vv
from dace.transformation.passes.vectorization.config import VectorizeConfig

pytest.importorskip("optarena")

N = dace.symbol("N")


@dace.program
def axpy(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    c[:] = a + b


def branch_free_sdfg():
    return axpy.to_sdfg(simplify=True)


def test_enumerate_prunes_dedups_and_names():
    """A branch-free nest gets no fp_factor cell; per (isa,width) exactly {base, even, posttail, fma,
    even+fma}; every cell has a unique greppable name and a distinct resolved key."""
    cells = vv.enumerate_vec_configs(branch_free_sdfg(), isas=("AVX512", "SCALAR"), widths=(8, 16))
    names = [c.name for c in cells]
    assert len(set(names)) == len(names)  # no duplicate names
    assert not any("fpfac" in n for n in names)  # branch-free -> fp_factor not enumerated
    assert len(cells) == 2 * 2 * 5  # 2 isas x 2 widths x {base, even, posttail, fma, even+fma}
    keys = {vv.resolved_key(c.config) for c in cells}
    assert len(keys) == len(cells)  # every enumerated cell is a genuinely distinct resolved config
    assert "cpu-avx512-w16-even-fma" in names and "cpu-scalar-w8" in names


def test_enumerate_offers_fp_factor_only_with_a_branch(monkeypatch):
    """fp_factor is enumerated exactly when the nest has a same-write-set branch."""
    monkeypatch.setattr(vv, "has_same_write_set_branch", lambda sdfg: True)
    names = [c.name for c in vv.enumerate_vec_configs(branch_free_sdfg(), isas=("AVX512", ), widths=(8, ))]
    assert any("fpfac" in n for n in names)


def test_resolved_key_collapses_dead_axes():
    """K>=2 makes target_isa dead (an AVX512 and an AVX2 (8,8) request resolve identically); assume_even
    makes the remainder strategy irrelevant -- so neither is recorded as a distinct variant."""
    k2_avx512 = vv.resolved_key(VectorizeConfig(widths=(8, 8), target_isa="AVX512"))
    k2_avx2 = vv.resolved_key(VectorizeConfig(widths=(8, 8), target_isa="AVX2"))
    assert k2_avx512 == k2_avx2  # target_isa dead at K>=2
    base = VectorizeConfig(widths=(8, ), target_isa="AVX512")
    even_masked = vv.resolved_key(dataclasses.replace(base, assume_even=True))
    even_postamble = vv.resolved_key(dataclasses.replace(base, assume_even=True, remainder_strategy="scalar_postamble"))
    assert even_masked == even_postamble  # remainder strategy dead under assume_even


def test_variant_name_reports_auto_for_k2():
    assert vv.variant_name(VectorizeConfig(widths=(8, 8), target_isa="AVX512")).startswith("cpu-auto-w8x8")


def test_coordinate_descent_finds_the_synthetic_optimum():
    """Against a synthetic cost table whose optimum is (AVX512, width 16, assume_even), multi-start descent
    from the scalar floor reaches it -- validating the search, not a compiler."""

    def cost(cfg: VectorizeConfig) -> float:
        c = 100.0
        c -= 40.0 if cfg.target_isa.value == "AVX512" else (20.0 if cfg.target_isa.value == "AVX2" else 0.0)
        c -= 15.0 if cfg.widths == (16, ) else 0.0
        c -= 8.0 if cfg.assume_even else 0.0
        return c

    axes = vv.descent_axes(isas=("SCALAR", "AVX2", "AVX512"), widths=(8, 16, 32))
    seeds = vv.default_seeds(isas=("AVX512", "AVX2", "SCALAR"), widths=(8, 16, 32))
    best, best_t = vv.multistart_descent(seeds, axes, cost)
    assert best.target_isa.value == "AVX512" and best.widths == (16, ) and best.assume_even
    assert best_t == pytest.approx(100.0 - 40.0 - 15.0 - 8.0)


def test_coordinate_descent_skips_unbuildable_cells():
    """A measure returning None (an unbuildable cell) never becomes the winner."""

    def measure(cfg: VectorizeConfig):
        return None if cfg.target_isa.value == "AVX512" else float(sum(cfg.widths))

    axes = vv.descent_axes(isas=("SCALAR", "AVX512"), widths=(8, 16))
    seed = VectorizeConfig(widths=(16, ), target_isa="SCALAR")
    best, best_t = vv.coordinate_descent(seed, axes, measure)
    assert best.target_isa.value != "AVX512" and best_t is not None
