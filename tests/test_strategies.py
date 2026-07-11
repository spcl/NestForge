"""Detection strategies: skip-taskloops (default), innermost maps, innermost loops."""
import pytest

pytest.importorskip("optarena")

from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion

from nestforge.corpus import iter_dace_kernels
from nestforge.strategies import get_strategy


def sdfg_for(short):
    return {k.short_name: k for k in iter_dace_kernels()}[short].to_sdfg(simplify=True)


def test_skip_taskloops_descends_through_a_map_only_time_loop():
    # jacobi's TSTEPS loop body is only maps -> a taskloop; skip it and offload the spatial maps.
    sdfg = sdfg_for("hpc/structured_grids/jacobi_1d/jacobi_1d")
    refs = get_strategy("skip-taskloops")(sdfg)
    assert refs, "expected the inner compute maps"
    assert all(isinstance(n, nodes.MapEntry) for _, n in refs)
    # outer would instead offload the loop wrapper itself.
    outer = get_strategy("outer")(sdfg)
    assert len(outer) == 1 and isinstance(outer[0][1], LoopRegion)


def test_innermost_yields_leaf_maps_when_present():
    # jacobi's loop holds maps, so the innermost units are those maps (not the loop wrapper).
    sdfg = sdfg_for("hpc/structured_grids/jacobi_1d/jacobi_1d")
    refs = get_strategy("innermost")(sdfg)
    assert refs and all(isinstance(n, nodes.MapEntry) for _, n in refs)


def test_innermost_yields_leaf_loops_for_loop_only_kernels():
    # lu has no maps and an outer loop wrapping two inner loops; the innermost units are those two.
    sdfg = sdfg_for("hpc/dense_linear_algebra/lu/lu")
    refs = get_strategy("innermost")(sdfg)
    assert len(refs) == 2 and all(isinstance(n, LoopRegion) for _, n in refs)
