"""Offloading granularity UNITS (paper Axis 2): the structural unit each external call wraps -- cfg / state
/ map, coarse -> fine. Unit set, no compile: candidate selection per unit, whole-state extraction, and
that lowering each unit still yields a valid SDFG. Composition with Axis 1 (fusion granularity) is checked
too -- finer fusion exposes more map-units."""
import numpy as np
import dace

from nestforge.offload import (OFFLOAD_UNITS, offload_candidates, offload_coarseness, offload_unit_axis)
from nestforge.extract import extract_state_nest
from nestforge.pass_lower import lower_nests_to_external_call
from nestforge.granularity import fuse_first_k
from nestforge.strategies import top_level_map_entries

N = dace.symbol('N')


@dace.program
def two_map(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    T = np.empty_like(A)  # 1 compute state, 2 maps, no control-flow region
    for i in dace.map[0:N]:
        T[i] = A[i] + B[i]
    for i in dace.map[0:N]:
        C[i] = T[i] * 2.0


@dace.program
def recur(A: dace.float64[N], B: dace.float64[N]):
    for i in range(1, N):  # loop-carried -> stays a LoopRegion (a cfg unit)
        B[i] = B[i - 1] + A[i]


def test_axis_is_coarse_to_fine():
    assert offload_unit_axis() == ["cfg", "state", "map"]
    assert [offload_coarseness(u) for u in OFFLOAD_UNITS] == [0, 1, 2]  # cfg coarsest, map finest


def test_unit_selection_on_flat_kernel():
    sdfg = two_map.to_sdfg(simplify=True)
    assert len(offload_candidates(sdfg, "map")) == 2  # two maps
    assert len(offload_candidates(sdfg, "state")) == 1  # one compute state wrapping both
    assert len(offload_candidates(sdfg, "cfg")) == 0  # no control-flow region in a flat kernel


def test_cfg_unit_selects_a_control_flow_region():
    sdfg = recur.to_sdfg(simplify=True)
    cands = offload_candidates(sdfg, "cfg")
    assert len(cands) == 1
    assert "loop" in cands[0].label  # a control-flow region, not a state or map


def test_state_extraction_yields_the_state_interface():
    sdfg = two_map.to_sdfg(simplify=True)
    state = next(st for st in sdfg.states() if top_level_map_entries(st))
    boundary = extract_state_nest(sdfg, state)
    assert set(boundary.inputs) == {"A", "B"}  # the state reads A, B
    assert boundary.outputs == ["C"]  # writes C; T is an internal transient, not on the boundary


def test_lowering_each_unit_keeps_the_sdfg_valid():
    for unit, expected in [("map", 2), ("state", 1)]:
        sdfg = two_map.to_sdfg(simplify=True)
        lowered = lower_nests_to_external_call(sdfg, unit)
        assert len(lowered) == expected
        sdfg.validate()  # the numpy-reference fallback keeps the lowered SDFG valid


def test_lowering_cfg_unit_keeps_the_sdfg_valid():
    # the cfg lowering path (extract_loop_nest via the unit strategy) -- distinct from map/state.
    sdfg = recur.to_sdfg(simplify=True)
    lowered = lower_nests_to_external_call(sdfg, "cfg")
    assert len(lowered) == 1  # the one LoopRegion externalized
    sdfg.validate()


def test_composes_with_fusion_granularity():
    # a fine (map) offload sees at least as many units at the atoms partition as at maximal fusion.
    atoms = two_map.to_sdfg(simplify=True)
    fuse_first_k(0)(atoms)
    maximal = two_map.to_sdfg(simplify=True)
    fuse_first_k(99)(maximal)
    assert len(offload_candidates(atoms, "map")) >= len(offload_candidates(maximal, "map"))
