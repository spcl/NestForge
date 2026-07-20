"""Fusion-granularity axis (paper Axis 1): the atoms->maximal partition ladder the arena sweeps. Unit set,
no compile -- these prove the ladder is well-formed (endpoints, bounded, monotone, idempotent w.r.t.
start), not that any point is fast (that is the measured evaluation). Ladder DEPTH is data-dependent
(fission explodes to statement atoms), so tests compute it rather than hardcode."""
import numpy as np
import dace

from nestforge.granularity import (GranularityPoint, count_nests, fuse_first_k, fusion_depth, granularity_ladder)

N = dace.symbol('N')


@dace.program
def two_map(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    T = np.empty_like(A)  # one transient chains two maps
    for i in dace.map[0:N]:
        T[i] = A[i] + B[i]
    for i in dace.map[0:N]:
        C[i] = T[i] * 2.0


@dace.program
def chain3(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N], out: dace.float64[N]):
    T1 = np.empty_like(A)
    T2 = np.empty_like(A)  # two transients chain three maps -> strictly deeper than two_map
    for i in dace.map[0:N]:
        T1[i] = A[i] + B[i]
    for i in dace.map[0:N]:
        T2[i] = T1[i] * C[i]
    for i in dace.map[0:N]:
        out[i] = T2[i] + 1.0


def test_atoms_are_finer_than_maximal():
    atoms = two_map.to_sdfg(simplify=True)
    fuse_first_k(0)(atoms)
    maximal = two_map.to_sdfg(simplify=True)
    fuse_first_k(99)(maximal)
    assert count_nests(atoms) > count_nests(maximal)  # atoms have more nests than the fused form
    assert count_nests(maximal) == 1  # everything legal fused into one nest


def test_fusion_depth_positive_and_grows_with_program():
    d2 = fusion_depth(two_map.to_sdfg(simplify=True))
    d3 = fusion_depth(chain3.to_sdfg(simplify=True))
    assert d2 >= 1
    assert d3 > d2  # more statements -> a deeper atoms->maximal ladder


def test_ladder_named_endpoints_and_length():
    sdfg = chain3.to_sdfg(simplify=True)
    depth = fusion_depth(sdfg)
    ladder = granularity_ladder(sdfg)
    assert all(isinstance(p, GranularityPoint) for p in ladder)
    assert ladder[0].name == "atoms" and ladder[-1].name == "maximal"
    assert len(ladder) == depth + 1  # one rung per greedy fusion step, inclusive


def test_ladder_monotonically_coarsens():
    ladder = granularity_ladder(chain3.to_sdfg(simplify=True))
    counts = []
    for point in ladder:
        sdfg = chain3.to_sdfg(simplify=True)
        point.apply(sdfg)
        counts.append(count_nests(sdfg))
    assert counts == sorted(counts, reverse=True)  # atoms -> maximal never increases nest count
    assert counts[0] > counts[-1]  # the endpoints genuinely differ


def test_max_points_subsamples_but_keeps_endpoints():
    ladder = granularity_ladder(chain3.to_sdfg(simplify=True), max_points=3)
    assert len(ladder) <= 3
    assert ladder[0].name == "atoms" and ladder[-1].name == "maximal"


def test_apply_is_idempotent_wrt_starting_granularity():
    # apply() fissions to atoms first, so the point reached is independent of the incoming granularity.
    from_frontend = chain3.to_sdfg(simplify=True)
    fuse_first_k(2)(from_frontend)
    already_coarsened = chain3.to_sdfg(simplify=True)
    fuse_first_k(99)(already_coarsened)  # fully fuse first
    fuse_first_k(2)(already_coarsened)  # then request the same rung
    assert count_nests(from_frontend) == count_nests(already_coarsened)
