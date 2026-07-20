"""Per-nest full-program differential measurement. The swap logic is a unit test (no compile); the
end-to-end build+run of the whole lowered program is an integration test (compiles + forks), gated on a
working C toolchain so it never fails on a machine without one."""
import numpy as np
import pytest
import dace

from nestforge.differential import ContextResult, NestVariant, measure_in_context, set_nest_variant
from nestforge.granularity import fuse_first_k
from nestforge.pass_lower import lower_nests_to_external_call
from nestforge.tsvc import TsvcKernel

N = dace.symbol('N')


@dace.program
def two_map(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    T = np.empty_like(A)
    for i in dace.map[0:N]:
        T[i] = A[i] + B[i]
    for i in dace.map[0:N]:
        C[i] = T[i] * 2.0


def test_set_nest_variant_swaps_only_the_target():
    sdfg = two_map.to_sdfg(simplify=True)
    calls = lower_nests_to_external_call(sdfg, "map")
    assert len(calls) == 2
    target, other = calls[0][0], calls[1][0]
    set_nest_variant(target, NestVariant("/abs/libk.a", "k", ["A", "B", "T", "N"]))
    assert target.lib_path == "/abs/libk.a" and target.symbol == "k"
    assert target.abi_order == ["A", "B", "T", "N"]
    assert other.lib_path == "" and other.symbol == ""  # untouched -> stays the numpy-reference fallback


@pytest.mark.integration
def test_all_reference_program_is_bit_exact_in_context(tmp_path):
    # no variants -> every nest at the numpy-reference fallback -> the lowered whole program must match the
    # whole-program oracle bit-exact. This exercises the full harness: lower, build, fork-run, validate, time.
    kernel = TsvcKernel(key="two_map", program=two_map, regime="1d", params={}, corpus="tsvc2")
    res = measure_in_context(kernel, tmp_path, variants={}, granularity="map", reps=3)
    assert isinstance(res, ContextResult)
    assert res.error is None, res.error
    assert res.ok and res.maxdiff == 0.0  # reference == oracle, bit-exact
    assert res.median_us < float("inf") and res.swapped == []


@pytest.mark.integration
def test_measures_at_each_granularity_rung(tmp_path):
    # Axis 1 <-> differential bridge: the same program measured at two granularity partitions (atoms vs
    # maximal fusion) both build, run in full-program context, and validate bit-exact. This is the E1
    # measurement primitive (times per granularity rung).
    kernel = TsvcKernel(key="two_map", program=two_map, regime="1d", params={}, corpus="tsvc2")
    for point in (fuse_first_k(0), fuse_first_k(99)):  # atoms, then maximal
        res = measure_in_context(kernel, tmp_path / "g", granularity="map", apply_granularity=point, reps=3)
        assert res.error is None, res.error
        assert res.ok and res.maxdiff == 0.0 and res.median_us < float("inf")
