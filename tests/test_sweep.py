"""The sweep matrix stays BOUNDED (the guard against a runaway experiment) and the measurement ledger
counts search cost. Unit set, no compile: cell counts and caps only."""
import numpy as np
import pytest
import dace

from nestforge.offload import OFFLOAD_UNITS
from nestforge.sweep import (DEFAULT_GRANULARITY_POINTS, MeasureLedger, SweepCell, bounded_kernels, parse_units,
                             sweep_cells, sweep_upper_bound)
from nestforge.tsvc import TsvcKernel

N = dace.symbol('N')


@dace.program
def two_map(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    T = np.empty_like(A)
    for i in dace.map[0:N]:
        T[i] = A[i] + B[i]
    for i in dace.map[0:N]:
        C[i] = T[i] * 2.0


def synthetic_kernels(n):
    return [TsvcKernel(key=f"k{i}", program=two_map, regime="1d", params={}, corpus="tsvc2") for i in range(n)]


def test_default_matrix_is_small():
    # the whole point: the default sweep is a handful of cells, not thousands.
    assert sweep_upper_bound(3) <= 27  # 3 kernels x 3 granularity points x <=3 units


def test_bounded_kernels_caps_after_discovery():
    kernels = bounded_kernels(limit=2)
    assert len(kernels) <= 2  # capped no matter how many the corpus discovers


def test_sweep_cells_never_exceed_the_bound():
    kernels = synthetic_kernels(2)
    cells = sweep_cells(kernels, max_granularity_points=2, units=("map", "state"))
    assert all(isinstance(c, SweepCell) for c in cells)
    assert len(cells) <= sweep_upper_bound(2, max_granularity_points=2, units=("map", "state"))
    assert {c.kernel for c in cells} == {"k0", "k1"}  # every kernel present
    assert {c.unit for c in cells} == {"map", "state"}


def test_granularity_points_cap_is_honored():
    cells = sweep_cells(synthetic_kernels(1), max_granularity_points=2, units=("map", ))
    grans = {c.granularity for c in cells}
    assert len(grans) <= 2  # ladder subsampled to the cap even if the true ladder is deeper


def test_ledger_counts_every_measurement():
    ledger = MeasureLedger()
    for cell in ["a", "b", "c"]:
        ledger.measure(cell, lambda: 1.0)
    assert ledger.measurements == 3 and ledger.seen == ["a", "b", "c"]


def test_default_points_constant_is_small():
    assert DEFAULT_GRANULARITY_POINTS <= 5  # a sane default cap


def test_parse_units_drops_blank_entries():
    # a trailing comma / empty env value used to yield a phantom '' unit that inflated the upper bound and
    # only failed later as get_strategy('') deep inside the sweep.
    assert parse_units("map,") == ("map", )
    assert parse_units(" map , state ") == ("map", "state")
    assert parse_units("") == OFFLOAD_UNITS  # empty falls back to the full axis, never an empty matrix


def test_parse_units_rejects_an_unknown_unit_up_front():
    with pytest.raises(ValueError, match="unknown offload unit"):
        parse_units("map,bogus")
