"""Stage-A device characterization: host-ISA expansion (shared by the vectorization sweep) and the
per-device vector-math-library ranking (throughput + ULP). The ISA + gating logic is pure and fast; the
one probe-compiling test is skipped without a C compiler."""
import shutil

import pytest

from nestforge import device_profile as dp

pytest.importorskip("optarena")


@pytest.mark.parametrize("detected,expected", [
    ("AVX512", ("AVX512", "SCALAR")),
    ("AVX2", ("AVX2", "SCALAR")),
    ("ARM_SVE", ("ARM_SVE", "ARM_NEON", "SCALAR")),
    ("ARM_NEON", ("ARM_NEON", "SCALAR")),
    ("SCALAR", ("SCALAR", )),
])
def test_host_isas_expands_detect_host_isa_with_scalar_floor(monkeypatch, detected, expected):
    """host_isas defers to DaCe's detect_host_isa (never re-sniffs the CPU) and expands it default-first
    with SCALAR always present; ARM_SVE keeps ARM_NEON too."""
    monkeypatch.setattr("nestforge.device_profile.detect_host_isa", lambda: detected)
    assert dp.host_isas() == expected
    assert dp.host_isas()[-1] == "SCALAR"  # floor always last


def test_host_isas_unknown_falls_back_to_scalar(monkeypatch):
    monkeypatch.setattr("nestforge.device_profile.detect_host_isa", lambda: "SOMETHING_NEW")
    assert dp.host_isas() == ("SCALAR", )


def test_pure_only_math_ops_are_the_veclib_captured_ops():
    """The ops routed to the pure tile loop for the vector-math library to capture -- pow/atan2/hypot in,
    the ISA-intrinsic sin/sqrt out -- so a nest free of these gets no veclib axis."""
    ops = dp.pure_only_math_ops()
    assert {"pow", "atan2", "hypot", "tan"} <= set(ops)
    assert "sin" not in ops and "sqrt" not in ops


def test_characterize_veclib_gates_without_compiling():
    """An incompatible or unknown veclib is reported ok=False WITH a reason -- no compile, no raise."""
    incompat = dp.characterize_veclib("g++", "sleef")  # sleef unsupported on gcc
    assert not incompat.ok and "incompatible" in incompat.reason
    unknown = dp.characterize_veclib("g++", "not_a_veclib")
    assert not unknown.ok and "unknown" in unknown.reason


@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not on PATH")
def test_rank_veclibs_orders_installed_and_records_skips():
    """rank_veclibs measures every installed+compatible veclib and records a reason for the rest; the
    installed ones (accuracy-gated) sort ahead of the skipped ones and carry a positive speedup."""
    ranked = dp.rank_veclibs("gcc", max_ulp=8.0)
    assert {p.name for p in ranked} == set(dp.VECTOR_LIBS)  # every library accounted for
    ok = [p for p in ranked if p.ok]
    skipped = [p for p in ranked if not p.ok]
    assert all(p.reason for p in skipped)  # every skip has a recorded reason
    if ok:  # at least glibc libmvec is usually present
        assert all(p.throughput_speedup > 0.0 for p in ok)
        assert ranked.index(ok[-1]) < ranked.index(skipped[0]) if skipped else True  # ok before skipped
