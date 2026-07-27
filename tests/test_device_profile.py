# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage-A device characterization: host-ISA expansion (shared by the vectorization sweep) and per-device
vector-math-library ranking (throughput + ULP). Pure and fast; the one probe-compiling test skips without
a C compiler."""
import shutil
import subprocess
from pathlib import Path

import pytest

from nestforge import device_profile as dp
from nestforge.toolchain import VECLIB_PROBE_OPS, VECTOR_LIBS, packed_ops_provided, veclib_library_path

pytest.importorskip("hpcagent_bench")


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
    incompat = dp.characterize_veclib("g++", "svml")  # gcc emits _ZGV*, never __svml_*, so svml is unusable
    assert not incompat.ok and "incompatible" in incompat.reason
    unknown = dp.characterize_veclib("g++", "not_a_veclib")
    assert not unknown.ok and "unknown" in unknown.reason


def test_probe_links_the_veclib_ahead_of_libm():
    """Order decides which library the loader binds ``_ZGV*`` to. glibc's libm.so pulls libmvec in, so
    with ``-lm`` first a sleef cell measures libmvec under sleef's name -- the speedup is real and the
    label is wrong, which no timing check can catch."""
    cmd = dp.probe_command("gcc", ["-ffast-math"], ["-lsleefgnuabi"], Path("p.c"), Path("p"))
    assert cmd.index("-lsleefgnuabi") < cmd.index("-lm")


@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not on PATH")
@pytest.mark.skipif(veclib_library_path(VECTOR_LIBS["sleef"], "gcc") is None, reason="sleef not installed")
def test_the_probe_link_order_really_keeps_libmvec_out_of_the_binary(tmp_path):
    """The command-order assertion above, proven on a real link: with the veclib first, libmvec is not
    even DT_NEEDED, so the packed calls can only have come from sleef. Reversing the order puts BOTH in,
    libmvec first, and the loader picks libmvec."""
    src = tmp_path / "p.c"
    src.write_text("#include <math.h>\n#include <stdio.h>\n"
                   "int main(){double s=0;for(int i=0;i<1000;i++)s+=sin(i*0.001);printf(\"%f\\n\",s);}\n")
    sleef = VECTOR_LIBS["sleef"]
    link = sleef.link_flags("gcc")
    exe = tmp_path / "p"
    cmd = dp.probe_command("gcc", ["-ffast-math", "-march=native"], link, src, exe)
    assert subprocess.run(cmd, capture_output=True, text=True).returncode == 0, cmd
    needed = subprocess.run(["objdump", "-p", str(exe)], capture_output=True, text=True).stdout
    assert "libsleefgnuabi" in needed
    assert "libmvec" not in needed, "libm pulled libmvec in anyway, so this timing is not sleef's"


@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not on PATH")
def test_a_library_is_credited_only_for_the_ops_it_exports():
    """Emitting a packed call proves the vectorizer fired, not that THIS library serves it -- another
    library on the link line may resolve it, and the timing would then carry the wrong name. The gate is
    what the library EXPORTS, so a library that serves nothing is credited with nothing."""
    path = veclib_library_path(VECTOR_LIBS["libmvec"], "gcc")
    if path is None:
        pytest.skip("libmvec not resolvable by gcc")
    assert packed_ops_provided("libmvec", path) == VECLIB_PROBE_OPS, "glibc serves every probe op"
    assert packed_ops_provided("libmvec", path, ops=("not_an_elemental", )) == ()


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
