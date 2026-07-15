"""Unit tests for the Pluto polyhedral lane helpers (:mod:`nestforge.perf.pluto_lane`).

These exercise the compiler-free / polycc-free logic: locating the emitted scop, deriving sibling paths,
parsing the authoritative ``_pluto_binding.json`` ABI, and the skip-gate precedence. The actual polycc
transform + compile + run is validated by the optarena numerical oracle where polycc is installed; here we
only prove nest-forge's plumbing (which never needs polycc to make its skip/marshal decisions)."""
import json
import shutil
import stat
from pathlib import Path

import pytest

from nestforge.perf import pluto_lane


def test_find_pluto_input_picks_the_scop_not_the_plain_c():
    srcs = [Path("k_fp64.c"), Path("k_fp64.cpp"), Path("k_fp64_pluto_input.c"), Path("k_fp64.f90")]
    assert pluto_lane.find_pluto_input(srcs) == Path("k_fp64_pluto_input.c")
    assert pluto_lane.find_pluto_input([Path("k_fp64.c"), Path("k_fp64.cpp")]) is None  # no scop -> None


def test_sibling_paths_derive_from_the_scop_name():
    pin = Path("/tmp/n0/k_fp64_pluto_input.c")
    assert pluto_lane.pluto_binding_path(pin) == Path("/tmp/n0/k_fp64_pluto_binding.json")
    assert pluto_lane.pluto_output_path(pin) == Path("/tmp/n0/k_fp64_pluto.c")


def test_read_binding_and_extract_symbol_and_size_first_order(tmp_path):
    pin = tmp_path / "k_fp64_pluto_input.c"
    pin.write_text("#pragma scop\n#pragma endscop\n")
    binding = {
        "kernel":
        "k_fp64",
        "abi":
        "c",
        # Pluto declares SIZE SYMBOLS first (the VLA dims), then arrays, then scalars.
        "args": [{
            "name": "N",
            "kind": "int64"
        }, {
            "name": "a",
            "kind": "ptr_double",
            "shape": ["N"]
        }, {
            "name": "b",
            "kind": "ptr_double",
            "shape": ["N"]
        }],
        "symbols": {
            "c": "k_fp64"
        },
        "sources": {
            "c": "k_fp64_pluto.c"
        },
    }
    pluto_lane.pluto_binding_path(pin).write_text(json.dumps(binding))
    got = pluto_lane.read_pluto_binding(pin)
    assert got == binding
    symbol, order = pluto_lane.binding_symbol_and_order(got)
    assert symbol == "k_fp64"
    assert order == ["N", "a", "b"]  # size symbol first, then arrays -- the VLA signature order


def test_read_binding_returns_none_when_absent(tmp_path):
    pin = tmp_path / "k_fp64_pluto_input.c"
    pin.write_text("#pragma scop\n#pragma endscop\n")
    assert pluto_lane.read_pluto_binding(pin) is None  # no sibling json -> None (not a crash)


def test_gate_reports_not_installed_first_when_polycc_absent(tmp_path, monkeypatch):
    # With polycc absent (the common case off the container), EVERY nest gets one uniform not-installed
    # reason -- even a non-affine or no-scop nest -- so the per-box story is unambiguous.
    monkeypatch.setattr(pluto_lane, "polycc_available", lambda: False)
    assert pluto_lane.pluto_gate_reason(None) == "skip:not-installed"
    aff = tmp_path / "aff_pluto_input.c"
    aff.write_text("#pragma scop\na[i] = b[i] + 1.0;\n#pragma endscop\n")
    assert pluto_lane.pluto_gate_reason(aff) == "skip:not-installed"


def test_gate_precedence_when_polycc_present(tmp_path, monkeypatch):
    monkeypatch.setattr(pluto_lane, "polycc_available", lambda: True)
    # no scop emitted -> unsupported:no-scop
    assert pluto_lane.pluto_gate_reason(None) == "skip:unsupported:no-scop"
    # a non-affine subscript (modulo) is outside Pluto's model -> skip, not a silent miscompile
    na = tmp_path / "na_pluto_input.c"
    na.write_text("#pragma scop\na[i % 4] = b[i];\n#pragma endscop\n")
    assert pluto_lane.pluto_gate_reason(na) == "skip:unsupported:non-affine:modulo"
    # a fully affine scop -> None (the lane proceeds to polycc)
    aff = tmp_path / "aff_pluto_input.c"
    aff.write_text("#pragma scop\na[i] = b[(i + 1)];\n#pragma endscop\n")
    assert pluto_lane.pluto_gate_reason(aff) is None


def test_run_polycc_reports_not_installed_when_binary_missing(tmp_path, monkeypatch):
    # OSError from a missing binary is turned into a recorded skip, never raised.
    monkeypatch.setattr(pluto_lane, "POLYCC", "definitely-not-a-real-polycc-binary")
    pin = tmp_path / "k_pluto_input.c"
    pin.write_text("#pragma scop\n#pragma endscop\n")
    ok, reason = pluto_lane.run_polycc(pin, tmp_path / "k_pluto.c", timeout=10.0)
    assert ok is False and reason.startswith("skip:not-installed")


def _fake_identity_polycc(dirpath: Path) -> str:
    """A stand-in ``polycc`` that just COPIES its ``--pet <src> -o <out>`` input to the output -- the
    identity 'transform'. Since the emitted ``_pluto_input.c`` is already a valid sequential C kernel with
    the VLA/size-first signature, compiling+running the copy exercises the WHOLE marshaling path (binding ->
    size-first order -> c_argtypes -> call_c) against the oracle, without needing a real Pluto install. Only
    the polyhedral RESCHEDULE (a perf transform) is unexercised -- that is validated in optarena's oracle."""
    p = dirpath / "polycc"
    p.write_text('#!/bin/sh\n# args: --pet <src> -o <out>\ncp "$2" "$4"\n')
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(p)


@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not on PATH")
def test_pluto_lane_marshals_the_size_first_vla_abi_end_to_end(tmp_path, monkeypatch):
    """The ABI trap the plan flags: Pluto's transformed signature puts SIZE SYMBOLS first with VLA array
    params. Prove nest-forge marshals it correctly by running the lane with an identity 'polycc' -- if the
    size-first ``_pluto_binding.json`` order were mis-bound (a size in a pointer slot), the result would be
    garbage / a crash, not a bit-exact match to the numpy oracle."""
    from nestforge import tsvc
    from nestforge.perf import tsvc_full
    from nestforge.perf.tsvc_arena import discover_toolchains
    if not discover_toolchains("gcc"):
        pytest.skip("no gcc toolchain")
    k = tsvc.iter_tsvc_kernels(only=["s000"])[0]
    ctxs = tsvc_full.build_opt_context(k, "simplify-parallel", "skip-taskloops", "S", ["c"], tmp_path)
    nc = ctxs[0]
    fake = _fake_identity_polycc(tmp_path)
    monkeypatch.setattr(pluto_lane, "POLYCC", fake)
    monkeypatch.setattr(pluto_lane, "polycc_available", lambda: True)
    res = tsvc_full.measure_pluto_lane(nc, "gcc", reps=3, workdir=tmp_path / "pluto")
    assert not res.get("skip"), f"lane skipped unexpectedly: {res.get('skip')}"
    assert res.get("error") is None, f"lane errored: {res.get('error')}"
    assert res["ok"] is True, f"pluto result did not validate (maxdiff={res.get('maxdiff')})"
    assert res["maxdiff"] <= 1e-6
