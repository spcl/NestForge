"""Unit tests for the runtime call-overhead job's pure logic (no compilation): the trampoline source
generation, the C-signature param parse, the markdown table, and the plot reader (invoked as a subprocess,
like the sbatch drivers do -- ``perf/`` is not a package). The full build+time path (three variants per
kernel) is exercised on the daint fleet, not in the unit set."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from nestforge import tsvc
from nestforge.perf import calloverhead as co
from nestforge.perf.tsvc_arena import discover_toolchains

REPO_ROOT = Path(__file__).resolve().parents[1]
PLOT_CALLOVERHEAD = REPO_ROOT / "perf" / "plot_calloverhead.py"

C_SRC = """
#include <math.h>
void s000_fp64(double * restrict a, double * restrict b, int64_t LEN_1D) {
    for (int64_t i = 0; i < LEN_1D; i++) a[i] = b[i] + 1.0;
}
"""


def test_signature_params_reads_full_declaration():
    params = co.signature_params(C_SRC, "s000_fp64")
    assert "double * restrict a" in params
    assert "int64_t LEN_1D" in params


def test_signature_params_missing_symbol_raises():
    with pytest.raises(LookupError):
        co.signature_params(C_SRC, "not_here")


def test_runner_source_inline_includes_kernel(tmp_path):
    kc = tmp_path / "s000.c"
    kc.write_text(C_SRC)
    src = co.runner_source("s000_fp64",
                           "double * restrict a, double * restrict b, int64_t LEN_1D", ["a", "b", "LEN_1D"],
                           kernel_c=kc)
    assert f'#include "{kc.resolve()}"' in src  # inline build includes the kernel TU
    assert "void run_s000_fp64(" in src
    assert "s000_fp64(a, b, LEN_1D)" in src  # forwards by name
    assert "for (int64_t r = 0; r < nreps; ++r)" in src


def test_runner_source_external_declares_extern(tmp_path):
    src = co.runner_source("s000_fp64", "double * a, int64_t n", ["a", "n"], kernel_c=None)
    assert "extern void s000_fp64(double * a, int64_t n);" in src
    assert "#include" not in src.replace("#include <stdint.h>", "")  # only the stdint include, no kernel TU


def test_render_tables_computes_ratios_and_geomean(tmp_path):
    (tmp_path / "tsvc2_s000.json").write_text(
        json.dumps({
            "key": "s000",
            "compiler": "gnu",
            "inline_us": 1.0,
            "external_lto_us": 1.02,
            "external_us": 1.5,
            "call_overhead": 1.5,
            "lto_overhead": 1.02
        }))
    (tmp_path / "tsvc2_bad.json").write_text(json.dumps({"key": "bad", "skipped": "emit: UnsupportedNest"}))
    report = co.render_tables(tmp_path)
    assert "1 kernels timed, 1 skipped" in report
    assert "| s000 | gnu | 1.0000 | 1.0200 | 1.5000 | 1.500 | 1.020 |" in report
    assert "Geomean call overhead" in report and "1.5000x" in report
    assert "`bad`" in report  # skipped kernel listed, not silently dropped


def test_plot_calloverhead_smoke(tmp_path):
    """The plot reader (subprocess, as the drivers call it) tolerates a completed kernel, a skipped one,
    and a non-finite external time -> PNG + CSV land, the skipped kernel is dropped, non-finite is empty."""
    (tmp_path / "tsvc2_s000.json").write_text(
        json.dumps({
            "key": "s000",
            "compiler": "gnu",
            "inline_us": 1.0,
            "external_lto_us": 1.02,
            "external_us": 1.5,
            "call_overhead": 1.5,
            "lto_overhead": 1.02
        }))
    (tmp_path / "tsvc2_s112.json").write_text('{"key": "s112", "compiler": "gnu", "inline_us": 1.0, '
                                              '"external_us": Infinity, "external_lto_us": null, '
                                              '"call_overhead": null, "lto_overhead": null}')
    (tmp_path / "tsvc2_bad.json").write_text(json.dumps({"key": "bad", "skipped": "emit failed"}))

    subprocess.run([sys.executable, str(PLOT_CALLOVERHEAD), "--results-dir",
                    str(tmp_path)],
                   capture_output=True,
                   text=True,
                   check=True)

    assert (tmp_path / "calloverhead.png").exists()
    with (tmp_path / "calloverhead.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = {r["key"] for r in rows}
    assert "s000" in keys and "s112" in keys and "bad" not in keys
    s112 = next(r for r in rows if r["key"] == "s112")
    assert s112["external_us"] == "" and s112["call_overhead"] == ""  # non-finite -> empty, not a crash


def test_calloverhead_multinest_s152_sums_over_nests(tmp_path):
    """The full build+time path on a MULTI-nest kernel (s152 = 2 compute nests): it is measured (not
    skipped) and each variant's per-call time is the SUM over both nests, so the schema is unchanged and
    the overhead ratios are finite. Gated on a C compiler being present (mirrors the driver tests)."""
    tcs = discover_toolchains("gcc")
    if not tcs:
        pytest.skip("no gcc")
    cc, family = co.resolve_cc("gcc")
    k = tsvc.iter_tsvc_kernels(only=["s152"])[0]
    res = co.run_kernel(k, cc, family, "baseline", "S", inner=200, reps=3, workdir=tmp_path)
    assert "skipped" not in res, res.get("skipped")
    # inline is the sum over both nests' per-call cost -> a real positive time; the ratios divide finite by finite.
    assert res["inline_us"] and res["inline_us"] > 0.0
    assert res["external_us"] and res["external_us"] > 0.0
    assert res["call_overhead"] and res["call_overhead"] > 0.0
