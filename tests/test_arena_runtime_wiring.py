# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The MAIN arena timing cells must link the OpenMP runtime the machine actually supports (from
``support_matrix.cached_default_runtime``), not the static ``flags.DEFAULT_OPENMP_RUNTIME`` -- mirroring
the pluto lane. ``enumerate_cells`` reads the machine runtime ONCE and threads it into timing cells only;
the sequential correctness GATE cell stays runtime-free. Pure enumeration path (dummy Paths, no source I/O).
"""
from pathlib import Path

from nestforge.build import LIBGOMP, LIBOMP
from nestforge.perf import flags, support_matrix, tsvc_full
from nestforge.perf.tsvc_arena import Toolchain

# The gnu link spelling for a runtime's soname (flags.openmp_runtime_flags): libgomp -> -lgomp, default
# libomp -> -lomp, both pinned via --push-state,--no-as-needed. Discriminates "injected" from "default".
GNU_LIBGOMP = "-Wl,--push-state,--no-as-needed,-lgomp,--pop-state"
GNU_LIBOMP = "-Wl,--push-state,--no-as-needed,-lomp,--pop-state"


def synthetic_omp_ctx():
    """A lane-3 context with one C source plus its omp-emit source -- dummy Paths, never read. Same shape
    as test_matrix_degradation._synthetic_opt_ctx, trimmed to C and given ``omp_src`` so omp-emit fires."""
    return {
        "lang_src": {
            "c": (Path("x.c"), ["a", "N"], [None, None]),
        },
        "omp_src": {
            "c": (Path("x_omp.c"), ["a", "N"], [None, None]),
        },
        "symbol": "s000_fp64",
    }


def omp_axes():
    """One omp-emit parallel point, one cost, one FP mode, gate ON -> the C lane yields exactly one omp-emit
    timing cell (links the machine runtime) plus one sequential gate cell (links none)."""
    return {
        "opt_mode": "simplify-parallel",
        "parallelism": ["omp-emit"],
        "cost_models": ["default"],
        "fp_modes": ["default-fp"],
        "gate": True,
    }


def gnu_toolchain():
    return Toolchain("gcc", cc="gcc", cxx="g++", version=(13, 0), source="path")


def omp_emit_timing_flags(pendings, jobs):
    """Compile flags of the single gnu omp-emit C TIMING cell, resolved through its compile job."""
    cells = [
        p for p in pendings if p.cell.language == "c" and p.cell.parallel == "omp-emit" and p.cell.role == "timing"
        and p.compile_key is not None
    ]
    assert len(cells) == 1, f"expected exactly one gnu omp-emit C timing cell, got {len(cells)}"
    return jobs[cells[0].compile_key]["flags"]


def test_arena_timing_cell_links_injected_machine_runtime(tmp_path, monkeypatch):
    """Inject LIBGOMP as the discovered machine runtime: the gnu omp-emit timing cell must link -lgomp, not
    the static-default -lomp -- fails before ``openmp=machine_runtime`` is threaded into lane_flags."""
    monkeypatch.setattr(support_matrix, "cached_default_runtime", lambda *a, **k: LIBGOMP)
    pendings, jobs = tsvc_full.enumerate_cells(synthetic_omp_ctx(), [gnu_toolchain()], {}, omp_axes(), 4, flags.CXX_STD,
                                               tmp_path)
    cflags = omp_emit_timing_flags(pendings, jobs)
    assert GNU_LIBGOMP in cflags, cflags  # the injected runtime's soname is what the cell links
    assert GNU_LIBOMP not in cflags, cflags  # and it is NOT the static default


def test_arena_timing_cell_passes_the_reported_runtime_through(tmp_path, monkeypatch):
    """Inject the static default (LIBOMP): the same cell links -lomp, never -lgomp -- proves the wiring
    passes whatever cached_default_runtime reports THROUGH, rather than hard-coding gomp (an uncharacterised
    machine still behaves exactly as before)."""
    monkeypatch.setattr(support_matrix, "cached_default_runtime", lambda *a, **k: LIBOMP)
    pendings, jobs = tsvc_full.enumerate_cells(synthetic_omp_ctx(), [gnu_toolchain()], {}, omp_axes(), 4, flags.CXX_STD,
                                               tmp_path)
    cflags = omp_emit_timing_flags(pendings, jobs)
    assert GNU_LIBOMP in cflags, cflags
    assert GNU_LIBGOMP not in cflags, cflags


def test_arena_gate_cell_links_no_runtime(tmp_path, monkeypatch):
    """The sequential correctness GATE cell links NO OpenMP runtime whatever the machine runtime is --
    enumerate_cells leaves the gate lane_flags call runtime-free."""
    monkeypatch.setattr(support_matrix, "cached_default_runtime", lambda *a, **k: LIBGOMP)
    pendings, jobs = tsvc_full.enumerate_cells(synthetic_omp_ctx(), [gnu_toolchain()], {}, omp_axes(), 4, flags.CXX_STD,
                                               tmp_path)
    gate = [p for p in pendings if p.cell.language == "c" and p.cell.role == "gate" and p.compile_key is not None]
    assert len(gate) == 1, f"expected exactly one C gate cell, got {len(gate)}"
    gflags = jobs[gate[0].compile_key]["flags"]
    assert GNU_LIBGOMP not in gflags and GNU_LIBOMP not in gflags, gflags
