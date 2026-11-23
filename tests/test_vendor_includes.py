# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A kernel calling a vendor library whose headers sit off the default search path (a module or spack OpenBLAS)
compiles with that library's ``-I``: on such a machine expanding one of pagerank's or atax's calls left a kernel
calling CBLAS that did not build (``cblas.h: No such file or directory``)."""

from pathlib import Path
from typing import cast

import pytest

from nestforge.build.sdfg import BuildOptions
from nestforge.ir.extract import Boundary
from nestforge.stages import kernel as kernel_stage
from nestforge.stages.kernel import KernelSource, build_kernel_library

#: An include directory off the default search path.
INCLUDE = "-I/opt/openblas/include"


def test_the_include_flags_follow_whatever_flags_a_build_takes() -> None:
    """The sweep passes its own flags; the vendor ``-I`` is added to those as to the defaults."""
    swept = BuildOptions(flags=["-O2"], includes=[INCLUDE]).resolved_flags()
    default = BuildOptions(includes=[INCLUDE]).resolved_flags()
    assert swept[0] == "-O2" and swept[-1] == INCLUDE
    assert default[-1] == INCLUDE


def test_a_kernel_source_builds_with_its_include_flags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    compiled: list[list[str]] = []

    def archive(units: object, folder: object, out: Path, shared: Path, opts: BuildOptions) -> None:
        compiled.append(opts.resolved_flags())

    monkeypatch.setattr(kernel_stage, "build_archive", archive)
    unit = tmp_path / "k.cpp"
    src = KernelSource("k", unit, [], cast(Boundary, None), "cpu", link=["-lopenblas"], includes=[INCLUDE])

    build_kernel_library(src, "g++", ["-O3"], tmp_path)

    (flags,) = compiled
    assert flags[0] == "-O3" and flags[-1] == INCLUDE, flags
