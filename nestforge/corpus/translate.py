# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""An extracted kernel's NumPy oracle: the Python function every build of the kernel is checked against."""

from __future__ import annotations

from dataclasses import dataclass

from nestforge.ir.emit_python import kernel_args, kernel_arrays, oracle_sdfg, render
from nestforge.ir.extract import Boundary


@dataclass(frozen=True, slots=True)
class Prepared:
    """A kernel's oracle: ``def <name>(<args>)`` and the argument order (the kernel's ABI order) it takes."""

    name: str
    numpy_source: str
    args: tuple[str, ...]


def prepare(boundary: Boundary, name: str) -> Prepared:
    """The NumPy oracle of one extracted kernel."""
    sdfg = oracle_sdfg(boundary)
    args = kernel_args(boundary, kernel_arrays(boundary, sdfg))
    return Prepared(name, render(name, args, sdfg), tuple(args))
