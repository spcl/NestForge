# Copyright 2021 ETH Zurich and the Tactics authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""An extracted kernel's NumPy oracle: the Python function every build of the kernel is checked against."""

from __future__ import annotations

from dataclasses import dataclass

from tactics.ir.emit_python import (
    RuntimeSizedScratch,
    UnsupportedNest,
    kernel_args,
    kernel_arrays,
    oracle_sdfg,
    render,
)
from tactics.ir.extract import Boundary


@dataclass(frozen=True, slots=True)
class Prepared:
    """A kernel's oracle: ``def <name>(<args>)`` and the argument order (the kernel's ABI order) it takes."""

    name: str
    numpy_source: str
    args: tuple[str, ...]
    #: Why the body has no NumPy spelling (a C++ tasklet: a vectorized body), so builds of the kernel go unchecked
    #: and the whole program checks them; empty when ``numpy_source`` is the oracle.
    unspelled: str = ""
    #: The temporaries the kernel sizes while it runs (:class:`RuntimeSizedScratch`); the kernel allocates them.
    runtime_sized: tuple[str, ...] = ()


def prepare(boundary: Boundary, name: str) -> Prepared:
    """The NumPy oracle of one extracted kernel; one whose body NumPy cannot spell, or whose temporaries are sized
    while it runs, keeps its arguments, no source."""
    try:
        sdfg = oracle_sdfg(boundary)
    except RuntimeSizedScratch as err:
        args = kernel_args(boundary, kernel_arrays(boundary, boundary.standalone_sdfg, scratch=False))
        why = f"the kernel sizes {', '.join(err.arrays)} while it runs, so it has no NumPy oracle: {err}"
        return Prepared(name, "", tuple(args), unspelled=why, runtime_sized=err.arrays)
    args = kernel_args(boundary, kernel_arrays(boundary, sdfg))
    try:
        return Prepared(name, render(name, args, sdfg), tuple(args))
    except UnsupportedNest as err:
        return Prepared(name, "", tuple(args), unspelled=f"the kernel has no NumPy oracle: {err}")


def runtime_sized_scratch(boundary: Boundary) -> tuple[str, ...]:
    """The temporaries of the kernel ``boundary`` bounds that it sizes while it runs; empty when a caller can size
    every one."""
    try:
        oracle_sdfg(boundary)
    except RuntimeSizedScratch as err:
        return err.arrays
    return ()
