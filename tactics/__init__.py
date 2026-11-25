# Copyright 2021 ETH Zurich and the Tactics authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tactics: staged whole-program optimization over DaCe SDFGs, driven through one :class:`Session`."""

# Must precede any ``dace.transformation.interstate`` import: extended's canonicalize -> vectorization ->
# interstate import cycle only resolves when ``passes`` loads first.
import dace.transformation.passes  # noqa: F401

from tactics.session import Session

__all__ = ["Session"]
