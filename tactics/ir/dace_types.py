# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A DaCe property whose annotation is looser than its values, narrowed for the type checker."""

from __future__ import annotations

from typing import cast


def strings(values: object) -> list[str]:
    """A ``ListProperty(element_type=str)``, which DaCe annotates as a list of ``type[str]``."""
    return cast(list[str], values)
