# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The one hardcoded location: the repository root, derived from the package."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
