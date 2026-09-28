# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The repo-wide pytest setup in ``conftest.py``."""

from pathlib import Path

import dace


def test_each_test_process_compiles_into_its_own_dace_build_folder(tmp_path_factory):
    folder = Path(dace.config.Config.get("default_build_folder"))

    assert folder.is_relative_to(tmp_path_factory.getbasetemp())
