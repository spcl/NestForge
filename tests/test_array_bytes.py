# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A transfer's bytes come from the session's sizes; a container the running program sizes has none (#96, gmres)."""

import dace

from nestforge.stages.performance import array_bytes


def test_a_container_sized_by_the_running_program_has_no_bytes_and_one_sized_by_the_sizes_has():
    sdfg = dace.SDFG("sizes")
    sdfg.add_array("h", ["m*(m + 1)"], dace.float64)
    sdfg.add_array("x", ["n"], dace.float64)

    assert array_bytes(sdfg, "h", {"n": 4}) is None
    assert array_bytes(sdfg, "x", {"n": 4}) == 32
