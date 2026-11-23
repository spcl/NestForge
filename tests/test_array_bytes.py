# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A transfer's bytes come from the session's sizes; a container the running program sizes has none (#96, gmres), and
then neither has the total."""

import dace

from tactics.stages.performance import array_bytes, offload_report
from tactics.stages.placement import Transfer


def test_a_container_sized_by_the_running_program_has_no_bytes_and_one_sized_by_the_sizes_has() -> None:
    sdfg = dace.SDFG("sizes")
    sdfg.add_array("h", ["m*(m + 1)"], dace.float64)
    sdfg.add_array("x", ["n"], dace.float64)

    assert array_bytes(sdfg, "h", {"n": 4}) is None
    assert array_bytes(sdfg, "x", {"n": 4}) == 32


def test_the_total_has_no_bytes_when_one_transfer_has_none() -> None:
    """Counting a transfer of unknown size as 0 made ``transfer_bytes`` smaller than the traffic, and nothing said so."""
    sdfg = dace.SDFG("totals")
    sdfg.add_array("h", ["m*(m + 1)"], dace.float64)
    sdfg.add_array("x", ["n"], dace.float64)
    to_gpu = Transfer("to_gpu", "x", "input", "extcall_0")
    devices = {"extcall_0": "gpu"}

    known = offload_report(sdfg, devices, (to_gpu,), {"n": 4})
    unknown = offload_report(sdfg, devices, (to_gpu, Transfer("to_gpu", "h", "input", "extcall_0")), {"n": 4})

    assert known.transfer_bytes == 32
    assert unknown.transfers is not None and [t.bytes for t in unknown.transfers] == [32, None]
    assert unknown.transfer_bytes is None
    assert offload_report(sdfg, devices, (), {"n": 4}).transfer_bytes == 0, "no transfer moves no bytes"
