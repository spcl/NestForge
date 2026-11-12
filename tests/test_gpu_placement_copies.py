# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The host/device copies placement inserts between two kernels, read off the SDFG without a GPU: a value passed
between two GPU kernels stays on the device, and a value crossing devices is copied once, in its direction."""

from pathlib import Path

import dace
import numpy as np
import pytest
from dace.libraries.standard.helper import GPU_RESIDENT_STORAGES

from nestforge.session import Session
from nestforge.stages.canonicalize import Targets

N = dace.symbol("N", dtype=dace.int64)


@dace.program
def chain(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    t = np.empty_like(a)
    for i in dace.map[0:N]:
        t[i] = a[i] + b[i]
    for i in dace.map[0:N]:
        c[i] = t[i] * 2.0


#: Devices of ``chain``'s two kernels, producer first; the ``(source, destination)`` copies placement inserts, to the
#: device then to the host; and the values the kernel DAG says cross, to the device then to the host.
PLACEMENTS = [
    (("gpu", "gpu"), [("a", "a_gpu"), ("b", "b_gpu")], [("c_gpu", "c")], ["a", "b"], ["c"]),
    (("cpu", "gpu"), [("t", "t_gpu")], [("c_gpu", "c")], ["t"], ["c"]),
    (("gpu", "cpu"), [("a", "a_gpu"), ("b", "b_gpu")], [("t", "t_host")], ["a", "b"], ["t"]),
    (("cpu", "cpu"), [], [], [], []),
]


@pytest.mark.parametrize(
    ("devices", "to_device", "to_host", "crossing_in", "crossing_out"),
    PLACEMENTS,
    ids=["-".join(p[0]) for p in PLACEMENTS],
)
def test_placing_two_chained_kernels_copies_each_crossing_value_once_and_nothing_between_two_gpu_kernels(
    tmp_path, devices, to_device, to_host, crossing_in, crossing_out
):
    session = Session(chain.to_sdfg(simplify=True), targets=Targets(gpu=True), work_dir=str(tmp_path))
    producer, consumer = (kernel.name for kernel in session.default_scopes())

    result = session.place(dict(zip((producer, consumer), devices)))

    arrays = session.sdfg.arrays
    assert result.devices == {producer: devices[0], consumer: devices[1]}
    assert sorted(result.copies) == sorted(to_device + to_host)
    assert all(arrays[dst].storage in GPU_RESIDENT_STORAGES for _, dst in to_device)
    assert not any(arrays[dst].storage in GPU_RESIDENT_STORAGES for _, dst in to_host)
    assert sorted(m.arg for m in result.transfers if m.direction == "to_gpu") == crossing_in
    assert sorted(m.arg for m in result.transfers if m.direction == "to_host") == crossing_out
    assert (arrays["t"].storage in GPU_RESIDENT_STORAGES) == (devices[0] == "gpu"), "t lives where it is written"
    session.sdfg.validate()


if __name__ == "__main__":
    import tempfile

    for placement in PLACEMENTS:
        test_placing_two_chained_kernels_copies_each_crossing_value_once_and_nothing_between_two_gpu_kernels(
            Path(tempfile.mkdtemp()), *placement
        )
