# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A GPU session's call narrows the visible GPUs to its slot's GPU only on a GPU slot: a CPU slot's index is a CPU
slot ordinal, and narrowing to it left a one-GPU judge's GPU target seeing no GPU. Read off the environment."""

import pytest
from hpcagent_bench.harness.judge_scheduler import DeviceSlot
from test_session_process import report_pin

from nestforge.server import SessionProcess, Setup

SETUP = Setup(kernel="none", preset="S", datatype="float64", gpu=True)


def test_a_gpu_session_sees_only_its_slots_gpu_on_a_gpu_slot_and_every_gpu_on_a_cpu_slot(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6")
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("DACE_compiler_cuda_backend", "cuda")
    monkeypatch.setenv("HPCAGENT_BENCH_JUDGE_GPUS_PER_NODE", "0")
    monkeypatch.setenv("HPCAGENT_BENCH_JUDGE_CPU_SLOTS_PER_NODE", "2")
    session = SessionProcess(SETUP, target=report_pin)

    on_gpu_slot = session.call("get_performance", {}, slot=DeviceSlot("gpu", 1)).result
    on_cpu_slot = session.call("get_performance", {}, slot=DeviceSlot("cpu", 1)).result
    unpinned = session.call("view_tree", {}).result
    session.close()

    for forked, spawned in (on_gpu_slot, on_cpu_slot, unpinned):
        assert forked == spawned, "a fork and a spawn inherit the same devices"
    assert on_gpu_slot[0]["CUDA_VISIBLE_DEVICES"] == "5", "the slot's GPU, a position in the inherited list"
    assert on_cpu_slot[0]["CUDA_VISIBLE_DEVICES"] == "4,5,6"
    assert unpinned[0]["CUDA_VISIBLE_DEVICES"] == "4,5,6"


if __name__ == "__main__":
    with pytest.MonkeyPatch.context() as patch:
        test_a_gpu_session_sees_only_its_slots_gpu_on_a_gpu_slot_and_every_gpu_on_a_cpu_slot(patch)
