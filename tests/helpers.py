# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Helpers several test modules share."""

import numpy as np

from nestforge.corpus.bench import CorpusKernel, iter_dace_kernels


def corpus_kernel(short_name: str) -> CorpusKernel:
    """The corpus kernel named ``short_name`` (``track/.../module``)."""
    return {k.short_name: k for k in iter_dace_kernels()}[short_name]


def loop_level_kernel(key: str) -> CorpusKernel:
    """The loop_level_reasoning kernel whose module is ``key``."""
    for kernel in iter_dace_kernels("loop_level_reasoning"):
        if kernel.short_name.rsplit("/", 1)[-1] == key:
            return kernel
    raise AssertionError(f"{key} is not in the loop_level_reasoning track; the corpus this test pins has changed")


def run(sdfg, inputs: dict[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
    """Run ``sdfg`` on copies of ``inputs`` with ``N=n``; returns the buffers after the call."""
    bufs = {k: v.copy() for k, v in inputs.items()}
    sdfg(**bufs, N=n)
    return bufs


def random_vectors(n: int = 48, names: tuple[str, ...] = ("a", "b", "c"), seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {k: rng.random(n) for k in names}
