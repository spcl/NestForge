# Copyright 2021 ETH Zurich and the Tactics authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage 6, configuration sweep: build a kernel per compiler x FP mode x cost model, keep the fastest correct build."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from dace.codegen import cpf
from dace.codegen.common import get_gpu_backend

from tactics.build import flags
from tactics.build.toolchain import (
    CudaToolchain,
    HipToolchain,
    Toolchain,
    discover_cuda_toolchains,
    discover_hip_toolchains,
    discover_toolchains,
)
from tactics.stages.kernel import (
    KernelSource,
    KernelVerdict,
    Probe,
    build_kernel_library,
    failed_verdict,
    measure_kernel,
)


@dataclass(frozen=True, slots=True)
class Variant:
    """One sweep cell: a compiler, the compile flags its axes compose to, and the toolchain name reports use."""

    compiler: str
    fp_mode: str
    cost_model: str
    flags: tuple[str, ...]
    toolchain: str

    @property
    def label(self) -> str:
        return f"{self.toolchain}:{self.fp_mode}:{self.cost_model}"


@dataclass(slots=True)
class VariantCell:
    """A variant's verdict, and its archive unless the build failed."""

    variant: Variant
    verdict: KernelVerdict
    archive: Path | None = None


@dataclass(slots=True)
class VariantResult:
    """Every cell and the fastest correct cell with the entry it links through."""

    cells: list[VariantCell]
    winner: VariantCell | None
    symbol: str
    abi_order: list[str]

    @property
    def library(self) -> Path | None:
        return self.winner.archive if self.winner is not None else None


def enumerate_variants(toolchains: Sequence[Toolchain]) -> list[Variant]:
    """Every compiler x FP mode x cost model cell the toolchains support, one per distinct flag set."""
    variants: dict[tuple[str, str, tuple[str, ...]], Variant] = {}
    for tc in toolchains:
        # flag_matrix already dedups a cost model the family has no knob for onto its default flags
        for fp_mode, cost_model, composed in flags.flag_matrix(tc.fp_family):
            variants.setdefault(
                (tc.cxx, fp_mode, tuple(composed)),
                Variant(tc.cxx, fp_mode, cost_model, tuple(composed), Path(tc.cxx).name),
            )
    return list(variants.values())


def enumerate_cuda_variants(toolchains: Sequence[CudaToolchain]) -> list[Variant]:
    """Every nvcc x GPU FP rung cell; a GPU cell has no cost model to sweep."""
    return [
        Variant(tc.nvcc, fp_mode, flags.NO_COST_MODEL, tuple(composed), tc.name)
        for tc in toolchains
        for fp_mode, composed in flags.cuda_flag_matrix(cpf.CUDA_BUILD_FLAGS)
    ]


def enumerate_hip_variants(toolchains: Sequence[HipToolchain]) -> list[Variant]:
    """Every hipcc x GPU FP rung cell; a GPU cell has no cost model to sweep."""
    return [
        Variant(tc.hipcc, fp_mode, flags.NO_COST_MODEL, tuple(composed), tc.name)
        for tc in toolchains
        for fp_mode, composed in flags.hip_flag_matrix()
    ]


def cpu_variants(compilers: Sequence[str] | None) -> list[Variant]:
    """The CPU cells of every toolchain on PATH, narrowed to the toolchain names in ``compilers``."""
    return enumerate_variants([tc for tc in discover_toolchains() if compilers is None or tc.name in compilers])


def gpu_variants(compilers: Sequence[str] | None) -> list[Variant]:
    """The GPU cells of every compiler on PATH for this machine's GPU backend: nvcc for CUDA, hipcc for HIP.
    ``compilers`` names one (``nvcc-13.1``, ``hipcc-6.3``) or all of a kind (``nvcc``, ``hipcc``)."""
    if get_gpu_backend() == "hip":
        hipccs = discover_hip_toolchains()
        return enumerate_hip_variants(
            [tc for tc in hipccs if compilers is None or "hipcc" in compilers or tc.name in compilers]
        )
    nvccs = discover_cuda_toolchains()
    return enumerate_cuda_variants(
        [tc for tc in nvccs if compilers is None or "nvcc" in compilers or tc.name in compilers]
    )


VARIANTS_BY_DEVICE: dict[str, Callable[[Sequence[str] | None], list[Variant]]] = {
    "cpu": cpu_variants,
    "gpu": gpu_variants,
}


def device_variants(device: str, compilers: Sequence[str] | None = None) -> list[Variant]:
    """Every sweep cell this machine offers for a kernel on ``device``."""
    return VARIANTS_BY_DEVICE[device](compilers)


def build_variant(src: KernelSource, variant: Variant, out_dir: Path) -> Path | str:
    """``variant``'s build of ``src`` under ``out_dir``: its archive, or the error of a failed build."""
    try:
        return build_kernel_library(src, variant.compiler, list(variant.flags), out_dir)
    except RuntimeError as err:
        return str(err)


def measure_variant(
    src: KernelSource, probe: Probe, reps: int, variant: Variant, out_dir: Path, built: Path | str
) -> VariantCell:
    """One sweep cell: ``variant``'s build ``built`` of ``src`` under ``out_dir``, measured."""
    if isinstance(built, str):
        return VariantCell(variant, failed_verdict(variant.fp_mode, built))
    return VariantCell(variant, measure_kernel(built, src, probe, reps, variant.fp_mode), built)


def build_workers() -> int:
    """Compilers a sweep runs at once: one per core this process may use."""
    return len(os.sched_getaffinity(0))


def select_variant(
    src: KernelSource,
    probe: Probe,
    reps: int,
    variants: Sequence[Variant],
    out_dir: Path,
    done: dict[Variant, VariantCell] | None = None,
) -> VariantResult:
    """Build every variant of ``src`` under ``out_dir``, time each against the NumPy oracle at its own FP rung, and
    return all cells with the fastest correct one as winner; a failed build is a cell without an archive. The builds
    run in parallel, the timings one after another, so no timed run shares the machine with a compiler.

    :param done: Cells of an earlier sweep of the same source that was stopped; each new cell joins them, so the next
        sweep goes on from there."""
    finished = done if done is not None else {}
    todo = [variant for variant in dict.fromkeys(variants) if variant not in finished]
    dirs = [out_dir / f"v{len(finished) + i}" for i in range(len(todo))]
    pool = ThreadPoolExecutor(build_workers())
    try:
        builds = list(pool.map(build_variant, [src] * len(todo), todo, dirs))
    finally:
        # a stop lets the running compilers finish in the background and starts no other
        pool.shutdown(wait=False, cancel_futures=True)
    for variant, folder, built in zip(todo, dirs, builds):
        finished[variant] = measure_variant(src, probe, reps, variant, folder, built)
    cells = [finished[variant] for variant in variants]
    correct = [cell for cell in cells if cell.verdict.ok]
    winner = min(correct, key=lambda cell: cell.verdict.time_us) if correct else None
    return VariantResult(cells, winner, src.symbol, list(src.abi_order))
