# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage 6, configuration sweep: build a kernel per compiler x FP mode x cost model, keep the fastest correct build."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from dace.codegen import cpf

from nestforge.build import flags
from nestforge.build.arena import make_inputs, run_oracle
from nestforge.build.toolchain import CudaToolchain, Toolchain, discover_cuda_toolchains, discover_toolchains
from nestforge.corpus.translate import Prepared
from nestforge.stages.kernel import (
    KernelSource,
    KernelVerdict,
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


def cpu_variants(compilers: Sequence[str] | None) -> list[Variant]:
    """The CPU cells of every toolchain on PATH, narrowed to the toolchain names in ``compilers``."""
    return enumerate_variants([tc for tc in discover_toolchains() if compilers is None or tc.name in compilers])


def gpu_variants(compilers: Sequence[str] | None) -> list[Variant]:
    """The GPU cells of every nvcc on PATH; ``compilers`` names one (``nvcc-13.1``) or all of them (``nvcc``)."""
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


def select_variant(
    src: KernelSource, prep: Prepared, sizes: dict[str, int], reps: int, variants: Sequence[Variant], out_dir: Path
) -> VariantResult:
    """Build every variant of ``src`` under ``out_dir``, time each against the NumPy oracle at its own FP rung, and
    return all cells with the fastest correct one as winner; a failed build is a cell without an archive."""
    inputs = make_inputs(src.boundary, sizes)
    oracle = run_oracle(prep, src.boundary, inputs, sizes)
    cells: list[VariantCell] = []
    for index, variant in enumerate(variants):
        try:
            archive = build_kernel_library(src, variant.compiler, list(variant.flags), out_dir / f"v{index}")
        except RuntimeError as err:
            cells.append(VariantCell(variant, failed_verdict(variant.fp_mode, str(err))))
            continue
        verdict = measure_kernel(archive, src, inputs, oracle, sizes, reps, variant.fp_mode)
        cells.append(VariantCell(variant, verdict, archive))
    correct = [cell for cell in cells if cell.verdict.ok]
    winner = min(correct, key=lambda cell: cell.verdict.time_us) if correct else None
    return VariantResult(cells, winner, src.symbol, list(src.abi_order))
