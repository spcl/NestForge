# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Repo-local benchmark kernels in ``benchmarking_creation/``: discovery, loading and the module cache key."""

from pathlib import Path

import dace
import pytest

from nestforge.corpus import bench
from nestforge.corpus.bench import (
    LOCAL_TRACK,
    iter_dace_kernels,
    iter_local_kernels,
    local_names,
    module_path,
    preset_sizes,
)

SHIPPED = f"{LOCAL_TRACK}/cpu_gpu_flip/cpu_gpu_flip"
#: Every shipped kernel with the sizes of preset S in its manifest.
SHIPPED_S = {
    f"{LOCAL_TRACK}/cpu_gpu_flip/cpu_gpu_flip": {"nlev": 32, "ncol": 512},
    f"{LOCAL_TRACK}/sweep_scale/sweep_scale": {"NJ": 64, "NI": 64},
}


def test_local_names_lists_the_shipped_kernels() -> None:
    assert set(SHIPPED_S) <= set(local_names())


def test_local_names_ignores_a_manifest_that_does_not_match_its_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "foo").mkdir()
    (tmp_path / "foo" / "bar.yaml").write_text("name: bar\n")
    monkeypatch.setattr(bench, "LOCAL_ROOT", tmp_path)
    assert local_names() == {}


@pytest.mark.parametrize("short_name", sorted(SHIPPED_S))
def test_local_kernel_loads_as_a_dace_program(short_name: str) -> None:
    kernel = next(k for k in iter_dace_kernels(LOCAL_TRACK) if k.short_name == short_name)
    assert isinstance(kernel.program(), dace.frontend.python.parser.DaceProgram)
    name = short_name.rsplit("/", 1)[-1]
    assert kernel.to_sdfg(simplify=True).name.startswith(f"benchmarking_creation_{name}")
    assert preset_sizes(kernel, "S") == SHIPPED_S[short_name]


def test_all_tracks_include_the_local_kernels() -> None:
    assert SHIPPED in {k.short_name for k in iter_dace_kernels()}


def test_local_and_corpus_cache_keys_are_distinct() -> None:
    assert module_path(SHIPPED) == "benchmarking_creation.cpu_gpu_flip.cpu_gpu_flip_dace"
    assert module_path("loop_level_reasoning/fuse_diamond/fuse_diamond").startswith("hpcagent_bench.benchmarks.")


def test_a_local_kernel_without_its_committed_dace_file_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "foo").mkdir()
    (tmp_path / "foo" / "foo.yaml").write_text("name: foo\n")
    monkeypatch.setattr(bench, "LOCAL_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="emit_local_dace"):
        list(iter_local_kernels())
