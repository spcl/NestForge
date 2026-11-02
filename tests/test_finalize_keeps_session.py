# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``finalize`` builds a copy: score and submit call it first, and the agent keeps working on the session after."""

from pathlib import Path

import pytest

from nestforge import session as session_module
from nestforge.corpus.bench import corpus_kernel, preset_sizes
from nestforge.session import Session, external_calls
from nestforge.stages.canonicalize import Targets
from nestforge.stages.finalize import BenchTarget

pytestmark = pytest.mark.e2e

KERNEL = "jacobi_1d"
PRESET = "S"


def bench_session(codegen: str) -> Session:
    kernel = corpus_kernel(KERNEL)
    assert kernel is not None, f"{KERNEL} is in the corpus"
    session = Session(kernel.to_sdfg(), Targets(gpu=False), sizes=preset_sizes(kernel, PRESET), codegen=codegen)
    session.bench = BenchTarget(kernel.short_name, "float64", False)
    session.canonicalize()
    return session


def test_finalize_leaves_the_moves_tree_and_run_defaults_scopes_after_it_works(tmp_path: Path) -> None:
    session = bench_session("first")
    tree, moves = session.describe(), session.list_moves()

    final = session.finalize(tmp_path / "final")

    assert final.library.exists() and final.kernels
    assert not external_calls(session.sdfg), "the kernels are the copy's"
    assert session.describe() == tree
    assert session.list_moves() == moves
    kernels = session.run_defaults("scopes")
    assert sorted(kernel.name for kernel in kernels) == sorted(final.kernels)
    assert all(ext.implementation != "ExternCall" for ext in external_calls(session.sdfg)), "built on the copy"


def test_a_second_finalize_relinks_the_swept_winners(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = bench_session("sweep")
    session.finalize(tmp_path / "first")
    swept = dict(session.swept)
    assert swept

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("an unchanged program is not swept again")

    monkeypatch.setattr(session_module, "select_variant", refuse)
    second = session.finalize(tmp_path / "second")

    assert second.library.exists() and session.swept == swept
