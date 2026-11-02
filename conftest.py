# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Repo-wide pytest setup. With ``NESTFORGE_CI_NO_SKIP`` set, as in CI's unit job, a skipped test fails the run."""

import os
from collections.abc import Iterator
from typing import TYPE_CHECKING

import dace
import pytest

from nestforge.corpus.bench import materialize_dace_corpus

if TYPE_CHECKING:
    import numpy as np

    from nestforge.session import Session
    from nestforge.stages.performance import Reference


def pytest_configure(config: pytest.Config) -> None:
    """Generate the corpus's gitignored ``_dace.py`` files once, in the controller before xdist workers start, since
    concurrent workers would race the write."""
    if vars(config).get("workerinput") is None:
        materialize_dace_corpus()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if "NESTFORGE_CI_NO_SKIP" not in os.environ:
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:
        return
    skipped = reporter.stats.get("skipped", [])
    if not skipped:
        return
    reporter.write_line(f"NESTFORGE_CI_NO_SKIP: {len(skipped)} skipped test(s) not allowed in the unit set:")
    for report in skipped:
        reporter.write_line(f"  SKIPPED {report.nodeid}")
    session.exitstatus = 1


@pytest.fixture(autouse=True, scope="session")
def private_dace_build_folder(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Each xdist worker compiles SDFGs into its own folder: two workers building a same-named program in the shared
    ``.dacecache`` overwrite each other's objects mid-build."""
    with dace.config.set_temporary("default_build_folder", value=str(tmp_path_factory.mktemp("dacecache"))):
        yield


@pytest.fixture(scope="session")
def cloudsc_monolith() -> tuple["Session", dict[str, "np.ndarray"], "Reference"]:
    """The bench's cloudsc_monolith canonicalized as an agent session starts it (preset S, float64, CPU), with the
    bench's inputs and reference; built once, as canonicalization takes minutes. Tests work on a ``trial()``."""
    from nestforge.server import Setup, start

    session, inputs, reference = start(Setup(kernel="cloudsc_monolith", preset="S", datatype="float64", gpu=False))
    assert reference is not None, "cloudsc_monolith lost its bench reference"
    return session, inputs, reference
