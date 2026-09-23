"""Repo-wide pytest setup. With ``NESTFORGE_CI_NO_SKIP`` set, as in CI's unit job, a skipped test fails the run."""

import os
from collections.abc import Iterator

import pytest

from nestforge.corpus.bench import materialize_dace_corpus
from nestforge.ir.libnode import ExternLibEnv


def pytest_configure(config: pytest.Config) -> None:
    """Generate the corpus's gitignored ``_dace.py`` files once, in the controller before xdist workers start, since
    concurrent workers would race the write."""
    if vars(config).get("workerinput") is None:
        materialize_dace_corpus()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if "NESTFORGE_CI_NO_SKIP" not in os.environ:
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    skipped = reporter.stats.get("skipped", []) if reporter is not None else []
    if reporter is None or not skipped:
        return
    reporter.write_line(f"NESTFORGE_CI_NO_SKIP: {len(skipped)} skipped test(s) not allowed in the unit set:")
    for report in skipped:
        reporter.write_line(f"  SKIPPED {report.nodeid}")
    session.exitstatus = 1


@pytest.fixture(autouse=True)
def reset_extern_lib_env() -> Iterator[None]:
    """``ExternLibEnv`` is a process-wide class that accumulates link items, so without a reset a test's outcome
    depends on which tests ran before it."""
    ExternLibEnv.reset()
    yield
    ExternLibEnv.reset()
