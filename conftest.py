"""Repo-root pytest config.

When NESTFORGE_CI_NO_SKIP is set (CI unit set), a skipped test is a failure: the unit set must run
with zero skips. Locally the env var is unset, so skips behave normally.
"""
import os


def pytest_configure(config):
    """Materialise the corpus's gitignored ``_dace.py`` ONCE, before collection. optarena regenerates
    them on demand (they are never committed), so a fresh checkout has none and the corpus tests would
    KeyError. Doing it here -- in the xdist CONTROLLER only, before workers fan out -- keeps parallel
    workers from racing on the non-atomic generate-and-write. Best-effort: if optarena/dace is not
    importable, the corpus tests report that themselves."""
    if vars(config).get("workerinput") is not None:
        return  # xdist worker: the controller already materialised the corpus
    try:
        from nestforge.corpus import materialize_dace_corpus
        materialize_dace_corpus()
    except Exception:
        pass


def pytest_sessionfinish(session, exitstatus):
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
