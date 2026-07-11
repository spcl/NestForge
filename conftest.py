"""Repo-root pytest config.

When NESTFORGE_CI_NO_SKIP is set (CI unit set), a skipped test is a failure: the unit set must run
with zero skips. Locally the env var is unset, so skips behave normally.
"""
import os


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
