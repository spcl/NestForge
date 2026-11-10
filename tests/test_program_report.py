# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A whole-program time taken with no reference says so in its report, so it cannot be read as a pass (#85)."""

from nestforge.stages.performance import NO_REFERENCE, ProgramTime, timed_report


def test_a_program_that_ran_with_no_reference_is_marked_unchecked():
    report = timed_report(ProgramTime(time_us=12.5, mismatched=None))
    assert report.correct is None and report.unchecked == NO_REFERENCE and report.time_us == 12.5


def test_a_program_that_matched_its_reference_is_correct_and_not_marked():
    report = timed_report(ProgramTime(time_us=12.5, mismatched=[]))
    assert report.correct is True and report.unchecked == ""


def test_a_program_that_differs_from_its_reference_names_the_outputs():
    report = timed_report(ProgramTime(time_us=12.5, mismatched=["out"]))
    assert report.correct is False and report.mismatched == ["out"] and report.unchecked == ""
