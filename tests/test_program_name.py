# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A program name CMake turns into a file name must stay under the 255 bytes a file name holds (#83)."""

from nestforge.build.sdfg import MAX_PROGRAM_NAME, short_program_name

CMAKE_FILE_PREFIX = "cmake_object_order_depends_target_dacestub_"
KERNEL = "conv2d_group_norm_tanh_hardswish_residual_add_logsumexp"
BENCH_NAME = f"hpcagent_bench_benchmarks_machine_learning_{KERNEL}_{KERNEL}_dace_{KERNEL}"


def test_the_program_name_of_the_issue_is_over_the_limit_before_and_under_it_after():
    assert len((CMAKE_FILE_PREFIX + BENCH_NAME).encode()) == 258
    assert len((CMAKE_FILE_PREFIX + short_program_name(BENCH_NAME)).encode()) <= 255


def test_a_name_that_fits_is_left_alone():
    name = "x" * MAX_PROGRAM_NAME
    assert short_program_name(name) == name


def test_the_shortened_name_is_deterministic_and_a_valid_identifier():
    short = short_program_name(BENCH_NAME)
    assert short == short_program_name(BENCH_NAME)
    assert short.isidentifier()
    assert len(short) == MAX_PROGRAM_NAME


def test_two_long_names_sharing_a_head_stay_apart():
    head = "h" * (MAX_PROGRAM_NAME + 10)
    assert short_program_name(head + "_a") != short_program_name(head + "_b")
