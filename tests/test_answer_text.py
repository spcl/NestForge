# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The answers the agent reads as lines (``plugin/nestforge/answers.py``): every fact of the record stays, the JSON
layout goes."""

import json
from collections.abc import Iterator
from types import ModuleType

import pytest
from hpcagent_agent.plugins import Plugin


@pytest.fixture(scope="module")
def answers(nestforge_plugin: tuple[Plugin, ...]) -> Iterator[ModuleType]:
    from hpcagent_bench_plugins.nestforge import answers  # importable once the bench loaded the plugin

    yield answers


@pytest.fixture(scope="module")
def defaults(nestforge_plugin: tuple[Plugin, ...]) -> dict[str, object]:
    """The defaults the agent side renders with (``tools/nestforge.py``)."""
    from hpcagent_bench_plugins.nestforge.wire import MOVE_DEFAULTS  # importable once the bench loaded the plugin

    return MOVE_DEFAULTS


def test_a_listed_move_is_its_kind_and_labels_then_the_arguments_it_was_checked_with(
    answers: ModuleType, defaults: dict[str, object]
) -> None:
    moves = [
        {"kind": "doall-fusion", "labels": ["doall0", "doall1"], "tile_sizes": []},
        {"kind": "tile-doall", "labels": ["doall3"], "tile_sizes": [32, 32], "max_iterations": 16},
        {"kind": "set-library-implementation", "labels": ["lib_gemm2"], "implementation": "MKL", "current": "pure"},
        {"kind": "relayout", "labels": [], "operation": {"action": "permute", "array": "A", "order": ["i1", "i0"]}},
        {"kind": "unroll-for", "labels": ["for4"], "max_iterations": 8},
    ]
    answer = answers.moves_answer({"tool": "list_moves", "result": moves, "performance": None}, defaults)
    expected = [
        "doall-fusion doall0 doall1",
        "tile-doall doall3 tile_sizes=[32,32]",
        "set-library-implementation lib_gemm2 current=pure implementation=MKL",
        'relayout operation={"action":"permute","array":"A","order":["i1","i0"]}',
        "unroll-for for4 max_iterations=8",
    ]
    assert answer == {"tool": "list_moves", "result": "\n".join(expected)}


def test_every_value_of_a_move_that_is_not_its_default_can_be_read_back_from_its_line(
    answers: ModuleType, defaults: dict[str, object]
) -> None:
    move = {"kind": "set-schedule-cpu", "labels": ["doall7"], "schedule": "dynamic", "chunk": 16, "threads": 8}
    head, label, *fields = answers.move_line(move, defaults).split(" ")
    read = dict(field.split("=", 1) for field in fields)
    assert (head, label) == ("set-schedule-cpu", "doall7")
    assert read == {"chunk": "16", "schedule": "dynamic", "threads": "8"}


def test_the_wire_defaults_are_those_apply_move_takes(defaults: dict[str, object]) -> None:
    assert defaults["tile_sizes"] == [] and defaults["max_iterations"] == 16 and defaults["isa"] == "auto"
    assert "kind" not in defaults and "labels" not in defaults


def test_no_moves_and_an_error_answer(answers: ModuleType, defaults: dict[str, object]) -> None:
    assert answers.moves_answer({"tool": "list_moves", "result": []}, defaults) == {
        "tool": "list_moves",
        "result": "none",
    }
    error = {"ok": False, "error": "Bad Request: list_moves: no such kind", "status": 400}
    assert answers.moves_answer(error, defaults) is error


def test_an_applied_list_keeps_status_reason_and_warning(answers: ModuleType) -> None:
    applied = [
        {"kind": "defaults-moves", "labels": [], "reason": "", "status": "applied"},
        {"kind": "doall-fission", "labels": ["doall1000"], "reason": "MapFission", "status": "applied"},
        {
            "kind": "for-to-doall",
            "labels": ["for3"],
            "reason": "LoopToMap",
            "status": "applied",
            "forced": True,
            "warning": "forced: the compiler did not prove this move legal.",
        },
        {
            "kind": "fuse",
            "labels": ["doall1", "doall2"],
            "reason": "kmin is read",
            "status": "refused",
            "refusal": "data",
        },
    ]
    expected = [
        "applied defaults-moves",
        "applied doall-fission doall1000: MapFission",
        "applied for-to-doall for3 forced=true: LoopToMap (warning: forced: the compiler did not prove this move "
        "legal.)",
        "refused fuse doall1 doall2 refusal=data: kmin is read",
    ]
    assert answers.results_answer({"tool": "applied_moves", "result": applied}) == {
        "tool": "applied_moves",
        "result": "\n".join(expected),
    }


def test_a_kind_is_its_rows_its_transformation_and_what_it_does(answers: ModuleType) -> None:
    kinds = [{"kind": "for-fusion", "rows": ["for", "for"], "runs": "LoopFusion", "does": "two loops become one."}]
    assert answers.kinds_answer({"tool": "list_move_kinds", "result": kinds, "performance": None}) == {
        "tool": "list_move_kinds",
        "result": "for-fusion [for for] LoopFusion: two loops become one.",
    }


def test_the_lines_are_shorter_than_the_json_the_bench_prints(answers: ModuleType, defaults: dict[str, object]) -> None:
    moves = [{"kind": "doall-fusion", "labels": [f"doall{i}", f"doall{i + 1}"], "tile_sizes": []} for i in range(20)]
    answer = {"tool": "list_moves", "result": moves, "performance": None}
    printed = json.dumps(answer, indent=2, sort_keys=True)
    rendered = json.dumps(answers.moves_answer(answer, defaults), indent=2, sort_keys=True)
    assert len(rendered) < len(printed) / 3


def test_a_score_reads_speedup_and_correctness_first_and_keeps_every_measured_field(answers: ModuleType) -> None:
    grade = {
        "baseline": "numba",
        "baseline_ns": 8720228534,
        "build_ok": True,
        "cells": [
            {
                "baseline": "numba",
                "correct": False,
                "label": "fuzzed:submit",
                "ratio": 0.5071802933643264,
                "shape": '{"dt": 60.0, "ke": 90}',
                "timing_reduction": "mwd-v3",
            },
        ],
        "correct": False,
        "detail": "vs numba: pre_gsp: numeric mismatch: 2 of 1415224 elements",
        "grading_protocol": "sealed-nonce-v1+host-monotonic",
        "harness_fault": False,
        "max_rel_error": 4.5781492697358144e-07,
        "seed_nonce": 1,
        "speedup": 0.5071802933643264,
        "speedups": {"numba": 0.14784201711876474},
        "timed_out": False,
        "timing_reduction": "mwd-v3",
        "weak_efficiency": None,
    }
    expected = [
        "speedup=0.507 correct=false build_ok=true timed_out=false",
        "detail: vs numba: pre_gsp: numeric mismatch: 2 of 1415224 elements",
        'baseline=numba baseline_ns=8720228534 harness_fault=false max_rel_error=4.58e-07 speedups={"numba":0.148}',
        'cell label=fuzzed:submit ratio=0.507 shape={"dt":60.0,"ke":90}',
    ]
    assert answers.score_answer(grade) == {"text": "\n".join(expected)}


def test_a_performance_report_is_hints_kernels_and_program_and_says_where_data_moves_only_off_the_cpu(
    answers: ModuleType,
) -> None:
    kernel = {"name": "k0", "device": "cpu", "correct": True, "time_us": 81323.94, "share": 0.77, "error": ""}
    report = {
        "hints": "kernel times: k0 81323.9 us",
        "kernels": [kernel],
        "offloading": {"devices": {"k0": "cpu"}, "transfers": [], "transfer_bytes": 0},
        "structured": {"time_us": 105612.3, "correct": True, "mismatched": [], "error": ""},
    }
    answer = {"tool": "get_performance", "result": None, "performance": report}
    expected = [
        "kernel times: k0 81323.9 us",
        "kernel k0 cpu correct=true share=0.77 time_us=8.13e+04",
        "program correct=true time_us=1.06e+05",
    ]
    assert answers.performance_answer(answer) == {"tool": "get_performance", "text": "\n".join(expected)}

    moved = {"direction": "h2d", "arg": "A", "producer": "host", "consumer": "k0", "bytes": 4096}
    report["offloading"] = {"devices": {"k0": "gpu"}, "transfers": [moved], "transfer_bytes": 4096}
    lines = answers.performance_answer(answer)["text"].splitlines()
    assert lines[-2:] == [
        'offloading devices={"k0":"gpu"} transfer_bytes=4096',
        "transfer arg=A bytes=4096 consumer=k0 direction=h2d producer=host",
    ]


def test_a_profile_keeps_its_report_text_and_drops_the_structured_copy(answers: ModuleType) -> None:
    report = {
        "build_ok": True,
        "threads": 24,
        "text": "hardware counters ...",
        "counters": {"cycles": 9},
        "hotspots": [],
    }
    assert answers.profile_answer(report) == {"build_ok": True, "threads": 24, "text": "hardware counters ..."}
    without_text = {"result": {"library": "/shared/lib.so"}, "profiled": True}
    assert answers.profile_answer(without_text) is without_text


def test_a_failed_measurement_reaches_the_agent_as_it_came(answers: ModuleType) -> None:
    timed_out = {"ok": False, "timed_out": True, "error": "The judge did not answer within 1800s"}
    assert answers.score_answer(timed_out) is timed_out
    assert answers.profile_answer(timed_out) is timed_out
    assert answers.performance_answer(timed_out) is timed_out
