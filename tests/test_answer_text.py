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
    assert answer == {"text": "\n".join(expected)}


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
    assert answers.moves_answer({"tool": "list_moves", "result": []}, defaults) == {"text": "none"}
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
    assert answers.results_answer({"tool": "applied_moves", "result": applied}) == {"text": "\n".join(expected)}


def test_a_kind_is_its_rows_its_transformation_and_what_it_does(answers: ModuleType) -> None:
    kinds = [{"kind": "for-fusion", "rows": ["for", "for"], "runs": "LoopFusion", "does": "two loops become one."}]
    assert answers.kinds_answer({"tool": "list_move_kinds", "result": kinds, "performance": None}) == {
        "text": "for-fusion [for for] LoopFusion: two loops become one.",
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
        "baseline=numba baseline_time=8.72s harness_fault=false max_rel_error=4.58e-07 speedups=numba:0.148",
        "cells:",
        "| label | ratio | shape |",
        "| --- | --- | --- |",
        "| fuzzed:submit | 0.507 | dt:60,ke:90 |",
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
        "kernels:",
        "| name | device | correct | share | time |",
        "| --- | --- | --- | --- | --- |",
        "| k0 | cpu | true | 0.77 | 81.3ms |",
        "program correct=true time=106ms",
    ]
    assert answers.performance_answer(answer) == {"text": "\n".join(expected)}

    moved = {"direction": "h2d", "arg": "A", "producer": "host", "consumer": "k0", "bytes": 4096}
    report["offloading"] = {"devices": {"k0": "gpu"}, "transfers": [moved], "transfer_bytes": 4096}
    lines = answers.performance_answer(answer)["text"].splitlines()
    assert lines[-5:] == [
        "offloading devices=k0:gpu transfer_bytes=4096",
        "transfers:",
        "| arg | direction | bytes | consumer | producer |",
        "| --- | --- | --- | --- | --- |",
        "| A | h2d | 4096 | k0 | host |",
    ]


def test_a_profile_keeps_its_report_text_and_drops_the_structured_copy(answers: ModuleType) -> None:
    report = {
        "build_ok": True,
        "kernel": "s231",
        "threads": 24,
        "text": "hardware counters ...",
        "counters": {"cycles": 9},
        "hotspots": [],
    }
    assert answers.profile_answer(report) == {"text": "build_ok=true threads=24\nhardware counters ..."}
    without_text = {"result": {"library": "/shared/lib.so"}, "profiled": True}
    assert answers.profile_answer(without_text) is without_text


def test_a_failed_measurement_reaches_the_agent_as_it_came(answers: ModuleType) -> None:
    timed_out = {"ok": False, "timed_out": True, "error": "The judge did not answer within 1800s"}
    assert answers.score_answer(timed_out) is timed_out
    assert answers.profile_answer(timed_out) is timed_out
    assert answers.performance_answer(timed_out) is timed_out


def test_a_score_leaves_out_the_kernel_and_hidden_counts_of_zero_but_keeps_the_language(answers: ModuleType) -> None:
    grade = {
        "speedup": 1.5,
        "correct": True,
        "kernel": "s231",
        "language": "cpp",
        "hidden_total": 0,
        "hidden_passed": 0,
        "hidden_correct": True,
        "native_ns": 1349396944,
    }
    assert answers.score_answer(grade)["text"].splitlines() == [
        "speedup=1.5 correct=true",
        "language=cpp native_time=1.35s",
    ]
    grade["hidden_total"] = grade["hidden_passed"] = 3
    assert "hidden_correct=true hidden_passed=3 hidden_total=3" in answers.score_answer(grade)["text"]


def test_times_read_in_one_unit_at_three_digits(answers: ModuleType) -> None:
    assert [answers.duration(seconds) for seconds in (1.755709786, 0.893448152, 8.13e-05, 4e-08, 0.9996)] == [
        "1.76s",
        "893ms",
        "81.3us",
        "40ns",
        "1s",
    ]
    assert answers.measured("baselines", {"c": 1391490146, "numba": 877854893}) == ("baselines", "c:1.39s,numba:878ms")


def test_the_cells_say_once_what_every_cell_shares(answers: ModuleType) -> None:
    cells = [
        {"label": "cfg0:large0", "graded": True, "ratio": 0.5088814558786084, "baseline": "numba"},
        {"label": "cfg0:large3", "graded": True, "ratio": 0.5167499458015494, "baseline": "numba"},
    ]
    text = answers.score_answer({"speedup": 0.51, "baseline": "numba", "cells": cells})["text"]
    assert text.splitlines()[-5:] == [
        "cells: graded=true",
        "| label | ratio |",
        "| --- | --- |",
        "| cfg0:large0 | 0.509 |",
        "| cfg0:large3 | 0.517 |",
    ]


def test_a_measurement_answer_has_no_quotes_and_a_value_with_spaces_gets_its_own_line(answers: ModuleType) -> None:
    grade = {"speedup": 1.0, "baselines": {"c": 10}, "shape": '{"N": 4}', "note": "two words"}
    text = answers.score_answer(grade)["text"]
    assert '"' not in text
    assert text.splitlines()[-2:] == ["baselines=c:10ns shape=N:4", "note: two words"]
