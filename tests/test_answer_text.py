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

    return dict(MOVE_DEFAULTS)


def pairs(line: str) -> dict[str, str]:
    """The ``name=value`` fields of ``line``, by name; its other words left out."""
    return dict(word.split("=", 1) for word in line.split() if "=" in word)


def table_rows(lines: list[str]) -> list[dict[str, str]]:
    """A markdown table's rows (``lines``: its header, the rule, then the rows) as one dict per row, by column."""
    columns = [name.strip() for name in lines[0].strip("|").split("|")]
    return [dict(zip(columns, (cell.strip() for cell in row.strip("|").split("|")), strict=True)) for row in lines[2:]]


def table_under(text: str, title: str) -> tuple[dict[str, str], list[dict[str, str]]]:
    """The table under the ``title:`` line of ``text``: the values its title line says once, and its rows by column."""
    lines = text.splitlines()
    start = next(index for index, line in enumerate(lines) if line.split(" ")[0] == f"{title}:")
    end = start + 1
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    return pairs(lines[start]), table_rows(lines[start + 1 : end])


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
    read = [([word for word in line.split() if "=" not in word], pairs(line)) for line in answer["text"].splitlines()]
    assert set(answer) == {"text"}
    assert read[0] == (["doall-fusion", "doall0", "doall1"], {})
    assert read[1] == (["tile-doall", "doall3"], {"tile_sizes": "[32,32]"})  # max_iterations is its default
    assert read[2] == (["set-library-implementation", "lib_gemm2"], {"current": "pure", "implementation": "MKL"})
    assert read[3][0] == ["relayout"] and json.loads(read[3][1]["operation"]) == {
        "action": "permute",
        "array": "A",
        "order": ["i1", "i0"],
    }
    assert read[4] == (["unroll-for", "for4"], {"max_iterations": "8"})


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
    lines = answers.results_answer({"tool": "applied_moves", "result": applied})["text"].splitlines()
    read = [(head.split(), said) for head, _, said in (line.partition(": ") for line in lines)]
    assert read == [
        (["applied", "defaults-moves"], ""),
        (["applied", "doall-fission", "doall1000"], "MapFission"),
        (
            ["applied", "for-to-doall", "for3", "forced=true"],
            "LoopToMap (warning: forced: the compiler did not prove this move legal.)",
        ),
        (["refused", "fuse", "doall1", "doall2", "refusal=data"], "kmin is read"),
    ]


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
    text = answers.score_answer(grade)["text"]
    head, detail, other, *_ = text.splitlines()
    assert pairs(head) == {"speedup": "0.507", "correct": "false", "build_ok": "true", "timed_out": "false"}
    assert detail == f"detail: {grade['detail']}"
    assert pairs(other) == {
        "baseline": "numba",
        "baseline_time": "8.72s",
        "harness_fault": "false",
        "max_rel_error": "4.58e-07",
        "speedups": "numba:0.148",
    }
    # the grading protocol and an unset field go; a cell leaves out what the grade already says
    assert not any(name in text for name in ("grading_protocol", "seed_nonce", "timing_reduction", "weak_efficiency"))
    assert table_under(text, "cells") == ({}, [{"label": "fuzzed:submit", "ratio": "0.507", "shape": "dt:60,ke:90"}])


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
    text = answers.performance_answer(answer)["text"]
    lines = text.splitlines()
    assert lines[0] == report["hints"]
    kernel_row = {"name": "k0", "device": "cpu", "correct": "true", "share": "0.77", "time": "81.3ms"}
    assert table_under(text, "kernels") == ({}, [kernel_row])
    assert lines[-1].split()[0] == "program" and pairs(lines[-1]) == {"correct": "true", "time": "106ms"}
    assert not any(line.startswith(("offloading", "transfers")) for line in lines)

    moved = {"direction": "h2d", "arg": "A", "producer": "host", "consumer": "k0", "bytes": 4096}
    report["offloading"] = {"devices": {"k0": "gpu"}, "transfers": [moved], "transfer_bytes": 4096}
    text = answers.performance_answer(answer)["text"]
    offloading = next(line for line in text.splitlines() if line.startswith("offloading"))
    assert pairs(offloading) == {"devices": "k0:gpu", "transfer_bytes": "4096"}
    transfer_row = {"arg": "A", "direction": "h2d", "bytes": "4096", "consumer": "k0", "producer": "host"}
    assert table_under(text, "transfers") == ({}, [transfer_row])


def test_a_profile_keeps_its_report_text_and_drops_the_structured_copy(answers: ModuleType) -> None:
    report = {
        "build_ok": True,
        "kernel": "s231",
        "threads": 24,
        "text": "hardware counters ...",
        "counters": {"cycles": 9},
        "hotspots": [],
    }
    answer = answers.profile_answer(report)
    scalars, *rest = answer["text"].splitlines()
    assert set(answer) == {"text"}
    assert pairs(scalars) == {"build_ok": "true", "threads": "24"} and rest == [report["text"]]
    without_text = {"result": {"library": "/shared/lib.so"}, "profiled": True}
    assert answers.profile_answer(without_text) is without_text


def test_a_profile_at_ranks_is_one_row_per_rank_count_with_the_time_inside_mpi(answers: ModuleType) -> None:
    run = {"time_us": 812.5, "mpi_time_us": 140.25, "correct": True, "mismatched": [], "error": ""}
    unfinished: dict[str, object] = {
        "ranks": 4,
        "time_us": None,
        "mpi_time_us": None,
        "correct": None,
        "mismatched": [],
        "error": "oom",
    }
    report = {"build_ok": True, "kernel": "jacobi_2d", "preset": "S", "runs": [{"ranks": 2, **run}, unfinished]}
    text = answers.profile_answer(report)["text"]
    shared, rows = table_under(text, "runs")
    assert pairs(text.splitlines()[0]) == {"build_ok": "true", "preset": "S"}
    assert [row["ranks"] for row in rows] == ["2", "4"]
    mpi_name, mpi_text = answers.measured("mpi_time_us", 140.25)
    assert mpi_name == "mpi_time" and rows[0][mpi_name] == mpi_text, rows
    assert rows[1]["error"] == "oom" and rows[1][mpi_name] == "-", rows


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
    said = pairs(answers.score_answer(grade)["text"])
    assert said == {"speedup": "1.5", "correct": "true", "language": "cpp", "native_time": "1.35s"}
    grade["hidden_total"] = grade["hidden_passed"] = 3
    hidden = pairs(answers.score_answer(grade)["text"])
    assert {name: hidden[name] for name in ("hidden_correct", "hidden_passed", "hidden_total")} == {
        "hidden_correct": "true",
        "hidden_passed": "3",
        "hidden_total": "3",
    }


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
    assert table_under(text, "cells") == (
        {"graded": "true"},
        [{"label": "cfg0:large0", "ratio": "0.509"}, {"label": "cfg0:large3", "ratio": "0.517"}],
    )


def test_a_measurement_answer_has_no_quotes_and_a_value_with_spaces_gets_its_own_line(answers: ModuleType) -> None:
    grade = {"speedup": 1.0, "baselines": {"c": 10}, "shape": '{"N": 4}', "note": "two words"}
    text = answers.score_answer(grade)["text"]
    assert '"' not in text
    inline, own = text.splitlines()[-2:]
    assert pairs(inline) == {"baselines": "c:10ns", "shape": "N:4"} and own == "note: two words"


def test_a_refused_batch_names_the_move_and_substep_first_then_each_result(answers: ModuleType) -> None:
    applied = {"kind": "tile-for", "labels": ["for0"], "reason": "LoopTiling", "status": "applied"}
    batch = {
        "status": "refused",
        "results": [applied],
        "failed_at": 1,
        "substep": "check",
        "reason": "wrong at the check sizes: mismatched ['A']",
        "unchecked": "",
    }
    answer = answers.batch_answer({"tool": "apply_moves", "result": batch, "performance": None})
    head, *rest = answer["text"].splitlines()
    assert set(answer) == {"text"}
    status, _, reason = head.partition(": ")
    assert status.split() == ["refused", "failed_at=1", "substep=check"] and reason == batch["reason"]
    assert rest == [answers.result_line(applied)]


def test_a_batch_refused_at_its_first_move_says_so_and_says_why_nothing_was_checked(answers: ModuleType) -> None:
    batch = {"status": "refused", "results": [], "failed_at": 0, "substep": "apply", "reason": "no such row"}
    head = answers.batch_answer({"result": batch})["text"].splitlines()[0]
    assert head.partition(": ")[0].split() == ["refused", "failed_at=0", "substep=apply"]
    unchecked = {"status": "applied", "results": [], "unchecked": "the bench has no reference"}
    assert answers.batch_answer({"result": unchecked})["text"].splitlines() == [
        "applied",
        "unchecked: the bench has no reference",
    ]


def test_a_tune_is_the_move_then_one_row_per_value_tried_the_program_as_it_stands_first(answers: ModuleType) -> None:
    values = [
        {"sizes": [], "status": "measured", "time_us": 1200.0, "correct": True, "reason": ""},
        {"sizes": [32], "status": "kept", "time_us": 800.0, "correct": True, "reason": ""},
        {"sizes": [64], "status": "refused", "time_us": None, "correct": None, "reason": "64 | 100 leaves a tail"},
    ]
    tuned = {"kind": "tile-for", "labels": ["for0"], "values": values}
    head, title, *table = answers.tune_answer({"tool": "tune_move", "result": tuned, "performance": None})[
        "text"
    ].split("\n")
    assert (head, title) == ("tile-for for0", "values:")
    rows = table_rows(table)
    assert [(row["sizes"], row["status"], row["time"]) for row in rows] == [
        ("now", "measured", "1.2ms"),
        ("[32]", "kept", "800us"),
        ("[64]", "refused", "-"),
    ]
    assert rows[2]["reason"] == "64 / 100 leaves a tail"


def test_a_phase_step_is_status_and_phase_then_the_summary_left_then_the_guide_reached(answers: ModuleType) -> None:
    summary = "structured: 1 move(s) applied\n  tile-for for0\nkernels: k0 on cpu"
    step = {"status": "moved", "phase": "scopes", "reason": "", "ran": ["moves"], "undone": [], "summary": summary}
    answer = answers.phase_answer({"tool": "advance_phase", "result": step, "guide": "# Scopes\nset_scope ..."})
    assert set(answer) == {"text"}
    assert answer["text"].splitlines() == [
        "moved scopes",
        "ran=[moves]",
        *summary.splitlines(),
        "# Scopes",
        "set_scope ...",
    ]
    refused = {"status": "refused", "phase": "structured", "reason": "structured is the first phase"}
    assert answers.phase_answer({"result": refused})["text"] == "refused structured: structured is the first phase"


def test_only_a_json_field_is_unpacked_and_a_reason_that_looks_like_json_stays_as_it_came(answers: ModuleType) -> None:
    grade = {"speedup": 1.0, "shape": '{"N": 4}', "note": '{"N": 4}'}
    inline, own = answers.score_answer(grade)["text"].splitlines()[-2:]
    assert pairs(inline) == {"shape": "N:4"} and own == f"note: {grade['note']}"
