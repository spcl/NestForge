# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""What the agent reads back from a tool: the judge's answer as short lines, the same facts without the JSON layout.
The judge, its journal and every other caller keep the records; only the agent's copy is rendered. Standard library
only, so the agent image (no DaCe, no NestForge) runs it."""

import contextlib
import json
from collections.abc import Callable, Collection, Mapping
from typing import Any

__all__ = [
    "RENDERED",
    "kinds_answer",
    "move_line",
    "moves_answer",
    "performance_answer",
    "profile_answer",
    "result_line",
    "results_answer",
    "score_answer",
]

#: What a list with nothing in it reads as.
NONE = "none"

#: The fields a move line prints first, in this order, instead of as ``name=value``.
MOVE_HEAD = ("kind", "labels")
#: The fields a result line prints in its own places.
RESULT_HEAD = ("status", "kind", "labels", "reason", "warning")

#: Significant digits a measured number keeps: timings repeat to about 1%, so more digits are noise.
DIGITS = 3
#: The ``score`` fields its first line prints, in this order.
SCORE_HEAD = ("speedup", "correct", "build_ok", "timed_out")
#: ``score`` fields about the grading protocol, the same in every answer of a run; never what the program did.
PROTOCOL = ("grading_protocol", "seed_nonce", "timing_reduction")
#: The ``score`` fields printed in their own places.
SCORE_OWN = (*SCORE_HEAD, *PROTOCOL, "detail", "cells")


def empty(value: object) -> bool:
    """Whether ``value`` says nothing: unset, false, or an empty string or collection."""
    return value is None or value is False or value in ("", [], {}, ())


def value_text(value: object) -> str:
    """``value`` as one token: a string without spaces as is, anything else as compact JSON."""
    if isinstance(value, str) and value and not any(char.isspace() for char in value):
        return value
    return json.dumps(value, separators=(",", ":"))


def extras(record: Mapping[str, Any], skip: Collection[str], defaults: Mapping[str, object]) -> list[str]:
    """``record``'s other fields as ``name=value``, by name; those that say nothing or hold their default go."""
    return [
        f"{name}={value_text(value)}"
        for name, value in sorted(record.items())
        if name not in skip and not empty(value) and not (name in defaults and defaults[name] == value)
    ]


def move_line(move: Mapping[str, Any], defaults: Mapping[str, object]) -> str:
    """A listed move as ``apply_move`` takes it back: its kind, its labels, then each argument it was checked with."""
    return " ".join([str(move["kind"]), *map(str, move.get("labels", ())), *extras(move, MOVE_HEAD, defaults)])


def result_line(result: Mapping[str, Any]) -> str:
    """What a move did: its status, kind and labels, the rest as ``name=value``, then its reason and any warning."""
    line = " ".join([str(result["status"]), str(result["kind"]), *map(str, result.get("labels", ()))])
    line = " ".join([line, *extras(result, RESULT_HEAD, {})])
    if result.get("reason"):
        line = f"{line}: {result['reason']}"
    return f"{line} (warning: {result['warning']})" if result.get("warning") else line


def kind_line(kind: Mapping[str, Any]) -> str:
    """A move kind: its name, the rows it takes in order, the transformation it runs and what it does."""
    return f"{kind['kind']} [{' '.join(kind['rows'])}] {kind['runs']}: {kind['does']}"


def lines(answer: dict[str, Any], line: Callable[[Mapping[str, Any]], str]) -> dict[str, Any]:
    """``answer`` with its list of records as one ``line`` each; an error, or any other shape, as it came."""
    result = answer.get("result")
    if "error" in answer or answer.get("ok") is False or not isinstance(result, list):
        return answer
    rendered = {name: value for name, value in answer.items() if name != "result" and value is not None}
    return {**rendered, "result": "\n".join(map(line, result)) if result else NONE}


def moves_answer(answer: dict[str, Any], defaults: Mapping[str, object]) -> dict[str, Any]:
    """``list_moves``' answer as one move per line."""
    return lines(answer, lambda move: move_line(move, defaults))


def results_answer(answer: dict[str, Any]) -> dict[str, Any]:
    """``applied_moves``' answer as one move result per line."""
    return lines(answer, result_line)


def kinds_answer(answer: dict[str, Any]) -> dict[str, Any]:
    """``list_move_kinds``' answer as one kind per line."""
    return lines(answer, kind_line)


def unset(value: object) -> bool:
    """Whether a measured field says nothing: unset, or an empty string or collection. False and 0 are measurements."""
    return value is None or value in ("", [], {}, ())


def rounded(value: object) -> object:
    """``value`` with every float in it kept to :data:`DIGITS` significant digits."""
    if isinstance(value, float):
        return float(f"{value:.{DIGITS}g}")
    if isinstance(value, dict):
        return {name: rounded(item) for name, item in value.items()}
    if isinstance(value, list):
        return [rounded(item) for item in value]
    return value


def measured_text(value: object) -> str:
    """A measured value as one token: numbers rounded, a JSON object sent as a string (a cell's shape) unpacked."""
    if isinstance(value, float):
        return f"{value:.{DIGITS}g}"
    if isinstance(value, str) and value.startswith("{"):
        with contextlib.suppress(json.JSONDecodeError):
            value = json.loads(value)
    return value_text(rounded(value))


def fields(record: Mapping[str, Any], skip: Collection[str] = ()) -> list[str]:
    """``record``'s measured fields as ``name=value``, by name, those that say nothing left out."""
    return [
        f"{name}={measured_text(value)}"
        for name, value in sorted(record.items())
        if name not in skip and not unset(value)
    ]


def failed(answer: Mapping[str, Any]) -> bool:
    """Whether ``answer`` is an error, which always reaches the agent as it came."""
    return answer.get("ok") is False or ("error" in answer and not answer.get("text"))


def score_answer(answer: dict[str, Any]) -> dict[str, Any]:
    """``score``'s grade as lines: speedup and correctness first, the judge's detail, the other fields, one line per
    graded cell. The grading protocol's own fields go."""
    if failed(answer) or "speedup" not in answer:
        return answer
    head = [f"{name}={measured_text(answer[name])}" for name in SCORE_HEAD if name in answer]
    text = [" ".join(head)]
    if answer.get("detail"):
        text.append(f"detail: {answer['detail']}")
    text.append(" ".join(fields(answer, SCORE_OWN)))
    for cell in answer.get("cells") or []:
        # what the grade's own fields already say is left out of each cell
        same = [name for name, value in cell.items() if name in answer and answer[name] == value]
        text.append(" ".join(["cell", *fields(cell, same)]))
    return {"text": "\n".join(line for line in text if line)}


def kernel_line(kernel: Mapping[str, Any]) -> str:
    """One kernel's time against its oracle."""
    return " ".join(["kernel", str(kernel.get("name")), str(kernel.get("device")), *fields(kernel, ("name", "device"))])


def offloaded(offloading: Mapping[str, Any]) -> bool:
    """Whether ``offloading`` says more than that every kernel runs on the CPU and nothing moves."""
    devices = offloading.get("devices") or {}
    moves = offloading.get("transfers") or offloading.get("transfer_bytes")
    return bool(moves) or any(device != "cpu" for device in devices.values()) or offloading.get("transfers") is None


def performance_answer(answer: dict[str, Any]) -> dict[str, Any]:
    """``get_performance``'s report as lines: the ranked hints, each kernel, the whole program, and where data moves
    when anything runs off the CPU."""
    performance = answer.get("performance")
    if failed(answer) or not isinstance(performance, dict):
        return answer
    text = [str(performance["hints"])] if performance.get("hints") else []
    text += [kernel_line(kernel) for kernel in performance.get("kernels") or []]
    text.append(" ".join(["program", *fields(performance.get("structured") or {})]))
    offloading = performance.get("offloading") or {}
    if offloaded(offloading):
        text.append(" ".join(["offloading", *fields(offloading, ("transfers",))]))
        text += [" ".join(["transfer", *fields(move)]) for move in offloading.get("transfers") or []]
    rest = {name: value for name, value in answer.items() if name != "performance" and value is not None}
    return {**rest, "text": "\n".join(text)}


def profile_answer(answer: dict[str, Any]) -> dict[str, Any]:
    """``profile``'s report: its own ``text`` and the scalar fields beside it. The structured copy of the same data
    (counters, call graph, hotspots) goes; an answer without a text comes as it came."""
    if failed(answer) or not isinstance(answer.get("text"), str):
        return answer
    return {name: value for name, value in answer.items() if not isinstance(value, (dict, list))}


#: Tool -> how its answer reaches the agent, given the move arguments' defaults; tools not here answer as they came.
RENDERED: dict[str, Callable[[dict[str, Any], Mapping[str, object]], dict[str, Any]]] = {
    "list_moves": moves_answer,
    "list_scope_moves": moves_answer,
    "applied_moves": lambda answer, defaults: results_answer(answer),
    "list_move_kinds": lambda answer, defaults: kinds_answer(answer),
    "get_performance": lambda answer, defaults: performance_answer(answer),
}
