# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""What the agent reads back from a tool: the judge's answer as short lines, the same facts without the JSON layout.
The judge, its journal and every other caller keep the records; only the agent's copy is rendered. Standard library
only, so the agent image (no DaCe, no NestForge) runs it."""

import json
from collections.abc import Callable, Collection, Mapping
from typing import Any

__all__ = ["RENDERED", "move_line", "moves_answer", "kinds_answer", "result_line", "results_answer"]

#: What a list with nothing in it reads as.
NONE = "none"

#: The fields a move line prints first, in this order, instead of as ``name=value``.
MOVE_HEAD = ("kind", "labels")
#: The fields a result line prints in its own places.
RESULT_HEAD = ("status", "kind", "labels", "reason", "warning")


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


#: Tool -> how its answer reaches the agent, given the move arguments' defaults; tools not here answer as they came.
RENDERED: dict[str, Callable[[dict[str, Any], Mapping[str, object]], dict[str, Any]]] = {
    "list_moves": moves_answer,
    "list_scope_moves": moves_answer,
    "applied_moves": lambda answer, defaults: results_answer(answer),
    "list_move_kinds": lambda answer, defaults: kinds_answer(answer),
}
