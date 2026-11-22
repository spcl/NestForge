# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""What the agent reads back from a tool: the judge's answer as short lines, the same facts without the JSON layout.
The judge, its journal and every other caller keep the records; only the agent's copy is rendered. Standard library
only, so the agent image (no DaCe, no NestForge) runs it."""

import contextlib
import json
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any

__all__ = [
    "RENDERED",
    "batch_answer",
    "kinds_answer",
    "move_line",
    "moves_answer",
    "performance_answer",
    "phase_answer",
    "profile_answer",
    "result_line",
    "results_answer",
    "score_answer",
    "tune_answer",
]

#: What a list with nothing in it reads as.
NONE = "none"

#: The fields a move line prints first, in this order, instead of as ``name=value``.
MOVE_HEAD = ("kind", "labels")
#: The fields a result line prints in its own places.
RESULT_HEAD = ("status", "kind", "labels", "reason", "warning", "tree")

#: Significant digits a measured number keeps: timings repeat to about 1%, so more digits are noise.
DIGITS = 3
#: The ``score`` fields its first line prints, in this order.
SCORE_HEAD = ("speedup", "correct", "build_ok", "timed_out")
#: ``score`` fields about the grading protocol, the same in every answer of a run; never what the program did.
PROTOCOL = ("grading_protocol", "seed_nonce", "timing_reduction")
#: The ``score`` fields printed in their own places.
SCORE_OWN = (*SCORE_HEAD, *PROTOCOL, "detail", "cells")
#: Answer fields the agent does not need beside the text: the tool's own name, and ``ok`` when it is true.
WRAPPER_DROP = ("tool",)
#: Seconds per unit of a time field, by the suffix its name ends with.
TIME_UNITS = {"_ns": 1e-9, "_us": 1e-6, "_ms": 1e-3}
#: Fields that hold a map of times in a unit their name does not say.
TIME_MAPS = {"baselines": 1e-9}
#: The units a time is printed in, largest first: each with its size in seconds.
PRINTED_UNITS = (("s", 1.0), ("ms", 1e-3), ("us", 1e-6), ("ns", 1e-9))
#: ``score`` fields the agent already has: the task prompt names the kernel.
GIVEN = ("kernel",)
#: ``score``'s counts of hidden inputs, which say nothing when there were none.
HIDDEN = ("hidden_correct", "hidden_passed", "hidden_total")
#: Measured fields the bench sends as a JSON object inside a string (a cell's shape); only these are unpacked.
JSON_FIELDS = ("shape",)

#: The ``apply_moves`` fields its first line prints after the status, in this order.
BATCH_HEAD = ("failed_at", "substep")
#: What the sizes ``tune_move`` tried read as for the program as it stands, which it tried at no size.
AS_IT_STANDS = "now"
#: The ``tune_move`` value fields its table leads with.
TUNE_LEAD = ("sizes", "status")
#: The phase tools' fields printed in their own places; the others are ``name=value`` after the status.
PHASE_OWN = ("status", "phase", "reason", "summary")


def unset_measurement(value: object) -> bool:
    """Whether a measured field says nothing: unset, or an empty string or collection. False and 0 are measurements."""
    return value is None or value in ("", [], {}, ())


def unset_argument(value: object) -> bool:
    """Whether a move argument says nothing: as :func:`unset_measurement`, and also ``False``, a flag left off."""
    return value is False or unset_measurement(value)


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
        if name not in skip and not unset_argument(value) and not (name in defaults and defaults[name] == value)
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
    return {**wrapper(answer, ("result",)), "text": "\n".join(map(line, result)) if result else NONE}


def wrapper(answer: Mapping[str, Any], rendered: Collection[str]) -> dict[str, Any]:
    """What of ``answer`` stays beside its text: every field that says something, but the ``rendered`` ones, the
    tool's name and a true ``ok``."""
    return {
        name: value
        for name, value in answer.items()
        if name not in (*rendered, *WRAPPER_DROP) and value is not None and not (name == "ok" and value is True)
    }


def moves_answer(answer: dict[str, Any], defaults: Mapping[str, object]) -> dict[str, Any]:
    """``list_moves``' answer as one move per line."""
    return lines(answer, lambda move: move_line(move, defaults))


def results_answer(answer: dict[str, Any]) -> dict[str, Any]:
    """``applied_moves``' answer as one move result per line."""
    return lines(answer, result_line)


def kinds_answer(answer: dict[str, Any]) -> dict[str, Any]:
    """``list_move_kinds``' answer as one kind per line."""
    return lines(answer, kind_line)


def number(value: float) -> str:
    """``value`` at :data:`DIGITS` significant digits, without an exponent where one is not needed."""
    text = f"{value:.{DIGITS}g}"
    return text if "e" not in text or abs(value) < 1e-4 or abs(value) >= 1e15 else f"{float(text):.0f}"


def duration(seconds: float) -> str:
    """A time in the largest unit it is at least one of, at :data:`DIGITS` significant digits: ``0.893s`` reads
    ``893ms``."""
    for unit, size in PRINTED_UNITS:
        scaled = float(f"{seconds / size:.{DIGITS}g}")
        if abs(scaled) >= 1 or unit == "ns":
            return f"{number(scaled)}{unit}"
    raise AssertionError("unreachable: the last unit always prints")


def plain(value: object) -> str:
    """A measured value without quotes: numbers rounded, a map as ``key:value,...``, a list as ``[a,b]``."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return number(value)
    if isinstance(value, dict):
        return ",".join(f"{key}:{plain(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return f"[{','.join(map(plain, value))}]"
    return str(value)


def measured(name: str, value: object) -> tuple[str, str]:
    """A measured field as the agent reads it: a time renamed without its unit and printed with one, a
    :data:`JSON_FIELDS` string unpacked."""
    if name in JSON_FIELDS and isinstance(value, str):
        with contextlib.suppress(json.JSONDecodeError):
            value = json.loads(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        for suffix, size in TIME_UNITS.items():
            if name.endswith(suffix):
                stem = name.removesuffix(suffix)
                return (stem if stem.endswith("time") else f"{stem}_time"), duration(float(value) * size)
    if name in TIME_MAPS and isinstance(value, dict):
        size = TIME_MAPS[name]
        return name, ",".join(f"{key}:{duration(float(item) * size)}" for key, item in value.items())
    return name, plain(value)


def fields(record: Mapping[str, Any], skip: Collection[str] = ()) -> list[str]:
    """``record``'s measured fields as ``name=value``, by name, those that say nothing left out. A value with a space
    in it goes on a line of its own, as ``name: value``, after the others."""
    pairs = [
        measured(name, value)
        for name, value in sorted(record.items())
        if name not in skip and not unset_measurement(value)
    ]
    inline = [f"{name}={text}" for name, text in pairs if not any(char.isspace() for char in text)]
    own = [f"{name}: {text}" for name, text in pairs if any(char.isspace() for char in text)]
    return [" ".join(inline), *own] if inline else own


def cell_text(text: str) -> str:
    """A table cell: a ``|`` in it would end the cell, so it reads ``/``."""
    return text.replace("|", "/")


def table(
    title: str, records: Sequence[Mapping[str, Any]], lead: tuple[str, ...], skip: Collection[str] = ()
) -> list[str]:
    """``records`` as a markdown table under ``title:``. With more than one record, a value every record shares is
    said once on the title line instead of in a column; a column no record fills goes; a missing value reads ``-``."""
    rows = [
        dict(
            measured(name, value) for name, value in record.items() if name not in skip and not unset_measurement(value)
        )
        for record in records
    ]
    names = {name for row in rows for name in row}
    shared = {
        name: rows[0][name]
        for name in sorted(names)
        if len(rows) > 1 and name not in lead and all(row.get(name) == rows[0].get(name) for row in rows)
    }
    columns = [name for name in lead if name in names] + sorted(names - set(lead) - set(shared))
    head = " ".join([f"{title}:", *(f"{name}={text}" for name, text in shared.items())])
    return [
        head,
        f"| {' | '.join(columns)} |",
        f"|{'|'.join(' --- ' for _ in columns)}|",
        *(f"| {' | '.join(cell_text(row.get(name, '-')) for name in columns)} |" for row in rows),
    ]


def failed(answer: Mapping[str, Any]) -> bool:
    """Whether ``answer`` is an error, which always reaches the agent as it came."""
    return answer.get("ok") is False or ("error" in answer and not answer.get("text"))


def score_skip(answer: Mapping[str, Any]) -> tuple[str, ...]:
    """The ``score`` fields its other-fields line leaves out."""
    hidden = HIDDEN if not answer.get("hidden_total") else ()
    return (*SCORE_OWN, *GIVEN, *hidden)


def score_answer(answer: dict[str, Any]) -> dict[str, Any]:
    """``score``'s grade as lines: speedup and correctness first, the judge's detail, the other fields, then the graded
    cells as a table. The grading protocol's fields, the kernel's name and hidden-input counts of zero go."""
    if failed(answer) or "speedup" not in answer:
        return answer
    text = [" ".join(f"{name}={plain(answer[name])}" for name in SCORE_HEAD if name in answer)]
    if answer.get("detail"):
        text.append(f"detail: {answer['detail']}")
    text += fields(answer, score_skip(answer))
    cells = answer.get("cells") or []
    if cells:
        # what the grade's own fields already say is left out of each cell
        same = {name for cell in cells for name, value in cell.items() if name in answer and answer[name] == value}
        text += table("cells", cells, ("label",), same)
    return {"text": "\n".join(line for line in text if line)}


def offloaded(offloading: Mapping[str, Any]) -> bool:
    """Whether a kernel runs on the GPU: only then does data move between memories."""
    return "gpu" in (offloading.get("devices") or {}).values()


def offloading_lines(offloading: Mapping[str, Any]) -> list[str]:
    """Where data moves, when anything runs off the CPU: the summary line, then the transfers as a table."""
    if not offloaded(offloading):
        return []
    text = [" ".join(["offloading", *fields(offloading, ("transfers",))])]
    transfers = offloading.get("transfers") or []
    return text + table("transfers", transfers, ("arg", "direction")) if transfers else text


def performance_answer(answer: dict[str, Any]) -> dict[str, Any]:
    """``get_performance``'s report as lines: the ranked hints, the kernels as a table, the whole program, and where
    data moves when anything runs off the CPU."""
    performance = answer.get("performance")
    if failed(answer) or not isinstance(performance, dict):
        return answer
    text = [str(performance["hints"])] if performance.get("hints") else []
    kernels = performance.get("kernels") or []
    if kernels:
        text += table("kernels", kernels, ("name", "device"))
    text.append(" ".join(["program", *fields(performance.get("structured") or {})]))
    text += offloading_lines(performance.get("offloading") or {})
    return {**wrapper(answer, ("performance",)), "text": "\n".join(text)}


def profile_answer(answer: dict[str, Any]) -> dict[str, Any]:
    """``profile``'s report: the scalar fields beside it on one line, then its own ``text``. The structured copy of
    the same data (counters, call graph, hotspots) and the kernel's name go; an answer without a text comes as it
    came."""
    if failed(answer) or not isinstance(answer.get("text"), str):
        return answer
    scalars = {
        name: value
        for name, value in answer.items()
        if name != "text" and name not in GIVEN and not isinstance(value, (dict, list))
    }
    return {"text": "\n".join([*fields(scalars), answer["text"]])}


def record_of(answer: Mapping[str, Any], own: str) -> dict[str, Any] | None:
    """``answer``'s result when it is a record holding the field ``own``; ``None`` for an error or any other shape."""
    result = answer.get("result")
    if failed(answer) or not isinstance(result, dict) or own not in result:
        return None
    return result


def status_line(record: Mapping[str, Any], head: Collection[str]) -> str:
    """A record's status, then its ``head`` fields that say something as ``name=value``, then ``: reason``."""
    said = [f"{name}={plain(record[name])}" for name in head if not unset_measurement(record.get(name))]
    line = " ".join([str(record["status"]), *said])
    return f"{line}: {record['reason']}" if record.get("reason") else line


def batch_answer(answer: dict[str, Any]) -> dict[str, Any]:
    """``apply_moves``' answer: its status (with the move that failed, from 0, the substep and why), why no check ran
    if none did, then one result line per move it applied."""
    batch = record_of(answer, "results")
    if batch is None:
        return answer
    text = [status_line(batch, BATCH_HEAD)]
    if batch.get("unchecked"):
        text.append(f"unchecked: {batch['unchecked']}")
    text += map(result_line, batch["results"])
    if batch.get("tree"):
        text.append(str(batch["tree"]))
    return {**wrapper(answer, ("result",)), "text": "\n".join(text)}


def move_answer(answer: dict[str, Any]) -> dict[str, Any]:
    """``apply_move``'s answer: its result line, then the subtree the move rewrote (or the whole tree)."""
    result = record_of(answer, "kind")
    if result is None:
        return answer
    text = [result_line(result), *([str(result["tree"])] if result.get("tree") else [])]
    return {**wrapper(answer, ("result",)), "text": "\n".join(text)}


def tune_answer(answer: dict[str, Any]) -> dict[str, Any]:
    """``tune_move``'s answer: the move's kind and labels, then the values it tried as a table (the program as it
    stands at the sizes ``now``)."""
    tuned = record_of(answer, "values")
    if tuned is None:
        return answer
    values = [{**value, "sizes": value.get("sizes") or AS_IT_STANDS} for value in tuned["values"]]
    head = " ".join([str(tuned["kind"]), *map(str, tuned.get("labels", ()))])
    return {**wrapper(answer, ("result",)), "text": "\n".join([head, *table("values", values, TUNE_LEAD)])}


def phase_answer(answer: dict[str, Any]) -> dict[str, Any]:
    """A phase tool's answer: its status and phase (``: reason`` when refused), the defaults it ran and the phases it
    took back, the summary of the phase it left, then the guide of the phase it reached."""
    step = record_of(answer, "phase")
    if step is None:
        return answer
    line = f"{step['status']} {step['phase']}"
    text = [f"{line}: {step['reason']}" if step.get("reason") else line, *fields(step, PHASE_OWN)]
    if step.get("summary"):
        text.append(str(step["summary"]))
    if answer.get("guide"):
        text.append(str(answer["guide"]))
    return {**wrapper(answer, ("result", "guide")), "text": "\n".join(text)}


#: Tool -> how its answer reaches the agent, given the move arguments' defaults; tools not here answer as they came.
#: The phase tools render with :func:`phase_answer` after their guide is added (``tools/nestforge.py``).
RENDERED: dict[str, Callable[[dict[str, Any], Mapping[str, object]], dict[str, Any]]] = {
    "list_moves": moves_answer,
    "list_scope_moves": moves_answer,
    "applied_moves": lambda answer, defaults: results_answer(answer),
    "apply_moves": lambda answer, defaults: batch_answer(answer),
    "apply_move": lambda answer, defaults: move_answer(answer),
    "tune_move": lambda answer, defaults: tune_answer(answer),
    "list_move_kinds": lambda answer, defaults: kinds_answer(answer),
    "get_performance": lambda answer, defaults: performance_answer(answer),
}
