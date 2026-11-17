# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Count the non-reasoning tokens of a fixed sequence of NestForge calls: deterministic, no model.

Two parts are counted per call: the call the agent writes (the tool's name and its arguments as JSON) and the answer
it reads back. Reasoning is not counted: it is the model's, and no layout of NestForge's changes it directly.

The calls run on a real session of a bench kernel through :func:`nestforge.server.answer`, as the plugin's route runs
them, and each answer is rendered exactly as the agent reads it: the route's ``ToolAnswer`` as JSON, which the
bench's MCP server re-serializes with ``json.dumps(indent=2, sort_keys=True)``. A tokenizer file (the run model's
``tokenizer.json``) counts it. Every answer stays in the context, so the script also reports the re-send cost: each
answer's tokens times the calls after it.

The calls come from a JSONL file of ``[tool, arguments]`` (a run's move journal, or a sequence written by hand), or
from an agent transcript (``--transcript``), whose NestForge calls are taken in order. Bench calls (``score``,
``submit``, ``profile``) are not NestForge's and are skipped; so are the timed tools unless ``--measure``.

    python scripts/answer_tokens.py --kernel pagerank --calls calls.jsonl --tokenizer tokenizer.json
    python scripts/answer_tokens.py --kernel pagerank --transcript transcript.jsonl --tokenizer tokenizer.json \\
        --format v1 --out answers.csv
"""

import argparse
import csv
import json
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import tiktoken
from hpcagent_bench.harness.judge_wire import JudgeAnswer
from pydantic import BaseModel
from tokenizers import Tokenizer

from nestforge.server import CALLS, TIMED, Reply, Setup, answer, session_data, started_session
from nestforge.stages.performance import Performance


class ToolAnswer(JudgeAnswer):
    """The plugin route's answer to a NestForge call (``plugin/nestforge/routes/nestforge.py``), which the bench loads
    as a plugin and this script cannot import; a test keeps the two alike."""

    tool: str
    result: Any = None
    performance: Performance | None = None


#: How an answer is turned into the text the agent reads, by format name.
Renderer = Callable[[str, Reply], str]

#: The prefix Claude Code gives an MCP tool's name (``mcp__<server>__<tool>``).
MCP_PREFIX = "mcp__"


def rendered_v1(tool: str, reply: Reply) -> str:
    """The answer as agents read it today: the route's JSON, indented and key-sorted by the bench's MCP server."""
    if reply.error:
        response: dict[str, Any] = {"ok": False, "error": f"{tool}: {reply.error}"}
    else:
        response = ToolAnswer(tool=tool, result=reply.result, performance=reply.performance).to_json()
    return json.dumps(response, indent=2, sort_keys=True)


#: Fields a v2 answer writes as their bare value: a word the agent reads first.
BARE_KEYS = frozenset({"kind", "status", "tool", "labels"})

#: Fields a v2 answer leaves out when they hold their default: nothing the agent acts on.
QUIET = (None, "", [], {}, False)


def compact(value: Any, depth: int = 0) -> list[str]:
    """``value`` as plain lines: one record per line, ``key value`` pairs, defaults left out, floats to 3 digits."""
    pad = "  " * depth
    if isinstance(value, str):
        return [pad + line for line in value.splitlines()] or [pad]
    if isinstance(value, float):
        return [pad + f"{value:.3g}"]
    if isinstance(value, list):
        flat = all(not isinstance(item, (dict, list)) for item in value)
        if flat:
            return [pad + " ".join(compact(item)[0].strip() for item in value)]
        return [line for item in value for line in compact(item, depth)]
    if isinstance(value, dict):

        def flat(item: Any) -> bool:
            if isinstance(item, str):
                return "\n" not in item
            if isinstance(item, dict):
                return False
            return not isinstance(item, list) or all(not isinstance(x, (dict, list)) for x in item)

        def word(item: Any) -> str:
            if isinstance(item, list):
                return ",".join(compact(x)[0].strip() for x in item)
            return compact(item)[0].strip()

        simple = {k: v for k, v in value.items() if v not in QUIET and flat(v)}
        nested = {k: v for k, v in value.items() if v not in QUIET and not flat(v)}
        words = [word(v) if k in BARE_KEYS else f"{k}={word(v)}" for k, v in simple.items()]
        lines = [pad + " ".join(words)] if words else []
        for key, item in nested.items():
            lines.append(f"{pad}{key}:")
            lines += compact(item, depth + 1)
        return lines
    return [pad + str(value)]


def rendered_v2(tool: str, reply: Reply) -> str:
    """A prototype of plain-text answers (not built in NestForge): a status word first, one line per record, no
    keys for kind and status, defaults and nulls left out, numbers to 3 significant digits, text tools as their
    text. The bench's MCP server still wraps it as one JSON string."""
    if reply.error:
        text = f"error: {tool}: {reply.error}"
    else:
        answer = ToolAnswer(tool=tool, result=reply.result, performance=reply.performance).to_json()
        lines = compact(answer["result"]) if answer.get("result") is not None else []
        if answer.get("performance") is not None:
            lines += ["performance:", *compact(answer["performance"], 1)]
        text = "\n".join(lines)
    return json.dumps({"ok": not reply.error, "text": text}, indent=2, sort_keys=True)


#: Every format the script can render; a new answer format adds its renderer here, and both are counted side by side.
RENDERERS: dict[str, Renderer] = {"v1": rendered_v1, "v2": rendered_v2}

#: Tokenizer names tiktoken knows: OpenAI's, a common reference when the run's own model is not the point.
TIKTOKEN_NAMES = frozenset({"o200k_base", "cl100k_base"})


def token_counter(name: str) -> Callable[[str], int]:
    """A token counter: a tiktoken encoding by name (``o200k_base``, GPT-4o's), or a ``tokenizer.json`` file."""
    if name in TIKTOKEN_NAMES:
        encoding = tiktoken.get_encoding(name)
        return lambda text: len(encoding.encode(text, disallowed_special=()))
    tokenizer = Tokenizer.from_file(name)
    return lambda text: len(tokenizer.encode(text).ids)


@dataclass(frozen=True, slots=True)
class Counted:
    """One call and the tokens of its answer, per format."""

    index: int
    tool: str
    arguments: str
    status: str
    call_tokens: int
    tokens: dict[str, int]


def status_of(reply: Reply) -> str:
    """``error``, a move's own status (``applied``, ``illegal``, ...), or ``ok``: a replay whose labels went astray
    shows as errors, not as cheaper answers."""
    if reply.error:
        return "error"
    status = vars(reply.result).get("status") if isinstance(reply.result, BaseModel) else None
    return str(status.name.lower() if isinstance(status, Enum) else status) if status is not None else "ok"


def tool_name(name: str) -> str:
    """``name`` without the MCP server's prefix."""
    return name.rsplit("__", 1)[-1] if name.startswith(MCP_PREFIX) else name


def transcript_calls(lines: Iterable[str]) -> Iterator[tuple[str, dict[str, Any]]]:
    """The NestForge calls of an agent transcript (Claude Code's JSONL), in order, each tool use once."""
    seen: set[str] = set()
    for line in lines:
        record = json.loads(line)
        if record.get("type") != "assistant":
            continue
        for block in record.get("message", {}).get("content", []):
            if block.get("type") != "tool_use" or block.get("id") in seen:
                continue
            seen.add(block.get("id"))
            name = tool_name(str(block.get("name", "")))
            if name in CALLS:
                yield name, dict(block.get("input", {}))


def journal_calls(lines: Iterable[str]) -> Iterator[tuple[str, dict[str, Any]]]:
    """``[tool, arguments]`` per line, as NestForge's move journal writes them."""
    for line in lines:
        if line.strip():
            tool, arguments = json.loads(line)
            yield str(tool), dict(arguments)


def call_text(tool: str, arguments: dict[str, Any]) -> str:
    """The call as the agent writes it: the tool's name and its arguments as JSON."""
    return json.dumps({"name": tool, "input": arguments})


def resend_cost(tokens: Sequence[int]) -> int:
    """The tokens of answers re-sent with every later call: each answer counts once per call after it."""
    return sum(count * (len(tokens) - 1 - index) for index, count in enumerate(tokens))


def count(
    calls: Iterable[tuple[str, dict[str, Any]]],
    run: Callable[[str, dict[str, Any]], Reply],
    renderers: dict[str, Renderer],
    tokens_of: Callable[[str], int],
    measure: bool,
    shown: dict[str, str] | None = None,
    show: int = -1,
) -> list[Counted]:
    """Each call's answer tokens per format; a timed tool only with ``measure``. The answer of call ``show`` is kept
    in ``shown``, per format."""
    counted: list[Counted] = []
    for tool, arguments in calls:
        if tool in TIMED and not measure:
            continue
        reply = run(tool, arguments)
        texts = {name: render(tool, reply) for name, render in renderers.items()}
        if shown is not None and len(counted) == show:
            shown.update(texts)
        tokens = {name: tokens_of(text) for name, text in texts.items()}
        call = tokens_of(call_text(tool, arguments))
        counted.append(
            Counted(len(counted), tool, json.dumps(arguments, sort_keys=True), status_of(reply), call, tokens)
        )
    return counted


def report(counted: list[Counted], formats: Sequence[str]) -> str:
    """Totals, re-send cost and tokens by tool, per format."""
    statuses = Counter(row.status for row in counted)
    lines = [f"{len(counted)} calls: " + ", ".join(f"{n} {status}" for status, n in sorted(statuses.items()))]
    calls = sum(row.call_tokens for row in counted)
    lines.append(f"calls written by the agent: {calls} tokens")
    for name in formats:
        tokens = [row.tokens[name] for row in counted]
        both = [row.tokens[name] + row.call_tokens for row in counted]
        by_tool: Counter[str] = Counter()
        for row in counted:
            by_tool[row.tool] += row.tokens[name]
        lines.append(
            f"{name}: answers {sum(tokens)} tokens; non-reasoning (calls + answers) {sum(both)}, "
            f"re-sent {resend_cost(both)}"
        )
        lines += [f"  {tool:20s} {total}" for tool, total in by_tool.most_common()]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kernel", required=True, help="The bench kernel the session serves.")
    parser.add_argument("--preset", default="S")
    parser.add_argument("--datatype", default="float64")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--calls", type=Path, help="JSONL of [tool, arguments], e.g. a move journal.")
    source.add_argument("--transcript", type=Path, help="An agent transcript; its NestForge calls are replayed.")
    parser.add_argument(
        "--tokenizer",
        default="o200k_base",
        help="o200k_base (default, OpenAI's GPT-4o tokenizer), cl100k_base, or a tokenizer.json (the run's model).",
    )
    parser.add_argument("--show", type=int, default=-1, help="Print call INDEX's answer in every format.")
    parser.add_argument("--format", default=",".join(RENDERERS), help="Formats to count, comma-separated.")
    parser.add_argument("--measure", action="store_true", help="Also run the timed tools (builds and runs code).")
    parser.add_argument("--out", type=Path, help="CSV of every call and its tokens per format.")
    args = parser.parse_args(argv)

    formats = [name for name in args.format.split(",") if name]
    unknown = [name for name in formats if name not in RENDERERS]
    if unknown:
        parser.error(f"unknown format(s) {unknown}; known: {sorted(RENDERERS)}")
    lines = (args.calls or args.transcript).read_text().splitlines()
    calls = journal_calls(lines) if args.calls else transcript_calls(lines)
    tokens_of = token_counter(args.tokenizer)
    session = started_session(Setup(args.kernel, args.preset, args.datatype, False), codegen="first")
    data = session_data(Setup(args.kernel, args.preset, args.datatype, False), session) if args.measure else None

    def run(tool: str, arguments: dict[str, Any]) -> Reply:
        return answer(session, tool, arguments, data, timed=args.measure)

    counted = count(
        calls,
        run,
        {name: RENDERERS[name] for name in formats},
        tokens_of,
        args.measure,
        shown := {},
        args.show,
    )
    for name, text in shown.items():
        print(f"----- call {args.show} as {name}\n{text}")
    if args.out:
        with args.out.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["index", "tool", "arguments", "status", "call", *formats])
            writer.writerows(
                [row.index, row.tool, row.arguments, row.status, row.call_tokens, *(row.tokens[f] for f in formats)]
                for row in counted
            )
    print(report(counted, formats), file=sys.stdout)


if __name__ == "__main__":
    main()
