# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Count the tokens of NestForge's answers to a fixed sequence of calls: deterministic, no model.

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


#: Every format the script can render; a new answer format adds its renderer here, and both are counted side by side.
RENDERERS: dict[str, Renderer] = {"v1": rendered_v1}


@dataclass(frozen=True, slots=True)
class Counted:
    """One call and the tokens of its answer, per format."""

    index: int
    tool: str
    arguments: str
    status: str
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


def resend_cost(tokens: Sequence[int]) -> int:
    """The tokens of answers re-sent with every later call: each answer counts once per call after it."""
    return sum(count * (len(tokens) - 1 - index) for index, count in enumerate(tokens))


def count(
    calls: Iterable[tuple[str, dict[str, Any]]],
    run: Callable[[str, dict[str, Any]], Reply],
    renderers: dict[str, Renderer],
    tokens_of: Callable[[str], int],
    measure: bool,
) -> list[Counted]:
    """Each call's answer tokens per format; a timed tool only with ``measure``."""
    counted: list[Counted] = []
    for tool, arguments in calls:
        if tool in TIMED and not measure:
            continue
        reply = run(tool, arguments)
        tokens = {name: tokens_of(render(tool, reply)) for name, render in renderers.items()}
        counted.append(Counted(len(counted), tool, json.dumps(arguments, sort_keys=True), status_of(reply), tokens))
    return counted


def report(counted: list[Counted], formats: Sequence[str]) -> str:
    """Totals, re-send cost and tokens by tool, per format."""
    statuses = Counter(row.status for row in counted)
    lines = [f"{len(counted)} calls: " + ", ".join(f"{n} {status}" for status, n in sorted(statuses.items()))]
    for name in formats:
        tokens = [row.tokens[name] for row in counted]
        by_tool: Counter[str] = Counter()
        for row in counted:
            by_tool[row.tool] += row.tokens[name]
        lines.append(f"{name}: answers {sum(tokens)} tokens, re-sent {resend_cost(tokens)}")
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
    parser.add_argument("--tokenizer", type=Path, required=True, help="The run model's tokenizer.json.")
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
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    session = started_session(Setup(args.kernel, args.preset, args.datatype, False), codegen="first")
    data = session_data(Setup(args.kernel, args.preset, args.datatype, False), session) if args.measure else None

    def run(tool: str, arguments: dict[str, Any]) -> Reply:
        return answer(session, tool, arguments, data, timed=args.measure)

    counted = count(
        calls,
        run,
        {name: RENDERERS[name] for name in formats},
        lambda text: len(tokenizer.encode(text).ids),
        args.measure,
    )
    if args.out:
        with args.out.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["index", "tool", "arguments", "status", *formats])
            writer.writerows(
                [row.index, row.tool, row.arguments, row.status, *(row.tokens[f] for f in formats)] for row in counted
            )
    print(report(counted, formats), file=sys.stdout)


if __name__ == "__main__":
    main()
