# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Attribute a finished run's speedup to its moves and their families by replay: the journal is replayed one call at
a time on the run's kernel, the program is measured after every prefix with the remaining phases on their defaults,
and each call's ratio to the prefix before it is its share, in the order the agent applied it. Builds only, no tokens.

    python scripts/attribute_moves.py --journal journal.jsonl --kernel jacobi_2d --preset S --out attribution.json
    python scripts/attribute_moves.py --db results.db --setup nestforge-llr --episode <label> --kernel jacobi_2d \\
        --preset S --out attribution.json

The replay stops at the first ``write_kernel``: the record keeps a kernel's source as its hash, so every call from
there on is attributed together as ``kernel code``.
"""

import argparse
import json
import math
import pathlib
from collections import defaultdict
from typing import Any

from nestforge.server import FEEDBACK_CODEGEN, Setup, answer, applied, start
from nestforge.session import Session
from nestforge.stages.families import family_of
from nestforge.stages.moves import journaled_arguments
from nestforge.stages.performance import Reference

#: What the calls from the first ``write_kernel`` on are summed as.
KERNEL_CODE = "kernel code"
#: The share of a call in no family (``run_defaults``, ``set_scope``, ``rollback``).
NO_FAMILY = "other"


def journal_from_file(path: pathlib.Path) -> list[list[Any]]:
    """The route's journal file: one ``[tool, arguments]`` per line."""
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def journal_from_db(db: pathlib.Path, setup: str, episode: str, kernel: str) -> list[list[Any]]:
    """The journal the episode's newest plugin record holds."""
    # deferred: only this way of reading needs the bench's statistics
    from hpcagent_bench.stats import plugin_records

    kept = plugin_records.load(db, setup=setup, episode=episode, plugin="nestforge")
    if not kept:
        raise SystemExit(f"no nestforge record of {episode} in {db}")
    data = kept[-1].record.data
    if not isinstance(data, dict) or data.get("kernel") != kernel:
        raise SystemExit(f"the record of {episode} is not of kernel {kernel}")
    journal = data["journal"]
    assert isinstance(journal, list), "a record's journal is a list"
    return [list(call) for call in journal if isinstance(call, list)]


def measured(session: Session, data: tuple[dict[str, Any], Reference | None], repeats: int) -> tuple[float, float]:
    """The program's best time over ``repeats`` measurements, and their spread (worst over best)."""
    times = []
    for _ in range(repeats):
        session.measured = None  # each repeat measures anew, never the cached measurement
        report = session.performance(*data, codegen=FEEDBACK_CODEGEN).structured
        if report.time_us is None:
            raise RuntimeError(f"the program did not run: {report.error}")
        times.append(report.time_us)
    return min(times), max(times) / min(times)


def attribute(
    session: Session, data: tuple[dict[str, Any], Reference | None], journal: list[list[Any]], repeats: int
) -> dict[str, Any]:
    """One row per replayed call and the log-share per family; the baseline is the program before the first call."""
    before, spread_before = measured(session, data, repeats)
    rows: list[dict[str, Any]] = []
    shares: dict[str, float] = defaultdict(float)
    replayed = 0
    for index, (tool, arguments) in enumerate(journal):
        if tool == "write_kernel":
            break
        arguments = journaled_arguments(arguments)
        reply = answer(session, tool, arguments, data, timed=False)
        if not applied(reply):
            raise RuntimeError(f"call {index} ({tool} {arguments}) does not replay: {reply.error or reply.result}")
        after, spread_after = measured(session, data, repeats)
        ratio = before / after
        noise = max(spread_before, spread_after)
        kind = arguments.get("kind")
        family = (family_of(kind) if isinstance(kind, str) else None) or NO_FAMILY
        counted = 1.0 if 1 / noise <= ratio <= noise else ratio
        shares[family] += math.log(counted)
        rows.append({"index": index, "tool": tool, "kind": kind, "family": family, "time_us": after, "ratio": counted})
        before, spread_before = after, spread_after
        replayed += 1
    # from the first written kernel on, the record holds no source to replay: those calls go unattributed, together
    unreplayed = {KERNEL_CODE: len(journal) - replayed} if replayed < len(journal) else {}
    return {"rows": rows, "log_share_per_family": dict(shares), "replayed": replayed, "unreplayed": unreplayed}


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--journal", type=pathlib.Path, help="the route's journal file of the episode")
    source.add_argument("--db", type=pathlib.Path, help="a results DB holding the episode's plugin record")
    parser.add_argument("--setup", default="", help="with --db: the episode's setup")
    parser.add_argument("--episode", default="", help="with --db: the episode's label")
    parser.add_argument("--kernel", required=True, help="the bench kernel the episode ran")
    parser.add_argument("--preset", default="S", help="the size preset to measure at")
    parser.add_argument("--datatype", default="fp64", help="the graded datatype")
    parser.add_argument("--gpu", action="store_true", help="the episode ran on the GPU")
    parser.add_argument("--repeats", type=int, default=2, help="measurements per prefix; their spread is the noise")
    parser.add_argument("--out", type=pathlib.Path, required=True, help="the JSON to write")
    args = parser.parse_args()
    if args.journal is not None:
        journal = journal_from_file(args.journal)
    else:
        journal = journal_from_db(args.db, args.setup, args.episode, args.kernel)
    session, inputs, reference = start(Setup(args.kernel, args.preset, args.datatype, args.gpu), codegen="first")
    result = attribute(session, (inputs, reference), journal, args.repeats)
    args.out.write_text(json.dumps({"kernel": args.kernel, "preset": args.preset, **result}, indent=2), "utf-8")
    print(f"{result['replayed']} of {len(journal)} calls replayed -> {args.out}")


if __name__ == "__main__":
    main()
