# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Clustering recorded transformation requests: requests whose example pairs coincide after canonicalization, up to
the names of arrays and symbols, are one candidate for the authoring step; a different transformation is another."""

import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

from tactics.paths import REPO_ROOT
from tactics.stages.canonicalize import Targets
from tactics.stages.requests import Asked, Example, Request, cluster_requests


def program(name: str, x: str, y: str, size: str, body: str) -> str:
    return (
        f"{size} = dace.symbol('{size}')\n\n"
        "@dace.program\n"
        f"def {name}({x}: dace.float64[{size}], {y}: dace.float64[{size}]):\n"
        f"    for i in dace.map[0:{size}]:\n"
        f"        {y}[i] = {body.format(x=x)}\n"
    )


def reciprocal(x: str, y: str, size: str, divisor: float) -> Example:
    """Division by a constant as multiplication by its reciprocal."""
    return Example(
        before=program("before", x, y, size, f"{{x}}[i] / {divisor}"),
        after=program("after", x, y, size, f"{{x}}[i] * {1.0 / divisor}"),
    )


def squared(x: str, y: str, size: str, offset: float) -> Example:
    """A power of two as a product: another transformation."""
    return Example(
        before=program("before", x, y, size, f"({{x}}[i] + {offset}) ** 2"),
        after=program("after", x, y, size, f"({{x}}[i] + {offset}) * ({{x}}[i] + {offset})"),
    )


def asked(episode: str, kernel: str, name: str, examples: list[Example]) -> Asked:
    return Asked(episode, kernel, Request(name=name, description=name, rows=("doall0",), examples=examples))


def test_requests_whose_examples_differ_in_names_and_constants_are_one_cluster_and_another_is_its_own() -> None:
    first = asked("e1", "jacobi_1d", "reciprocal", [reciprocal("a", "b", "N", 2.0), reciprocal("a", "b", "N", 4.0)])
    renamed = asked("e2", "gemver", "times-inverse", [reciprocal("u", "v", "M", 8.0), reciprocal("u", "v", "M", 16.0)])
    other = asked("e3", "jacobi_1d", "square", [squared("a", "b", "N", 1.0), squared("a", "b", "N", 2.0)])

    clusters = cluster_requests([first, other, renamed], Targets())

    assert [[a.request.name for a in c.asked] for c in clusters] == [["reciprocal", "times-inverse"], ["square"]]
    assert clusters[0].kernels == {"jacobi_1d", "gemver"} and clusters[0].failed == []


def test_the_script_reads_request_files_and_reports_a_broken_example(tmp_path: Path) -> None:
    broken = Example(before="this is not python", after="neither")
    calls = [
        {
            "name": "reciprocal",
            "description": "x / c",
            "rows": ["doall0"],
            "reference": "",
            "examples": [e.model_dump() for e in (reciprocal("a", "b", "N", 2.0), broken)],
        },
    ]
    folder = tmp_path / "run" / "nestforge"
    folder.mkdir(parents=True)
    (folder / f"requests-{quote('e1@jacobi_1d', safe='')}.jsonl").write_text(
        "".join(json.dumps(c) + "\n" for c in calls)
    )
    out = tmp_path / "clusters.json"

    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "cluster_requests.py"), str(tmp_path), "--out", str(out)],
        check=True,
    )

    (cluster,) = json.loads(out.read_text())
    assert cluster["kernels"] == ["jacobi_1d"] and cluster["requests"][0]["name"] == "reciprocal"
    assert len(cluster["failed_examples"]) == 1 and "does not run" in cluster["failed_examples"][0][3]
