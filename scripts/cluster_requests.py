# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cluster the transformation requests a round of episodes recorded, for the authoring step between rounds
(``docs/design/transformation-requests.md``): requests that share an example pair after canonicalization, up to the
names of arrays and symbols, are one candidate. Candidates come out most asked first (kernels, then episodes).

    python scripts/cluster_requests.py <run folder>... --out clusters.json

A request file is ``requests-<episode>@<kernel>.jsonl`` (quoted), one recorded request per line, anywhere under the
folders given.
"""

import argparse
import json
import pathlib
from collections.abc import Iterator
from urllib.parse import unquote

from tactics.stages.canonicalize import Targets
from tactics.stages.requests import Asked, Cluster, Request, cluster_requests

#: The prefix of a request file's name, as the plugin route writes it.
PREFIX = "requests-"


def recorded(folders: list[pathlib.Path]) -> Iterator[Asked]:
    """Every request kept under ``folders``, with its episode and kernel from the file name."""
    for folder in folders:
        for path in sorted(folder.rglob(f"{PREFIX}*.jsonl")):
            episode, _, kernel = unquote(path.stem.removeprefix(PREFIX)).rpartition("@")
            for line in path.read_text("utf-8").splitlines():
                if line.strip():
                    arguments = json.loads(line)  # the tool call's arguments; ``reference`` is no part of a request
                    fields = {key: arguments[key] for key in Request.model_fields if key in arguments}
                    yield Asked(episode, kernel, Request.model_validate(fields))


def summary(cluster: Cluster) -> dict[str, object]:
    return {
        "kernels": sorted(cluster.kernels),
        "episodes": len(cluster.episodes),
        "requests": [
            {"episode": a.episode, "kernel": a.kernel, "name": a.request.name, "description": a.request.description}
            for a in cluster.asked
        ],
        "failed_examples": [list(failure) for failure in cluster.failed],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="+", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    clusters = cluster_requests(recorded(args.folders), Targets())
    args.out.write_text(json.dumps([summary(c) for c in clusters], indent=2) + "\n")
    print(f"{sum(len(c.asked) for c in clusters)} requests in {len(clusters)} clusters -> {args.out}")


if __name__ == "__main__":
    main()
