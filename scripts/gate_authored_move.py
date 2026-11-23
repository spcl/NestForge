# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Gate one authored move before it enters the catalogue (``docs/design/transformation-requests.md``): its cluster's
examples, a fixed point of the cleanup after every application, every site of the corpus kernels against the bench's
reference, and NestForge's unit suite with the move loaded. The owner's review comes last: the report says
``review: pending`` and the move enters the catalogue only when the owner approves it.

    python scripts/gate_authored_move.py candidates/<kind> --examples examples.json \\
        [--kernels jacobi_2d heat_3d ...] [--preset S] [--suite] --out gate.json

``examples.json`` is a list of ``{"before": <source>, "after": <source>}``, the pairs of the requests in the move's
cluster. Without ``--kernels`` every corpus kernel is tried, which takes hours: canonicalization is minutes each.
"""

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

from tactics.corpus.bench import iter_dace_kernels
from tactics.paths import REPO_ROOT
from tactics.server import Setup, start
from tactics.stages.authored import CATALOGUE_ENV, load_move
from tactics.stages.gate import GateReport, example_failures, site_failures
from tactics.stages.requests import Example

#: The tests the suite step runs: the unit marker set of CI.
SUITE_MARKERS = "not integration and not judge and not cloudsc and not gpu and not vendor"


def suite_passes(catalogue: pathlib.Path) -> bool:
    """NestForge's unit suite with ``catalogue`` loaded in every session it starts."""
    env = os.environ | {CATALOGUE_ENV: str(catalogue)}
    command = [sys.executable, "-m", "pytest", "-q", "-x", "-m", SUITE_MARKERS, "-n", "auto"]
    return subprocess.run(command, cwd=REPO_ROOT, env=env, check=False).returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("entry", type=pathlib.Path, help="the candidate's folder, holding move.py")
    parser.add_argument("--examples", type=pathlib.Path, required=True)
    parser.add_argument("--kernels", nargs="*", help="corpus kernels to try every site of; default: all")
    parser.add_argument("--preset", default="S")
    parser.add_argument("--suite", action="store_true", help="also run the unit suite with the move loaded")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    move = load_move(args.entry)
    examples = [Example.model_validate(pair) for pair in json.loads(args.examples.read_text("utf-8"))]
    work = pathlib.Path(tempfile.mkdtemp(prefix="gate-"))
    catalogue = work / "catalogue"
    shutil.copytree(args.entry, catalogue / move.kind)
    os.environ[CATALOGUE_ENV] = str(catalogue)  # every session the corpus step starts loads this move alone

    report = GateReport(move.kind, example_failures(move, examples, work))
    for kernel in args.kernels or [k.short_name for k in iter_dace_kernels()]:
        session, inputs, reference = start(Setup(kernel=kernel, preset=args.preset, datatype="float64", gpu=False))
        if reference is None:
            report.site_failures.append(f"{kernel}: the bench has no reference to check against")
            continue
        sites, failures = site_failures(session, move.kind, inputs, reference)
        report.sites += sites
        report.site_failures += failures
    suite = suite_passes(catalogue) if args.suite else None
    passed = report.passed and suite is not False
    args.out.write_text(
        json.dumps(
            {
                "kind": move.kind,
                "examples": len(examples),
                "example_failures": report.example_failures,
                "sites": report.sites,
                "site_failures": report.site_failures,
                "suite": suite,
                "gate": "passed" if passed else "failed",
                "review": "pending",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"{move.kind}: gate {'passed' if passed else 'failed'} ({report.sites} sites) -> {args.out}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
