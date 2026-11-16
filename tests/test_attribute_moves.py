# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``scripts/attribute_moves.py`` replays a journal one call at a time and names each call's family; it stops at the
first written kernel, whose source the record keeps only as a hash."""

import json
import pathlib
import subprocess
import sys

import pytest

from nestforge.paths import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "attribute_moves.py"

pytestmark = pytest.mark.e2e

#: Three moves on jacobi_1d's canonical form as the route journals them, then a written kernel.
JOURNAL = [
    ["apply_move", {"kind": "doall-fusion", "labels": ["doall9", "doall10"]}],
    ["apply_move", {"kind": "tile-doall", "labels": ["doall6"], "tile_sizes": [32]}],
    ["apply_move", {"kind": "doall-fusion", "labels": ["doall7", "doall8"]}],
    ["write_kernel", {"name": "kernel_0", "source": "sha256:00", "language": "cpp"}],
]


def test_each_replayed_call_is_timed_and_named_by_family_up_to_the_first_written_kernel(tmp_path: pathlib.Path) -> None:
    journal, out = tmp_path / "journal.jsonl", tmp_path / "attribution.json"
    journal.write_text("".join(json.dumps(call) + "\n" for call in JOURNAL))

    subprocess.run(
        [sys.executable, str(SCRIPT), "--journal", str(journal), "--kernel", "jacobi_1d", "--preset", "S",
         "--repeats", "1", "--out", str(out)],
        check=True,
    )  # fmt: skip

    result = json.loads(out.read_text())
    assert [(row["index"], row["family"]) for row in result["rows"]] == [(0, "fusion"), (1, "tiling"), (2, "fusion")]
    assert all(row["time_us"] > 0 and row["ratio"] > 0 for row in result["rows"])
    assert (result["replayed"], result["unreplayed"]) == (3, {"kernel code": 1})
    assert set(result["log_share_per_family"]) == {"fusion", "tiling"}
