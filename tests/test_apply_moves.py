# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``apply_moves``: several moves in order, all or nothing. After each move the program is built and run at the small
check sizes against the reference; the first move refused or found wrong takes the whole batch back, and the answer
names it, the substep and why. Each applied move stays its own change for ``rollback``."""

from pathlib import Path
from typing import Any

import dace
import numpy as np
from dace.sdfg import nodes

from tactics.session import CheckData, Session
from tactics.stages.performance import Reference

N = dace.symbol("N", dtype=dace.int64)
SIZE = 16


@dace.program
def three_stages(A: dace.float64[N], C: dace.float64[N]) -> None:
    T = np.empty_like(A)
    U = np.empty_like(A)
    for i in dace.map[0:N]:
        T[i] = A[i] + 1.0
    for i in dace.map[0:N]:
        U[i] = T[i] * 2.0
    for i in dace.map[0:N]:
        C[i] = U[i] - 3.0


def session(tmp_path: Path) -> Session:
    sut = Session(three_stages.to_sdfg(simplify=True), work_dir=str(tmp_path), sizes={"N": SIZE})
    sut.canonicalize()
    return sut


def check(offset: float = 0.0) -> CheckData:
    """The data at the check size; ``offset`` makes the reference wrong."""
    a = np.linspace(0.0, 1.0, SIZE)
    want = (a + 1.0) * 2.0 - 3.0 + offset
    return CheckData({"A": a, "C": np.zeros(SIZE)}, Reference({"C": want}, 1e-12, 1e-12), {"N": SIZE})


def fusions(sut: Session) -> list[dict[str, Any]]:
    """The first fusion offered, as an agent passes it to apply_moves."""
    first = sut.list_moves("doall-fusion")[0]
    return [{"kind": first.kind, "labels": list(first.labels)}]


def doalls(sut: Session) -> int:
    return sum(isinstance(n, nodes.MapEntry) for n, _ in sut.sdfg.all_nodes_recursive())


def test_a_batch_applies_every_move_checks_each_and_rollback_takes_back_the_last(tmp_path: Path) -> None:
    sut = session(tmp_path)
    before = doalls(sut)
    (fusion,) = fusions(sut)
    untouched = next(m.labels[0] for m in sut.list_moves("tile-doall") if m.labels[0] not in fusion["labels"])
    tile = {"kind": "tile-doall", "labels": [untouched], "tile_sizes": [4]}

    done = sut.apply_moves([fusion, tile], check())

    assert (done.status, len(done.results), done.unchecked) == ("applied", 2, "")
    assert doalls(sut) == before - 1 + 1, "one fusion took a doall away, the tile added one"
    sut.apply_move("rollback", [])
    assert doalls(sut) == before - 1, "rollback takes back the tile only"


def test_a_refused_move_names_its_index_and_takes_the_whole_batch_back(tmp_path: Path) -> None:
    sut = session(tmp_path)
    tree = sut.describe()
    (fusion,) = fusions(sut)

    done = sut.apply_moves([fusion, {"kind": "doall-fusion", "labels": ["no_such_row", "doall0"]}], check())

    assert (done.status, done.failed_at, done.substep) == ("refused", 1, "apply")
    assert "no_such_row" in done.reason
    assert sut.describe() == tree and sut.applied == []


def test_a_move_that_leaves_wrong_numbers_is_named_and_taken_back(tmp_path: Path) -> None:
    sut = session(tmp_path)
    tree = sut.describe()

    done = sut.apply_moves(fusions(sut), check(offset=1.0))

    assert (done.status, done.failed_at, done.substep) == ("refused", 0, "check")
    assert "['C'] differ" in done.reason
    assert sut.describe() == tree


def test_without_a_reference_the_batch_applies_and_says_it_went_unchecked(tmp_path: Path) -> None:
    sut = session(tmp_path)
    data = check()

    done = sut.apply_moves(fusions(sut), CheckData(data.inputs, None, data.sizes))

    assert done.status == "applied" and "no reference" in done.unchecked
