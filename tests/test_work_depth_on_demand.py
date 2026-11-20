# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Work and depth are not in the tree by default; ``get_work_depth`` gives them for the rows asked, at the session's
sizes or as formulas."""

from pathlib import Path

import dace

from nestforge.ir import introspect
from nestforge.session import Session

N = dace.symbol("N", dtype=dace.int64)
SIZE = 100


@dace.program
def scaled_sum(a: dace.float64[N], b: dace.float64[N]) -> None:
    for i in dace.map[0:N]:
        b[i] = 2.0 * a[i] + 1.0


def session(tmp_path: Path, sizes: dict[str, int] | None = None) -> Session:
    sut = Session(scaled_sum.to_sdfg(simplify=True), work_dir=str(tmp_path), sizes=sizes)
    sut.canonicalize()
    return sut


def doall(sut: Session) -> str:
    (label,) = [label for label in sut.row_index() if label.startswith("doall")]
    return label


def test_the_tree_keeps_reads_and_writes_and_leaves_out_work_and_depth(tmp_path: Path) -> None:
    sut = session(tmp_path)

    tree = sut.describe()

    assert "reads=" in tree and "writes=" in tree
    assert "work=" not in tree and "depth=" not in tree


def test_work_and_depth_come_at_the_session_sizes_or_as_formulas(tmp_path: Path) -> None:
    sut = session(tmp_path, {"N": SIZE})
    label = doall(sut)

    at_sizes = sut.work_depth([label])
    formulas = sut.work_depth([label], formula=True)

    assert f"{label}  work=" in at_sizes and "N" not in at_sizes.splitlines()[1], at_sizes
    assert "at N=100" in at_sizes.splitlines()[0]
    assert "N" in formulas.splitlines()[1], formulas


def test_an_unknown_label_is_named_and_nothing_is_computed_for_it(tmp_path: Path) -> None:
    sut = session(tmp_path)

    answer = sut.work_depth(["doall99"])

    assert answer.startswith("not-found") or "doall99" in answer, answer


def test_a_small_whole_cost_prints_as_an_integer_and_a_large_one_to_three_digits() -> None:
    n, t = dace.symbol("N"), dace.symbol("T")
    facts = dace.symbolic.Facts((), frozenset())

    small = introspect.cost_text(t + 1, facts, {"T": 10})
    large = introspect.cost_text(n * t, facts, {"N": 1000, "T": 10})
    open_count = introspect.cost_text(n * t, facts, {"N": 1000})

    assert (small, large, open_count) == ("11", "1.00e+4", "1.0e+3*T")
