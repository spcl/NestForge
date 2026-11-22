# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A move answers with the subtree it rewrote: the tree the agent holds, with that subtree and the rows around it put
in, is the new tree, and a change inside one loop answers with less than the whole tree."""

import inspect

import dace
import pytest
import test_session_moves
from dace.frontend.python.parser import DaceProgram

from nestforge.ir.subtree import ancestors, changed_view, rewritten_root, spans, subtree
from nestforge.records import MoveStatus
from nestforge.session import Session
from nestforge.stages.moves import ROLLBACK

N = dace.symbol("N", dace.int64, positive=True)

OLD = """SDFG 'p'
|- for0  i in [1, N)
|  `- state1  reads=['a'] writes=['a']
`- for2  j in [0, N)
   |- state3  reads=['b'] writes=['c']
   |  `- doall0  [k in 0:N]  reads=['b'] writes=['c']
   |     : c[k] = b[k]
   `- state4  reads=['c'] writes=['d']"""


def applied(old: str, answer: str) -> str:
    """The tree an agent holding ``old`` has after reading ``answer``: the whole tree, or ``old`` with the answer's
    subtree and the rows around it put in place."""
    head, *rest = answer.splitlines()
    if head == "The whole tree:":
        return "\n".join(rest)
    if head == "The tree is unchanged.":
        return old
    lines = old.splitlines()
    found = spans(lines)
    root = head.split()[2]
    path = ancestors(found, root)
    for name, line in zip(path, rest, strict=False):
        lines[found[name].start] = line
    span = found[root]
    return "\n".join([*lines[: span.start], *rest[len(path) :], *lines[span.end :]])


def test_a_subtree_is_its_row_and_every_line_indented_under_it() -> None:
    assert subtree(OLD, "state3") == "\n".join(OLD.splitlines()[4:7])
    assert subtree(OLD, "for0") == "\n".join(OLD.splitlines()[1:3])
    assert subtree(OLD, "nothing") is None


def test_a_change_inside_one_loop_answers_with_that_loop_under_no_other_row() -> None:
    new = OLD.replace("c[k] = b[k]", "c[k] = 2 * b[k]")

    answer = changed_view(OLD, new)

    assert rewritten_root(OLD, new) == "doall0"
    assert "for0" not in answer and "c[k] = 2 * b[k]" in answer
    assert applied(OLD, answer) == new


def test_a_row_gone_from_the_top_level_answers_with_the_whole_tree() -> None:
    lines = OLD.splitlines()
    new = "\n".join([lines[0], *lines[3:]])

    answer = changed_view(OLD, new)

    assert answer.startswith("The whole tree:")
    assert applied(OLD, answer) == new


def test_an_agent_that_read_no_tree_gets_the_whole_tree() -> None:
    assert changed_view(None, OLD) == f"The whole tree:\n{OLD}"


FIXTURES = [
    (name, value)
    for name, value in inspect.getmembers(test_session_moves)
    if isinstance(value, DaceProgram) and name in ("two_loops", "vertical_pair", "two_indep", "calls_then_loops")
]


@pytest.mark.parametrize("name, program", FIXTURES)
def test_every_listed_move_answers_with_what_brings_the_held_tree_up_to_date(name: str, program: DaceProgram) -> None:
    session = Session(program.to_sdfg(simplify=True), sizes={"N": 8})
    session.canonicalize()
    for move in session.list_moves():
        held = session.cpf()
        result = session.apply_move(move.kind, move.labels)
        if result.status is not MoveStatus.APPLIED:
            continue
        answer = session.changed_tree()
        assert applied(held, answer) == session.describe(), (move, answer)
        session.apply_move(ROLLBACK, [])
