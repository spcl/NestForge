# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A move answers with the subtree it rewrote: the tree the agent holds, with that subtree and the rows around it put
in, is the new tree, and a change inside one loop answers with less than the whole tree."""

import inspect
from typing import Any

import dace
import pytest
import test_session_moves
from dace.frontend.python.parser import DaceProgram
from dace.sdfg import nodes

from tactics import server
from tactics.ir.introspect import enclosing_rows
from tactics.ir.subtree import ancestors, changed_lines, changed_view, rewritten_root, spans, subtree
from tactics.records import MoveStatus
from tactics.session import Move, Session
from tactics.stages.moves import ROLLBACK

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
        if not apply(session, move):
            continue
        answer = session.changed_tree()
        assert applied(held, answer) == session.describe(), (move, answer)
        session.apply_move(ROLLBACK, [])


T = dace.symbol("T", dace.int64, positive=True)


@dace.program
def time_steps(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]) -> None:
    for i in dace.map[0:N]:
        c[i] = a[i] * 0.5
    for _ in range(T):
        for i in dace.map[1 : N - 1]:
            b[i] = (a[i - 1] + a[i] + a[i + 1]) / 3.0
        for i in dace.map[1 : N - 1]:
            a[i] = b[i] * c[i]


@dace.program
def nested_time_steps(a: dace.float64[N, N], b: dace.float64[N, N]) -> None:
    for _ in range(T):
        for i in range(1, N):
            for j in dace.map[0:N]:
                b[i, j] = a[i - 1, j] + a[i, j]
            for j in dace.map[0:N]:
                a[i, j] = b[i, j] * 0.5


def one_top_row(old: str, new: str) -> str | None:
    """The top-level row whose span, in both trees, holds every line the change touched, found without
    :func:`rewritten_root`; ``None`` when the change touches more than one top-level row or none."""
    before, after = old.splitlines(), new.splitlines()
    touched_old, touched_new = changed_lines(before, after)
    old_rows, new_rows = spans(before), spans(after)
    for label, span in new_rows.items():
        if span.depth != 0 or label not in old_rows:
            continue
        held = old_rows[label]
        if all(held.start <= line < held.end for line in touched_old) and all(
            span.start <= line < span.end for line in touched_new
        ):
            return label
    return None


def check_answer(held: str, answer: str, new: str, what: object) -> bool:
    """The answer brings the held tree up to ``new``, and is a subtree whenever the change stays inside one
    top-level row; whether it was a subtree."""
    assert applied(held, answer) == new, (what, answer)
    whole = answer.startswith("The whole tree:")
    if one_top_row(held, new) is not None:
        assert not whole, f"{what}: the change stays inside {one_top_row(held, new)}, yet the answer is the whole tree"
    if not whole and answer != "The tree is unchanged.":
        assert len(answer.splitlines()) < len(new.splitlines()) + 1, what
    return not whole


#: The sizes the time-loop programs are canonicalized at.
SIZES = {"N": 8, "T": 2}
#: The tile size and the vector lanes a listed move is tried with when it takes them.
TILE = 2
LANES = 4
#: How many listed moves the batch test pairs.
PAIRED = 6
#: How many moves the chain test applies at most.
CHAIN = 8


def arguments(session: Session, move: Move) -> dict[str, Any]:
    """The arguments that apply ``move``: its kind, labels and kernel, and the sizes a tiling or vectorizing kind
    needs (one tile per dimension of the doall, when the program shows it)."""
    given: dict[str, Any] = {"kind": move.kind, "labels": list(move.labels)}
    if move.scope is not None:
        given["scope"] = move.scope
    if move.kind == "tile-doall":
        row = session.row_index().get(move.labels[0], (None, None))[0]
        dims = len(row.map.params) if isinstance(row, nodes.MapEntry) else 1
        given["tile_sizes"] = list(move.tile_sizes) or [TILE] * dims
    elif move.kind == "tile-for":
        given["tile_sizes"] = list(move.tile_sizes) or [TILE]
    elif move.kind == "vectorize-doall":
        given["lanes"] = list(move.lanes) or [LANES]
    return given


def apply(session: Session, move: Move) -> bool:
    """Whether ``move``, with the sizes it needs, applied."""
    return session.apply_move(**arguments(session, move)).status is MoveStatus.APPLIED


def rows(session: Session) -> set[str]:
    """Every row label of the program, the collapsed doalls the tree does not print included."""
    return set(session.row_index())


def time_session(program: DaceProgram = time_steps) -> Session:
    session = Session(program.to_sdfg(simplify=True), sizes=SIZES)
    session.canonicalize()
    return session


PROGRAMS = pytest.mark.parametrize("program", [time_steps, nested_time_steps], ids=["time_steps", "nested_time_steps"])


@PROGRAMS
def test_a_move_inside_a_loop_answers_with_that_subtree_never_the_whole_tree(program: DaceProgram) -> None:
    """Every listed move (with the sizes it needs), its rollback, and pairs of moves as one batch: the agent's tree
    is brought up to date, and the answer is the whole tree only when the change reaches more than one top-level
    row."""
    session = time_session(program)
    session.cpf()
    subtrees = 0
    moves = session.list_moves()
    for move in moves:
        held = session.cpf()
        if not apply(session, move):
            continue
        subtrees += check_answer(held, session.changed_tree(), session.describe(), move)
        after = session.describe()
        session.apply_move(ROLLBACK, [])
        check_answer(after, session.changed_tree(), session.describe(), ("rollback", move))
        assert session.describe() == held, ("rollback restores the tree", move)
    assert subtrees > 0, "no move inside the time loop answered with a subtree"
    for first in moves[:PAIRED]:
        for second in moves[:PAIRED]:
            if first == second:
                continue
            held = session.cpf()
            if session.apply_moves([arguments(session, first), arguments(session, second)]).status != "applied":
                continue
            check_answer(held, session.changed_tree(), session.describe(), (first, second))
            session.apply_move(ROLLBACK, [])
            session.apply_move(ROLLBACK, [])


@PROGRAMS
def test_a_label_a_rollback_drops_is_never_handed_out_again(program: DaceProgram) -> None:
    """Labels only grow: a row a move made and its rollback took away keeps its label out of every later program,
    so a label the agent read never comes to mean another row."""
    session = time_session(program)
    dropped: set[str] = set()
    for move in session.list_moves():
        held = rows(session)
        if not apply(session, move):
            continue
        made = rows(session)
        assert not made & dropped, (move, made & dropped)
        session.apply_move(ROLLBACK, [])
        assert rows(session) == held, move
        dropped |= made - held
    assert dropped, "no move made a row of its own"


@PROGRAMS
def test_a_removed_label_never_comes_back_along_a_chain_of_moves(program: DaceProgram) -> None:
    """Along a chain of moves, a removed row's label never names a new row. Only an undo brings labels back, with
    the rows they named (an exact inverse of the last move is answered by one, ``UNDONE_BY``)."""
    session = time_session(program)
    removed: set[str] = set()
    for _ in range(CHAIN):
        before = rows(session)
        results = (session.apply_move(**arguments(session, m)) for m in session.list_moves() if m.kind != ROLLBACK)
        result = next((done for done in results if done.status is MoveStatus.APPLIED), None)
        if result is None:
            break
        after = rows(session)
        if result.kind == ROLLBACK or result.reason.startswith(f"{ROLLBACK}:"):  # an undo, or an inverse it answered
            removed -= after  # the rows the undo restored, with the labels they had
        assert not after & removed, (result, after & removed)
        removed |= before - after
    assert removed, "no move of the chain removed a row"


def tiles(session: Session) -> list[Move]:
    found = [move for move in session.list_moves() if move.kind == "tile-doall"]
    assert found, "the program offers tile-doall"
    return found


def test_an_expanded_row_a_move_replaces_leaves_its_replacement_expanded() -> None:
    """Expanding a doall and then tiling it keeps the tiled rows open: the agent does not expand again."""
    session = time_session()
    applied_any = False
    for move in tiles(session):
        session.expand(list(move.labels))
        held = rows(session)
        if not apply(session, move):
            session.collapse(list(move.labels))
            continue
        applied_any = True
        made = rows(session) - held
        assert made and all(session.is_open(label) for label in made), (move, made)
        session.apply_move(ROLLBACK, [])
        session.collapse(list(move.labels))
    assert applied_any, "no tile-doall applied"


def test_a_row_inside_an_expanded_row_stays_open_after_a_move_inside_it() -> None:
    """Expanding a state and then tiling a doall in it: the state, and with it the tiled rows, stay open."""
    session = time_session()
    applied_any = False
    for move in tiles(session):
        obj, state = session.row_index()[move.labels[0]]
        around = next(row for row in enclosing_rows(obj, state) if row.startswith("state"))
        session.expand([around])
        held = rows(session)
        if not apply(session, move):
            session.collapse([around])
            continue
        applied_any = True
        made = rows(session) - held
        assert session.is_open(around) and all(session.is_open(label) for label in made), move
        session.apply_move(ROLLBACK, [])
        session.collapse([around])
    assert applied_any, "no tile-doall applied"


def test_rows_outside_the_rewritten_subtree_keep_their_labels() -> None:
    """A move inside the time loop leaves the first state's lines as they were, labels included, so the agent's
    copy of them stays valid."""
    session = time_session()
    held = session.cpf()
    printed = spans(held.splitlines())
    first = min((label for label, span in printed.items() if span.depth == 0), key=lambda label: printed[label].start)
    inside = [
        move
        for move in session.list_moves()
        if move.labels
        and all(first not in enclosing_rows(*session.row_index()[label]) for label in move.labels)
        and all(label != first for label in move.labels)
    ]
    applied_any = False
    for move in inside:
        if not apply(session, move):
            continue
        applied_any = True
        assert subtree(session.describe(), first) == subtree(held, first), move
        session.apply_move(ROLLBACK, [])
    assert applied_any, "no move inside the time loop applied"


def test_what_the_agent_opened_after_a_move_stays_open_after_its_rollback() -> None:
    session = time_session()
    opened = next(label for label in rows(session) if label.startswith("doall"))
    move = next(m for m in session.list_moves() if opened not in m.labels and apply(session, m))
    assert opened in session.row_index(), move
    session.expand([opened])

    session.apply_move(ROLLBACK, [])

    assert session.is_open(opened), move


def scoped_then_rewound(session: Session) -> dict[str, int]:
    """The namer's counters after the default scopes and one move inside a kernel, then ``rewind`` (back to before
    the kernels); returns the counters the kernel phase reached."""
    before = rows(session)
    session.run_defaults("scopes")
    for kernel in session.list_kernels():
        if any(apply(session, move) for move in session.list_moves(scope=kernel.name) if move.scope == kernel.name):
            break
    reached = dict(session.namer.counters)
    session.rewind()
    assert rows(session) == before, "rewind gives the program before its kernels back"
    return reached


def test_after_rewind_no_counter_falls_below_what_the_kernel_phase_reached() -> None:
    """A label the kernel phase handed out is never handed out again after ``rewind``."""
    session = time_session()
    reached = scoped_then_rewound(session)
    assert all(session.namer.counters.get(kind, 0) >= count for kind, count in reached.items()), (
        reached,
        session.namer.counters,
    )


def test_a_version_one_journal_rewinds_to_the_namer_it_was_written_with() -> None:
    """Before labels only grew, ``rewind`` put the namer back as the program before the kernels had it, and a
    version-1 journal names the rows after it so."""
    legacy = time_session()
    legacy.reissue_dropped_labels = True
    kept = dict(legacy.namer.counters)
    scoped_then_rewound(legacy)
    assert legacy.namer.counters == kept


def test_a_session_rebuilt_from_its_journal_is_the_program_the_agent_left_labels_included() -> None:
    """The judge rebuilds a dead session by repeating its journal (rollbacks included, refusals left out): the
    rebuilt program, labels and namer, is the one the agent holds, so its next call names the same rows."""
    live = time_session()
    journal: list[tuple[str, dict[str, Any]]] = []

    def change(given: dict[str, Any]) -> bool:
        done = server.applied(server.answer(live, "apply_move", given, ({}, None), timed=False))
        if done:
            journal.append(("apply_move", given))
        return done

    moves = [arguments(live, move) for move in live.list_moves() if move.kind != ROLLBACK]
    first = next(given for given in moves if change(given))
    assert change({"kind": ROLLBACK, "labels": []})
    assert any(change(given) for given in moves if given != first)
    refused = live.apply_moves([first, {"kind": "no-such-move", "labels": []}])
    assert refused.status == "refused", refused  # not journaled, and it changes no label
    change(first)  # the first move again on the program the second left; journaled only when it applies

    rebuilt = time_session()
    for tool, given in journal:
        assert server.applied(server.answer(rebuilt, tool, given, ({}, None), timed=False)), (tool, given)

    assert rebuilt.describe() == live.describe() and rows(rebuilt) == rows(live)
    assert rebuilt.namer.counters == live.namer.counters and rebuilt.namer.retired == live.namer.retired


def test_a_tree_read_with_bodies_is_answered_with_bodies_and_only_the_subtree() -> None:
    """The answer prints as the agent read the tree; printed plainly it would differ on every body line and come
    back whole."""
    session = time_session(nested_time_steps)
    held = session.cpf(bodies=True)
    subtrees = 0
    for move in session.list_moves():
        if not apply(session, move):
            continue
        subtrees += check_answer(held, session.changed_tree(), session.describe(bodies=True), move)
        session.apply_move(ROLLBACK, [])
        held = session.cpf(bodies=True)
    assert subtrees > 0
