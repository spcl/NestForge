# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Parts of the printed tree: one row's subtree, and what a change rewrote. A move answers with the subtree it
rewrote instead of the whole program, since the agent holds the rest from the tree it last read
(``docs/design/subtree-answers.md``)."""

import difflib
from dataclasses import dataclass

from nestforge.ir.introspect import BLANK, ELBOW, PIPE, TEE

#: Characters of one level of the tree's indentation.
LEVEL = len(PIPE)


@dataclass(frozen=True, slots=True)
class Span:
    """A row's lines: its own at ``start``, its subtree up to ``end`` (exclusive), at ``depth``."""

    start: int
    end: int
    depth: int


def indent(line: str) -> int:
    """How many indentation levels open ``line``."""
    levels = 0
    while line[levels * LEVEL : (levels + 1) * LEVEL] in (PIPE, BLANK):
        levels += 1
    return levels


def row_label(line: str) -> str | None:
    """The label of the row ``line`` prints, or ``None`` for a line that is no row (the header, a body, a note)."""
    rest = line[indent(line) * LEVEL :]
    if not rest.startswith((TEE, ELBOW)):
        return None
    text = rest[len(TEE) :]
    if text.startswith("[") and "] " in text:  # a session id stamped in front
        text = text.split("] ", 1)[1]
    return text.split(maxsplit=1)[0] if text.strip() else None


def spans(lines: list[str]) -> dict[str, Span]:
    """Every row's span, by label."""
    found: dict[str, Span] = {}
    for start, line in enumerate(lines):
        label = row_label(line)
        if label is None:
            continue
        depth = indent(line)
        end = start + 1
        while end < len(lines) and indent(lines[end]) > depth:
            end += 1
        found[label] = Span(start, end, depth)
    return found


def subtree(tree: str, label: str) -> str | None:
    """The lines of row ``label`` and everything under it, or ``None`` when no row has that label."""
    lines = tree.splitlines()
    span = spans(lines).get(label)
    return None if span is None else "\n".join(lines[span.start : span.end])


def ancestors(found: dict[str, Span], label: str) -> list[str]:
    """The labels of the rows around ``label``, outermost first."""
    span = found[label]
    around = [
        (other.depth, name)
        for name, other in found.items()
        if other.start < span.start and other.end >= span.end and other.depth < span.depth
    ]
    return [name for _, name in sorted(around)]


def changed_lines(old: list[str], new: list[str]) -> tuple[set[int], set[int]]:
    """The lines of ``old`` and of ``new`` a line diff does not match; an insertion counts the line before it."""
    before: set[int] = set()
    after: set[int] = set()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        before |= set(range(i1, i2)) if i2 > i1 else {max(i1 - 1, 0)}
        after |= set(range(j1, j2)) if j2 > j1 else {max(j1 - 1, 0)}
    return before, after


def covers(found: dict[str, Span], label: str, changed: set[int], path: list[str]) -> bool:
    """Whether the span of ``label``, with the rows ``path`` around it, holds every line of ``changed``."""
    span = found[label]
    allowed = {found[name].start for name in path}
    return all(span.start <= line < span.end or line in allowed for line in changed)


def rewritten_root(old: str, new: str) -> str | None:
    """The deepest row whose subtree holds every line the change from ``old`` to ``new`` touched, in both trees,
    besides the lines of the rows around it; ``None`` when no row does (the change reaches the top level)."""
    before, after = old.splitlines(), new.splitlines()
    old_rows, new_rows = spans(before), spans(after)
    touched_old, touched_new = changed_lines(before, after)
    shared = [label for label in new_rows if label in old_rows]
    for label in sorted(shared, key=lambda name: -new_rows[name].depth):
        path = ancestors(new_rows, label)
        if path != ancestors(old_rows, label):
            continue
        if covers(old_rows, label, touched_old, path) and covers(new_rows, label, touched_new, path):
            return label
    return None


def changed_view(old: str | None, new: str) -> str:
    """What the agent needs to bring the tree ``old`` it holds up to ``new``: the rewritten subtree under the rows
    around it, or the whole tree when the change reaches the top level (or the agent holds none)."""
    if old == new:
        return "The tree is unchanged."
    root = None if old is None else rewritten_root(old, new)
    if root is None:
        return f"The whole tree:\n{new}"
    lines = new.splitlines()
    found = spans(lines)
    path = [lines[found[name].start] for name in ancestors(found, root)]
    span = found[root]
    head = f"Rows outside {root} are unchanged; {root} now reads, under the rows around it:"
    return "\n".join([head, *path, *lines[span.start : span.end]])
