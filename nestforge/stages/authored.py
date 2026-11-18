# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Authored moves: moves written from agents' transformation requests (``docs/design/transformation-requests.md``)
and kept in a catalogue folder that every session loads. Each is a DaCe pattern transformation; NestForge offers it
wherever DaCe's pattern matcher finds it, applies it through ``apply_to`` (its ``can_be_applied`` checked first), and
checks the result against the oracle like every move. They are tested, not proven.

A catalogue entry is a folder ``<kind>/move.py`` defining:

- ``TRANSFORMATION``: the DaCe ``PatternTransformation`` subclass;
- ``ROWS``: the names of its pattern nodes the move's rows bind to, in the order ``apply_move`` takes the labels;
- ``TEXT``: what the move does, in one or two sentences, as ``list_move_kinds`` shows it.
"""

import importlib.util
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import dace
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowBlock, LoopRegion, SDFGState
from dace.transformation import transformation as xf
from dace.transformation.passes.pattern_matching import match_patterns

from nestforge.paths import REPO_ROOT

#: The family every authored move belongs to; a run switches them off together.
AUTHORED_FAMILY = "authored"
#: The environment variable naming the catalogue folder; unset, the repository's ``catalogue``.
CATALOGUE_ENV = "NESTFORGE_CATALOGUE"
#: The row keyword of each pattern node type, as the tree and ``list_move_kinds`` name rows.
ROW_KINDS: dict[type, str] = {
    nodes.MapEntry: "doall",
    LoopRegion: "for",
    SDFGState: "state",
    nodes.LibraryNode: "lib_<family>",
}


@dataclass(frozen=True, slots=True)
class AuthoredMove:
    """One catalogue entry, loaded."""

    kind: str
    transformation: type[xf.PatternTransformation]
    rows: tuple[str, ...]
    text: str

    def pattern_node(self, field: str) -> xf.PatternNode:
        """The pattern node ``field`` of the transformation, declared on it or a base class."""
        for cls in self.transformation.__mro__:
            if field in vars(cls):
                return vars(cls)[field]
        raise LookupError(f"{self.transformation.__name__} has no pattern node {field!r}")

    def row_kinds(self) -> tuple[str, ...]:
        """The row keyword of each row, from its pattern node's type."""
        return tuple(row_kind(self.pattern_node(field).node) for field in self.rows)


def row_kind(node_type: type) -> str:
    return next((name for cls, name in ROW_KINDS.items() if issubclass(node_type, cls)), node_type.__name__)


class CatalogueError(ValueError):
    """A catalogue entry that does not load; the message names the entry and what is wrong."""


def catalogue_dir() -> Path:
    return Path(os.environ.get(CATALOGUE_ENV) or REPO_ROOT / "catalogue")


def load_move(folder: Path) -> AuthoredMove:
    """The authored move of ``folder`` (``<kind>/move.py``).

    :raises CatalogueError: The file does not run or lacks a field, or a row is no pattern node."""
    path = folder / "move.py"
    # a catalogue entry is a file, not a package module: loaded by path, as replay loads a solution
    spec = importlib.util.spec_from_file_location(f"nestforge_authored_{folder.name.replace('-', '_')}", path)
    if spec is None or spec.loader is None:
        raise CatalogueError(f"{folder.name}: no move.py")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as err:  # noqa: BLE001 -- any failure of an entry is its error, named
        raise CatalogueError(f"{folder.name}: move.py does not run: {type(err).__name__}: {err}") from err
    fields = vars(module)
    missing = [name for name in ("TRANSFORMATION", "ROWS", "TEXT") if name not in fields]
    if missing:
        raise CatalogueError(f"{folder.name}: move.py defines no {missing}")
    transformation = fields["TRANSFORMATION"]
    if not (isinstance(transformation, type) and issubclass(transformation, xf.PatternTransformation)):
        raise CatalogueError(f"{folder.name}: TRANSFORMATION is no DaCe PatternTransformation")
    move = AuthoredMove(folder.name, transformation, tuple(fields["ROWS"]), str(fields["TEXT"]))
    try:
        move.row_kinds()
    except LookupError as err:
        raise CatalogueError(f"{folder.name}: {err}") from err
    return move


def load_catalogue(folder: Path) -> dict[str, AuthoredMove]:
    """Every entry of ``folder`` by kind; an absent folder is an empty catalogue.

    :raises CatalogueError: An entry does not load."""
    if not folder.is_dir():
        return {}
    return {move.kind: move for move in map(load_move, sorted(p for p in folder.iterdir() if p.is_dir()))}


def bound_node(match: xf.PatternTransformation, move: AuthoredMove, field: str) -> nodes.Node | ControlFlowBlock:
    """The node or block of the program ``match`` binds the pattern node ``field`` to: the pattern node resolves its
    match itself."""
    return move.pattern_node(field).__get__(match, move.transformation)


def matches(sdfg: dace.SDFG, move: AuthoredMove) -> Iterator[tuple[nodes.Node | ControlFlowBlock, ...]]:
    """The rows of every place in ``sdfg`` the move's pattern matches and ``can_be_applied`` accepts."""
    for match in match_patterns(sdfg, move.transformation):
        yield tuple(bound_node(match, move, field) for field in move.rows)


def apply_authored(sdfg: dace.SDFG, move: AuthoredMove, rows: Sequence[nodes.Node | ControlFlowBlock]) -> str | None:
    """Apply the move on ``rows`` of ``sdfg``; ``None`` when applied, else why it is refused (nothing changed)."""
    where = dict(zip(move.rows, rows, strict=True))
    owner = rows[0].sdfg if isinstance(rows[0], ControlFlowBlock) else None
    try:
        move.transformation.apply_to(owner or sdfg_of(sdfg, rows[0]), verify=True, save=False, **where)
    except ValueError as err:
        return f"blocked by {move.transformation.__name__}: {err}"
    return None


def sdfg_of(sdfg: dace.SDFG, node: nodes.Node | ControlFlowBlock) -> dace.SDFG:
    """The (nested) SDFG of ``sdfg`` that holds ``node``."""
    for nested in sdfg.all_sdfgs_recursive():
        if any(n is node for state in nested.states() for n in state.nodes()):
            return nested
    raise LookupError(f"{node} is in no SDFG of {sdfg.name}")
