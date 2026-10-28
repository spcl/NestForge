# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""velocity_tendencies: one 64-wide column tile around the edge phase.

``default_moves`` strip-mines each nest on its own, so the next sweep of ``vn`` and ``vt`` reloads them.
The edge states become one kernel, and that kernel is rewritten as one OpenMP loop over tiles of 64
columns, with the level loops inside the tile. No menu move expresses the tile. ``place`` runs before the
source is bound, because a later ``place`` would drop the library.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from dace.libraries.standard.nodes.external_call import external_calls
from dace.sdfg.state import ControlFlowBlock

from nestforge.ir.names import in_order
from nestforge.replay import Context, ReplayError, Script

KERNEL = "velocity_tendencies"
CONFIGS = (Context("cpu", 72, "M"),)
DECOYS: dict = {}

_TILE = 64
_EDGE = re.compile(r"(?:^|_)(?:vt|vn_ie|z_kin_hor_e|z_vt_ie|z_w_concorr_me|z_v_grad_w)(?:_|$)")
_POWERS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


class FuseError(Exception):
    """The outlined source is not a sequence of ``__dace_tile`` loops this rewrite understands."""


def _dump(name: str, text: str) -> None:
    folder = os.environ.get("VT_DUMP")
    if folder:
        Path(folder, name).write_text(text)


def _brace_end(text: str, open_at: int) -> int:
    """Index of the ``}`` matching the ``{`` at ``open_at``."""
    depth = 0
    i = open_at
    limit = len(text)
    while i < limit:
        char = text[i]
        if char == '"':
            i += 1
            while i < limit and text[i] != '"':
                if text[i] == "\\":
                    i += 1
                i += 1
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise FuseError("a tile loop's brace is not closed")


def _tile_regions(text: str) -> list[tuple[int, int, int]]:
    """``(pragma start, brace index, closing brace)`` of each outermost ``__dace_tile`` parallel loop."""
    regions: list[tuple[int, int, int]] = []
    start = 0
    needle = "#pragma omp parallel for"
    while True:
        at = text.find(needle, start)
        if at < 0:
            return regions
        line_end = text.find("\n", at)
        if line_end < 0:
            raise FuseError("a pragma is not followed by a for")
        header = text[line_end + 1 : line_end + 400]
        if not re.match(r"[ \t]*for \([^;]*\b__dace_tile\b", header):
            start = at + len(needle)
            continue
        brace = text.find("{", line_end)
        if brace < 0:
            raise FuseError("a tile for has no brace")
        end = _brace_end(text, brace)
        regions.append((at, brace, end))
        start = end + 1


def _tile_size(body: str, header: str) -> int:
    """The column stride this tile loop was generated with."""
    ceil = re.search(r"int_ceil\s*\([^,]+,\s*(\d+)\s*\)", header)
    if ceil and int(ceil.group(1)) in _POWERS:
        return int(ceil.group(1))
    products = [int(a) * int(b) for a, b in re.findall(r"\(\s*(\d+)\s*\*\s*__dace_tile\s*\)\s*\*\s*(\d+)", body)]
    if products and len(set(products)) == 1 and products[0] in _POWERS:
        return products[0]
    direct = [int(n) for n in re.findall(r"__dace_tile\s*\*\s*(\d+)", body)]
    direct += [int(n) for n in re.findall(r"(?<![\w.])(\d+)\s*\*\s*__dace_tile", body)]
    direct = [n for n in direct if n in _POWERS]
    if direct and len(set(direct)) == 1:
        return direct[0]
    shown = header.strip().splitlines()
    raise FuseError(f"tile stride is not one power of two; header: {shown[0] if shown else header!r}")


def _retile(text: str, old: int, new: int = _TILE) -> str:
    """Rewrite tile stride ``old`` to ``new`` in bounds that name ``__dace_tile``."""
    if old == new:
        return text

    def product(match: re.Match[str]) -> str:
        if int(match.group(1)) * int(match.group(2)) == old:
            return f"(__dace_tile * {new})"
        return match.group(0)

    text = re.sub(r"\(\s*(\d+)\s*\*\s*__dace_tile\s*\)\s*\*\s*(\d+)", product, text)
    text = text.replace(f"(__dace_tile + 1) * {old}", f"(__dace_tile + 1) * {new}")
    text = text.replace(f"__dace_tile * {old}", f"__dace_tile * {new}")
    text = re.sub(rf"(?<![\w.]){old}\s*\*\s*__dace_tile\b", f"{new} * __dace_tile", text)
    text = re.sub(rf"int_ceil\s*\(([^,]+),\s*{old}\s*\)", rf"int_ceil(\1, {new})", text)
    text = re.sub(
        rf"\(\s*([^()]+)\s*\+\s*{old - 1}\s*\)\s*/\s*{old}\b",
        rf"(\1 + {new - 1}) / {new}",
        text,
    )
    return text


def _scope_gap(gap: str) -> bool:
    """A gap of only state-scope braces, with as many opens as closes."""
    return all(char in "{}\n\t " for char in gap) and gap.count("{") == gap.count("}")


def _paren_end(text: str, open_at: int) -> int:
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    raise FuseError("a call's parenthesis is not closed")


def _static_functions(text: str) -> list[str]:
    return re.findall(r"static inline void (\w+)\(", text)


def _pass_columns(text: str, names: list[str]) -> str:
    """Give each helper that walks a tile the tile's column bounds, and pass them at every call."""
    needing: set[str] = set()
    changed = True
    while changed:
        changed = False
        for name in names:
            if name in needing:
                continue
            body = _function_body(text, name)
            calls = any(re.search(rf"\b{re.escape(other)}\s*\(", body) for other in needing)
            if "__col_lo" in body or calls:
                needing.add(name)
                changed = True
    for name in needing:
        text = re.sub(
            rf"(static inline void {name}\([^;]*?)\) \{{",
            r"\1, int64_t __col_lo, int64_t __col_hi) {",
            text,
            count=1,
        )
    for name in needing:
        text = _append_call_args(text, name)
    return text


def _function_body(text: str, name: str) -> str:
    match = re.search(rf"static inline void {name}\([^;]*\) \{{", text)
    if match is None:
        return ""
    end = _brace_end(text, match.end() - 1)
    return text[match.end() : end]


def _append_call_args(text: str, name: str) -> str:
    out: list[str] = []
    token = name + "("
    cursor = 0
    while True:
        at = text.find(token, cursor)
        if at < 0:
            out.append(text[cursor:])
            return "".join(out)
        if text[max(0, at - 6) : at].rstrip().endswith("void"):
            out.append(text[cursor : at + len(token)])
            cursor = at + len(token)
            continue
        close = _paren_end(text, at + len(name))
        out.append(text[cursor:close])
        out.append(", __col_lo, __col_hi")
        cursor = close


def _narrow_columns(text: str) -> str:
    """A full ``0 .. nproma`` sweep becomes the surrounding tile's columns."""
    text = re.sub(
        r"for \(int64_t (\w+) = 0; \1 < nproma; \1 \+= 1\)",
        r"for (int64_t \1 = __col_lo; \1 < __col_hi; \1 += 1)",
        text,
    )
    return re.sub(
        r"for \(int64_t (\w+) = 0; \(\1 < nproma\); \1 = \(\1 \+ 1\)\)",
        r"for (int64_t \1 = __col_lo; \1 < __col_hi; \1 += 1)",
        text,
    )


def _strip_teams(text: str) -> str:
    """Drop team and worksharing pragmas. ``simd`` stays: the tile loop is the one parallel region."""
    return re.sub(r"[ \t]*#pragma omp (?:parallel for|parallel|for)\n", "", text)


def tile_columns(source: str) -> str:
    """One OpenMP loop over tiles of 64 columns around the outlined phase.

    The rendered kernel has no ``__dace_tile``. Its parallel axis is the block, which is one trip at
    preset M, and each nest then walks every column. This puts those column walks inside one tile loop.
    """
    entry = re.search(r"extern \"C\" void (\w+)\([^;]*\)\s*\{", source)
    if entry is None:
        raise FuseError('the kernel has no extern "C" entry')
    sweeps = len(re.findall(r"< nproma", source))
    if sweeps < 2:
        raise FuseError(f"the kernel walks nproma {sweeps} times; a phase tile needs at least two")
    body_open = entry.end() - 1
    body_close = _brace_end(source, body_open)
    signature = source[entry.start() : body_open + 1]
    narrowed = _narrow_columns(source)
    if "__col_lo" not in narrowed:
        raise FuseError("no full nproma loop was narrowed to a tile")
    narrowed = _strip_teams(narrowed)
    narrowed = _pass_columns(narrowed, _static_functions(narrowed))
    body_open = narrowed.find(signature) + len(signature) - 1
    if body_open < len(signature) - 1:
        raise FuseError("the entry signature moved while the helpers were rewritten")
    body_close = _brace_end(narrowed, body_open)
    tiled = (
        "\n#pragma omp parallel for\n"
        "for (int64_t __tile = 0; __tile < ((nproma + 63) / 64); __tile += 1) {\n"
        "int64_t __col_lo = __tile * 64;\n"
        "int64_t __col_hi = __col_lo + 64;\n"
        "if (__col_hi > nproma) __col_hi = nproma;\n"
    )
    rewritten = (
        narrowed[: body_open + 1] + tiled + narrowed[body_open + 1 : body_close] + "\n}\n" + narrowed[body_close:]
    )
    if rewritten.count("{") != rewritten.count("}"):
        raise FuseError("the tiled source has unbalanced braces")
    if rewritten.count("#pragma omp parallel for") != 1:
        raise FuseError("the tiled source does not have exactly one parallel loop")
    if signature not in rewritten:
        raise FuseError("the entry signature changed")
    return rewritten


def rewrite_kernel(source: str) -> str:
    """Tile the phase, whether the kernel was rendered with ``__dace_tile`` or with full column sweeps."""
    if source.count("__dace_tile") >= 2:
        return fuse_tiles(source)
    return tile_columns(source)


def fuse_tiles(source: str) -> str:
    """One ``#pragma omp parallel for`` over tiles of 64, containing every tile loop's body."""
    regions = _tile_regions(source)
    if len(regions) < 2:
        raise FuseError(f"the kernel has {len(regions)} __dace_tile loops; the phase needs at least two")
    pieces: list[tuple[int, int, int, str, str]] = []
    for pragma, brace, end in regions:
        header, body = source[pragma:brace], source[brace + 1 : end]
        old = _tile_size(body, header)
        retiled = _retile(header, old), _retile(body, old)
        if old != _TILE and re.search(rf"(?<![\w.]){old}(?![\w.])", retiled[0]):
            raise FuseError(f"tile bound still uses {old} after rewriting the header: {retiled[0]!r}")
        pieces.append((pragma, brace, end, retiled[0], retiled[1]))
    for earlier, later in zip(pieces, pieces[1:]):
        gap = source[earlier[2] + 1 : later[0]]
        if gap.strip() and not _scope_gap(gap):
            raise FuseError("code between tile loops is not a state scope:\n" + gap[:500])
    first = pieces[0]
    out = [source[: first[0]], first[3], "{", first[4]]
    for earlier, later in zip(pieces, pieces[1:]):
        out.append(later[4])
    out.append("}")
    out.append(source[pieces[-1][2] + 1 :])
    fused = "".join(out)
    if fused.count("{") != fused.count("}"):
        raise FuseError("the fused source has unbalanced braces")
    if fused.count("#pragma omp parallel for") != source.count("#pragma omp parallel for") - len(regions) + 1:
        raise FuseError("the fused source dropped a parallel loop that is not a tile loop")
    return fused


def _remember(script: Script, labels: tuple[str, ...], reason: str) -> None:
    cls = getattr(sys.modules.get("__main__"), "AppliedMove", None)
    if cls is None:
        cls = script.apply.__globals__["AppliedMove"]
    script.applied.append(cls(kind="define-scope", labels=labels, reason=reason))


def _edge(block: ControlFlowBlock) -> bool:
    try:
        _reads, writes = block.read_and_write_sets()
    except (AttributeError, TypeError, ValueError):
        return False
    return any(_EDGE.search(str(name)) for name in writes)


def _straight(blocks: list[ControlFlowBlock]) -> bool:
    """The same test ``block_group`` uses: each block falls into the next with one unconditional edge."""
    if len(blocks) < 2:
        return bool(blocks)
    region = blocks[0].parent_graph
    for first, second in zip(blocks, blocks[1:]):
        out = region.out_edges(first)
        straight = [edge.dst for edge in out] == [second] and out[0].data.is_unconditional()
        if not straight or len(region.in_edges(second)) != 1:
            return False
    return True


def _runs(session: Script) -> list[list[ControlFlowBlock]]:
    sdfg = session.session.sdfg
    found: list[list[ControlFlowBlock]] = []
    for region in sdfg.all_control_flow_regions(recursive=True):
        blocks = [
            block
            for block in in_order(region)
            if isinstance(block, ControlFlowBlock) and block.sdfg is sdfg and _edge(block)
        ]
        run: list[ControlFlowBlock] = []
        for block in blocks:
            if run and not _straight([run[-1], block]):
                found.append(run)
                run = []
            run.append(block)
        if run:
            found.append(run)
    found.sort(key=len, reverse=True)
    return found


def _subruns(runs: list[list[ControlFlowBlock]]) -> list[list[ControlFlowBlock]]:
    """Longest contiguous piece first, so a guard that splits a run still leaves its longest legal part."""
    ordered: list[list[ControlFlowBlock]] = []
    seen: set[tuple[str, ...]] = set()
    for run in runs:
        for length in range(len(run), 0, -1):
            for start in range(0, len(run) - length + 1):
                piece = run[start : start + length]
                key = tuple(block.label for block in piece)
                if key not in seen:
                    seen.add(key)
                    ordered.append(piece)
    return ordered


def _drop_unused_boundary_copies(sdfg: object) -> None:
    """An outlined nest copies its connector names onto the parent as transients no edge reads.

    Expanding the call turns the library node into a tasklet with those connector names, and validation
    then rejects a connector that is also an array. The memlets carry the original fields, so a transient
    nothing accesses can go.
    """
    for graph in sdfg.all_sdfgs_recursive():
        names = [
            name
            for name, desc in graph.arrays.items()
            if desc.transient and (name.startswith("_in_") or name.startswith("_out_"))
        ]
        for name in names:
            try:
                graph.remove_data(name, validate=True)
            except ValueError:
                continue


def _bind(session: Script, name: str, source: str) -> None:
    outcome = session.session.set_kernel_source(name, source, "cpp")
    if outcome.status.name != "OK":
        raise ReplayError(f"{name}: {outcome.status.name.lower()}: {outcome.reason}")


def solve(script: Script, ctx: Context) -> None:
    session = script.session
    session.default_moves()
    refusals: list[str] = []
    for piece in _subruns(_runs(script)):
        labels = tuple(block.label for block in piece)
        try:
            result = session.set_scope(labels, session.epoch)
        except (KeyError, RuntimeError, ValueError) as err:
            raise ReplayError(f"define-scope {list(labels)} raised {type(err).__name__}: {err}") from err
        if result.status.name != "APPLIED":
            refusals.append(f"{list(labels)}: {result.reason}")
            continue
        name = str(result.reason)
        source = ""
        try:
            source = session.kernel_source(name)
            rewrite_kernel(source)
        except (FuseError, KeyError, RuntimeError, ValueError) as err:
            _dump("original.cpp", source)
            refusals.append(f"{list(labels)} outlined {name}, rewrite refused: {err}")
            session.unset_scope(name, session.epoch)
            continue
        session.default_scopes()
        _drop_unused_boundary_copies(session.sdfg)
        placed = session.place()
        if placed.status.name != "APPLIED":
            raise ReplayError(f"place: {placed.reason}")
        if name not in {ext.name for ext in external_calls(session.sdfg)}:
            raise ReplayError(f"{name} is gone after placement")
        source = session.kernel_source(name)
        _dump("original.cpp", source)
        try:
            rewritten = rewrite_kernel(source)
        except FuseError as err:
            raise ReplayError(f"{name}: rewrite after placement failed: {err}") from err
        _dump("rewritten.cpp", rewritten)
        _bind(script, name, rewritten)
        _remember(script, labels, name)
        script.finished = True
        return
    raise ReplayError("no edge-phase run became one 64-wide tile. " + " | ".join(refusals[:8]))
