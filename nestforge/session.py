"""Session: the consolidated agent-facing API over the 4-phase optimizer.

A stateless model cannot hold a live SDFG node across turns -- a reference goes stale the moment any
transformation mutates the graph, and a stale reference is a silently-wrong move, not an error. The
:class:`Session` closes that gap: the SDFG lives here (server-side), and the agent names graph objects
only by **epoch-stamped string ids**.

The contract:

  * A ``list_*`` call mints ids at the session's current epoch and returns plain data (labels, read/write
    sets, reasons) -- never a node.
  * A mutating call (:meth:`fuse`, :meth:`fission_all`, :meth:`fission_map`, :meth:`externalize`) bumps the
    epoch and clears every handle, then returns the fresh graph.
  * An id from a past epoch no longer resolves: :meth:`resolve` raises :class:`StaleHandle` telling the
    agent to re-list. That is the safety net -- the "enumerate -> apply one -> re-enumerate" discipline the
    arms require, enforced by the id scheme instead of by the agent remembering to do it.

Every method is one agent tool; every return value is JSON-serializable, so a transport layer (MCP, RPC)
is a thin wrapper over this class and holds no graph state of its own. The same calls drive the
deterministic path -- a strategy is just a scripted policy over ``list_fusions`` / ``fuse``.
"""
from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import dace
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation.dataflow.map_fission import MapFission

from nestforge.arena import Cell, run_arena
from nestforge.extract import find_state_of_node
from nestforge.feedback import run_feedback_loop
from nestforge.fusion import FusionMove, apply_fusion, can_fuse, enumerate_fusions, fission_to_statements, map_fission_moves
from nestforge.introspect import describe_graph, nest_reads_writes
from nestforge.offload import DEFAULT_GRANULARITY, label_nest, lower_nests_to_external_call, offload_candidates
from nestforge.region_arms import RegionMove, apply_region_fusion, enumerate_region_fusions
from nestforge.strategies import is_parallel_nest, top_level_map_entries
from nestforge.translate import Prepared, emit_sources, prepare

try:
    from dace.sdfg.state import ConditionalBlock
except ImportError:  # older DaCe without first-class conditional regions
    ConditionalBlock = None


class StaleHandle(KeyError):
    """Raised when an id from a past epoch is used -- the graph moved under it; re-list and retry."""


def unit_reads_writes(parent: dace.SDFG, node) -> Tuple[List[str], List[str]]:
    """``(reads, writes)`` for one nest, whichever kind it is (map needs its state; loop reads its own
    read/write sets). Thin adapter over :func:`introspect.nest_reads_writes`."""
    container = find_state_of_node(parent, node) if isinstance(node, nodes.MapEntry) else parent
    return nest_reads_writes(container, node)


class Session:
    """Server-side owner of one SDFG and the id registry the agent drives it through."""

    def __init__(self, sdfg: dace.SDFG, name: Optional[str] = None, work_dir: Optional[str] = None):
        self.sdfg = sdfg
        self.name = name or sdfg.label
        self.epoch = 0
        self.handles: Dict[str, object] = {}  # id -> node/move/nest, valid only at self.epoch
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="nfsession_"))
        self.prepared: Dict[str, Prepared] = {}  # nest id -> its emitted numpy/yaml package (Phase 3)

    # --- handle machinery -------------------------------------------------------------------------

    def mint(self, kind: str, obj: object) -> str:
        """Register ``obj`` under a fresh epoch-stamped id and return it."""
        hid = f"e{self.epoch}:{kind}:{len(self.handles)}"
        self.handles[hid] = obj
        return hid

    def resolve(self, hid: str, kind: Optional[str] = None) -> object:
        """The object ``hid`` names, or raise. A past-epoch id is a :class:`StaleHandle` (re-list); a
        wrong-``kind`` id is a plain error (the agent mixed a move id with a region id)."""
        if hid not in self.handles:
            # Only a WELL-FORMED past-epoch id is stale ("re-list and retry" is then the right advice).
            # A malformed id is a caller bug, and reporting it as stale would send the agent into a
            # re-list/retry loop that can never succeed.
            stamp = hid.split(":", 1)[0] if ":" in hid else ""
            if not re.fullmatch(r"e\d+", stamp):
                raise KeyError(f"malformed id {hid!r}; expected 'e<epoch>:<kind>:<n>' from a list_* call")
            if stamp != f"e{self.epoch}":
                raise StaleHandle(f"id {hid!r} is from a past epoch (now e{self.epoch}); re-list and retry")
            raise KeyError(f"unknown id {hid!r}")
        obj = self.handles[hid]
        if kind is not None and not hid.split(":")[1] == kind:
            raise KeyError(f"id {hid!r} is not a {kind} handle")
        return obj

    def bump(self) -> None:
        """A mutation happened: advance the epoch and drop every handle (all now stale)."""
        self.epoch += 1
        self.handles = {}

    # --- Phase 0: see the graph -------------------------------------------------------------------

    def describe(self) -> str:
        """The control-flow-region tree as TEXT, each nest with its parallel/sequential nature and read/write
        sets. Read-only; safe at any epoch. See :meth:`region_tree` for the same tree as structured data."""
        return describe_graph(self.sdfg)

    def region_tree(self) -> dict:
        """The control-flow REGION tree as nested data, one id per container. A ``region`` is a control-flow
        container (``SDFGState`` / ``LoopRegion`` / ``ConditionalBlock`` / ``ControlFlowRegion``) -- the box a
        nest lives in, NOT the nest itself. A ``SDFGState`` is a ``barrier`` and lists the nests it holds;
        the others recurse via ``children``. Two nests fuse only inside one region, so this is the map the
        agent reads before deciding whether a fuse needs a region merge first (:meth:`list_region_fusions`)."""
        return self.region_node(self.sdfg)

    # --- Level 1: region structure (merge the containers) -----------------------------------------

    def list_region_fusions(self) -> List[dict]:
        """Legal region merges right now -- adjacent ``SDFGState`` pairs ``StateFusion`` accepts, each with an
        id for :meth:`fuse_regions`. This is the level ABOVE nest fusion: merging two states so the maps that
        were barred by the state boundary become fusable siblings. (Loop-region merges ride :meth:`fuse` --
        fuse the enclosing loops.)"""
        out: List[dict] = []
        for move in enumerate_region_fusions(self.sdfg):
            hid = self.mint("regmove", move)
            out.append({"id": hid, "kind": move.kind, "label": move.label()})
        return out

    def fuse_regions(self, move_id: str) -> dict:
        """Commit one region merge (from a current :meth:`list_region_fusions`) -- dissolve a state barrier so
        the nests inside can then fuse. Bumps the epoch (all prior ids stale); returns the fresh region tree."""
        move: RegionMove = self.resolve(move_id, "regmove")
        apply_region_fusion(self.sdfg, move)
        self.bump()
        return self.region_tree()

    # --- Level 2: nest fusion (fuse maps / loops within a region) ----------------------------------

    def list_nests(self) -> List[dict]:
        """Every nest (top-level map-nest or loop-nest) with a fresh id, label, parallel flag, and read/write
        sets. Ids feed :meth:`can_fuse`; the sets tell the agent what each nest computes. These are the units
        Level-2 fusion operates on -- distinct from the offload candidates of :meth:`list_offload_candidates`."""
        out: List[dict] = []
        for nest in fusion_units(self.sdfg):
            reads, writes = unit_reads_writes(self.sdfg, nest)
            hid = self.mint("nest", nest)
            out.append({
                "id": hid,
                "kind": "map" if isinstance(nest, nodes.MapEntry) else "loop",
                "label": label_nest(nest),
                "parallel": is_parallel_nest(nest),
                "reads": reads,
                "writes": writes,
            })
        return out

    def can_fuse(self, first_id: str, second_id: str) -> str:
        """``"yes"`` if the two nests may fuse, else a one-line reason -- the same gate :meth:`fuse` applies,
        so ``"yes"`` is exactly an applicable move. Diagnostic: works on any pair, legal or not. When the
        blocker is a region boundary (two maps in different states), the reason says to merge the enclosing
        regions first (:meth:`list_region_fusions` / :meth:`fuse_regions`)."""
        first = self.resolve(first_id, "nest")
        second = self.resolve(second_id, "nest")
        return can_fuse(self.sdfg, first, second)

    def list_fusions(self) -> List[dict]:
        """Every legal fusion move right now (all three arms), each with an id for :meth:`fuse`. This is
        ``can_fuse == "yes"`` enumerated -- the agent applies one, then re-lists (the epoch bump forces it)."""
        out: List[dict] = []
        for move in enumerate_fusions(self.sdfg):
            hid = self.mint("move", move)
            out.append({"id": hid, "kind": move.kind, "label": move.label()})
        return out

    def fuse(self, move_id: str) -> str:
        """Commit one fusion move (from a current :meth:`list_fusions`). Bumps the epoch -- all prior ids go
        stale -- and returns the fresh graph tree."""
        move: FusionMove = self.resolve(move_id, "move")
        apply_fusion(self.sdfg, move)
        self.bump()
        return self.describe()

    def fission_all(self) -> str:
        """Explode the whole SDFG to statement granularity (the inverse of max-fuse), then re-fuse up to
        taste. Bumps the epoch; returns the fresh tree."""
        fission_to_statements(self.sdfg)
        self.bump()
        return self.describe()

    def list_map_fissions(self) -> List[dict]:
        """Maps ``MapFission`` can split one at a time (a map whose nested-SDFG body has independent output
        groups), each with an id for :meth:`fission_map` -- fine-grained control vs :meth:`fission_all`."""
        out: List[dict] = []
        for map_entry, nsdfg in map_fission_moves(self.sdfg):
            hid = self.mint("fission", (map_entry, nsdfg))
            out.append({"id": hid, "label": label_nest(map_entry)})
        return out

    def fission_map(self, move_id: str) -> str:
        """Split one map (from a current :meth:`list_map_fissions`). Bumps the epoch; returns the fresh tree."""
        map_entry, nsdfg = self.resolve(move_id, "fission")
        MapFission.apply_to(self.sdfg, expr_index=1, map_entry=map_entry, nested_sdfg=nsdfg)
        self.bump()
        return self.describe()

    # --- Phase 2: externalize (hand a nest to the next phase) --------------------------------------

    def list_offload_candidates(self, granularity: str = DEFAULT_GRANULARITY) -> List[dict]:
        """Non-mutating preview: what ``granularity`` WOULD externalize -- each candidate nest labeled,
        parallel/sequential, with its read/write sets. The Phase-2 analog of :meth:`list_fusions`. A
        DISTINCT decision from fusion: which nests leave as library calls, not how nests are fused."""
        out: List[dict] = []
        for cand in offload_candidates(self.sdfg, granularity):
            reads, writes = unit_reads_writes(cand.parent_sdfg, cand.node)
            hid = self.mint("cand", cand)
            out.append({
                "id": hid,
                "label": cand.label,
                "parallel": cand.parallel,
                "reads": reads,
                "writes": writes,
            })
        return out

    def externalize(self, granularity: str = DEFAULT_GRANULARITY) -> List[dict]:
        """Commit Phase 2: swap every selected nest for an ``ExternalCall`` (still runs, numpy-reference
        fallback, so the SDFG stays bit-exact). Bumps the epoch, then mints an id per externalized nest at
        the new epoch -- each carrying the boundary's read (``inputs``) / write (``outputs``) sets, the
        interface Phase 3 compiles against."""
        lowered = lower_nests_to_external_call(self.sdfg, granularity)
        if lowered:
            self.bump()  # only a REAL change costs the agent its handles; a no-op must not strand them
        out: List[dict] = []
        for ext, boundary in lowered:
            hid = self.mint("extnest", (ext, boundary))
            out.append({
                "id": hid,
                "name": ext.name,
                "reads": list(boundary.inputs),
                "writes": list(boundary.outputs),
                "symbols": list(boundary.symbols),
            })
        return out

    # --- Phase 3: optimize a nest's kernel (two evaluation modes) ----------------------------------

    def nest_boundary(self, nest_id: str) -> dict:
        """The externalized nest's interface -- read (``inputs``) / write (``outputs``) / ``symbols`` in
        boundary order, plus the kernel name. What an agent authoring a Mode-A kernel compiles against; the
        boundary order is also the order :meth:`set_kernel`'s ``abi_order`` must match."""
        ext, boundary = self.resolve(nest_id, "extnest")
        return {
            "name": ext.name,
            "inputs": list(boundary.inputs),
            "outputs": list(boundary.outputs),
            "symbols": list(boundary.symbols),
            "boundary_order": list(boundary.inputs) + list(boundary.outputs) + list(boundary.symbols),
        }

    def emit_reference(self, nest_id: str) -> str:
        """Write the nest's numpy reference (the correctness oracle) and return its path. Both modes
        validate against it."""
        return str(self.prepare_nest(nest_id).numpy_path)

    def emit_variant(self, nest_id: str, target: str = "c", precision: str = "float64") -> List[str]:
        """Emit the nest's kernel in ``target`` (``"numpy"`` | ``"c"`` | ``"cpp"`` | ``"fortran"``) and return
        the written source paths. Mode B sweeps these; a Mode-A agent may start from one."""
        prep = self.prepare_nest(nest_id)
        gen = self.work_dir / prep.name / target
        return [str(p) for p in emit_sources(prep, gen, target=target, precision=precision)]

    def set_kernel(self, nest_id: str, lib_path: str, symbol: str, abi_order: List[str], fp_mode: str = "") -> dict:
        """Mode A -- point the nest at an agent-authored compiled kernel: a ``.a`` (static, linked into the
        parent ``.so``) or ``.so`` exposing ``symbol`` as an ``extern "C"`` entry. Sets leaf fields only
        (no topology change, so the epoch is unchanged and ids stay valid).

        ``abi_order`` is the SILENT-BREAK field: it must be the arg order the compiled symbol expects, not
        the manifest/role order. Returned alongside ``boundary_order`` so the agent can check them; a
        mismatch is undiagnosed ABI corruption, never an error."""
        ext, boundary = self.resolve(nest_id, "extnest")
        ext.lib_path = lib_path
        ext.symbol = symbol
        ext.abi_order = list(abi_order)
        if fp_mode:
            ext.fp_mode = fp_mode
        return {
            "nest": ext.name,
            "abi_order": list(ext.abi_order),
            "boundary_order": list(boundary.inputs) + list(boundary.outputs) + list(boundary.symbols),
        }

    def sweep(self, nest_id: str, sizes: Dict[str, int], reps: int = 100, seed: int = 0) -> dict:
        """Mode B -- sweep ``compiler x fp-mode`` over the framework-emitted C kernel, validate each cell
        bit-exact vs the numpy oracle, time it, and return the winning cell per fp-mode. The framework does
        the building; the agent only supplies the code (or lets the framework emit it)."""
        ext, boundary = self.resolve(nest_id, "extnest")
        prep = self.prepare_nest(nest_id)
        gen = self.work_dir / prep.name / "c"
        sources = emit_sources(prep, gen, target="c")
        c_source = next((Path(p) for p in sources if str(p).endswith(".c")), None)
        if c_source is None:  # bare StopIteration here would surface as an opaque tool error
            raise ValueError(f"emitting {prep.name!r} as C produced no .c source (got {[str(p) for p in sources]}); "
                             "nothing to sweep")
        res = run_arena(prep, boundary, c_source, self.work_dir / prep.name / "arena", sizes, reps=reps, seed=seed)
        return {fp: cell_summary(cell) for fp, cell in res.winners.items()}

    # --- Phase 4: feed measurements back ----------------------------------------------------------

    def feedback(self, measure: Callable, max_rounds: int = 8) -> dict:
        """Re-adjust granularity (re-fuse move by move) until measured time plateaus. ``measure`` is the
        framework's build+time step (supplied by the transport, never the model) -- Phase 4 is driven by
        MEASUREMENT, so it cannot be a pure-data tool. Bumps the epoch (granularity changed)."""
        res = run_feedback_loop(self.sdfg, measure, max_rounds=max_rounds)
        self.bump()
        best = res.best
        return {
            "rounds": res.rounds,
            "best_name": best.proposal.name if best is not None else None,
            "best_us": best.median_us if best is not None else None,
        }

    def prepare_nest(self, nest_id: str) -> Prepared:
        """The nest's numpy+yaml package (:func:`translate.prepare`), memoized per id so emit and sweep
        share one."""
        if nest_id not in self.prepared:
            ext, boundary = self.resolve(nest_id, "extnest")
            self.prepared[nest_id] = prepare(boundary, ext.name, self.work_dir / ext.name)
        return self.prepared[nest_id]

    # --- region-tree walk (used by region_tree / fuse_regions) ------------------------------------

    def region_id(self, block) -> str:
        """A STABLE, purely descriptive id for a control-flow container in :meth:`region_tree`.

        Deliberately not a minted handle: ``region_tree`` is a read view, and no method resolves a
        ``region`` kind. Minting here would grow ``self.handles`` on every inspection call and hand back ids
        that raise on the kind-guard if an agent tried to use one. The label already names the block
        uniquely within its SDFG, which is all a tree reader needs.
        """
        return f"region:{block.label}"

    def region_node(self, cfg) -> dict:
        """One control-flow region as structured data: an id, its label/type, and its child blocks."""
        return {
            "id": self.region_id(cfg),
            "label": cfg.label,
            "type": type(cfg).__name__,
            "children": [self.block_node(block) for block in cfg.nodes()],
        }

    def block_node(self, block) -> dict:
        """One block inside a region: a State (a barrier, listing its nests) or a nested region (recurse)."""
        if isinstance(block, SDFGState):
            nests = []
            for entry in top_level_map_entries(block):
                reads, writes = nest_reads_writes(block, entry)
                nests.append({
                    "label": label_nest(entry),
                    "parallel": is_parallel_nest(entry),
                    "reads": reads,
                    "writes": writes,
                })
            return {
                "id": self.region_id(block),
                "label": block.label,
                "type": "SDFGState",
                "barrier": True,
                "nests": nests
            }
        if ConditionalBlock is not None and isinstance(block, ConditionalBlock):
            branches = [{
                "condition": "else" if cond is None else f"when {cond.as_string}",
                "region": self.region_node(branch),
            } for cond, branch in block.branches]
            return {
                "id": self.region_id(block),
                "label": block.label,
                "type": "ConditionalBlock",
                "selector": "stays in core SDFG",
                "branches": branches
            }
        if isinstance(block, (LoopRegion, ControlFlowRegion)):
            node = self.region_node(block)
            node["sequential"] = isinstance(block, LoopRegion)
            return node
        return {"id": self.region_id(block), "label": block.label, "type": type(block).__name__}


def cell_summary(cell: Cell) -> dict:
    """The agent-relevant fields of an arena :class:`Cell` -- correctness + time + who won."""
    return {
        "compiler": cell.compiler,
        "fp_mode": cell.fp_mode,
        "ok": cell.ok,
        "maxdiff": cell.maxdiff,
        "time_us": cell.time_us,
        "error": cell.error,
    }


def fusion_units(sdfg: dace.SDFG) -> List:
    """The fusable units of ``sdfg``: every top-level map-nest (per state) plus every loop-nest (recursive)
    -- exactly the node kinds :func:`can_fuse` accepts. Matches what :func:`introspect.describe_graph` walks."""
    units: List = []
    for cfg in sdfg.all_control_flow_regions(recursive=True):
        for node in cfg.nodes():
            if isinstance(node, LoopRegion):
                units.append(node)
    for state in sdfg.all_states():
        units.extend(top_level_map_entries(state))
    return units
