# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Staged screening for the DaCe tile-op vectorization axis: turn the VectorizeConfig product into a small,
statically-pruned candidate set, then a coordinate-descent search that the arena times.

The vectorizer itself is user-owned; this module only *drives* it -- it builds :class:`VectorizeConfig`
objects (its public knobs) and hands them to :func:`build.apply_vectorizer`. It never edits the vectorizer.

THREE axes are searched -- tile ``width``, the remainder arm, and ``branch_mode``. Everything else the
config exposes is decided rather than explored:

  * ``widths`` is always ``(W,)``. This is the CPU SIMD lane count; a K>=2 tile is a multi-dim blocking
    question, needs a map with >=2 params to apply at all (every TSVC nest tiles a 1-param map, so
    ``MarkTileDims`` raises), and makes ``target_isa`` dead on top.
  * ``target_isa`` is pinned to ``ISA.AUTO``, which resolves to the host's best ISA at expansion. Searching
    it measured the host's own detection against itself, plus a SCALAR floor that the plain (non-vectorized)
    DaCe lane already provides as the baseline.
  * ``fuse_multiply_add`` follows the FP rung (:func:`fma_allowed`): contraction is exactly what every rung
    above ``strict-ieee`` permits, so an FMA cell is not a choice to measure -- it is what the requested
    numerics already allow.
  * ``assume_even`` is not a remainder arm. It ASSERTS the extent is divisible instead of emitting a
    remainder, so it does not answer "how is the remainder emitted"; and nothing here can establish the
    assertion, since the extent arrives as a symbol and the validating size and the timing size differ.

Two stages, cheapest first:

  A. **free static pruning** (:func:`enumerate_vec_configs`, no compile) -- read the nest's SDFG and drop
     the one optional axis it cannot move: ``fp_factor`` needs a same-write-set branch writing an INTEGER
     (the vectorizer blends integer ``ITE`` only). Deduped by :func:`resolved_key`, so a silently-collapsed
     config is never recorded as a distinct variant.
  B. **coordinate descent** (:func:`coordinate_descent`) -- sweep one axis at a time holding the rest, keep
     the winner, advance. ``O(sum of axis sizes)`` instead of ``O(product)``; multi-start so a single greedy
     path does not design away a width x remainder interaction.

Both entry points draw their remainder arms from :data:`REMAINDER_ARMS`, so the enumeration and the descent
cannot disagree about which arms exist -- they did, and each searched an arm the other never saw.

Names are stable and greppable: ``cpu-w16-posttail-fma``."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import dace
from dace.transformation.passes.vectorization.config import VectorizeConfig

#: Width ladder for the tile dim: powers of two, all divisible by 8 so an AVX512 (8xfp64) tile is legal at
#: every rung; 64 is opt-in (pass it explicitly) since it rarely pays on memory-bound nests.
WIDTH_LADDER: Tuple[int, ...] = (8, 16, 32)

#: Timed cells per nest that a sweep will tolerate. The pruned product is ``widths x arms x branch`` -- 24 at
#: the default ladder -- but the ladder is caller-supplied, and every cell is a codegen + compile + forked run.
MAX_CELLS: int = 256

#: FP rungs that forbid FMA contraction. Everything above ``strict-ieee`` permits it (``-ffp-contract=fast``
#: / ``-Mfma``), so the vectorizer's own fuse is granted exactly when the compiler's already is -- denying it
#: would make the tile path numerically stricter than the scalar baseline it is measured against.
NO_FMA_FP_MODES: frozenset = frozenset({"strict-ieee"})

#: The remainder axis: how a non-divisible extent is emitted. Four distinct emissions, verified distinct at
#: the generated-C++ level (see docs/VECTORIZATION_KNOBS.md) -- not a strategy enum with two live values:
#:
#:   * ``masked``   -- mask-free ``__tile_main`` interior + W-strided masked boundary slabs (the default)
#:   * ``fullmask`` -- NO split: one ``0:N:W`` map, mask computed and applied on EVERY tile
#:   * ``posttail`` -- interior + a plain step-1 scalar loop tail
#:   * ``k1tail``   -- interior + a single-lane (K=1, w=1) tile-op tail
#:
#: Each arm sets both fields the axis owns, so switching arms cannot leave the previous arm's
#: ``scalar_remainder_emit`` behind -- the bug a per-field descent axis would reintroduce.
#:
#: ``fullmask`` is the arm whose cost is ISA-dependent (AVX512 ``k``-registers make a per-tile mask nearly
#: free; AVX2 has no mask registers and must blend; SVE is predicated by nature), which is why the optimum is
#: measured on the target rather than assumed.
REMAINDER_ARMS: Tuple[Tuple[str, Dict[str, object]], ...] = (
    ("masked", {
        "remainder_strategy": "masked_tail",
        "scalar_remainder_emit": "scalar"
    }),
    ("fullmask", {
        "remainder_strategy": "full_mask",
        "scalar_remainder_emit": "scalar"
    }),
    ("posttail", {
        "remainder_strategy": "scalar_postamble",
        "scalar_remainder_emit": "scalar"
    }),
    ("k1tail", {
        "remainder_strategy": "scalar_postamble",
        "scalar_remainder_emit": "tile_k1"
    }),
)


@dataclass(slots=True)
class VecVariant:
    """One named vectorization cell: a greppable name and the VectorizeConfig that produces it."""
    name: str
    config: VectorizeConfig


def fma_allowed(fp_mode: str) -> bool:
    """Whether the FP rung ``fp_mode`` permits FMA contraction, and so whether the tile path fuses too."""
    return fp_mode not in NO_FMA_FP_MODES


def base_config(width: int, fp_mode: str) -> VectorizeConfig:
    """The K=1 config for one tile width at one FP rung, with every non-searched knob decided."""
    return VectorizeConfig(widths=(width, ), target_isa="AUTO", fuse_multiply_add=fma_allowed(fp_mode))


def conditional_blocks(sdfg: dace.SDFG) -> List["object"]:
    """Every ``ConditionalBlock`` in ``sdfg``, INCLUDING the ones inside nested SDFGs.

    ``all_control_flow_regions`` defaults to ``recursive=False``, and a map body's conditional lives in a
    NestedSDFG -- so the shallow walk finds conditionals only in nests that have no map, i.e. exactly the
    ones the tile path cannot vectorize. Returns ``[]`` when the DaCe build predates the node type."""
    try:
        from dace.sdfg.state import ConditionalBlock
    except ImportError:
        return []
    return [b for b in sdfg.all_control_flow_regions(recursive=True) if isinstance(b, ConditionalBlock)]


def has_same_write_set_branch(sdfg: dace.SDFG) -> bool:
    """True when the SDFG has a conditional whose arms write data. Conservative: any ``ConditionalBlock``
    counts. Necessary for ``branch_mode`` to be live, but NOT sufficient -- see
    :func:`has_integer_conditional_write`."""
    return bool(conditional_blocks(sdfg))


def has_integer_conditional_write(sdfg: dace.SDFG) -> bool:
    """True when a conditional writes an INTEGER-dtype array -- the only case where ``fp_factor`` differs
    from ``merge``.

    ``LowerITEToFpFactor`` rewrites ``ITE(c, t, e)`` to ``c*t + (1-c)*e`` for an integer output ONLY: the
    blend evaluates BOTH arms, and ``0 * inf`` is NaN, so a float arm keeps its ``ITE`` and lowers to a real
    per-lane select under either branch mode. On an all-float nest the two modes emit byte-identical C++
    (measured on 7/7 nests), so enumerating both doubles the branch cells to measure the same build twice.
    """
    for block in conditional_blocks(sdfg):
        owner = block.sdfg
        for state in block.all_states():
            for node in state.nodes():
                if not isinstance(node, dace.nodes.AccessNode) or not state.in_degree(node):
                    continue  # in_degree: a WRITTEN access node; a read cannot be the ITE's output
                desc = owner.arrays.get(node.data)
                if desc is not None and desc.dtype in dace.dtypes.INTEGER_TYPES:
                    return True
    return False


def resolved_key(cfg: VectorizeConfig) -> tuple:
    """A dedup key reflecting what the vectorizer ACTUALLY does, so silently-collapsed configs map to one
    cell. ``scalar_remainder_emit`` is only read under ``scalar_postamble`` (the orchestrator validates
    that), so it is normalized away elsewhere -- otherwise ``masked_tail`` would enumerate twice under two
    emit spellings that produce one build."""
    remainder = cfg.remainder_strategy.value
    emit = cfg.scalar_remainder_emit if remainder == "scalar_postamble" else None
    return (tuple(cfg.widths), cfg.target_isa.value, remainder, emit, cfg.branch_mode.value, cfg.assume_even,
            cfg.fuse_multiply_add)


def remainder_arm(cfg: VectorizeConfig) -> str:
    """The :data:`REMAINDER_ARMS` label ``cfg`` sits on (``masked`` when it matches none, which is the
    vectorizer's own default)."""
    for label, delta in REMAINDER_ARMS:
        if cfg.remainder_strategy.value != delta["remainder_strategy"]:
            continue
        if delta["remainder_strategy"] == "scalar_postamble" and cfg.scalar_remainder_emit != delta[
                "scalar_remainder_emit"]:
            continue
        return label
    return "masked"


def variant_name(cfg: VectorizeConfig) -> str:
    """A stable, greppable name: ``cpu-w<width>-<remainder arm>[-fpfac][-fma]``. No ISA token: every cell is
    ``AUTO``, so one would be constant noise in every name."""
    parts = ["cpu-w%s" % "x".join(str(w) for w in cfg.widths), remainder_arm(cfg)]
    if cfg.branch_mode.value == "fp_factor":
        parts.append("fpfac")
    if cfg.fuse_multiply_add:
        parts.append("fma")
    return "-".join(parts)


def live_axes(sdfg: dace.SDFG) -> Dict[str, bool]:
    """Which optional axes can change THIS nest's generated code: ``{"fp_factor": ...}``.

    Read once and shared by :func:`enumerate_vec_configs` and :func:`descent_axes` so the two entry points
    prune identically."""
    return {"fp_factor": has_integer_conditional_write(sdfg)}


def enumerate_vec_configs(sdfg: dace.SDFG,
                          widths: Sequence[int] = WIDTH_LADDER,
                          fp_mode: str = "contract-fma") -> List[VecVariant]:
    """The statically-pruned candidate set for a nest: ``widths x`` :data:`REMAINDER_ARMS` ``x branch_mode``,
    with ``branch_mode`` offered only where an integer conditional makes it live. Deduped by
    :func:`resolved_key`, so no silently-identical cells."""
    live = live_axes(sdfg)
    out: List[VecVariant] = []
    seen = set()

    def add(cfg: VectorizeConfig) -> None:
        key = resolved_key(cfg)
        if key in seen:
            return
        seen.add(key)
        out.append(VecVariant(variant_name(cfg), cfg))

    for w in widths:
        base = base_config(w, fp_mode)
        for _, delta in REMAINDER_ARMS:
            arm = dataclasses.replace(base, **delta)
            add(arm)
            if live["fp_factor"]:
                add(dataclasses.replace(arm, branch_mode="fp_factor"))
    return out


def cap_variants(variants: Sequence[VecVariant], max_cells: int = MAX_CELLS) -> Tuple[List[VecVariant], List[str]]:
    """``(kept, notes)`` -- the first ``max_cells`` variants, plus a note naming what was dropped.

    A cap is a coverage claim the tables would otherwise make silently: a truncated sweep reports the best of
    what it tried in the same column as an exhaustive one. The note exists so the caller can print it.
    Truncation is by enumeration order (width-major, then remainder arm), so the narrowest widths keep their
    full arm set rather than every width losing its tail."""
    if len(variants) <= max_cells:
        return list(variants), []
    dropped = variants[max_cells:]
    return list(variants[:max_cells]), [
        f"capped at {max_cells} cells: dropped {len(dropped)} ({dropped[0].name} .. {dropped[-1].name})"
    ]


def descent_axes(widths: Sequence[int] = WIDTH_LADDER,
                 live: Optional[Dict[str, bool]] = None) -> List[Tuple[str, List[Dict[str, object]]]]:
    """The one-axis-at-a-time move set for :func:`coordinate_descent`: each entry is ``(axis name, candidate
    DELTAS)``, a delta being the field assignments ``dataclasses.replace`` applies.

    Deltas rather than single field values because the remainder axis owns two fields at once
    (:data:`REMAINDER_ARMS`) -- moving to ``posttail`` while the previous arm's ``scalar_remainder_emit``
    stays set is not a point on the axis. ``live`` is :func:`live_axes`' verdict; ``None`` offers
    ``branch_mode``."""
    live = live or {"fp_factor": True}
    axes: List[Tuple[str, List[Dict[str, object]]]] = [
        ("widths", [{
            "widths": (w, )
        } for w in widths]),
        ("remainder", [dict(delta) for _, delta in REMAINDER_ARMS]),
    ]
    if live["fp_factor"]:
        axes.append(("branch_mode", [{"branch_mode": "merge"}, {"branch_mode": "fp_factor"}]))
    return axes


def coordinate_descent(seed: VectorizeConfig,
                       axes: List[Tuple[str, List[Dict[str, object]]]],
                       measure: Callable[[VectorizeConfig], Optional[float]],
                       rounds: int = 2) -> Tuple[VectorizeConfig, Optional[float]]:
    """Greedy coordinate descent from ``seed``: sweep one axis at a time holding the rest at the current
    best, keep the fastest improving move, advance; repeat up to ``rounds`` passes or until a pass finds no
    improvement. ``measure(config)`` returns a time (lower is better) or ``None`` for an unbuildable cell.
    ``O(rounds * sum of axis sizes)`` measurements, not the full product. Returns ``(best_config, best_time)``."""
    current = seed
    current_key = resolved_key(current)
    best = measure(current)
    for _ in range(rounds):
        improved = False
        for _, deltas in axes:
            for delta in deltas:
                cand = dataclasses.replace(current, **delta)
                if resolved_key(cand) == current_key:
                    continue  # a no-op move (the arm already held); no measure
                t = measure(cand)
                if t is not None and (best is None or t < best):
                    best, current, improved = t, cand, True
                    current_key = resolved_key(current)  # recompute only on an accepted move
        if not improved:
            break
    return current, best


def multistart_descent(seeds: Sequence[VectorizeConfig],
                       axes: List[Tuple[str, List[Dict[str, object]]]],
                       measure: Callable[[VectorizeConfig], Optional[float]],
                       rounds: int = 2) -> Tuple[VectorizeConfig, Optional[float]]:
    """Run :func:`coordinate_descent` from several seeds and keep the global best -- so a width x remainder
    interaction is not designed away by a single greedy path."""
    best_cfg, best_t = None, None
    for seed in seeds:
        cfg, t = coordinate_descent(seed, axes, measure, rounds)
        if t is not None and (best_t is None or t < best_t):
            best_cfg, best_t = cfg, t
    return (best_cfg if best_cfg is not None else seeds[0]), best_t


def default_seeds(widths: Sequence[int] = WIDTH_LADDER, fp_mode: str = "contract-fma") -> List[VectorizeConfig]:
    """Diverse starting points for :func:`multistart_descent`: the narrowest and the widest rung on the
    default arm, plus a mid rung on the scalar-tail arm -- so the descent enters the space from both ends of
    the width ladder and from more than one remainder arm."""
    narrow, wide = widths[0], widths[-1]
    mid = widths[len(widths) // 2]
    posttail = dict(next(delta for label, delta in REMAINDER_ARMS if label == "posttail"))
    return [
        base_config(narrow, fp_mode),
        base_config(wide, fp_mode),
        dataclasses.replace(base_config(mid, fp_mode), **posttail),
    ]
