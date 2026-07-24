# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Staged screening for the DaCe multi-dim tile-op vectorization axis: turn the ~hundreds-of-cells
VectorizeConfig product into a small, statically-pruned candidate set, then a coordinate-descent search
that the arena times.

The vectorizer itself is user-owned; this module only *drives* it -- it builds :class:`VectorizeConfig`
objects (its public knobs) and hands them to :func:`build.apply_vectorizer`. It never edits the vectorizer.

Four stages, cheapest first (the plan):

  A. **device characterization** -- :func:`nestforge.device_profile.host_isas` (cached) gives the ISAs to
     emit (widest first, SCALAR floor). Done once per device.
  B. **free static pruning** (:func:`enumerate_vec_configs`, no compile) -- read the nest's SDFG and drop
     dead axes: ``assume_even`` bypasses the remainder strategy entirely (one cell per ``(isa,width)``, not
     one per strategy); ``K>=2`` makes ``target_isa`` dead (one AUTO cell); ``fp_factor`` only when the nest
     has a same-write-set branch; dedup by the *resolved* config so a silently-collapsed cell is not
     recorded as a distinct variant.
  C. **coordinate descent** (:func:`coordinate_descent`, ~16-30 cells) -- sweep one axis at a time holding
     the rest, keep the winner, advance. ``O(sum of axis sizes)`` instead of ``O(product)``; multi-start so
     a single greedy path does not design away a width x ISA interaction.
  D. **deep sweep** on the top-N nests -- the full pruned product (:func:`enumerate_vec_configs`) where C
     showed the widest spread. Anything capped is the caller's to log, never silently dropped.

Names are stable and greppable: ``cpu-avx512-w16-even-fma``."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import dace
from dace.transformation.passes.vectorization.config import VectorizeConfig

from nestforge.device_profile import host_isas

#: Width ladder for the innermost tile dim: powers of two, all divisible by 8 so an AVX512 (8xfp64) tile is
#: legal at every rung; 64 is opt-in (pass it explicitly) since it rarely pays on memory-bound nests.
WIDTH_LADDER: Tuple[int, ...] = (8, 16, 32)


@dataclass(slots=True)
class VecVariant:
    """One named vectorization cell: a greppable name and the VectorizeConfig that produces it."""
    name: str
    config: VectorizeConfig


def has_same_write_set_branch(sdfg: dace.SDFG) -> bool:
    """True when the SDFG has a conditional whose arms write data (so ``branch_mode=fp_factor`` -- per-lane
    ``c*x + (1-c)*y`` arithmetic vs the ``merge`` blend -- is a REAL axis, not a dead one). Conservative:
    any ``ConditionalBlock`` in the control-flow tree counts; if the DaCe build predates that node type the
    detection returns False (fp_factor simply is not enumerated), never an error."""
    try:
        from dace.sdfg.state import ConditionalBlock
    except ImportError:
        return False
    return any(isinstance(b, ConditionalBlock) for b in sdfg.all_control_flow_regions())


def resolved_key(cfg: VectorizeConfig) -> tuple:
    """A dedup key reflecting what the vectorizer ACTUALLY does, so silently-collapsed configs map to one
    cell: ``assume_even`` makes the remainder strategy irrelevant, and ``K>=2`` (multi-dim widths) makes
    ``target_isa`` dead (the K>=2 tile op is always the pure expansion). Mirrors the vectorizer's dispatch
    so a ``(K=2, AVX512)`` request is recorded as its resolved AUTO/pure form, never a distinct AVX512 cell."""
    isa = "AUTO" if len(cfg.widths) >= 2 else cfg.target_isa.value
    remainder = None if cfg.assume_even else cfg.remainder_strategy.value
    return (tuple(cfg.widths), isa, remainder, cfg.branch_mode.value, cfg.scalar_remainder_emit, cfg.assume_even,
            cfg.fuse_multiply_add)


def variant_name(cfg: VectorizeConfig) -> str:
    """A stable, greppable name: ``cpu-<isa>-w<widths>[-even][-posttail][-fpfac][-fma]``. ``K>=2`` reports
    ``auto`` (target_isa is dead there)."""
    isa = "auto" if len(cfg.widths) >= 2 else cfg.target_isa.value.lower()
    parts = ["cpu-%s-w%s" % (isa, "x".join(str(w) for w in cfg.widths))]
    if cfg.assume_even:
        parts.append("even")
    elif cfg.remainder_strategy.value == "scalar_postamble":
        parts.append("posttail")
    if cfg.branch_mode.value == "fp_factor":
        parts.append("fpfac")
    if cfg.fuse_multiply_add:
        parts.append("fma")
    return "-".join(parts)


def enumerate_vec_configs(sdfg: dace.SDFG,
                          isas: Optional[Sequence[str]] = None,
                          widths: Sequence[int] = WIDTH_LADDER,
                          allow_fma: bool = True) -> List[VecVariant]:
    """The statically-pruned candidate set for a nest (stage B / the stage-D deep sweep). ``isas`` defaults
    to :func:`host_isas` (SCALAR floor included). Deduped by :func:`resolved_key`, so no silently-identical
    cells. FMA cells are included (they validate on the ``contract-fma`` FP rung, not gated unsafe)."""
    if isas is None:
        isas = host_isas()
    branches = has_same_write_set_branch(sdfg)
    out: List[VecVariant] = []
    seen = set()

    def add(cfg: VectorizeConfig) -> None:
        key = resolved_key(cfg)
        if key in seen:
            return
        seen.add(key)
        out.append(VecVariant(variant_name(cfg), cfg))

    for isa in isas:
        for w in widths:
            base = VectorizeConfig(widths=(w, ), target_isa=isa)
            add(base)  # masked tail, merge blend -- the reference cell
            add(dataclasses.replace(base, assume_even=True))  # one even cell per (isa,width): bypasses remainder
            add(dataclasses.replace(base, remainder_strategy="scalar_postamble", scalar_remainder_emit="tile_k1"))
            if allow_fma:
                add(dataclasses.replace(base, fuse_multiply_add=True))
                add(dataclasses.replace(base, assume_even=True, fuse_multiply_add=True))
            if branches:  # fp_factor is a real axis ONLY with a same-write-set branch
                add(dataclasses.replace(base, branch_mode="fp_factor"))
    return out


def descent_axes(isas: Optional[Sequence[str]] = None,
                 widths: Sequence[int] = WIDTH_LADDER,
                 with_fp_factor: bool = False) -> List[Tuple[str, list]]:
    """The one-axis-at-a-time move set for :func:`coordinate_descent`: each entry is ``(field_name,
    candidate values)`` applied via ``dataclasses.replace``. ``target_isa`` and ``widths`` are the big
    levers; the rest are booleans / small enums. ``fp_factor`` is offered only when a branch makes it live."""
    if isas is None:
        isas = host_isas()
    axes: List[Tuple[str, list]] = [
        ("target_isa", list(isas)),
        ("widths", [(w, ) for w in widths]),
        ("assume_even", [False, True]),
        ("fuse_multiply_add", [False, True]),
        ("remainder_strategy", ["masked_tail", "scalar_postamble"]),
    ]
    if with_fp_factor:
        axes.append(("branch_mode", ["merge", "fp_factor"]))
    return axes


def coordinate_descent(seed: VectorizeConfig,
                       axes: List[Tuple[str, list]],
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
        for field_name, values in axes:
            for value in values:
                cand = dataclasses.replace(current, **{field_name: value})
                if resolved_key(cand) == current_key:
                    continue  # a no-op move (e.g. flipping remainder under assume_even); skip the measure
                t = measure(cand)
                if t is not None and (best is None or t < best):
                    best, current, improved = t, cand, True
                    current_key = resolved_key(current)  # recompute only on an accepted move
        if not improved:
            break
    return current, best


def multistart_descent(seeds: Sequence[VectorizeConfig],
                       axes: List[Tuple[str, list]],
                       measure: Callable[[VectorizeConfig], Optional[float]],
                       rounds: int = 2) -> Tuple[VectorizeConfig, Optional[float]]:
    """Run :func:`coordinate_descent` from several seeds and keep the global best -- so a width x ISA
    interaction is not designed away by a single greedy path. Seeds are typically the scalar floor, the
    ISA-native width, and ``assume_even``+FMA."""
    best_cfg, best_t = None, None
    for seed in seeds:
        cfg, t = coordinate_descent(seed, axes, measure, rounds)
        if t is not None and (best_t is None or t < best_t):
            best_cfg, best_t = cfg, t
    return (best_cfg if best_cfg is not None else seeds[0]), best_t


def default_seeds(isas: Optional[Sequence[str]] = None, widths: Sequence[int] = WIDTH_LADDER) -> List[VectorizeConfig]:
    """Diverse starting points for :func:`multistart_descent`: the scalar floor, the widest-ISA native
    width, and an ``assume_even``+FMA aggressive start."""
    if isas is None:
        isas = host_isas()
    native = isas[0]  # host_isas is widest-first
    mid = widths[len(widths) // 2]
    return [
        VectorizeConfig(widths=(widths[0], ), target_isa="SCALAR"),
        VectorizeConfig(widths=(mid, ), target_isa=native),
        VectorizeConfig(widths=(mid, ), target_isa=native, assume_even=True, fuse_multiply_add=True),
    ]
