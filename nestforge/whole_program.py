# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Whole-program baseline lane: optimize the ENTIRE un-split program as one unit and measure it.

The baseline a per-nest result must beat -- otherwise a "win" may be something a whole-program optimizer
gets for free. Any :class:`~nestforge.optimizers.Optimizer` proposing ``scope='whole-program'`` plugs in.
Build + validate is the per-nest lane's path, pointed at
:func:`~nestforge.extract.whole_program_boundary`; the kernel runs forked.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import dace

from nestforge import build, tsvc
from nestforge.arena import make_inputs, run_oracle, validate_and_time
from nestforge.extract import Boundary, find_state_of_node, whole_program_boundary
from nestforge.isolation import run_isolated
from nestforge.libnode import ExternalCall
from nestforge.optimizers import Optimizer
from nestforge.pass_lower import lower_nests_to_external_call
from nestforge.perf import flags
from nestforge.split_unsupported import isolate_into_own_state
from nestforge.translate import prepare_whole_program


@dataclass(slots=True)
class WholeProgramResult:
    """One whole-program measurement. ``median_us`` is ``inf`` and ``error`` is set when the build or the
    forked run failed; ``ok`` is the bit-exact verdict vs the whole-program numpy oracle."""
    optimizer: str
    opt_mode: str
    ok: bool
    maxdiff: float
    median_us: float
    reps: int
    error: Optional[str] = None


def measure_whole_program(optimizer: Optimizer,
                          kernel: tsvc.TsvcKernel,
                          out_dir: Union[str, Path],
                          preset: str = "S",
                          reps: int = 7,
                          seed: int = 0,
                          atol: Optional[float] = None,
                          timeout: float = 900.0) -> WholeProgramResult:
    """Build + validate + time ``optimizer``'s whole-program proposal for ``kernel``.

    The proposal must be ``scope='whole-program'``, DaCe lane (the external lane is future work). The
    oracle is emitted from the same optimized SDFG, so the check is codegen-vs-emit on identical
    semantics. ``atol`` defaults to the strict-ieee tolerance.
    """
    proposal = optimizer.propose()
    if proposal is None:
        return WholeProgramResult(optimizer.name, "", False, float("inf"), float("inf"), 0, "optimizer declined")
    if proposal.scope != "whole-program":
        raise ValueError(f"{optimizer.name} proposes scope {proposal.scope!r}; measure_whole_program needs "
                         f"'whole-program' (use the per-nest arena for a per-nest optimizer)")
    if proposal.lane != "dace":
        raise NotImplementedError(f"whole-program lane builds the DaCe scope; {proposal.lane!r} is future work")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    atol = flags.FP_ATOL["strict-ieee"] if atol is None else atol

    sdfg = tsvc.build_sdfg(kernel, proposal.opt_mode)  # optimize the WHOLE program (across nests)
    boundary = whole_program_boundary(sdfg)
    sizes = tsvc.sample_sizes(kernel, boundary, preset=preset)
    inputs = make_inputs(boundary, sizes, seed=seed, given=tsvc.index_fills(kernel, boundary, sizes, seed=seed))
    prep = prepare_whole_program(sdfg, kernel.key, out_dir, sizes=sizes)
    oracle = run_oracle(prep, boundary, inputs, sizes)

    def work() -> Dict[str, object]:
        # BUILD inside the fork, not just the run: dace codegen and the C++ compile are where a candidate
        # hangs or dies natively, which a per-kernel try/except cannot catch. Only this summary crosses
        # back, so the artifact (and its dlopen mapping) dying with the child costs nothing.
        built = build.build_sdfg(boundary.standalone_sdfg, out_dir / "build", opts=proposal.build)
        # bind_program binds only the SDFG's own parameters, so make_inputs' extra scratch is ignored.
        # arena.validate_and_time owns the absent-output guard, the report-absolute/gate-relative split and
        # the per-rep rewind -- differential.py runs the identical body, and they drifted while duplicated.
        return validate_and_time(built, boundary, inputs, sizes, oracle, atol, reps)

    res = run_isolated(work, timeout=timeout)
    if "error" in res:
        return WholeProgramResult(optimizer.name, proposal.opt_mode, False, float("inf"), float("inf"), reps,
                                  res["error"])
    return WholeProgramResult(optimizer.name, proposal.opt_mode, bool(res["ok"]), float(res["maxdiff"]),
                              float(res["median_us"]), reps)


# --- offload analysis: externalize before deciding offload -------------------------------------------
# Order is fixed: externalize each nest into a call FIRST, then let a tool decide offloadability. A lane
# that pre-decides offload would have its decision invalidated by a later extraction. Each call gets its
# own state -- the scope where the host<->device transfer lives -- so calls decide independently.
@dataclass(slots=True)
class OffloadScope:
    """One externalized call as an INDEPENDENT offload unit. ``inputs`` cross INTO the scope
    (host->device on offload), ``outputs`` cross OUT; ``offloadable`` is the per-call decision (injected,
    see :func:`offload_scopes`), ``reason`` says why not."""
    call: str
    inputs: List[str]
    outputs: List[str]
    offloadable: bool = True
    reason: str = ""


def default_offloadable(call: ExternalCall, boundary: Boundary) -> Tuple[bool, str]:
    """Default decision: an externalized compute nest may offload. The real per-tool GPU-viability check
    plugs in here -- the analysis provides the scope, the tool provides the verdict."""
    return True, ""


def offload_scopes(
    sdfg: dace.SDFG,
    strategy: str = "skip-taskloops",
    offloadable: Optional[Callable[[ExternalCall, Boundary], Tuple[bool, str]]] = None
) -> Tuple[dace.SDFG, List[OffloadScope]]:
    """Whole-program offload analysis: externalize each nest into a call, then put each call in its OWN
    state so it is an independent offload unit.

    NON-DESTRUCTIVE: works on a deepcopy, so the caller's SDFG is untouched. Returns the transformed SDFG
    (still runnable via the ``DaceReference`` expansion) and one :class:`OffloadScope` per call, in
    extraction order. ``offloadable`` defaults to :func:`default_offloadable`.
    """
    decide = offloadable or default_offloadable
    work = copy.deepcopy(sdfg)
    calls = lower_nests_to_external_call(work, strategy)
    scopes: List[OffloadScope] = []
    for ext, boundary in calls:
        # Re-find the state each time: isolating a prior call fissions states, but the node object is stable.
        state = find_state_of_node(work, ext)
        isolate_into_own_state(state, ext)
        ok, reason = decide(ext, boundary)
        scopes.append(OffloadScope(ext.name, list(boundary.inputs), list(boundary.outputs), ok, reason))
    return work, scopes
