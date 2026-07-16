"""Whole-program baseline lane: optimize the ENTIRE un-split program as one unit and measure it.

The per-nest arena extracts each loop nest and tunes it in isolation. That risks crediting a "win" that a
whole-program optimizer -- DaCe ``auto-opt`` across all nests, or a compiler auto-parallelizing the whole
emitted source -- already gets for free. This lane is the honest baseline: hand the whole program to ONE
optimizer, build it, validate it bit-exact, and time it, so a per-nest result has something real to beat.

Any :class:`~nestforge.optimizers.Optimizer` proposing ``scope='whole-program'`` plugs in -- the
deterministic :class:`~nestforge.optimizers.WholeProgramOptimizer` (auto-opt) now, an agent later ("agent
or anything"). The build + validate path is the per-nest lane's, pointed at the WHOLE-program boundary
(:func:`~nestforge.extract.whole_program_boundary`) instead of an extracted nest; the compiled kernel runs
FORKED (:func:`~nestforge.isolation.run_isolated`) so a crash in fresh code never takes down the caller.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import dace

from nestforge import build, tsvc
from nestforge.arena import make_inputs, maxdiff, run_oracle
from nestforge.extract import Boundary, find_state_of_node, whole_program_boundary
from nestforge.isolation import run_isolated
from nestforge.libnode import ExternalCall
from nestforge.optimizers import Optimizer
from nestforge.pass_lower import lower_nests_to_external_call
from nestforge.perf import flags
from nestforge.perf.harness import median
from nestforge.split_unsupported import isolate_into_own_state
from nestforge.translate import prepare_whole_program


@dataclass
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
                          out_dir,
                          preset: str = "S",
                          reps: int = 7,
                          seed: int = 0,
                          atol: Optional[float] = None,
                          timeout: float = 900.0) -> WholeProgramResult:
    """Build + validate + time ``optimizer``'s whole-program proposal for ``kernel``.

    The proposal must be ``scope='whole-program'``, DaCe lane (the external whole-program lane -- gcc /
    clang+Polly / gcc+Graphite / pluto over the whole emitted source -- is future work). The optimizer's
    ``opt_mode`` optimizes the WHOLE kernel SDFG (``auto-opt`` fuses/tiles across nests); the numpy oracle
    is emitted from that same optimized program, so the check is codegen-vs-emit on identical semantics.
    ``atol`` defaults to the strict-ieee tolerance.
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
    built = build.build_sdfg(boundary.standalone_sdfg, out_dir / "build", opts=proposal.build)

    def work():
        # Validate: one in-place run, compared to the oracle. bind_program binds only the SDFG's own
        # parameters, so make_inputs' extra scratch buffers (internal transients of the whole program) are
        # harmlessly ignored.
        vbuf = {k: v.copy() for k, v in inputs.items()}
        built.run(vbuf, sizes)
        outs = {o: vbuf[o] for o in boundary.outputs if o in vbuf}
        if outs:
            md = float(maxdiff({o: oracle[o] for o in outs}, outs))
            verdict = {"ok": bool(md <= atol), "maxdiff": md}
        else:
            verdict = {"ok": False, "maxdiff": float("inf")}
        # Time: init once, bind once, call the bare kernel in the rep loop (no per-rep marshaling).
        tbuf = {k: v.copy() for k, v in inputs.items()}
        built.init(sizes)
        try:
            fn, cargs = built.bind_program(tbuf, sizes)
            fn(*cargs)  # warm
            samples: List[float] = []
            for _ in range(reps):
                t0 = time.perf_counter()
                fn(*cargs)
                samples.append((time.perf_counter() - t0) * 1e6)
        finally:
            built.close()
        return {**verdict, "median_us": median(samples)}

    res = run_isolated(work, timeout=timeout)
    if "error" in res:
        return WholeProgramResult(optimizer.name, proposal.opt_mode, False, float("inf"), float("inf"), reps,
                                  res["error"])
    return WholeProgramResult(optimizer.name, proposal.opt_mode, bool(res["ok"]), float(res["maxdiff"]),
                              float(res["median_us"]), reps)


# --- offload analysis: externalize before deciding offload -------------------------------------------
# Whole-program scope is the full un-split program (`whole_program_boundary`). Offloading works at a FINER
# scope: each loop nest is externalized into a call first, and only THEN does a tool decide whether that
# call is offloadable. The order is fixed -- if DaCe offloaded a loop and a polyhedral tool then found the
# same loop GPU-viable, the offload decision would change underneath us -- so no lane may pre-decide
# offload before extraction. Each externalized call is put in its OWN state: that isolated state is the
# scope "between" the call and the host program, where the host<->device transfer lives, so every call
# decides offload independently.
@dataclass
class OffloadScope:
    """One externalized call as an INDEPENDENT offload unit -- the scope between the call and the host
    program. ``inputs`` cross INTO the scope (host->device on offload), ``outputs`` cross OUT
    (device->host). ``offloadable`` is the per-call decision (a tool's, injected -- see
    :func:`offload_scopes`); ``reason`` says why not when it is ``False``."""
    call: str
    inputs: List[str]
    outputs: List[str]
    offloadable: bool = True
    reason: str = ""


def default_offloadable(call: ExternalCall, boundary: Boundary) -> Tuple[bool, str]:
    """Default decision: an externalized compute nest may offload. The real per-tool GPU-viability check
    (polyhedral affinity, data-parallelism, cost) plugs in here -- the analysis provides the scope, the
    tool provides the verdict."""
    return True, ""


def offload_scopes(
    sdfg: dace.SDFG,
    strategy: str = "skip-taskloops",
    offloadable: Optional[Callable[[ExternalCall, Boundary], Tuple[bool, str]]] = None
) -> Tuple[dace.SDFG, List[OffloadScope]]:
    """Whole-program offload analysis: externalize each nest into a call, then put each call in its OWN
    state so it is an independent offload unit.

    NON-DESTRUCTIVE: works on a deepcopy, so the caller's SDFG is untouched (an analysis that does not
    commit must not mutate the input). Returns the transformed SDFG (externalized + isolated, still runnable
    via the ``DaceReference`` expansion) and one :class:`OffloadScope` per call, in extraction order.

    ``offloadable`` is the per-call decision, injected because the invariant is that each TOOL decides
    independently after extraction; it defaults to :func:`default_offloadable`.
    """
    decide = offloadable or default_offloadable
    work = copy.deepcopy(sdfg)
    calls = lower_nests_to_external_call(work, strategy)
    scopes: List[OffloadScope] = []
    for ext, boundary in calls:
        # Re-find the state each time: isolating a prior call fissions states, but the node object is stable.
        state = find_state_of_node(work, ext)
        isolate_into_own_state(work, state, ext)
        ok, reason = decide(ext, boundary)
        scopes.append(OffloadScope(ext.name, list(boundary.inputs), list(boundary.outputs), ok, reason))
    return work, scopes
