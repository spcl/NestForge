# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The optimizer contract (:mod:`nestforge.optimizers`): each arena variant is a deterministic optimizer,
and the agent is one more under the same contract.

Two stubs, doing nothing in two different ways, are what let the agent path run in CI with no model:
:class:`NoOpAgent` is one-shot (propose the Phase-1 baseline, done), while :class:`StubAgent` is the whole
agentic LOOP -- propose, observe the measured outcome, stop. The loop is where an agent integration breaks
(a round that never ends, an outcome the agent cannot read, a proposal issued after ``stop``), and none of
it needs inference to test. The unit tests drive the loop against a FAKE measure so it runs with no
compiler at all; two integration tests then prove both stubs' proposals survive a real build and validate
bit-exact.
"""
import numpy as np
import pytest

from nestforge import tsvc
from nestforge.build import BuildOptions
from nestforge.optimizers import (BASELINE_OPT_MODE, AgenticOptimizer, DaceOptimizer, ExternalOptimizer, NoOpAgent,
                                  Optimizer, Outcome, Proposal, StubAgent, deterministic_optimizers, run_agent_loop)


# --- the stub agent: the agentic LOOP with nothing in it ---------------------------------------------
def fake_measure(median_us=1.0, ok=True):
    """A measure step with no compiler: the loop is what is under test, not the build. Records every
    proposal it was handed, so a test can assert what the agent actually asked for."""
    seen = []

    def measure(proposal):
        seen.append(proposal)
        return Outcome(proposal=proposal, ok=ok, median_us=median_us)

    measure.seen = seen
    return measure


def test_stub_agent_runs_the_whole_loop_and_stops():
    # propose -> measure -> observe -> stop. No model, no inference, no compiler: exactly what CI needs to
    # prove the agent plumbing works before any real agent is wired to it.
    agent = StubAgent()
    measure = fake_measure()
    outcomes = run_agent_loop(agent, nest=None, measure=measure)
    assert len(outcomes) == 1 and outcomes[0].ok
    assert len(measure.seen) == 1, "the stub must make exactly one proposal, then stop"
    assert measure.seen[0].opt_mode == BASELINE_OPT_MODE  # it does NOTHING: the Phase-1 baseline, unchanged


def test_stub_agent_observes_the_outcome_it_was_given():
    # The agent must be able to READ its result -- that is the only thing separating the agentic loop from
    # a one-shot optimizer, so an outcome that never reaches the agent is the loop being fake.
    agent = StubAgent()
    run_agent_loop(agent, nest=None, measure=fake_measure(median_us=42.0))
    assert len(agent.observed) == 1
    assert agent.observed[0].median_us == 42.0 and agent.observed[0].ok


def test_stub_agent_stops_even_when_its_round_budget_is_larger():
    # The budget is a CEILING, not a target: having seen one result and having no move to make, the stub
    # stops at once rather than re-proposing the same cell until the budget runs out.
    agent = StubAgent(max_rounds=5)
    measure = fake_measure()
    outcomes = run_agent_loop(agent, nest=None, measure=measure)
    assert len(outcomes) == 1 and len(measure.seen) == 1


def test_stub_agent_still_stops_when_its_proposal_fails_the_correctness_gate():
    # A losing candidate must not put the loop in a spin. Correctness is a HARD gate, so the agent only
    # gets to react by proposing something else -- and the stub has nothing else.
    agent = StubAgent()
    outcomes = run_agent_loop(agent, nest=None, measure=fake_measure(ok=False, median_us=float("inf")))
    assert len(outcomes) == 1 and not outcomes[0].ok


def test_agent_loop_raises_rather_than_hangs_when_an_agent_never_stops():
    """THE reason the bound is enforced by the loop and not left to the agent: a buggy agent that keeps
    proposing must fail with a name, not time the CI job out with none."""

    class RunawayAgent(AgenticOptimizer):
        name = "runaway"

        def propose(self, nest=None):
            return Proposal(self.name, "dace", opt_mode=BASELINE_OPT_MODE, build=BuildOptions())

    with pytest.raises(RuntimeError, match="without stopping"):
        run_agent_loop(RunawayAgent(max_rounds=3), nest=None, measure=fake_measure())


def test_agentic_optimizer_is_an_optimizer_and_rejects_a_useless_budget():
    assert isinstance(StubAgent(), Optimizer)  # same contract: the agent is just another optimizer
    with pytest.raises(ValueError, match="max_rounds"):
        StubAgent(max_rounds=0)  # a loop that cannot run a single round is a silent no-op, not a stub


# --- the no-op agent: the identity, proposes the Phase-1 baseline -------------------------------------
def test_noop_agent_proposes_the_phase1_baseline():
    agent = NoOpAgent()
    assert agent.name == "noop"
    proposal = agent.propose()
    assert proposal.lane == "dace"
    assert proposal.opt_mode == BASELINE_OPT_MODE
    assert proposal.build == BuildOptions()  # unchanged, default codegen -- the denominator
    assert proposal.fp_mode == "strict-ieee"  # the baseline carries no FP error


def test_noop_agent_is_deterministic():
    a, b = NoOpAgent().propose(), NoOpAgent().propose()
    assert a == b  # same recipe every time (frozen dataclass equality)


def test_noop_agent_is_an_optimizer():
    assert isinstance(NoOpAgent(), Optimizer)


# --- each variant is an optimizer --------------------------------------------------------------------
def test_dace_optimizer_proposes_its_variant():
    opt = DaceOptimizer("canonicalize", BuildOptions(compiler="g++"))
    proposal = opt.propose()
    assert proposal.lane == "dace" and proposal.opt_mode == "canonicalize"
    assert "opt=canonicalize" in opt.name and "cc=g++" in opt.name


def test_dace_optimizer_rejects_an_unknown_opt_mode():
    with pytest.raises(ValueError, match="opt_mode"):
        DaceOptimizer("no-such-mode", BuildOptions())


def test_external_optimizer_proposes_a_composed_flag_cell():
    opt = ExternalOptimizer("c", "gnu", "gcc", fp_mode="strict-ieee", cost_model="cheap")
    proposal = opt.propose()
    assert proposal is not None and proposal.lane == "external"
    assert proposal.language == "c" and proposal.compiler == "gcc"
    assert "-O3" in proposal.flags  # every cell starts from base_flags
    assert proposal.fp_mode == "strict-ieee" and proposal.cost_model == "cheap"


def test_external_optimizer_declines_when_its_flags_are_unavailable():
    # When lane_flags reports a combination unsupported, the optimizer keeps being an optimizer -- it just
    # declines: propose() -> None. Force that state directly rather than depend on which combos a given
    # toolchain rejects (that varies, e.g. Polly makes llvm auto-par supported).
    opt = ExternalOptimizer("c", "gnu", "gcc")
    assert opt.propose() is not None  # a supported cell proposes
    opt.flags = None
    opt.skip_reason = "forced-unsupported"
    assert opt.propose() is None


def test_each_optimizer_is_deterministic_and_uniquely_named():
    opts = deterministic_optimizers()
    names = [o.name for o in opts]
    assert len(names) == len(set(names)), f"duplicate optimizer names: {names}"
    for o in opts:
        assert o.propose() == o.propose()  # deterministic


def test_enumeration_covers_the_dace_and_external_lanes():
    opts = deterministic_optimizers(opt_modes=("simplify-parallel", "canonicalize"),
                                    external=(("c", "gnu", "gcc"), ("fortran", "gnu", "gfortran")))
    lanes = {p.lane for p in (o.propose() for o in opts) if p is not None}
    assert lanes == {"dace", "external"}
    # one DaCe optimizer per (opt_mode, compiler); the two opt-modes are both present.
    dace_modes = {o.opt_mode for o in opts if isinstance(o, DaceOptimizer)}
    assert dace_modes == {"simplify-parallel", "canonicalize"}


def test_proposal_validates_its_lane_fields():
    with pytest.raises(ValueError, match="dace proposal"):
        Proposal("bad", "dace")  # missing opt_mode + build
    with pytest.raises(ValueError, match="external proposal"):
        Proposal("bad", "external")  # missing language/compiler/flags
    with pytest.raises(ValueError, match="unknown lane"):
        Proposal("bad", "quantum")


# --- end-to-end: the no-op agent's proposal builds + validates bit-exact ------------------------------
@pytest.mark.integration  # compiles a nest -- excluded from the fast unit set
def test_noop_agent_proposal_builds_and_validates(tmp_path):
    from nestforge import build
    from nestforge.arena import make_inputs, run_oracle
    from nestforge.multinest import extract_all_nests
    from nestforge.translate import prepare

    kernel = tsvc.iter_tsvc_kernels(only=["s000"])[0]
    proposal = NoOpAgent().propose()

    nests = extract_all_nests(lambda: tsvc.build_sdfg(kernel, proposal.opt_mode), "outer", kernel.key)
    assert nests, "s000 produced no nest"
    _, name, _, boundary = nests[0]
    sizes = tsvc.sample_sizes(kernel, boundary, preset="S")
    inputs = make_inputs(boundary, sizes, given=tsvc.index_fills(kernel, boundary, sizes))

    prep = prepare(boundary, name, tmp_path, sizes=sizes)
    oracle = run_oracle(prep, boundary, inputs, sizes)

    built = build.build_sdfg(boundary.standalone_sdfg, tmp_path / "build", opts=proposal.build)
    buffers = {k: v.copy() for k, v in inputs.items()}
    built.run(buffers, sizes)
    got = {o: buffers[o] for o in boundary.outputs}
    assert all(np.allclose(got[o], oracle[o]) for o in oracle), "no-op agent proposal diverged from oracle"


@pytest.mark.integration  # compiles a nest -- excluded from the fast unit set
def test_stub_agent_loop_builds_validates_and_times_for_real(tmp_path):
    """The agentic loop end to end against a REAL measure: propose -> build -> validate bit-exact -> time
    -> observe -> stop. The unit tests above drive the loop against a fake measure, which proves the
    plumbing but never proves an agent's proposal survives a compiler. This is the CI guarantee that a
    model dropped into this seat has a working loop to sit in -- no inference involved.
    """
    from nestforge import build
    from nestforge.arena import make_inputs, run_oracle
    from nestforge.multinest import extract_all_nests
    from nestforge.translate import prepare

    kernel = tsvc.iter_tsvc_kernels(only=["s000"])[0]
    nests = extract_all_nests(lambda: tsvc.build_sdfg(kernel, BASELINE_OPT_MODE), "outer", kernel.key)
    assert nests, "s000 produced no nest"
    _, name, _, boundary = nests[0]
    sizes = tsvc.sample_sizes(kernel, boundary, preset="S")
    inputs = make_inputs(boundary, sizes, given=tsvc.index_fills(kernel, boundary, sizes))
    oracle = run_oracle(prepare(boundary, name, tmp_path, sizes=sizes), boundary, inputs, sizes)

    def measure(proposal):
        built = build.build_sdfg(boundary.standalone_sdfg,
                                 tmp_path / proposal.name.replace(":", "_"),
                                 opts=proposal.build)
        buffers = {k: v.copy() for k, v in inputs.items()}
        elapsed = built.run(buffers, sizes)
        ok = all(np.allclose(buffers[o], oracle[o]) for o in oracle)  # the HARD gate: wrong never competes
        return Outcome(proposal=proposal, ok=ok, median_us=float(elapsed) if elapsed else 1.0)

    agent = StubAgent()
    outcomes = run_agent_loop(agent, nest=boundary, measure=measure)
    assert len(outcomes) == 1, "the stub proposes once and stops"
    assert outcomes[0].ok, "stub-agent proposal diverged from the oracle"
    assert agent.observed and agent.observed[0].ok  # the result reached the agent
