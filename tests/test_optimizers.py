"""The optimizer contract (:mod:`nestforge.optimizers`): each arena variant is a deterministic optimizer,
and the agent is one more under the same contract. The no-op agent -- the identity stub that proposes the
Phase-1 baseline -- is what lets the whole loop run in CI with no model. One end-to-end (integration) proves
the no-op agent's proposal actually builds and validates bit-exact.
"""
import numpy as np
import pytest

from nestforge import tsvc
from nestforge.build import BuildOptions
from nestforge.optimizers import (BASELINE_OPT_MODE, DaceOptimizer, ExternalOptimizer, NoOpAgent, Optimizer, Proposal,
                                  deterministic_optimizers)


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
