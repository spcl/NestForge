"""The whole-program baseline lane (:mod:`nestforge.whole_program`): optimize the entire program as one
unit and measure it, so the per-nest arena has an honest baseline to beat. Unit tests cover the
whole-program optimizer + scope contract; one integration test builds + validates + times a real kernel's
whole-program DaCe auto-opt baseline bit-exact vs the whole-program oracle.
"""
import pytest

from nestforge import tsvc
from nestforge.build import BuildOptions
from nestforge.optimizers import NoOpAgent, Optimizer, Proposal, WholeProgramOptimizer


# --- the whole-program optimizer + scope contract ----------------------------------------------------
def test_whole_program_optimizer_proposes_whole_program_scope():
    opt = WholeProgramOptimizer("auto-opt")
    assert isinstance(opt, Optimizer)
    proposal = opt.propose()
    assert proposal.scope == "whole-program" and proposal.lane == "dace"
    assert proposal.opt_mode == "auto-opt"
    assert "whole-program" in opt.name and "opt=auto-opt" in opt.name


def test_whole_program_optimizer_defaults_to_auto_opt():
    assert WholeProgramOptimizer().propose().opt_mode == "auto-opt"  # the strong cross-nest baseline


def test_whole_program_optimizer_rejects_an_unknown_opt_mode():
    with pytest.raises(ValueError, match="opt_mode"):
        WholeProgramOptimizer("no-such-mode")


def test_whole_program_optimizer_is_deterministic():
    assert WholeProgramOptimizer("canonicalize").propose() == WholeProgramOptimizer("canonicalize").propose()


def test_proposal_rejects_an_unknown_scope():
    with pytest.raises(ValueError, match="unknown scope"):
        Proposal("bad", "dace", scope="galaxy", opt_mode="auto-opt", build=BuildOptions())


def test_per_nest_optimizers_default_to_per_nest_scope():
    # the no-op agent and the per-nest variants are per-nest; only WholeProgramOptimizer is whole-program.
    assert NoOpAgent().propose().scope == "per-nest"


# --- end-to-end: the whole-program baseline builds + validates + times --------------------------------
@pytest.mark.integration  # compiles the whole program -- excluded from the fast unit set
@pytest.mark.parametrize("opt_mode", ["simplify-parallel", "auto-opt"])
def test_whole_program_baseline_builds_validates_and_times(opt_mode, tmp_path):
    from nestforge.whole_program import measure_whole_program

    kernel = tsvc.iter_tsvc_kernels(only=["s000"])[0]
    result = measure_whole_program(WholeProgramOptimizer(opt_mode), kernel, tmp_path, preset="S", reps=3)
    assert result.error is None, result.error
    assert result.ok, f"whole-program {opt_mode} diverged from the oracle (maxdiff={result.maxdiff})"
    assert result.median_us > 0.0  # a real, positive measurement
    assert result.opt_mode == opt_mode
