"""Baseline comparison lanes (paper C1/C2). Unit set, no compile: the baseline set is well-formed, each
lane either proposes a recipe or records a skip_reason (never crashes), whole-program proposes at
whole-program scope, and the Pluto lane is gated on its tool with a recorded reason."""
from nestforge.baselines import PLUTO_TOOL, baseline_names, baseline_optimizers, pluto_available
from nestforge.optimizers import ExternalOptimizer, Optimizer, WholeProgramOptimizer


def test_baseline_set_is_named_and_complete():
    names = baseline_names()
    assert names == ["gcc-O3", "llvm-O3", "graphite", "polly", "whole-program", "pluto"]


def test_every_optimizer_proposes_or_records_a_skip_reason():
    for opt in baseline_optimizers():
        assert isinstance(opt, Optimizer)
        proposal = opt.propose()
        if proposal is None:  # only an external auto-par lane declines -- with a reason, not a crash
            assert isinstance(opt, ExternalOptimizer) and opt.skip_reason, f"{opt.name} declined without a reason"
        else:
            assert proposal.name == opt.name


def test_whole_program_lane_proposes_whole_program_scope():
    wp = next(o for o in baseline_optimizers() if o.name == "whole-program")
    assert isinstance(wp, WholeProgramOptimizer)
    assert wp.propose().scope == "whole-program"


def test_pluto_lane_is_tool_gated_with_a_reason():
    available, reason = pluto_available()
    assert isinstance(available, bool)
    if available:
        assert reason is None
    else:
        assert reason and PLUTO_TOOL in reason  # names the missing tool
