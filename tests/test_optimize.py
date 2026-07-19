"""Phase-3 per-nest optimization API (:mod:`nestforge.optimize`): the knob-bundle grid the agent reads
before choosing (``optimization_choices``), and the verb that turns a chosen bundle into ONE nest's
build recipe (``optimize``). Recipe-level -- nothing compiles here except the one buildability gate that
proves a chosen DaCe recipe actually generates + links on an extracted nest.
"""
from pathlib import Path

import numpy as np
import pytest

import dace
from nestforge.build import build_sdfg
from nestforge.extract import extract_nest_to_sdfg
from nestforge.optimize import (DEFAULT_OPT_MODE, OPT_MODES, DaceOptimizer, ExternalOptimizer, Optimizer, Proposal,
                                optimization_choices, optimize)
from nestforge.strategies import outer

N = dace.symbol("N")
f64 = dace.float64


@dace.program
def two_nests(a: f64[N], c: f64[N]):
    tmp = np.empty_like(a)
    for i in dace.map[0:N]:
        tmp[i] = a[i] * 2.0
    for i in dace.map[0:N]:
        c[i] = tmp[i] + 1.0


class DecliningOptimizer(Optimizer):
    """A knob bundle that declines every nest -- e.g. an unsupported flag combo on this compiler."""
    name = "declining"

    def propose(self, nest=None):
        return None


def first_nest_boundary():
    sdfg = two_nests.to_sdfg(simplify=True)
    parent, node = outer(sdfg)[0]
    return extract_nest_to_sdfg(parent, node, name="two_nests_n0")


def test_default_opt_mode_is_a_known_opt_mode():
    assert DEFAULT_OPT_MODE in OPT_MODES


def test_optimize_dace_choice_yields_dace_recipe():
    dace_choices = [o for o in optimization_choices() if isinstance(o, DaceOptimizer)]
    assert dace_choices  # the grid has a DaCe lane
    prop = optimize(first_nest_boundary(), dace_choices[0])
    assert isinstance(prop, Proposal)
    assert prop.lane == "dace"
    assert prop.opt_mode in OPT_MODES
    assert prop.build is not None


def test_optimize_external_choice_yields_external_recipe():
    ext_choices = [o for o in optimization_choices() if isinstance(o, ExternalOptimizer) and o.flags is not None]
    assert ext_choices  # the grid has a (supported) external lane
    prop = optimize(None, ext_choices[0])  # deterministic bundle: nest-independent
    assert isinstance(prop, Proposal)
    assert prop.lane == "external"
    assert prop.language and prop.compiler and prop.flags is not None


def test_optimize_returns_none_when_bundle_declines():
    assert optimize(first_nest_boundary(), DecliningOptimizer()) is None


def test_optimize_rejects_non_optimizer_knobs():
    with pytest.raises(TypeError):
        optimize(None, "not-an-optimizer")


def test_choices_are_named_optimizers_with_unique_names():
    choices = optimization_choices()
    assert all(isinstance(o, Optimizer) for o in choices)
    names = [o.name for o in choices]
    assert len(names) == len(set(names))  # each knob bundle is distinctly named


def test_dace_recipe_builds_on_extracted_nest(tmp_path: Path):
    boundary = first_nest_boundary()
    dace_choice = next(o for o in optimization_choices() if isinstance(o, DaceOptimizer))
    prop = optimize(boundary, dace_choice)
    built = build_sdfg(boundary.standalone_sdfg, tmp_path, prop.build)  # the chosen recipe generates + links
    assert built.so_path.exists()
