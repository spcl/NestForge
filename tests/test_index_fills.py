# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Integer and bool inputs of the arena: an index array filled with zeros turns a gather into one repeated read, and
a kernel that indexes wrongly then matches its oracle. ``make_inputs`` fills an integer array as the bench does,
with subscripts its manifests keep in bounds, and a bool array with both truth values."""

import dace
import numpy as np
import pytest
from dace.transformation.passes.canonicalize import canonicalize
from helpers import corpus_kernel, extract_nest_to_sdfg
from test_kernel_stage import lowered_kernel

from nestforge.build.arena import make_inputs, run_oracle
from nestforge.corpus.bench import preset_sizes
from nestforge.corpus.translate import prepare
from nestforge.stages.kernel import build_kernel_library, schedule_kernel, validate_kernel
from nestforge.stages.scopes import parallel_top_level_maps

#: (kernel, index array). Every one is a gather/scatter whose index array is the whole point of the test;
#: loop_level_reasoning is a superset of TSVC-2, so these are TSVC kernels.
GATHER_KERNELS = ["tsvc_2_vag", "tsvc_2_s4113", "tsvc_2_s353", "reroll_gather"]
INDEX_ARRAY = "ip"
N = dace.symbol("N")


# These kernels are named constants of the corpus this repo pins, and every one is a gather/scatter whose
# index array is the whole point of the test. A missing kernel or a nest-less kernel is therefore a broken
# corpus or a broken splitter -- a failure to surface, never a skip to hide behind. (A skip here would also
# fail CI, which runs the unit set under NESTFORGE_CI_NO_SKIP=1.)
def first_nest(kernel):
    sdfg = kernel.to_sdfg(simplify=True)
    canonicalize(sdfg, target="cpu")
    refs = parallel_top_level_maps(sdfg)
    assert refs, f"{kernel.short_name}: the splitter found no compute nest -- this kernel has one"
    return extract_nest_to_sdfg(refs[0][0], refs[0][1], name=kernel.short_name.rsplit("/", 1)[-1])


@pytest.mark.parametrize("key", GATHER_KERNELS)
def test_an_index_array_takes_many_values_and_stays_in_bounds(key):
    """Every subscript lies inside every vector the nest reads (a length-1 array is a scalar, never subscripted),
    and the oracle runs on them: NumPy raises on an index past the end."""
    kernel = corpus_kernel(key)
    boundary = first_nest(kernel)
    sizes = preset_sizes(kernel, "S")
    inputs = make_inputs(boundary, sizes, seed=0)
    ip = inputs[INDEX_ARRAY]
    assert ip.size > 1 and len(np.unique(ip)) > 1
    extent = min(a.shape[0] for name, a in inputs.items() if name != INDEX_ARRAY and a.size > 1)
    assert 0 <= ip.min() and ip.max() < extent, (ip.min(), ip.max(), extent)
    run_oracle(prepare(boundary, "k"), boundary, inputs, {s: sizes[s] for s in boundary.symbols})


@dace.program
def masked_copy(m: dace.bool[N], x: dace.float64[N], y: dace.float64[N]):
    for i in dace.map[0:N]:
        y[i] = x[i] if m[i] else 0.0


def test_a_bool_input_takes_both_truth_values():
    _, _, boundary = lowered_kernel(masked_copy)
    mask = make_inputs(boundary, {"N": 64}, seed=0)["m"]
    assert mask.dtype == np.bool_ and mask.any() and not mask.all()


@dace.program
def gather(x: dace.float64[N], idx: dace.int64[N], out: dace.float64[N]):
    for i in dace.map[0:N]:
        out[i] = x[idx[i]]


@dace.program
def hoisted_gather(x: dace.float64[N], idx: dace.int64[N], out: dace.float64[N]):
    for i in dace.map[0:N]:
        out[i] = x[idx[0]]


@pytest.mark.e2e
def test_a_gather_that_hoists_its_index_load_fails_validation(tmp_path):
    """A gather miscompiled to read ``x[idx[0]]`` agrees with ``x[idx[i]]`` on all-zero indices; it must not
    match the gather's oracle, while the right gather still does."""
    _, ext, boundary = lowered_kernel(gather)
    oracle = prepare(boundary, ext.name)
    for program, ok in ((gather, True), (hoisted_gather, False)):
        _, wrong_ext, wrong_boundary = lowered_kernel(program)
        src = schedule_kernel(wrong_ext, wrong_boundary, tmp_path / program.name / "gen")
        archive = build_kernel_library(src, "g++", None, tmp_path / program.name / "lib")
        verdict = validate_kernel(archive, src, oracle, {"N": 1037}, reps=1)
        assert verdict.error == "", verdict.error
        assert verdict.ok is ok, (program.name, verdict)
