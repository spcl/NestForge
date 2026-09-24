# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The default random fill of an integer index array is all zeros, which turns a gather into one repeated read
and a conflict-free scatter into writes to one element; the oracle reads the same zeros, so validation cannot
notice."""

import numpy as np
import pytest

from dace.transformation.passes.canonicalize import canonicalize

from nestforge.build.arena import make_inputs
from nestforge.corpus.bench import preset_sizes
from nestforge.ir.extract import extract_nest_to_sdfg
from nestforge.phases.scopes import parallel_top_level_maps

from helpers import loop_level_kernel

#: (kernel, index array). Every one is a gather/scatter whose index array is the whole point of the test;
#: loop_level_reasoning is a superset of TSVC-2, so these are TSVC kernels.
GATHER_KERNELS = ["tsvc_2_vag", "tsvc_2_s4113", "tsvc_2_s353", "reroll_gather"]
INDEX_ARRAY = "ip"


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
def test_the_default_fill_of_an_index_array_is_all_zeros(key):
    # the plain float fill cast to an int dtype is all zeros, so every iteration reads b[0]
    kernel = loop_level_kernel(key)
    boundary = first_nest(kernel)
    sizes = preset_sizes(kernel, "S")
    ip = make_inputs(boundary, sizes, seed=0)[INDEX_ARRAY]
    assert ip.size > 1 and len(np.unique(ip)) == 1 and ip.ravel()[0] == 0
