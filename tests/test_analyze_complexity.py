# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The static work-exponent rule (``scripts/analyze_complexity.py``): every clause of it, on the loop
shape that motivated it. Pure text analysis -- no compile, no run."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_complexity import scalar_params, strip_comments, with_derived_locals, work_exponent  # noqa: E402


def k(src, params=("n", )):
    """The exponent as :func:`analyze_complexity.analyze` computes it: strip comments, follow local
    rebindings of the size, then walk the loops."""
    text = strip_comments(src)
    return work_exponent(text, with_derived_locals(text, set(params)))


def test_sequential_nests_add_they_do_not_multiply():
    """Two loops in sequence are 2n, not n^2 -- so the exponent is the deeper of the two, not the sum."""
    assert k("void f(){ for (int i=0;i<n;++i){ x[i]=1; } for (int j=0;j<n;++j){ y[j]=2; } }") == 1


def test_nesting_multiplies():
    assert k("void f(){ for (int i=0;i<n;++i){ for (int j=0;j<n;++j){ a[i][j]=0; } } }") == 2


def test_literal_bound_is_repetition_not_work():
    """A loop bounded by a constant repeats fixed work; scaling the problem does not lengthen it."""
    assert k("void f(){ for (int i=0;i<n;++i){ for (int u=0;u<4;++u){ a[i]+=u; } } }") == 1


def test_tiled_nest_counts_tiles_not_depth():
    """The shape this rule exists for: six nested loops, O(n^2) work. The inner bounds name the enclosing
    index of a STRIDED loop, so they cover one tile rather than the whole range."""
    src = """void f(){
      for (int ii=0;ii<n;ii+=t){ for (int jj=0;jj<n;jj+=t){
        for (int i=ii;i<ii+t;++i){ for (int j=jj;j<jj+t;++j){ a[i][j]=0; } } } } }"""
    assert k(src, params=("n", "t")) == 2


def test_triangular_nest_is_quadratic():
    """Same "bound names the enclosing index" shape as the tiled case, but the outer loop steps by ONE,
    so the inner trip count grows with n. The bound names no parameter at all -- only the outer index."""
    assert k("void f(){ for (int i=1;i<n;++i){ for (int j=0;j<=i-1;++j){ a[i][j]=0; } } }") == 2


def test_size_rebound_to_a_local_is_followed():
    """``s471`` does ``int m = len_1d;`` and loops on ``m``. Matching parameter names alone read that as
    unbounded-by-size and reported the kernel O(1)."""
    assert with_derived_locals("int m = n; int h = m / 2;", {"n"}) == {"n", "m", "h"}
    assert k("void f(){ int m = n; for (int i=0;i<m;++i){ a[i]=0; } }") == 1


def test_unbraced_body_is_not_counted():
    """A single-statement loop opens no scope. It is left uncounted (and so surfaces as a skip) rather
    than swallowing every later loop into a phantom nest."""
    assert k("void f(){ for (int i=0;i<n;++i) a[i]=0; for (int j=0;j<n;++j){ b[j]=0; } }") == 1


def test_pointer_parameters_are_not_bounds():
    """Only by-value scalars can bound a loop; a pointer is the data."""
    assert scalar_params("void f(double *__restrict__ a, int iterations, int len_1d) {",
                         "f") == {"iterations", "len_1d"}


def test_comments_are_not_code():
    """Kernel comments carry prose like ``// for i: ...`` and the attribution header's parentheses."""
    assert k("void f(){ // for (int i=0;i<n;++i){ nope\n for (int i=0;i<n;++i){ a[i]=0; } }") == 1
