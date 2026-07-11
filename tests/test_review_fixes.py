"""Regression tests for the code-review findings (corpus selection, C-style emission, emit guards)."""
import numpy as np
import pytest
import dace

from nestforge.emit_numpy import sdfg_to_numpy, UnsupportedNest

N = dace.symbol('N')


# ----- corpus: pick the kernel's entry @dace.program, not the first helper (Finder B#1) ----------
def test_corpus_program_is_the_entry_not_a_helper():
    pytest.importorskip("optarena")
    from nestforge.corpus import iter_dace_kernels
    ks = {k.short_name: k for k in iter_dace_kernels()}
    # mlp_dace defines relu, softmax, then mlp; resnet has resnet_basicblock + a _gpu variant after it.
    assert ks["ml/mlp/mlp"].program().name.endswith("mlp")
    assert ks["ml/resnet/resnet"].program().name.endswith("resnet_basicblock")


def test_corpus_module_path_independent_of_namespace_path():
    pytest.importorskip("optarena")
    from nestforge.corpus import module_path
    # Derived from the registry key, not optarena.benchmarks.__path__ (which can be stale/multi-root).
    assert module_path("hpc/dense_linear_algebra/gemm/gemm") == \
        "optarena.benchmarks.hpc.dense_linear_algebra.gemm.gemm_dace"


# ----- C-style emission: pre-allocated buffers, no internal allocation ----------------------------
def run(src, fn, **buffers):
    ns = {"np": np}
    exec(src, ns)
    ns[fn](**buffers)


@dace.program
def dot_scale(x: dace.float64[N], y: dace.float64[N], z: dace.float64[N], out: dace.float64[N]):
    s = np.dot(x, y)
    for i in dace.map[0:N]:
        out[i] = z[i] * s


def test_scalar_transient_consistent_between_libnode_and_tasklet():
    # Dot writes a scalar transient `s`; the map tasklet reads it. Both must name it identically.
    src = sdfg_to_numpy(dot_scale.to_sdfg(simplify=True), "k")
    n = 6
    rng = np.random.default_rng(0)
    x, y, z, out = rng.random(n), rng.random(n), rng.random(n), np.zeros(n)
    run(src, "k", x=x, y=y, z=z, out=out, N=n)
    np.testing.assert_allclose(out, z * (x @ y))


@dace.program
def matvec_return(A: dace.float64[N, N], v: dace.float64[N]):
    return A @ v


def test_return_and_scratch_are_inplace_buffer_params_no_allocation():
    sdfg = matvec_return.to_sdfg(simplify=True)
    src = sdfg_to_numpy(sdfg, "k")
    assert "np.empty" not in src and "np.zeros" not in src, "C-style: caller pre-allocates, kernel must not"
    assert "return " not in src, "C-style: __return is an in-place output buffer param, not a python return"
    # __return is a parameter written in place.
    header = src.splitlines()[0]
    assert "__return" in header
    n = 5
    rng = np.random.default_rng(1)
    A, v = rng.random((n, n)), rng.random(n)
    ret = np.zeros(n)
    run(src, "k", A=A, v=v, __return=ret, N=n)
    np.testing.assert_allclose(ret, A @ v)


# ----- emit guards: nested constructs must raise, not silently mis-emit (Finder A#6 / C#7) --------
def nested_map_sdfg():
    sdfg = dace.SDFG("nested")
    sdfg.add_array("A", [N, N], dace.float64)
    sdfg.add_array("B", [N, N], dace.float64)
    st = sdfg.add_state()
    me_i, mx_i = st.add_map("outer", dict(i="0:N"))
    me_j, mx_j = st.add_map("inner", dict(j="0:N"))
    t = st.add_tasklet("t", {"a"}, {"b"}, "b = a * 2.0")
    rA, wB = st.add_read("A"), st.add_write("B")
    st.add_memlet_path(rA, me_i, me_j, t, dst_conn="a", memlet=dace.Memlet("A[i, j]"))
    st.add_memlet_path(t, mx_j, mx_i, wB, src_conn="b", memlet=dace.Memlet("B[i, j]"))
    return sdfg


def test_nested_map_in_map_raises_instead_of_dropping_inner_loop():
    with pytest.raises(UnsupportedNest):
        sdfg_to_numpy(nested_map_sdfg(), "k")


# ----- emit staging: a map-entry-sourced scalar read (`b_index = b[i]`) must be emitted -----------
@dace.program
def gather(a: dace.float64[N], b: dace.int64[N], out: dace.float64[N]):
    for i in dace.map[0:N]:
        out[i] = a[b[i]]


def test_indirect_gather_stages_map_entry_read():
    """DaCe requires every array access to be symbolic, so an indirect read ``a[b[i]]`` is staged as a
    scalar access node fed by the map entry (``<sym> = b[i]``) before the gather ``a[<sym>]``. The
    ``copy_lines`` map-entry branch emits that load; without it the gather names an undefined symbol and
    the kernel raises ``NameError`` at exec. Guards the map-entry-staging fix (baseline opt mode)."""
    sdfg = gather.to_sdfg(simplify=True)
    src = sdfg_to_numpy(sdfg, "k")
    assert "= b[i]" in src, f"map-entry-sourced staging load not emitted:\n{src}"
    n = 8
    rng = np.random.default_rng(0)
    a, b, out = rng.random(n), rng.integers(0, n, size=n).astype(np.int64), np.zeros(n)
    run(src, "k", a=a, b=b, out=out, N=n)  # NameError here if the staged read is dropped
    np.testing.assert_allclose(out, a[b])


# ----- arena regression: a value-returning kernel no longer crashes run_oracle (Finder C#1) -------
@dace.program
def scaley(A: dace.float64[N]):
    return A * 2.0


def test_returning_kernel_survives_arena_oracle_and_manifest_matches(tmp_path):
    from nestforge.pass_lower import lower_nests_to_external_call
    from nestforge.translate import prepare
    from nestforge.arena import make_inputs, run_oracle

    sdfg = scaley.to_sdfg(simplify=True)
    ext, boundary = lower_nests_to_external_call(sdfg)[0]
    assert boundary.outputs == ["__return"]
    prep = prepare(boundary, ext.name, tmp_path / "k")
    # __return is an in-place buffer parameter in the numpy signature AND the manifest -- aligned.
    assert "__return" in prep.numpy_source.splitlines()[0]
    assert "return " not in prep.numpy_source
    assert prep.manifest["input_args"] == list(prep.manifest["input_args"])  # well-formed
    assert "__return" in prep.manifest["input_args"]
    sizes = {"N": 16}
    out = run_oracle(prep, boundary, make_inputs(boundary, sizes), sizes)  # crashed pre-fix
    assert "__return" in out
