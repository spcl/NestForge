"""Splitting the computation around an unsupported (non-emittable) library node into distinct states, so
the externalize lane can offload the pure compute before/after it while the node itself stays native
(:mod:`nestforge.split_unsupported`). An MPI ``Bcast`` stands in for the general unsupported node.
"""
import dace
import pytest
from dace.sdfg import nodes

from nestforge.split_unsupported import (isolate_into_own_state, isolate_unsupported_library_nodes,
                                         unsupported_library_nodes)

N = dace.symbol("N")


def linear_chain_sdfg():
    """A single state with A -> t1 -> tmp -> M -> tmp2 -> t3 -> B (M a plain tasklet stand-in)."""
    sdfg = dace.SDFG("chain")
    for a in ("A", "B"):
        sdfg.add_array(a, [N], dace.float64)
    for a in ("tmp", "tmp2"):
        sdfg.add_transient(a, [N], dace.float64)
    st = sdfg.add_state("main")
    tmp, tmp2 = st.add_access("tmp"), st.add_access("tmp2")
    t1 = st.add_tasklet("t1", {"i0"}, {"o0"}, "o0 = i0 + 1.0")
    m = st.add_tasklet("M", {"i0"}, {"o0"}, "o0 = i0 * 2.0")
    t3 = st.add_tasklet("t3", {"i0"}, {"o0"}, "o0 = i0 - 3.0")
    st.add_edge(st.add_read("A"), None, t1, "i0", dace.Memlet("A[0]"))
    st.add_edge(t1, "o0", tmp, None, dace.Memlet("tmp[0]"))
    st.add_edge(tmp, None, m, "i0", dace.Memlet("tmp[0]"))
    st.add_edge(m, "o0", tmp2, None, dace.Memlet("tmp2[0]"))
    st.add_edge(tmp2, None, t3, "i0", dace.Memlet("tmp2[0]"))
    st.add_edge(t3, "o0", st.add_write("B"), None, dace.Memlet("B[0]"))
    return sdfg, st, m


def test_isolate_into_own_state_produces_valid_three_way_split():
    sdfg, st, m = linear_chain_sdfg()
    isolate_into_own_state(sdfg, st, m)
    states = list(sdfg.states())
    assert len(states) == 3
    # the node's state holds only the node (plus its in/out access nodes) -- no other tasklet.
    m_states = [s for s in states if m in s.nodes()]
    assert len(m_states) == 1
    others = [n for n in m_states[0].nodes() if n is not m and not isinstance(n, nodes.AccessNode)]
    assert others == []
    sdfg.validate()  # the split must leave a valid SDFG


def bcast_sdfg():
    """A single state mixing a producer tasklet, an MPI ``Bcast`` (unsupported), and a consumer tasklet."""
    from dace.libraries.mpi.nodes.bcast import Bcast
    sdfg = dace.SDFG("with_bcast")
    for a in ("A", "B"):
        sdfg.add_array(a, [8], dace.float64)
    for a in ("tmp", "tmp2"):
        sdfg.add_transient(a, [8], dace.float64)
    sdfg.add_array("root", [1], dace.int32)
    st = sdfg.add_state("main")
    tmp, tmp2 = st.add_access("tmp"), st.add_access("tmp2")
    prod = st.add_tasklet("prod", {"i0"}, {"o0"}, "o0 = i0 + 1.0")
    cons = st.add_tasklet("cons", {"i0"}, {"o0"}, "o0 = i0 - 1.0")
    bcast = Bcast("bcast")
    st.add_edge(st.add_read("A"), None, prod, "i0", dace.Memlet("A[0]"))
    st.add_edge(prod, "o0", tmp, None, dace.Memlet("tmp[0]"))
    st.add_edge(tmp, None, bcast, "_inbuffer", dace.Memlet("tmp[0:8]"))
    st.add_edge(st.add_read("root"), None, bcast, "_root", dace.Memlet("root[0]"))
    st.add_edge(bcast, "_outbuffer", tmp2, None, dace.Memlet("tmp2[0:8]"))
    st.add_edge(tmp2, None, cons, "i0", dace.Memlet("tmp2[0]"))
    st.add_edge(cons, "o0", st.add_write("B"), None, dace.Memlet("B[0]"))
    return sdfg, st, bcast


def test_unsupported_predicate_flags_mpi_not_registered_nodes():
    sdfg, st, bcast = bcast_sdfg()
    assert bcast in unsupported_library_nodes(st)
    # a registered node (MatMul) is NOT flagged.
    from dace.libraries.blas.nodes.matmul import MatMul
    st.add_node(MatMul("mm"))
    flagged = unsupported_library_nodes(st)
    assert bcast in flagged and all(type(n).__name__ != "MatMul" for n in flagged)


def test_pass_isolates_bcast_and_is_idempotent():
    sdfg, st, bcast = bcast_sdfg()
    n_isolated = isolate_unsupported_library_nodes(sdfg)
    assert n_isolated == 1
    states = list(sdfg.states())
    assert len(states) == 3  # producers | bcast island | consumers
    island = [s for s in states if bcast in s.nodes()]
    assert len(island) == 1
    # the Bcast is alone with its access nodes -- the producer/consumer tasklets are elsewhere.
    non_access = [n for n in island[0].nodes() if not isinstance(n, nodes.AccessNode)]
    assert non_access == [bcast]
    # every other state is free of unsupported nodes.
    assert all(not unsupported_library_nodes(s) for s in states if s is not island[0])
    # idempotent: a second run finds nothing to split.
    assert isolate_unsupported_library_nodes(sdfg) == 0
