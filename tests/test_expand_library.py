# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The ``expand-library`` move: a library call shows as a ``lib_<family>`` row and expands into canonical maps."""

import re

import dace
import numpy as np
from dace.libraries.standard.nodes.external_call import external_calls
from dace.sdfg import nodes

from nestforge.records import MoveStatus
from nestforge.session import Session

N = dace.symbol("N", dtype=dace.int64)
SIZE = 24


@dace.program
def product_total(a: dace.float64[N, N], b: dace.float64[N, N], c: dace.float64[N, N], total: dace.float64[1]):
    c[:] = a @ b
    total[0] = dace.reduce(lambda x, y: x + y, c)


def library_session(tmp_path) -> Session:
    return Session(product_total.to_sdfg(simplify=True), work_dir=str(tmp_path), sizes={"N": SIZE})


def tree(session: Session) -> str:
    states = re.findall(r"\b(state\d+_\d+)\b", session.describe())
    session.expand(states, session.epoch)
    return session.describe()


def library_nodes(sdfg: dace.SDFG) -> list[str]:
    return [type(n).__name__ for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.LibraryNode)]


def row(text: str, family: str) -> str:
    match = re.search(rf"\b(lib_{family}\d+_\d+)\b", text)
    assert match is not None, text
    return match.group(1)


def test_library_calls_are_lib_rows_named_by_their_family_and_each_can_expand(tmp_path):
    sut = library_session(tmp_path)

    text = tree(sut)

    blas, reduce = row(text, "blas"), row(text, "reduce")
    offered = {move.labels for move in sut.list_moves("expand-library")}
    assert offered == {(blas,), (reduce,)}, (offered, text)


def test_a_collapsed_state_counts_its_library_calls_as_doalls(tmp_path):
    sut = library_session(tmp_path)

    text = sut.describe()

    assert sum(int(n) for n in re.findall(r"\bdoalls=(\d+)", text)) == 2, text


def test_an_expanded_product_is_maps_that_compute_it_and_no_library_call(tmp_path):
    sut = library_session(tmp_path)
    blas = row(tree(sut), "blas")

    result = sut.apply_move("expand-library", [blas], sut.epoch)

    assert result.status is MoveStatus.APPLIED, result
    assert library_nodes(sut.sdfg) == ["Reduce"], "only the reduction stays a library call"
    text = tree(sut)
    assert "lib_blas" not in text
    assert re.search(r"doall\d+_\d+  \[\w+=0:N, \w+=0:N, \w+=0:N\]  reduce=\(\+ over", text), text
    rng = np.random.default_rng(3)
    a, b = rng.random((SIZE, SIZE)), rng.random((SIZE, SIZE))
    c, total = np.zeros((SIZE, SIZE)), np.zeros(1)
    sut.sdfg(a=a, b=b, c=c, total=total, N=SIZE)
    np.testing.assert_allclose(c, a @ b, rtol=1e-12)
    np.testing.assert_allclose(total[0], (a @ b).sum(), rtol=1e-12)


def test_a_kernel_nestforge_built_is_not_a_library_call_to_expand(tmp_path):
    sut = library_session(tmp_path)
    sut.apply_move("expand-library", [row(tree(sut), "blas")], sut.epoch)
    (kernel, *_) = sut.default_scopes()

    result = sut.apply_move("expand-library", [kernel.name], sut.epoch)

    assert result.status is MoveStatus.ILLEGAL
    assert "not a library call" in result.reason


def kernel_libraries(sut: Session) -> dict[str, list[nodes.LibraryNode]]:
    """Each kernel's library calls, as its standalone SDFG holds them."""
    return {
        ext.name: [n for n, owner in ext.standalone_sdfg.all_nodes_recursive() if isinstance(n, nodes.LibraryNode)]
        for ext in external_calls(sut.sdfg)
    }


def test_each_library_call_is_a_default_kernel_of_its_own_like_a_doall(tmp_path):
    sut = library_session(tmp_path)

    sut.default_scopes()

    held = sorted(type(n).__name__ for calls in kernel_libraries(sut).values() for n in calls)
    assert held == ["MatMul", "Reduce"], held
    assert not [n for n, owner in sut.sdfg.all_nodes_recursive() if type(n).__name__ in ("MatMul", "Reduce")]


def test_a_library_call_offers_one_move_per_implementation_a_cpu_kernel_renders(tmp_path):
    sut = library_session(tmp_path)
    reduce = row(tree(sut), "reduce")

    offered = {m.implementation for m in sut.list_moves("set-library-implementation") if m.labels == (reduce,)}

    assert "pure" in offered, offered
    assert not offered & {"cuBLAS", "hipBLAS", "CUDA", "GPUAuto"}, offered


def test_a_pinned_implementation_keeps_the_row_and_reaches_the_kernel_that_builds(tmp_path):
    sut = library_session(tmp_path)
    reduce = row(tree(sut), "reduce")

    result = sut.apply_move("set-library-implementation", [reduce], sut.epoch, implementation="pure")

    assert result.status is MoveStatus.APPLIED, result
    assert row(tree(sut), "reduce") == reduce
    sut.default_scopes()
    (kernel,) = [name for name, calls in kernel_libraries(sut).items() if type(calls[0]).__name__ == "Reduce"]
    (call,) = kernel_libraries(sut)[kernel]
    assert (call.implementation, call.auto_select_implementation) == ("pure", False)
    (evidence,) = [e for e in sut.kernel_evidence(reps=1) if e.kernel == kernel]
    assert evidence.verdict.ok, evidence.verdict


def test_a_device_library_on_a_cpu_only_target_is_refused(tmp_path):
    sut = library_session(tmp_path)
    reduce = row(tree(sut), "reduce")

    result = sut.apply_move("set-library-implementation", [reduce], sut.epoch, implementation="CUDA (device)")

    assert result.status is MoveStatus.ILLEGAL
    assert "on this target" in result.reason


if __name__ == "__main__":
    import pathlib
    import tempfile

    for test in (
        test_library_calls_are_lib_rows_named_by_their_family_and_each_can_expand,
        test_a_collapsed_state_counts_its_library_calls_as_doalls,
        test_an_expanded_product_is_maps_that_compute_it_and_no_library_call,
        test_a_kernel_nestforge_built_is_not_a_library_call_to_expand,
        test_each_library_call_is_a_default_kernel_of_its_own_like_a_doall,
        test_a_library_call_offers_one_move_per_implementation_a_cpu_kernel_renders,
        test_a_pinned_implementation_keeps_the_row_and_reaches_the_kernel_that_builds,
        test_a_device_library_on_a_cpu_only_target_is_refused,
    ):
        test(pathlib.Path(tempfile.mkdtemp()))
