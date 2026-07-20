"""Unit tests for the whole-program-scope plumbing (Phase 3): the whole-kernel Boundary factory and the
whole-program prepare path. These build a tiny two-nest SDFG inline (fast ``to_sdfg``) -- they do NOT run
the corpus or numpyto/gcc, so they stay light; the end-to-end compile+time is exercised by the arena on a
quiet box."""
import numpy as np

import dace

from nestforge.extract import whole_program_boundary
from nestforge.translate import prepare_regions, prepare_whole_program

N = dace.symbol("N")


@dace.program
def two_nest(a: dace.float64[N], out: dace.float64[N]):
    tmp = np.empty_like(a)
    for i in dace.map[0:N]:
        tmp[i] = a[i] * 2.0
    for i in dace.map[0:N]:
        out[i] = tmp[i] + 1.0


def _sdfg():
    sdfg = two_nest.to_sdfg(simplify=True)
    return sdfg


def test_whole_program_boundary_interface_from_read_write_sets():
    b = whole_program_boundary(_sdfg())
    # a is read, out is written; tmp is a transient (scratch) -> excluded from the caller interface.
    assert "a" in b.inputs
    assert "out" in b.outputs
    assert "tmp" not in b.inputs and "tmp" not in b.outputs
    # the size symbol is a symbol, not an array arg.
    assert "N" in b.symbols
    # a whole-program boundary has no replacement handles (it emits + compiles, never swaps a libnode).
    assert b.nsdfg_node is None and b.state is None and b.parent_sdfg is None
    # the standalone SDFG is a detached copy, not the original object.
    assert isinstance(b.standalone_sdfg, dace.SDFG)


def test_prepare_whole_program_emits_named_numpy_and_manifest(tmp_path):
    prep = prepare_whole_program(_sdfg(), "two_nest", tmp_path, sizes={"N": 64})
    # the whole-program numpy source is a single function over the un-split program.
    assert "def two_nest(" in prep.numpy_source
    assert prep.numpy_path.exists() and prep.yaml_path.exists()
    # the manifest names the kernel, its arrays, and the written output.
    assert prep.manifest["func_name"] == "two_nest"
    assert "a" in prep.manifest["array_args"] and "out" in prep.manifest["array_args"]
    assert prep.manifest["output_args"] == ["out"]  # only `out` is written; `a` is read-only, `tmp` scratch


def test_whole_program_boundary_detaches_no_parent_links():
    sdfg = _sdfg()
    b = whole_program_boundary(sdfg)
    # detached copy must not alias the source object (mutations in emit must not touch the original).
    assert b.standalone_sdfg is not sdfg
    assert b.standalone_sdfg.parent is None and b.standalone_sdfg.parent_sdfg is None


def test_prepare_regions_pure_program_is_one_region(tmp_path):
    # a program with no unsupported node is a single externalizable region == the whole program.
    prepared, islands = prepare_regions(_sdfg(), "two_nest", tmp_path, sizes={"N": 64})
    assert islands == []
    assert len(prepared) == 1
    assert "def two_nest(" in prepared[0].numpy_source
    assert prepared[0].manifest["output_args"] == ["out"]
