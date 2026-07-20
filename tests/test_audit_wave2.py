"""Second audit wave: the remaining findings from the full-repo review. Unit set, no compile.

Covers the contract/robustness bugs that silently mislead rather than crash -- ``can_fuse`` disagreeing
with ``enumerate_fusions``, a read view mutating session state, a documented FP level crashing flag
composition, an auto-par lane reporting parallel without verifying, a runtime linked without an rpath,
a rank>=2 size-1 buffer indexed as a sub-array, and extern-call ABI args with no connector.
"""
import re
from pathlib import Path

import numpy as np
import pytest
import dace

import nestforge

from nestforge.build import LIBOMP, OpenMPRuntime
from nestforge.emit_libnode import scalar_elem
from nestforge.fusion_arms import can_fuse, enumerate_fusions
from nestforge.libnode import ExternLibEnv, ExternalCall, proto_and_call
from nestforge.perf import flags
from nestforge.session import Session
from nestforge.strategies import top_level_map_entries

N = dace.symbol('N')


@dace.program
def live_and_transient(A: dace.float64[N], B: dace.float64[N], live_out: dace.float64[N], C: dace.float64[N]):
    T = np.empty_like(A)  # transient intermediate
    for i in dace.map[0:N]:
        T[i] = A[i] + B[i]
        live_out[i] = A[i] * 3.0  # a NON-transient result of the same producer map
    for i in dace.map[0:N]:
        C[i] = T[i] * 2.0 + live_out[i]  # consumer reads BOTH intermediates


def map_pairs(sdfg):
    for state in sdfg.all_states():
        entries = top_level_map_entries(state)
        for first in entries:
            for second in entries:
                if first is not second:
                    yield first, second


def test_can_fuse_agrees_with_enumerate_fusions():
    # THE contract: can_fuse == "yes" exactly when an applicable move exists. vertical_reason used to return
    # on the FIRST intermediate, so a live (non-transient) output could mask a legal move that
    # enumerate_fusions still offered via the transient -- the agent told "cannot fuse" about a listed move.
    sdfg = live_and_transient.to_sdfg(simplify=True)
    listed = enumerate_fusions(sdfg)
    for first, second in map_pairs(sdfg):
        verdict = can_fuse(sdfg, first, second)
        if verdict == "yes":
            assert listed, "can_fuse said yes but enumerate_fusions offered nothing"
        else:
            assert isinstance(verdict, str) and verdict  # always an explaining reason, never a bare False


def test_live_output_does_not_mask_a_transient_fusion():
    # the specific shape: if any move is offered for the producer/consumer pair, can_fuse must not report the
    # live output as the blocker.
    sdfg = live_and_transient.to_sdfg(simplify=True)
    assert enumerate_fusions(sdfg), "fixture must produce a fusable pair, else it tests nothing"
    verdicts = [can_fuse(sdfg, a, b) for a, b in map_pairs(sdfg)]
    assert any(v == "yes" for v in verdicts), f"a move is offered but no pair says yes: {verdicts}"


def test_region_tree_is_a_read_view_and_mints_nothing():
    # region_tree is documented read-only, but minted 'region' handles into self.handles -- ids that no
    # method resolves, growing the registry on every inspection call.
    session = Session(live_and_transient.to_sdfg(simplify=True))
    before = dict(session.handles)
    tree = session.region_tree()
    assert session.handles == before  # a read view mutates nothing
    session.region_tree()
    assert session.handles == before  # and stays stable when called repeatedly
    assert tree["id"].startswith("region:")  # descriptive, not a handle pretending to be resolvable


def test_lane_flags_accepts_every_documented_fp_level():
    # ExternalOptimizer/Proposal document fp_mode as FP_LEVELS, but anything except 'strict-ieee' fell into
    # the REDUCED table and raised KeyError during optimizer construction.
    for level in flags.FP_LEVELS:
        out, reason = flags.lane_flags("gnu", level, "none", "sequential", "c", 1)
        assert out is not None, f"{level} declined: {reason}"


def test_lane_flags_declines_an_unknown_fp_mode():
    out, reason = flags.lane_flags("gnu", "not-a-mode", "none", "sequential", "c", 1)
    assert out is None and "unknown fp_mode" in reason  # declines like any unsupported axis, never KeyError


def test_autopar_declines_when_the_backend_is_absent():
    # nvidia/intel used to return their flags UNPROBED, so a compiler that accepts the flag but emits no
    # parallel loop was still labelled 'auto-par'. Every family now goes through the same probes.
    for family in ("gnu", "llvm", "nvidia", "intel"):
        out, reason = flags.autopar_flags(family, 4, compiler="/nonexistent/cc")
        assert out is None and reason, f"{family} claimed auto-par with no working compiler"


def test_autopar_pure_composition_still_works_without_a_compiler():
    for family in ("gnu", "llvm", "nvidia", "intel"):
        out, reason = flags.autopar_flags(family, 4, compiler=None)
        assert out and reason is None  # unprobed composition (for tests/figures) is unchanged


def test_openmp_link_flags_carry_an_rpath_for_a_pinned_dir():
    # -L satisfies the LINKER only; without -rpath the built .so has DT_NEEDED and no RUNPATH, so the
    # ctypes.CDLL right after the build fails to find libomp.
    runtime = OpenMPRuntime(name=LIBOMP.name, soname=LIBOMP.soname, lib_dir="/opt/llvm/lib")
    linked = runtime.link_flags("clang")
    assert "-L/opt/llvm/lib" in linked
    assert "-Wl,-rpath,/opt/llvm/lib" in linked


def test_openmp_link_flags_omit_libdir_when_not_pinned():
    runtime = OpenMPRuntime(name=LIBOMP.name, soname=LIBOMP.soname, lib_dir="")
    assert not [f for f in runtime.link_flags("clang") if f.startswith(("-L", "-Wl,-rpath"))]


def desc_of_shape(shape):
    sdfg = dace.SDFG("d")
    sdfg.add_array("s", shape, dace.float64)
    return sdfg.arrays["s"]


def test_scalar_elem_indexes_every_dimension():
    # is_scalar is rank-agnostic (total_size == 1), so a keepdims (1,1) buffer landed here too; name[0]
    # selects a shape-(1,) SUB-ARRAY, not the element.
    assert scalar_elem("s", desc_of_shape([1])) == "s[0]"
    assert scalar_elem("s", desc_of_shape([1, 1])) == "s[0, 0]"
    assert scalar_elem("s", desc_of_shape([1, 1, 1])) == "s[0, 0, 0]"


def extern_call(abi_order, inputs):
    manifest = {
        "array_args": list(abi_order),
        "output_args": [],
        "init": {
            "arrays": {
                a: {
                    "dtype": "float64"
                }
                for a in abi_order
            },
            "scalars": {}
        },
    }
    node = ExternalCall("k", inputs=set(inputs), outputs=set(), config=manifest)
    node.symbol, node.abi_order = "k_fp64", list(abi_order)
    return node


def test_abi_arg_without_a_connector_is_refused():
    # the compiled signature exposes a caller-allocated scratch buffer that never crosses the ExternalCall
    # boundary; emitting the call anyway referenced an undefined identifier in the tasklet.
    with pytest.raises(ValueError, match="no '_in_scratch' connector"):
        proto_and_call(extern_call(["A", "scratch"], inputs=["_in_A"]))


def test_extern_lib_env_accumulates_every_nest_library():
    # one shared environment class: assigning (not appending) kept only the LAST expanded nest's library, so
    # every earlier nest's extern-C symbol was unresolved at link.
    ExternLibEnv.reset()
    assert ExternLibEnv.cmake_libraries == []
    ExternLibEnv.configure("/tmp/libone_nest.so")
    ExternLibEnv.configure("/tmp/libtwo_nest.so")
    assert ExternLibEnv.cmake_libraries == ["/tmp/libone_nest.so", "/tmp/libtwo_nest.so"]
    ExternLibEnv.configure("/tmp/libone_nest.so")  # deduplicated
    assert len(ExternLibEnv.cmake_libraries) == 2
    ExternLibEnv.reset()
    assert ExternLibEnv.cmake_libraries == [] and ExternLibEnv.cmake_link_flags == []


def test_every_link_search_path_is_paired_with_an_rpath():
    """Repo-wide invariant: any ``-L<dir>`` we emit is accompanied by ``-Wl,-rpath,<dir>``.

    ``-L`` satisfies the LINKER only. Without the matching rpath the built artifact needs
    LD_LIBRARY_PATH to run, which broke the OpenMP lane (the ctypes.CDLL right after the build could not
    find libomp). Pinning it here keeps every future link site honest -- a built .so must just run.
    """
    package = Path(nestforge.__file__).parent
    offenders = []
    for path in sorted(package.rglob("*.py")):
        lines = path.read_text().splitlines()
        for num, line in enumerate(lines):
            if not re.search(r"""["']-L""", line):
                continue
            window = " ".join(lines[num:num + 2])  # the flag list may wrap onto the next line
            if "rpath" not in window:
                offenders.append(f"{path.name}:{num + 1}: {line.strip()}")
    assert not offenders, "a -L without a paired -Wl,-rpath:\n" + "\n".join(offenders)
