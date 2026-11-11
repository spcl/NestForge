# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regressions of the build layer: persistent promotion inside parallel maps, device runtime discovery, init/exit
status, entry binding, complex tolerance floors, compile timeouts, the forked child's setup and a missing frame."""

import ctypes
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import dace
import numpy as np
import pytest

import nestforge.build.sdfg as build_sdfg
from nestforge.build import flags, toolchain
from nestforge.build.arena import dtype_floor
from nestforge.build.sdfg import BuiltSDFG, generate_program_folder, make_persistent

N = dace.symbol("N")
SCRATCH = 1 << 18  # a 2 MB scratch row, above PERSISTENT_MIN_BYTES


def scratch_in_map(schedule: dace.ScheduleType) -> dace.SDFG:
    """``B[i] = A[i, 0]`` through a heap scratch array allocated inside the map, once per iteration."""
    sdfg = dace.SDFG(f"scratch_{schedule.name}")
    sdfg.add_array("A", [N, N], dace.float64)
    sdfg.add_array("B", [N], dace.float64)
    sdfg.add_transient("tmp", [N], dace.float64, storage=dace.StorageType.CPU_Heap)
    state = sdfg.add_state()
    entry, exit_ = state.add_map("m", {"i": "0:N"}, schedule=schedule)
    tmp = state.add_access("tmp")
    load = state.add_tasklet("load", {"x"}, {"y"}, "y = x")
    store = state.add_tasklet("store", {"x"}, {"y"}, "y = x")
    state.add_memlet_path(state.add_read("A"), entry, load, dst_conn="x", memlet=dace.Memlet("A[i, 0]"))
    state.add_edge(load, "y", tmp, None, dace.Memlet("tmp[0]"))
    state.add_edge(tmp, None, store, "x", dace.Memlet("tmp[0]"))
    state.add_memlet_path(store, exit_, state.add_write("B"), src_conn="y", memlet=dace.Memlet("B[i]"))
    sdfg.validate()
    return sdfg


def test_scratch_inside_a_parallel_map_stays_per_thread() -> None:
    sdfg = scratch_in_map(dace.ScheduleType.CPU_Multicore)
    assert make_persistent(sdfg, {"N": SCRATCH}) == []
    assert sdfg.arrays["tmp"].lifetime == dace.AllocationLifetime.Scope


def test_scratch_inside_a_sequential_map_becomes_persistent() -> None:
    sdfg = scratch_in_map(dace.ScheduleType.Sequential)
    assert make_persistent(sdfg, {"N": SCRATCH}) == [f"{sdfg.label}.tmp"]
    assert sdfg.arrays["tmp"].lifetime == dace.AllocationLifetime.Persistent


def fake_tool(directory: Path, name: str, body: str) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    tool = directory / name
    tool.write_text(f"#!/bin/sh\n{body}\n")
    tool.chmod(0o755)
    return str(tool)


def test_hip_runtime_found_off_the_link_line_with_a_versioned_soname(tmp_path: Path) -> None:
    runtime = tmp_path / "rocm" / "lib64"  # neither <hipcc>/../lib nor an unversioned libamdhip64.so
    runtime.mkdir(parents=True)
    (runtime / "libamdhip64.so.7").touch()
    hipcc = fake_tool(tmp_path / "bin", "hipcc", f'echo "ld -L{tmp_path}/elsewhere -L{runtime} -lamdhip64" >&2')
    assert toolchain.hip_runtime_dir(hipcc) == str(runtime.resolve())


def test_discovery_skips_an_nvcc_that_cannot_link_cudart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_tool(
        tmp_path,
        "nvcc",
        'case "$1" in --version) echo "Cuda compilation tools, release 13.0";; '
        '*) echo "ld: cannot find -lcudart" >&2; exit 1;; esac',
    )
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.warns(UserWarning, match="cannot find -lcudart"):
        assert toolchain.discover_cuda_toolchains() == []


STATUS_LIBRARY = """
#include <stddef.h>
static int handle;
static int calls;
void *__dace_init_ok(void) { return &handle; }
void *__dace_init_null(void) { return NULL; }
void __program_ok(void *h) { calls += h == &handle; }
int __dace_exit_ok(void *h) { return h == &handle ? 3 : 0; }
int program_calls(void) { return calls; }
"""


@pytest.fixture(scope="module")
def status_library(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("status")
    source, so = folder / "status.c", folder / "status.so"
    source.write_text(STATUS_LIBRARY)
    cc = shutil.which("cc") or shutil.which("gcc")
    assert cc is not None, "no C compiler on PATH"
    subprocess.run([cc, "-shared", "-fPIC", str(source), "-o", str(so)], check=True)
    return so


class CountingCDLL(ctypes.CDLL):
    """A library that counts its entry-point lookups."""

    lookups = 0

    def __getitem__(self, name: str) -> object:
        self.lookups += 1
        return super().__getitem__(name)


def built(so: Path, name: str) -> BuiltSDFG:
    return BuiltSDFG(name, so, CountingCDLL(str(so)), [], [])


def test_a_null_init_handle_raises(status_library: Path) -> None:
    with pytest.raises(RuntimeError, match="__dace_init_null failed"):
        built(status_library, "null").init({})


def test_a_failing_exit_status_raises(status_library: Path) -> None:
    program = built(status_library, "ok")
    program.init({})
    with pytest.raises(RuntimeError, match=r"__dace_exit_ok failed \(status 3\)"):
        program.close()
    assert program.handle is None, "a failed exit still drops the handle"


def test_program_entry_is_bound_once(status_library: Path) -> None:
    program = built(status_library, "ok")
    lib = program.lib
    assert isinstance(lib, CountingCDLL)
    program.init({})
    before = lib.lookups
    for _ in range(3):
        program.program({}, {})
    assert lib.lookups - before == 1, "__program_ok is looked up and typed once, not per call"
    program.bind({}, {})()
    assert lib.lookups - before == 1
    assert lib.program_calls() == 4


@pytest.mark.parametrize(("complex_dtype", "component"), [(np.complex64, "float32"), (np.complex128, "float64")])
def test_complex_outputs_take_their_component_floor(complex_dtype: type, component: str) -> None:
    assert dtype_floor({"z": np.zeros(2, dtype=complex_dtype)}) == flags.DTYPE_RTOL[component]


def test_compile_timeout_kills_the_whole_process_group(tmp_path: Path) -> None:
    pidfile = tmp_path / "grandchild"
    # the timeout must outlast the shell starting up and writing the pidfile under CPU load
    start = time.perf_counter()
    with pytest.raises(RuntimeError, match="timed out"):
        toolchain.run(["sh", "-c", f"sleep 30 & echo $! > {pidfile}; wait"], timeout=5)
    assert time.perf_counter() - start < 25, "run waited for the grandchild holding its pipe"
    grandchild = Path(f"/proc/{pidfile.read_text().strip()}/stat")
    deadline = time.perf_counter() + 10
    while grandchild.exists() and grandchild.read_text().split(")")[-1].split()[0] != "Z":
        assert time.perf_counter() < deadline, "the compiler's child outlived the timeout"
        time.sleep(0.05)


CHILD_SETUP_FAILS = """
import nestforge.build.isolation as isolation
from nestforge.build.isolation import Failure, Result, run_isolated

def refuse(cpus):
    raise OSError("pin refused")

isolation.pin_cpus = refuse
result = run_isolated(lambda: Result(), Result)
print("RESULT", isinstance(result, Failure) and result.error)
"""


def test_a_failing_child_setup_is_reported_not_returned_into_the_caller() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(CHILD_SETUP_FAILS)], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines() == ["RESULT OSError: pin refused"], "only the parent prints, with the child's error"


def test_a_missing_frame_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sdfg = scratch_in_map(dace.ScheduleType.Sequential)

    def other_frame(sdfg: dace.SDFG, code_objects: object, out: str) -> str:
        (Path(out) / "src" / "cpu").mkdir(parents=True, exist_ok=True)
        (Path(out) / "src" / "cpu" / "other.cpp").write_text("// not the program\n")
        return out

    monkeypatch.setattr(build_sdfg.dace_compiler, "generate_program_folder", other_frame)
    with pytest.raises(FileNotFoundError, match=f"{sdfg.name}.cpp"):
        generate_program_folder(sdfg, tmp_path / "out")
