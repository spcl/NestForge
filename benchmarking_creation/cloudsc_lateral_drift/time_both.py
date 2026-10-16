"""Time the baseline and the reference of cloudsc_lateral_drift on one GPU at preset M.

Baseline is stage 1 plus default_moves(). Reference is two interchange-loop-map moves, then finish_moves().
Each variant is checked against a vectorized NumPy of the same loops, then timed for 5 calls.
Host-loop symbols that ExternalCall does not see are declared on the parent SDFG, as in the 2026-10-09 run.
"""

import copy
import importlib.util
import statistics
import sys
import time
import traceback
from pathlib import Path

import dace
import numpy as np
from dace.libraries.standard.nodes.external_call import ExternalCall

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from nestforge.build.sdfg import compile_linked_program  # noqa: E402
from nestforge.corpus.bench import iter_dace_kernels, preset_sizes  # noqa: E402
from nestforge.session import Session  # noqa: E402
from nestforge.stages.canonicalize import Targets  # noqa: E402

KERNEL = REPO / "benchmarking_creation" / "cloudsc_lateral_drift"
PRESET = "M"
REPS = 5
ROOT = Path("out/time-both")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


numpy_mod = load_module(KERNEL / "cloudsc_lateral_drift_numpy.py", "cloudsc_lateral_drift_numpy")
init_mod = load_module(KERNEL / "cloudsc_lateral_drift.py", "cloudsc_lateral_drift_init")

PTSPHY = numpy_mod.PTSPHY
RG = numpy_mod.RG
RD = numpy_mod.RD
VQR = numpy_mod.VQR
CFL = numpy_mod.CFL
ARG_NAMES = [
    "pap", "paph", "ztp1", "zqr", "pfsnow", "pfplsx", "flx", "outE", "outW", "out", "cap", "east",
]


def inputs(sizes: dict) -> dict:
    values = init_mod.initialize(sizes["NLEV"], sizes["KLON"], sizes["NCLDTOP"])
    return dict(zip(ARG_NAMES, values))


def reference_out(inp: dict, sizes: dict) -> np.ndarray:
    """Same loops as the oracle, vectorized on the free axis. This is the check used on 2026-10-09."""
    nlev, klon, top = sizes["NLEV"], sizes["KLON"], sizes["NCLDTOP"]
    paph, pap, ztp1, zqr = inp["paph"], inp["pap"], inp["ztp1"], inp["zqr"]
    pf = inp["pfplsx"].copy()
    for jk in range(top, nlev):
        zdtgdp = PTSPHY * RG / (paph[jk + 1] - paph[jk])
        zsink = zdtgdp * VQR * pap[jk] / (RD * ztp1[jk])
        zqn = (zqr[jk] + pf[jk] * zdtgdp) / (1.0 + zsink)
        pf[jk + 1] = zsink * zqn / zdtgdp
    flx = pf + inp["pfsnow"]
    cap = inp["cap"]
    oe, ow = flx.copy(), flx.copy()
    for j in range(1, klon):
        oe[:, j] = (flx[:, j] + np.minimum(CFL * oe[:, j - 1], cap[:, j])) / (1.0 + CFL)
    for j in range(klon - 2, -1, -1):
        ow[:, j] = (flx[:, j] + np.minimum(CFL * ow[:, j + 1], cap[:, j])) / (1.0 + CFL)
    east = inp["east"][:, None].astype(np.float64)
    return east * oe + (1.0 - east) * ow


def declare_symbols(sdfg: dace.SDFG) -> list[str]:
    added: list[str] = []
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, ExternalCall):
            conns = set(node.in_connectors) | set(node.out_connectors)
            for arg in node.abi_order:
                if arg in conns or f"_in_{arg}" in conns or f"_out_{arg}" in conns:
                    continue
                if arg not in sdfg.symbols and arg not in sdfg.arrays:
                    sdfg.add_symbol(arg, dace.int64)
                    added.append(arg)
    return added


def kernel():
    short = "benchmarking_creation/cloudsc_lateral_drift/cloudsc_lateral_drift"
    return next(item for item in iter_dace_kernels("benchmarking_creation") if item.short_name == short)


def build(variant: str, sizes: dict) -> tuple[Session, str]:
    work = (ROOT / variant).resolve()
    work.mkdir(parents=True, exist_ok=True)
    dace.Config.set("default_build_folder", value=str(work / "dacecache"))
    session = Session(
        kernel().to_sdfg(),
        targets=Targets(gpu=True),
        name=f"lateral_{variant}",
        work_dir=str(work),
        sizes=sizes,
    )
    session.canonicalize()
    print(f"[{variant}] stage 1\n{session.describe()}", flush=True)
    print(f"[{variant}] offered {session.list_moves()}", flush=True)
    notes: list[str] = []
    if variant == "baseline":
        session.default_moves()
    else:
        for step in range(2):
            moves = session.list_moves("interchange-loop-map")
            if not moves:
                raise RuntimeError(f"step {step}: interchange-loop-map is not offered")
            result = session.apply_move("interchange-loop-map", moves[0]["labels"], moves[0]["epoch"])
            notes.append(f"{result.status}:{list(result.labels)}")
            if result.status != "applied":
                raise RuntimeError(f"{notes} reason={result.reason}")
        session.finish_moves()
    print(f"[{variant}] moves {notes}\n{session.describe()}", flush=True)
    session.define_scopes()
    added = declare_symbols(session.sdfg)
    session.place()
    added += declare_symbols(session.sdfg)
    for item in session.list_kernels():
        info = session.optimize_kernel(item["name"])
        print(f"[{variant}] kernel {info}", flush=True)
    print(f"[{variant}] declared symbols {added}", flush=True)
    return session, str(work)


def run(variant: str, sizes: dict, inp: dict, ref: np.ndarray) -> None:
    started = time.time()
    stage = "build"
    try:
        session, work = build(variant, sizes)
        stage = "compile"
        sdfg = copy.deepcopy(session.sdfg)
        sdfg.expand_library_nodes()
        compiled = compile_linked_program(sdfg, Path(work) / "parent")
        stage = "check"
        arrays = {key: value.copy() for key, value in inp.items()}
        compiled(**arrays, **sizes)
        error = float(np.max(np.abs(arrays["out"] - ref) / (np.abs(ref) + 1e-30)))
        times = []
        stage = "time"
        for _ in range(REPS):
            tick = time.perf_counter()
            compiled(**arrays, **sizes)
            times.append(time.perf_counter() - tick)
        millis = [1e3 * item for item in times]
        print(
            f"RESULT {variant}: median {statistics.median(millis):.2f} ms  "
            f"min {min(millis):.2f}  max {max(millis):.2f}  "
            f"rel err {error:.2e}  ({time.time() - started:.0f}s)",
            flush=True,
        )
    except BaseException as err:
        traceback.print_exc()
        print(
            f"RESULT {variant}: FAIL at {stage}: {type(err).__name__}: {str(err)[:400]} "
            f"({time.time() - started:.0f}s)",
            flush=True,
        )


def main() -> None:
    sizes = preset_sizes(kernel(), PRESET)
    print(f"preset {PRESET} {sizes}", flush=True)
    inp = inputs(sizes)
    started = time.time()
    ref = reference_out(inp, sizes)
    print(f"numpy reference: {time.time() - started:.1f}s", flush=True)
    for variant in ("baseline", "reference"):
        run(variant, sizes, inp, ref)


if __name__ == "__main__":
    main()
