# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage 2 by the static score against stage 2 by default, on HPCAgent-Bench kernels.

For every kernel, four programs leave stage 2: ``stage1`` (no move), ``default`` (canonicalization's fusion stage and
the passes after it, ``Session.default_moves``), ``guided`` (``Session.guided_moves``: apply the legal move with the
lowest score change while it is negative, nothing is built to choose; then ``Session.finish_moves``) and
``default+guided`` (the default, then the score's moves on top). Each then takes the stage 3 to 5 defaults, every kernel is checked
against the NumPy oracle and timed, and the program time is the sum of kernel time times calls. Prints one table and
writes ``<out>/score_demo.json``. ``--static`` stops after stage 2 (no build): a cheap probe of what each
pipeline chooses.

Run it alone on a node: the timings are serial. OpenMP kernels use the threads of ``OMP_NUM_THREADS``."""

import argparse
import json
import statistics
import time
import traceback
from pathlib import Path

import dace

from nestforge.build.arena import TIMED_REPS
from nestforge.corpus.bench import iter_dace_kernels, preset_sizes
from nestforge.ir.extract import find_state_of_node
from nestforge.session import Session
from nestforge.stages.canonicalize import Targets
from nestforge.stages.kernel import validate_kernel
from nestforge.stages.score import enclosing_trips, outer_iterations

#: Kernels the score was fitted and tested on (in sample) and kernels it never saw (new).
KERNELS = dict.fromkeys(
    (
        "tsvc_2_s231",
        "tsvc_2_s233",
        "tsvc_2_s235",
        "tsvc_2_s2233",
        "tsvc_2_s256",
        "column_scan_nproma_nlev",
        "column_scan_nlev_nproma",
    ),
    ("loop_level_reasoning", "in sample"),
)
PIPELINES = ("stage1", "default", "guided", "default+guided")
#: Independent measurements per kernel; the median is kept.
REPEATS = 3


def find_kernel(name: str):
    for track in ("loop_level_reasoning", "scientific_computing", "machine_learning"):
        for kernel in iter_dace_kernels(track):
            if kernel.short_name.rsplit("/", 1)[-1] == name:
                return kernel
    raise LookupError(f"no HPCAgent-Bench kernel named {name}")


def stage2(session: Session, pipeline: str) -> dict:
    t = time.perf_counter()
    moves: list[dict] = []
    if pipeline in ("default", "default+guided"):
        session.default_moves()
    if pipeline in ("guided", "default+guided"):
        moves = [{"kind": m.kind, "labels": list(m.labels), "d_score": m.d_score} for m in session.guided_moves()]
    if pipeline == "guided":
        session.finish_moves()
    return {"moves": moves, "choose_s": time.perf_counter() - t, **session.score_terms()}


def kernel_calls(session: Session, name: str) -> float:
    state = find_state_of_node(session.sdfg, session.kernel(name))
    sizes = {k: float(v) for k, v in session.sizes.items()}
    trips, _ = enclosing_trips(state, sizes)
    outer, _ = outer_iterations(state.sdfg, sizes)
    return trips * outer


def time_kernels(session: Session) -> list[dict]:
    session.default_scopes()
    session.place()
    out = []
    for info in session.list_kernels():
        name = info.name
        session.optimize_kernel(name)
        build = session.builds[name]
        prep, sizes = session.prepare_kernel(name), session.kernel_sizes(name)
        runs = [
            validate_kernel(build.archive, build.source, prep, sizes, TIMED_REPS, build.variant.fp_mode)
            for _ in range(REPEATS)
        ]
        out.append(
            {
                "name": name,
                "calls": kernel_calls(session, name),
                "time_us": statistics.median(r.time_us for r in runs),
                "spread_us": [min(r.time_us for r in runs), max(r.time_us for r in runs)],
                "ok": all(r.ok for r in runs),
            }
        )
    return out


def run(name: str, preset: str, gpu: bool, pipeline: str, out: Path, static: bool) -> dict:
    kernel = find_kernel(name)
    work = out / "work" / f"{name}.{pipeline}"
    dace.Config.set("default_build_folder", value=str(work / "dacecache"))
    session = Session(
        kernel.to_sdfg(), targets=Targets(gpu=gpu), work_dir=str(work), sizes=preset_sizes(kernel, preset)
    )
    session.canonicalize()
    record = {"kernel": name, "pipeline": pipeline, "preset": preset, **stage2(session, pipeline)}
    if static:
        return record
    try:
        kernels = time_kernels(session)
        record.update(kernels=kernels, ok=all(k["ok"] for k in kernels))
        record["total_us"] = sum(k["time_us"] * k["calls"] for k in kernels)
    except Exception as err:  # keep the static part of a program whose build or check fails
        record.update(ok=False, error=f"{type(err).__name__}: {err}", trace=traceback.format_exc()[-1500:])
    return record


def table(records: list[dict]) -> str:
    lines = [
        f"{'kernel':28s} {'set':10s} {'pipeline':14s} {'moves':>5s} {'E':>10s} {'lines':>10s} {'ms':>10s} vs default"
    ]
    by = {(r["kernel"], r["pipeline"]): r for r in records}
    for r in records:
        base = by.get((r["kernel"], "default"), {}).get("total_us")
        ms = r.get("total_us")
        ratio = f"{ms / base:6.2f}x" if ms and base else "   n/a"
        shown = f"{ms / 1e3:10.3f}" if ms else f"{'fail':>10s}"
        lines.append(
            f"{r['kernel']:28s} {KERNELS.get(r['kernel'], ('', 'new'))[1]:10s} {r['pipeline']:14s} "
            f"{len(r['moves']):5d} {r['E']:10.3g} {r['lines']:10.3g} {shown} {ratio}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "kernels", nargs="*", default=["tsvc_2_s231", "column_scan_nproma_nlev"], help="HPCAgent-Bench names"
    )
    parser.add_argument("--preset", default="M")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--out", default="score_demo_out")
    parser.add_argument("--static", action="store_true", help="stop after stage 2, build nothing")
    parser.add_argument("--pipelines", default=",".join(PIPELINES), help="comma list of " + ", ".join(PIPELINES))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for name in args.kernels:
        for pipeline in args.pipelines.split(","):
            try:
                record = run(name, args.preset, args.device == "gpu", pipeline, out, args.static)
            except Exception as err:  # a kernel that does not lower or canonicalize: report it and go on
                record = {
                    "kernel": name,
                    "pipeline": pipeline,
                    "moves": [],
                    "E": float("nan"),
                    "lines": float("nan"),
                    "ok": False,
                    "error": f"{type(err).__name__}: {err}",
                }
            records.append(record)
            print(json.dumps({k: v for k, v in record.items() if k not in ("kernels", "trace")}), flush=True)
            (out / "score_demo.json").write_text(json.dumps(records, indent=2, default=str))
    print(table(records))


if __name__ == "__main__":
    main()
