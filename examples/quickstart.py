# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Walk an HPCAgent-Bench kernel through the seven NestForge stages, one Session call each, and save what each
stage made. Every stage runs its deterministic default; ``--agent`` hands stages 2 to 5 to a model instead.

``fuse_diamond`` fuses into a single kernel. ``jacobi_1d`` keeps two maps in its time loop, one reading the other's
neighbours, so it shows two kernels and the dependency carried between them."""

import argparse
import functools
import json
import shutil
from pathlib import Path

import dace

from nestforge.agent import llm, loop
from nestforge.build.sdfg import generate_program
from nestforge.corpus.bench import iter_dace_kernels, preset_sizes
from nestforge.session import Session
from nestforge.stages.canonicalize import Targets

KERNELS = {
    "fuse_diamond": "loop_level_reasoning/fuse_diamond/fuse_diamond",
    "jacobi_1d": "scientific_computing/structured_grids/jacobi_1d/jacobi_1d",
}
PRESET = "S"


def save(session: Session, out: Path, label: str) -> None:
    session.sdfg.save(str(out / f"{label}.sdfg"))


def show_tree(session: Session, out: Path, label: str) -> None:
    """Print the program's structure tree and save it as ``trees/<label>.txt``."""
    tree = session.describe()
    (out / "trees").mkdir(parents=True, exist_ok=True)
    (out / "trees" / f"{label}.txt").write_text(tree + "\n")
    print(tree)


def map_count(session: Session) -> int:
    return sum(isinstance(row[0], dace.nodes.MapEntry) for row in session.row_index().values())


def default_stages_2_to_5(session: Session, out: Path) -> None:
    before = map_count(session)
    session.default_moves()
    save(session, out, "2-moves")
    print(f"2 moves       {before} -> {map_count(session)} maps")
    show_tree(session, out, "2-moved")
    kernels = session.define_scopes()
    save(session, out, "3-scopes")
    names = ", ".join(f"{k['name']} (reads {', '.join(k['reads'])}; writes {', '.join(k['writes'])})" for k in kernels)
    print(f"3 scopes      {len(kernels)} kernel(s): {names}")
    placement = session.place()
    devices = ", ".join(f"{k['name']} on {k['device']}" for k in placement["kernels"])
    print(f"4 placement   {devices}; copies: {len(placement['copies'])}")
    for kernel in session.list_kernels():
        info = session.optimize_kernel(kernel["name"])
        print(f'5 kernel      {info["kernel"]}: extern "C" {info["entry"]} by {info["variant"]}')


def agent_stages_2_to_5(session: Session, out: Path, args: argparse.Namespace) -> None:
    client = llm.Client(args.agent, args.model, args.base_url)
    task = f"Make {session.name} as fast as possible on the {args.device} at sizes {session.sizes}."
    for line in loop.run(session, functools.partial(llm.chat, client), task, args.turns, args.analyst):
        print(f"  {line}")
    show_tree(session, out, "2-moved")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--kernel", choices=sorted(KERNELS), default="fuse_diamond")
    parser.add_argument("--out", type=Path, default=Path("quickstart_out"))
    parser.add_argument("--agent", choices=("openai", "anthropic"), help="let a model drive stages 2 to 5")
    parser.add_argument("--model", help="model name, required with --agent")
    parser.add_argument("--base-url", help="an OpenAI-compatible server (vLLM, SGLang)")
    parser.add_argument("--turns", type=int, default=4, help="model calls per stage")
    parser.add_argument("--analyst", action="store_true", help="one more model call rewrites each feedback report")
    args = parser.parse_args()
    if args.agent and not args.model:
        parser.error("--agent needs --model")
    out, work = args.out.resolve(), args.out.resolve() / "work"
    dace.Config.set("default_build_folder", value=str(work / "dacecache"))
    short_name = KERNELS[args.kernel]
    kernel = next(k for k in iter_dace_kernels(short_name.split("/", 1)[0]) if k.short_name == short_name)
    sizes = preset_sizes(kernel, PRESET)
    print(f"{kernel.short_name}, preset {PRESET} {sizes}, device {args.device}")
    targets = Targets(gpu=args.device == "gpu")
    session = Session(kernel.to_sdfg(), targets=targets, work_dir=str(work), sizes=sizes)
    show_tree(session, out, "0-input")
    session.canonicalize()
    save(session, out, "1-canonicalize")
    print(f"1 canonicalize {map_count(session)} maps")
    show_tree(session, out, "1-canonical")
    if args.agent:
        agent_stages_2_to_5(session, out, args)
    else:
        default_stages_2_to_5(session, out)
    deps = session.kernel_graph().lines()
    (out / "kernel_deps.txt").write_text("\n".join(deps) + "\n")
    print("\n".join(f"  {line}" for line in deps))
    save(session, out, "5-kernels")
    for name, build in session.builds.items():
        (out / "kernels" / name).mkdir(parents=True, exist_ok=True)
        for artifact in (build.source.unit, build.archive):
            shutil.copy2(artifact, out / "kernels" / name)
    (out / "program").mkdir(exist_ok=True)
    for source in sorted(generate_program(session.sdfg, work / "program").folder.glob("src/*/*")):
        shutil.copy2(source, out / "program" / source.name)
    configs: dict[str, dict] = {}
    for kernel_info in session.list_kernels():
        result = session.sweep(kernel_info["name"])
        if result["winner"] is None:
            raise SystemExit(f"stage 6 found no configuration of {result['kernel']} that matches its oracle")
        configs[result["kernel"]] = result["config"]
        print(f"6 variants    {result['kernel']}: {result['cells']} cells, winner {result['winner']}")
    (out / "6-variants.json").write_text(json.dumps(configs, indent=2) + "\n")
    report = session.feedback()
    (out / "7-feedback.txt").write_text(report + "\n")
    print("7 feedback\n" + "\n".join(f"  {line}" for line in report.splitlines()))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
