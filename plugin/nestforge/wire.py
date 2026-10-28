# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Every NestForge tool: its phase, its arguments and what the model is told. Pure pydantic, so the agent image
(no DaCe, no NestForge) and the judge route read the same table."""

import enum
from typing import Literal, NamedTuple

import pydantic

__all__ = ["IDENTITY", "ROUTE", "TOOLS", "Args", "Phase", "ToolDef"]

#: The judge route every tool posts to, as ``/nestforge/<tool>``.
ROUTE = "nestforge"

#: Body fields the transport adds to every request; never a tool argument.
IDENTITY = frozenset({"kernel", "rank", "episode_id", "optimizer", "tokens"})


class Phase(enum.StrEnum):
    STRUCTURED = "structured"
    SCOPES = "scopes"
    UNSTRUCTURED = "unstructured"
    CODEGEN = "codegen"
    ANY = "any"


class Args(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)


class NoArgs(Args):
    pass


class ViewTree(Args):
    expand: list[str] = pydantic.Field(default_factory=list, description="State or map labels to open.")
    collapse: list[str] = pydantic.Field(default_factory=list, description="Labels to close again.")
    epoch: int | None = pydantic.Field(None, description="The epoch the labels were read at; needed with expand.")


class ListMoves(Args):
    kind: str | None = pydantic.Field(None, description="Only moves of this kind.")


class Labeled(Args):
    labels: list[str] = pydantic.Field(description="Tree labels as view_tree printed them.")
    epoch: int = pydantic.Field(description="The epoch those labels were read at.")


class ApplyMove(Labeled):
    kind: str = pydantic.Field(description="A move kind list_moves offered.")


class RunDefaults(Args):
    stage: Literal["moves", "finish", "scopes", "placement"] = pydantic.Field(
        description="moves: default fusion and the stages after it; finish: the stages after your own moves; "
        "scopes: the default kernels (parallel maps; sequential and mixed loops whole); placement: parallel kernels on "
        "the GPU when it is a target."
    )


class UnsetScope(Args):
    name: str = pydantic.Field(description="The kernel to inline back.")
    epoch: int


class KernelName(Args):
    name: str = pydantic.Field(description="A kernel list_kernels named.")


class WriteKernel(KernelName):
    source: str
    language: Literal["cpp", "cuda", "hip"] = pydantic.Field(
        description="cpp for a CPU kernel; for a GPU kernel the dialect read_kernel showed (cuda or hip)."
    )


class TuneKernel(KernelName):
    compilers: list[str] | None = None


class ToolDef(NamedTuple):
    phase: Phase
    args: type[Args]
    description: str


TOOLS: dict[str, ToolDef] = {
    "get_cpf": ToolDef(Phase.STRUCTURED, NoArgs, "The program in canonical parallel form, as a labeled tree."),
    "view_tree": ToolDef(
        Phase.STRUCTURED,
        ViewTree,
        "Expand or collapse tree rows (loops, states, maps, library calls), then show the tree.",
    ),
    "list_move_kinds": ToolDef(
        Phase.STRUCTURED, NoArgs, "Every move kind: the tree rows it takes, the transformation it runs, what it does."
    ),
    "list_moves": ToolDef(
        Phase.STRUCTURED, ListMoves, "Every legal fusion, fission, interchange and library expansion right now."
    ),
    "apply_move": ToolDef(Phase.STRUCTURED, ApplyMove, "Apply one legal move; returns the new tree and performance."),
    "run_defaults": ToolDef(Phase.STRUCTURED, RunDefaults, "Finish one stage with NestForge's default heuristic."),
    "set_scope": ToolDef(Phase.SCOPES, Labeled, "Make one kernel of the regions the labels name."),
    "unset_scope": ToolDef(Phase.SCOPES, UnsetScope, "Inline a kernel back into the program."),
    "list_kernels": ToolDef(Phase.UNSTRUCTURED, NoArgs, "Every kernel with its device and arguments."),
    "read_kernel": ToolDef(Phase.UNSTRUCTURED, KernelName, "A kernel's current source."),
    "write_kernel": ToolDef(Phase.UNSTRUCTURED, WriteKernel, "Replace a kernel's source; it is built and checked."),
    "tune_kernel": ToolDef(Phase.CODEGEN, TuneKernel, "Sweep compilers and flags for one kernel."),
    "get_performance": ToolDef(
        Phase.ANY, NoArgs, "Whole-program time, offloading (devices, transfers) and per-kernel times, with hints."
    ),
    "finalize": ToolDef(
        Phase.ANY, NoArgs, "Build the program as one library in the bench C ABI; send its library and workspace_bytes."
    ),
}
