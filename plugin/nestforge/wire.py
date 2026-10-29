# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Every NestForge tool: its phase, its arguments and what the model is told. Pure pydantic, so the agent image
(no DaCe, no NestForge) and the judge route read the same table."""

import enum
from typing import Literal, NamedTuple

import pydantic

__all__ = ["IDENTITY", "ROUTE", "TOOLS", "Args", "Effect", "Phase", "ToolDef"]

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


class ListMoves(Args):
    kind: str | None = pydantic.Field(None, description="Only moves of this kind.")


class Labeled(Args):
    labels: list[str] = pydantic.Field(description="Tree labels as view_tree printed them.")


class ApplyMove(Labeled):
    kind: str = pydantic.Field(description="A move kind list_moves offered.")
    tile_sizes: list[int] = pydantic.Field(
        default_factory=list,
        description="tile-doall: one tile size per dimension, e.g. [32, 32]; tile-for: one size, e.g. [64].",
    )
    implementation: str | None = pydantic.Field(
        None, description="set-library-implementation: one implementation list_moves offered for the row."
    )


class RunDefaults(Args):
    stage: Literal["moves", "finish", "scopes", "placement"] = pydantic.Field(
        description="moves: default fusion and the stages after it; finish: the stages after your own moves; "
        "scopes: the default kernels (parallel maps; sequential and mixed loops whole); placement: parallel kernels on "
        "the GPU when it is a target."
    )


class UnsetScope(Args):
    name: str = pydantic.Field(description="The kernel to inline back, as list_kernels names it.")


class KernelName(Args):
    name: str = pydantic.Field(description="A kernel list_kernels named.")


class WriteKernel(KernelName):
    source: str = pydantic.Field(
        description='The whole translation unit, with the entry read_kernel showed (same extern "C" name and '
        "parameters, in order); a GPU kernel's host and device units joined as read_kernel joined them."
    )
    language: Literal["cpp", "cuda", "hip"] = pydantic.Field(
        description="cpp for a CPU kernel; for a GPU kernel the dialect read_kernel showed (cuda or hip)."
    )


class TuneKernel(KernelName):
    compilers: list[str] | None = pydantic.Field(
        None, description="Toolchains to try (gcc, clang, nvcc, hipcc, ...); every one found for the device if omitted."
    )


class Effect(enum.StrEnum):
    """What a call does to the session, which the agent side lists as MCP annotations."""

    READ = "read"  # changes nothing
    ADD = "add"  # changes the program or a kernel; an inverse or another call undoes it
    REPLACE = "replace"  # overwrites earlier work (a kernel's source, a scope's kernel)


class ToolDef(NamedTuple):
    phase: Phase
    args: type[Args]
    description: str
    effect: Effect = Effect.ADD


TOOLS: dict[str, ToolDef] = {
    "get_cpf": ToolDef(
        Phase.STRUCTURED,
        NoArgs,
        "Start here: the program in canonical parallel form, as a labeled tree.",
        Effect.READ,
    ),
    "view_tree": ToolDef(
        Phase.STRUCTURED,
        ViewTree,
        "Expand or collapse tree rows (loops, states, doalls, library calls), then show the tree.",
        Effect.READ,
    ),
    "list_move_kinds": ToolDef(
        Phase.STRUCTURED,
        NoArgs,
        "Every move kind: the tree rows it takes, the transformation it runs, what it does.",
        Effect.READ,
    ),
    "list_moves": ToolDef(
        Phase.STRUCTURED,
        ListMoves,
        "Every legal move right now: fusion, fission, interchange, tiling, library expansion and implementation.",
        Effect.READ,
    ),
    "apply_move": ToolDef(
        Phase.STRUCTURED,
        ApplyMove,
        "Apply one move as list_moves listed it (or with your own tile_sizes); answers with the outcome and the "
        "whole-program performance. The rows it took are renamed: read the tree again before the next move.",
    ),
    "run_defaults": ToolDef(Phase.STRUCTURED, RunDefaults, "Finish one stage with NestForge's default heuristic."),
    "set_scope": ToolDef(
        Phase.SCOPES,
        Labeled,
        "Make one kernel of several top-level doalls of one state, or of a straight run of blocks; answers with the "
        "kernel's name.",
    ),
    "unset_scope": ToolDef(Phase.SCOPES, UnsetScope, "Inline a kernel back into the program.", Effect.REPLACE),
    "list_kernels": ToolDef(Phase.UNSTRUCTURED, NoArgs, "Every kernel with its device and arguments.", Effect.READ),
    "read_kernel": ToolDef(Phase.UNSTRUCTURED, KernelName, "A kernel's current source.", Effect.READ),
    "write_kernel": ToolDef(
        Phase.UNSTRUCTURED,
        WriteKernel,
        "Replace a kernel's source; it is built, checked against the kernel's oracle and linked only when it matches.",
        Effect.REPLACE,
    ),
    "tune_kernel": ToolDef(Phase.CODEGEN, TuneKernel, "Sweep compilers and flags for one kernel."),
    "get_performance": ToolDef(
        Phase.ANY,
        NoArgs,
        "Whole-program time, offloading (devices, transfers) and per-kernel times, with hints.",
        Effect.READ,
    ),
    "finalize": ToolDef(
        Phase.ANY, NoArgs, "Build the program as one library in the bench C ABI; send its library and workspace_bytes."
    ),
}
