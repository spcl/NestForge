# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The record every NestForge answer is: a frozen pydantic model, and the outcome enums its ``status`` takes. An
enum serializes as its lowercase name (``NOT_FOUND`` as ``not-found``), so JSON readers see words, not numbers."""

import enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, PlainSerializer


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


def spelled(member: enum.Enum) -> str:
    return member.name.lower().replace("_", "-")


class MoveStatus(enum.Enum):
    """What a labeled call did; only ``APPLIED`` changed the program."""

    APPLIED = enum.auto()
    ILLEGAL = enum.auto()
    STALE = enum.auto()
    NOT_FOUND = enum.auto()


class KernelStatus(enum.Enum):
    """What a kernel source did; only ``OK`` changed the kernel's library."""

    OK = enum.auto()
    REFUSED = enum.auto()
    BUILD_FAILED = enum.auto()
    WRONG = enum.auto()


MoveOutcome = Annotated[MoveStatus, PlainSerializer(spelled, return_type=str)]
KernelOutcomeStatus = Annotated[KernelStatus, PlainSerializer(spelled, return_type=str)]
