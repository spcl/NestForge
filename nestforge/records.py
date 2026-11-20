# Copyright 2021 ETH Zurich and the NestForge authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The record every NestForge answer is: a frozen pydantic model, and the outcome enums its ``status`` takes. An
enum serializes as its lowercase name (``NOT_FOUND`` as ``not-found``), so JSON readers see words, not numbers."""

import enum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, PlainSerializer


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


def spelled(member: enum.Enum) -> str:
    return member.name.lower().replace("_", "-")


class MoveStatus(enum.Enum):
    """What a labeled call did; only ``APPLIED`` changed the program."""

    APPLIED = enum.auto()
    ILLEGAL = enum.auto()
    NOT_FOUND = enum.auto()


class KernelStatus(enum.Enum):
    """What a kernel source did. ``OK`` matched the oracle and ``UNCHECKED`` had no oracle to match: only these two
    changed the kernel's library."""

    OK = enum.auto()
    UNCHECKED = enum.auto()
    REFUSED = enum.auto()
    BUILD_FAILED = enum.auto()
    WRONG = enum.auto()


class Refusal(enum.Enum):
    """Why a labeled call changed nothing, as a class the plugin record counts; the text says the rest."""

    #: No row holds the label, or a change retired it.
    UNKNOWN_LABEL = enum.auto()
    #: The rows are not of the kinds, or not in the places, the move takes.
    WRONG_ROW_KIND = enum.auto()
    #: The packet switched the move's family off.
    FAMILY_OFF = enum.auto()
    #: DaCe's legality check, or its dependence analysis, said no.
    DEPENDENCE = enum.auto()
    #: A crossed block touches what the row reads or writes.
    EFFECTS = enum.auto()
    #: A rule or an invariant of communicating programs.
    COMMUNICATION = enum.auto()
    #: A bad tile size, cap, implementation, value, layout, or scope.
    ARGUMENT = enum.auto()
    #: Not classified; the test suite refuses none of these.
    OTHER = enum.auto()


class Refused(str):
    """A refusal's text with its class: a ``str``, so every caller of a planner's ``Rewrite | str`` reads the text as
    before. A ``str`` subtype takes no ``__slots__``, so the class sits in the instance dict."""

    refusal: Refusal

    def __new__(cls, text: str, refusal: Refusal) -> Self:
        made = super().__new__(cls, text)
        made.refusal = refusal
        return made

    def __reduce__(self) -> tuple[type[Self], tuple[str, Refusal]]:  # pickling passes the class too
        return type(self), (str(self), self.refusal)


class UnclassedRefusal(UserWarning):
    """A refusal of class :attr:`Refusal.OTHER`. The test suite turns it into an error, so every refusal a test meets
    carries its class."""


def refusal_of(reason: str) -> Refusal:
    """The class of ``reason``: its own for a :class:`Refused`, else :attr:`Refusal.OTHER`."""
    return reason.refusal if isinstance(reason, Refused) else Refusal.OTHER


MoveOutcome = Annotated[MoveStatus, PlainSerializer(spelled, return_type=str)]
KernelOutcomeStatus = Annotated[KernelStatus, PlainSerializer(spelled, return_type=str)]
RefusalOutcome = Annotated[Refusal, PlainSerializer(spelled, return_type=str)]


#: What a failing transformation, build or run may raise without ending the session: a library's ``sys.exit`` too.
CRASHES = (Exception, SystemExit)
