"""Compatibility helpers for test environments older than the package target."""

from enum import Enum


class StrEnum(str, Enum):
    """Small `enum.StrEnum` compatible base for Python 3.10 test runners."""

    def __str__(self) -> str:
        return self.value
