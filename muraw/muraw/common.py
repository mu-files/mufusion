# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files

"""Common utilities for muraw."""

from __future__ import annotations

import logging
import sys
from enum import Enum
from typing import TypeVar

_EnumT = TypeVar("_EnumT", bound=Enum)


def enum_display_name(enum_class: type[Enum], value: int, suffix: str = "") -> str:
    """
    Get display name for an enum value.

    Converts enum member name to display format (e.g., MAIN_IMAGE -> MainImage).
    Returns formatted value string if enum member not found.

    Args:
        enum_class: The enum class to look up
        value: The numeric value to find
        suffix: Optional suffix to append (e.g., " compression")

    Returns:
        Display name string (e.g., "MainImage") or "Type{value}" if not found
    """
    try:
        member = enum_class(value)
        display = "".join(word.capitalize() for word in member.name.split("_"))
        return f"{display}{suffix}" if suffix else display
    except ValueError:
        return f"Type{value}{suffix}" if suffix else f"Type{value}"


def enum_from_value(enum_class: type[_EnumT], value: int) -> _EnumT | None:
    """
    Get enum member from numeric value.

    Args:
        enum_class: The enum class to look up
        value: The numeric value to find

    Returns:
        Enum member or None if not found
    """
    try:
        return enum_class(value)
    except ValueError:
        return None


def enum_from_string(enum_class: type[_EnumT], value: str) -> _EnumT:
    """
    Get enum member from string value.

    For string enums (inheriting from str and Enum), this looks up the member
    by its string value. Raises KeyError if not found.

    Args:
        enum_class: The enum class to look up
        value: The string value to find

    Returns:
        Enum member

    Raises:
        KeyError: If value not found in enum
    """
    try:
        return enum_class(value)
    except ValueError as e:
        raise KeyError(f"'{value}' is not a valid {enum_class.__name__}") from e


def setup_logging(verbosity: int = 0) -> None:
    """
    Set up logging configuration based on verbosity level.

    Args:
        verbosity: Verbosity level (0=ERROR, 1=WARNING, 2=INFO, 3+=DEBUG)
    """
    if verbosity == 0:
        level = logging.ERROR
    elif verbosity == 1:
        level = logging.WARNING
    elif verbosity == 2:
        level = logging.INFO
    else:
        level = logging.DEBUG

    # Configure logging
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )

    # Set specific logger levels
    logger = logging.getLogger("muraw")
    logger.setLevel(level)
