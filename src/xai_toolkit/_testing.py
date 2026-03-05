"""Test utilities — helpers for deterministic assertions.

This module provides shared utilities used across the test suite.
It is NOT part of the public API (underscore prefix convention).

The key problem these utilities solve: ToolMetadata.timestamp is the
only non-deterministic field in the entire response schema. These helpers
strip volatile fields so that snapshot tests and equality assertions
work deterministically.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def strip_volatile_fields(response: dict[str, Any]) -> dict[str, Any]:
    """Remove non-deterministic fields from a ToolResponse dict.

    Currently strips: metadata.timestamp

    This enables deterministic comparison of full ToolResponse objects
    in snapshot tests or equality assertions, without the timestamp
    breaking reproducibility.

    Args:
        response: A ToolResponse.model_dump() dict.

    Returns:
        A deep copy with volatile fields removed.

    Example:
        >>> r1 = tool_response.model_dump()
        >>> r2 = tool_response.model_dump()  # different timestamp
        >>> strip_volatile_fields(r1) == strip_volatile_fields(r2)
        True
    """
    result = deepcopy(response)
    if "metadata" in result and isinstance(result["metadata"], dict):
        result["metadata"].pop("timestamp", None)
    return result
