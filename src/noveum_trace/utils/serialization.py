"""
Serialization utilities for Noveum Trace SDK.

This module provides utilities for safely serializing data structures
for tracing, ensuring compatibility with JSON format.
"""

import json
from typing import Any


def convert_to_json_string(value: Any) -> Any:
    """
    Convert Python dictionaries and lists to JSON strings for safe serialization.

    This function ensures that complex data structures (dicts and lists) are
    converted to JSON strings before being stored in span attributes, preventing
    serialization errors when sending traces to the Noveum platform.

    Args:
        value: Value to potentially convert

    Returns:
        JSON string if value is a dict or list, otherwise the original value

    Example:
        >>> convert_to_json_string({"key": "value"})
        '{"key": "value"}'
        >>> convert_to_json_string([1, 2, 3])
        '[1, 2, 3]'
        >>> convert_to_json_string("simple string")
        'simple string'
    """
    if isinstance(value, dict):
        try:
            return json.dumps(value, default=str)
        except (TypeError, ValueError):
            # If JSON serialization fails, fall back to string representation
            return str(value)
    elif isinstance(value, (list, tuple)):
        try:
            return json.dumps(value, default=str)
        except (TypeError, ValueError):
            # If JSON serialization fails, fall back to string representation
            return str(value)
    return value
