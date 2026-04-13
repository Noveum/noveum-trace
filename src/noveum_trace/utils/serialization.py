"""
Serialization utilities for Noveum Trace SDK.

This module provides utilities for safely serializing data structures
for tracing, ensuring compatibility with JSON format.
"""

import json
import warnings
from typing import Any, Optional

# Depth/size limits for attribute serialization (used by context managers, etc.)
DEFAULT_MAX_DEPTH = 10
DEFAULT_MAX_SIZE_BYTES = 1048576  # 1MB
WARNING_STACKLEVEL = 4


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


def _serialize_value(
    value: Any,
    max_depth: int = DEFAULT_MAX_DEPTH,
    current_depth: int = 0,
    _visited: Optional[set[int]] = None,
) -> Any:
    """
    Safely serialize a value for tracing, returning JSON-serializable objects.

    Preserves structure for dicts and lists, extracts meaningful data from
    complex objects, and warns on very large serialized payloads.
    """
    if _visited is None:
        _visited = set()

    if current_depth >= max_depth:
        return f"<max_depth_reached:{type(value).__name__}>"

    try:
        if value is None:
            return None

        if isinstance(value, (int, float, bool)):
            return value

        if isinstance(value, str):
            _check_serialized_size(value, value)
            return value

        if isinstance(value, dict):
            dict_result: dict[str, Any] = {}
            for key, val in value.items():
                str_key = str(key) if not isinstance(key, str) else key
                dict_result[str_key] = _serialize_value(
                    val,
                    max_depth=max_depth,
                    current_depth=current_depth + 1,
                    _visited=_visited,
                )
            _check_serialized_size(dict_result, value)
            return dict_result

        if isinstance(value, (list, tuple)):
            list_result: list[Any] = [
                _serialize_value(
                    item,
                    max_depth=max_depth,
                    current_depth=current_depth + 1,
                    _visited=_visited,
                )
                for item in value
            ]
            _check_serialized_size(list_result, value)
            return list_result

        if hasattr(value, "to_dict") and callable(value.to_dict):
            try:
                dict_value = value.to_dict()
                serialized_result: Any = _serialize_value(
                    dict_value,
                    max_depth=max_depth,
                    current_depth=current_depth + 1,
                    _visited=_visited,
                )
                _check_serialized_size(serialized_result, value)
                return serialized_result
            except Exception:
                pass

        if hasattr(value, "__dict__"):
            obj_id = id(value)
            if obj_id in _visited:
                return f"<circular_reference:{type(value).__name__}>"
            _visited.add(obj_id)

            try:
                attrs: dict[str, Any] = {}
                for key, val in value.__dict__.items():
                    if not key.startswith("_"):
                        attrs[key] = _serialize_value(
                            val,
                            max_depth=max_depth,
                            current_depth=current_depth + 1,
                            _visited=_visited,
                        )
                _visited.remove(obj_id)
                _check_serialized_size(attrs, value)
                return attrs
            except Exception:
                _visited.discard(obj_id)
                pass

        try:
            result_str: str = str(value)
            _check_serialized_size(result_str, value)
            return result_str
        except Exception:
            return f"<{type(value).__name__} object>"

    except Exception as e:
        return f"<serialization_error:{type(value).__name__}:{str(e)}>"


def _check_serialized_size(serialized: Any, original: Any) -> None:
    """Warn if serialized data exceeds the configured size threshold."""
    try:
        if isinstance(serialized, (dict, list)):
            json_str = json.dumps(serialized)
            size_bytes = len(json_str.encode("utf-8"))
        elif isinstance(serialized, str):
            size_bytes = len(serialized.encode("utf-8"))
        else:
            size_bytes = len(str(serialized).encode("utf-8"))

        if size_bytes > DEFAULT_MAX_SIZE_BYTES:
            size_mb = size_bytes / (1024 * 1024)
            warnings.warn(
                f"Serialized value is large ({size_mb:.2f} MB). "
                f"This may impact trace performance. "
                f"Type: {type(original).__name__}",
                UserWarning,
                stacklevel=WARNING_STACKLEVEL,
            )
    except Exception:
        pass
