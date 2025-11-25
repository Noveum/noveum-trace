"""
Base trace decorator for Noveum Trace SDK.

This module provides the fundamental @trace decorator that can be applied
to any function to add comprehensive tracing capabilities.
"""

import functools
import inspect
import json
import time
import warnings
from typing import Any, Callable, Optional, Union

from noveum_trace.core.context import attach_context_to_span
from noveum_trace.core.span import SpanStatus

# Serialization configuration constants
DEFAULT_MAX_DEPTH = 10
DEFAULT_MAX_SIZE_BYTES = 1048576  # 1MB
WARNING_STACKLEVEL = 4


def trace(
    func: Optional[Callable[..., Any]] = None,
    *,
    name: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    capture_args: bool = True,
    capture_result: bool = True,
    capture_errors: bool = True,
    capture_stack_trace: bool = False,
    capture_performance: bool = False,
    sample_fn: Optional[Callable[..., Any]] = None,
    tags: Optional[dict[str, str]] = None,
) -> Union[Callable[..., Any], Callable[[Callable[..., Any]], Callable[..., Any]]]:
    """
    Decorator to add tracing to any function.

    This decorator automatically creates a span for the decorated function,
    capturing inputs, outputs, timing, and error information.

    Args:
        func: Function to decorate (when used as @trace)
        name: Custom span name (defaults to function name)
        metadata: Additional metadata to attach to the span
        capture_args: Whether to capture function arguments
        capture_result: Whether to capture function return value
        capture_errors: Whether to capture exceptions
        capture_stack_trace: Whether to capture stack traces on errors
        capture_performance: Whether to capture performance metrics
        sample_fn: Custom sampling function
        tags: Tags to add to the span

    Returns:
        Decorated function or decorator

    Example:
        >>> @trace
        >>> def process_data(data: str) -> dict:
        ...     return {"processed": data}

        >>> @trace(name="custom_operation", capture_performance=True)
        >>> def expensive_operation(data: list) -> dict:
        ...     return complex_processing(data)
    """

    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        # Get function metadata
        func_name = name or f.__name__
        func_module = f.__module__
        func_qualname = f.__qualname__

        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Import here to avoid circular imports
            from noveum_trace import get_client, is_initialized
            from noveum_trace.core.context import get_current_trace

            # Check if SDK is initialized
            if not is_initialized():
                return f(*args, **kwargs)

            client = get_client()

            # Apply custom sampling if provided
            if sample_fn and not sample_fn(args, kwargs):
                return f(*args, **kwargs)

            # Auto-create trace if none exists
            auto_created_trace = False
            current_trace = get_current_trace()
            if current_trace is None:
                auto_created_trace = True
                current_trace = client.start_trace(
                    name=f"auto_trace_{func_name}",
                    attributes={"auto_created": True, "function": func_name},
                )

            # Create span attributes
            attributes = {
                "function.name": func_name,
                "function.module": func_module,
                "function.qualname": func_qualname,
                "function.type": "user_function",
            }

            # Add metadata
            if metadata:
                for key, value in metadata.items():
                    attributes[f"metadata.{key}"] = value

            # Add tags
            if tags:
                for key, value in tags.items():
                    attributes[f"tag.{key}"] = value

            # Capture function arguments
            if capture_args:
                try:
                    # Get function signature
                    sig = inspect.signature(f)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()

                    # Add arguments to attributes
                    for param_name, param_value in bound_args.arguments.items():
                        # Serialize argument value safely
                        serialized_value = _serialize_value(param_value)
                        attributes[f"function.args.{param_name}"] = serialized_value

                except Exception as e:
                    attributes["function.args.error"] = str(e)

            # Start the span
            span = client.start_span(
                name=func_name,
                attributes=attributes,
            )

            # Attach context to span
            attach_context_to_span(span)

            start_time = time.perf_counter()

            try:
                # Execute the function
                result = f(*args, **kwargs)

                # Capture performance metrics
                if capture_performance:
                    end_time = time.perf_counter()
                    execution_time = (end_time - start_time) * 1000  # Convert to ms
                    span.set_attributes(
                        {
                            "performance.execution_time_ms": execution_time,
                            "performance.cpu_time_ms": execution_time,  # Simplified for now
                        }
                    )

                # Capture result
                if capture_result:
                    try:
                        serialized_result = _serialize_value(result)
                        span.set_attribute("function.result", serialized_result)
                        span.set_attribute(
                            "function.result.type", type(result).__name__
                        )
                    except Exception as e:
                        span.set_attribute("function.result.error", str(e))

                # Set success status
                span.set_status(SpanStatus.OK)

                return result

            except Exception as e:
                # Handle errors
                if capture_errors:
                    span.record_exception(e, capture_stack_trace=capture_stack_trace)
                    span.set_status(SpanStatus.ERROR, str(e))

                # Re-raise the exception
                raise

            finally:
                # Always finish the span
                client.finish_span(span)

                # Finish auto-created trace
                if auto_created_trace and current_trace:
                    client.finish_trace(current_trace)

        # Add metadata to the wrapper
        wrapper._noveum_traced = True  # type: ignore
        wrapper._noveum_trace_config = {  # type: ignore
            "name": func_name,
            "metadata": metadata,
            "capture_args": capture_args,
            "capture_result": capture_result,
            "capture_errors": capture_errors,
            "capture_stack_trace": capture_stack_trace,
            "capture_performance": capture_performance,
            "tags": tags,
        }

        return wrapper

    # Handle both @trace and @trace() usage
    if func is None:
        # Called as @trace() with arguments
        return decorator
    else:
        # Called as @trace without arguments
        return decorator(func)


def _serialize_value(
    value: Any,
    max_depth: int = DEFAULT_MAX_DEPTH,
    current_depth: int = 0,
    _visited: Optional[set[int]] = None,
) -> Any:
    """
    Safely serialize a value for tracing, returning JSON-serializable objects.

    This function preserves structure for dicts and lists, extracts meaningful
    data from complex objects, and provides size warnings for large data.

    Args:
        value: Value to serialize
        max_depth: Maximum recursion depth (default: 10)
        current_depth: Current recursion depth (internal use)
        _visited: Set of object IDs to detect circular references (internal use)

    Returns:
        Serialized string representation (dicts are converted to JSON strings)
    """
    import json
        JSON-serializable representation (dict, list, str, int, float, bool, None)
    """
    if _visited is None:
        _visited = set()

    # Check recursion depth
    if current_depth >= max_depth:
        return f"<max_depth_reached:{type(value).__name__}>"

    try:
        # Handle None
        if value is None:
            return None

        # Handle primitive types - return as-is (already JSON-serializable)
        if isinstance(value, (int, float, bool)):
            return value

        # Handle strings - check size before returning
        if isinstance(value, str):
            _check_serialized_size(value, value)
            return value

        # Handle dict - preserve structure and recursively serialize values
        if isinstance(value, dict):
            dict_result: dict[str, Any] = {}
            for key, val in value.items():
                # Convert keys to strings if needed
                str_key = str(key) if not isinstance(key, str) else key
                dict_result[str_key] = _serialize_value(
                    val,
                    max_depth=max_depth,
                    current_depth=current_depth + 1,
                    _visited=_visited,
                )
            # Check size before returning
            _check_serialized_size(dict_result, value)
            return dict_result

        # Handle list/tuple - preserve structure and recursively serialize items
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
            # Check size before returning
            _check_serialized_size(list_result, value)
            return list_result

        # Handle objects with to_dict() method
        if hasattr(value, "to_dict") and callable(value.to_dict):
            try:
                dict_value = value.to_dict()
                serialized_result: Any = _serialize_value(
                    dict_value,
                    max_depth=max_depth,
                    current_depth=current_depth + 1,
                    _visited=_visited,
                )
                # Check size before returning
                _check_serialized_size(serialized_result, value)
                return serialized_result
            except Exception:
                pass  # Fall through to __dict__ extraction

        # Handle objects with __dict__ - extract attributes
        if hasattr(value, "__dict__"):
            # Check for circular references
            obj_id = id(value)
            if obj_id in _visited:
                return f"<circular_reference:{type(value).__name__}>"
            _visited.add(obj_id)

            try:
                attrs: dict[str, Any] = {}
                for key, val in value.__dict__.items():
                    # Skip private attributes (starting with _)
                    if not key.startswith("_"):
                        attrs[key] = _serialize_value(
                            val,
                            max_depth=max_depth,
                            current_depth=current_depth + 1,
                            _visited=_visited,
                        )
                _visited.remove(obj_id)
                # Check size before returning
                _check_serialized_size(attrs, value)
                return attrs
            except Exception:
                _visited.discard(obj_id)
                pass  # Fall through to string conversion

        # Fallback: convert to string representation
        try:
            result_str: str = str(value)
            # Check size and warn if too large
            _check_serialized_size(result_str, value)
            return result_str
        except Exception:
            return f"<{type(value).__name__} object>"

    except Exception as e:
        # Final fallback for any unexpected errors
        return f"<serialization_error:{type(value).__name__}:{str(e)}>"


def _check_serialized_size(serialized: Any, original: Any) -> None:
    """
    Check the size of serialized data and warn if it's too large.

    Args:
        serialized: The serialized value
        original: The original value (for context)
    """
    try:
        # Estimate size by converting to JSON string
        if isinstance(serialized, (dict, list)):
            json_str = json.dumps(serialized)
            size_bytes = len(json_str.encode("utf-8"))
        elif isinstance(serialized, str):
            size_bytes = len(serialized.encode("utf-8"))
        else:
            # For other types, estimate based on string representation
            size_bytes = len(str(serialized).encode("utf-8"))

        # Warn if size exceeds threshold
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
        # Silently ignore size checking errors
        pass


def is_traced(func: Callable[..., Any]) -> bool:
    """
    Check if a function has been decorated with @trace.

    Args:
        func: Function to check

    Returns:
        True if function is traced, False otherwise
    """
    return hasattr(func, "_noveum_traced") and func._noveum_traced


def get_trace_config(func: Callable[..., Any]) -> Optional[dict[str, Any]]:
    """
    Get the trace configuration for a decorated function.

    Args:
        func: Function to get config for

    Returns:
        Trace configuration dictionary or None if not traced
    """
    if is_traced(func):
        return func._noveum_trace_config  # type: ignore
    return None
