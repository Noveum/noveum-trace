"""
Base trace decorator for Noveum Trace SDK.

This module provides the fundamental @trace decorator that can be applied
to any function to add comprehensive tracing capabilities.
"""

import functools
import inspect
import time
from typing import Any, Callable, Optional, Union

from noveum_trace.core.context import attach_context_to_span
from noveum_trace.core.span import SpanStatus


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


def _serialize_value(value: Any, max_length: int = 1000) -> str:
    """
    Safely serialize a value for tracing.

    Args:
        value: Value to serialize
        max_length: Maximum length of serialized string

    Returns:
        Serialized string representation
    """
    try:
        # Handle common types
        if value is None:
            return "None"
        elif isinstance(value, (str, int, float, bool)):
            result = str(value)
        elif isinstance(value, (list, tuple)):
            if len(value) <= 10:  # Small collections
                result = str(value)
            else:
                result = f"{type(value).__name__}(length={len(value)})"
        elif isinstance(value, dict):
            if len(value) <= 10:  # Small dictionaries
                result = str(value)
            else:
                result = f"dict(keys={len(value)})"
        else:
            # For other types, use repr but limit length
            result = repr(value)

        # Truncate if too long
        if len(result) > max_length:
            result = result[: max_length - 3] + "..."

        return result

    except Exception:
        # Fallback for objects that can't be serialized
        return f"<{type(value).__name__} object>"


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
