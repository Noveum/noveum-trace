"""
Tool-specific trace decorator for Noveum Trace SDK.

This module provides the @trace_tool decorator for tracing
tool usage in LLM applications and agent systems.
"""

import functools
import inspect
from typing import Any, Callable, Optional, Union

from noveum_trace.core.context import attach_context_to_span
from noveum_trace.core.span import SpanStatus
from noveum_trace.decorators.base import _serialize_value


def trace_tool(
    func: Optional[Callable[..., Any]] = None,
    *,
    tool_name: Optional[str] = None,
    name: Optional[str] = None,
    tool_type: Optional[str] = None,
    description: Optional[str] = None,
    capture_inputs: bool = True,
    capture_outputs: bool = True,
    capture_errors: bool = True,
    metadata: Optional[dict[str, Any]] = None,
    tags: Optional[dict[str, str]] = None,
) -> Union[Callable[..., Any], Callable[[Callable[..., Any]], Callable[..., Any]]]:
    """
    Decorator to add tool-specific tracing to functions.

    This decorator automatically captures tool usage metadata including
    tool identity, inputs, outputs, and execution details.

    Args:
        func: Function to decorate (when used as @trace_tool)
        tool_name: Name of the tool being used
        name: Custom span name (defaults to function name)
        tool_type: Type of tool (e.g., "api", "database", "file_system")
        description: Description of what the tool does
        capture_inputs: Whether to capture tool inputs
        capture_outputs: Whether to capture tool outputs
        capture_errors: Whether to capture tool errors
        metadata: Additional metadata to attach to the span
        tags: Tags to add to the span

    Returns:
        Decorated function or decorator

    Example:
        >>> @trace_tool
        >>> def search_web(query: str) -> list:
        ...     return perform_search(query)

        >>> @trace_tool(tool_name="web_search", tool_type="api")
        >>> def search_web(query: str) -> list:
        ...     # Web search implementation
        ...     return search_results

        >>> @trace_tool(
        ...     "database_query",
        ...     tool_type="database",
        ...     description="Execute SQL queries"
        ... )
        >>> def query_database(sql: str) -> list:
        ...     # Database query implementation
        ...     return results
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        func_name = name or func.__name__
        actual_tool_name = tool_name or func_name

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            from noveum_trace import get_client, is_initialized
            from noveum_trace.core.context import get_current_trace

            if not is_initialized():
                return func(*args, **kwargs)

            client = get_client()

            # Auto-create trace if none exists
            auto_created_trace = False
            current_trace = get_current_trace()
            if current_trace is None:
                auto_created_trace = True
                current_trace = client.start_trace(
                    name=f"auto_trace_{func_name}",
                    attributes={
                        "auto_created": True,
                        "function": func_name,
                        "type": "tool_call",
                        "tool_name": actual_tool_name,
                    },
                )

            # Create span attributes
            attributes = {
                "function.name": func_name,
                "function.module": func.__module__,
                "function.type": "tool_call",
                "tool.name": actual_tool_name,
                "tool.operation": func_name,
            }

            if tool_type:
                attributes["tool.type"] = tool_type
            if description:
                attributes["tool.description"] = description

            # Add metadata and tags
            if metadata:
                for key, value in metadata.items():
                    attributes[f"metadata.{key}"] = value
            if tags:
                for key, value in tags.items():
                    attributes[f"tag.{key}"] = value

            # Capture tool inputs
            if capture_inputs:
                try:
                    sig = inspect.signature(func)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()

                    for param_name, param_value in bound_args.arguments.items():
                        serialized_value = _serialize_value(param_value)
                        attributes[f"tool.input.{param_name}"] = serialized_value

                except Exception as e:
                    attributes["tool.input.error"] = str(e)

            # Start the span
            span = client.start_span(
                name=f"tool:{tool_name}:{func_name}",
                attributes=attributes,
            )

            attach_context_to_span(span)

            try:
                result = func(*args, **kwargs)

                # Capture tool outputs
                if capture_outputs:
                    try:
                        output_data = _extract_tool_output(result)
                        span.set_attributes(output_data)
                    except Exception as e:
                        span.set_attribute("tool.output.error", str(e))

                span.set_status(SpanStatus.OK)
                return result

            except Exception as e:
                if capture_errors:
                    span.record_exception(e)
                    span.set_status(SpanStatus.ERROR, str(e))
                    span.set_attributes(
                        {
                            "tool.error.type": type(e).__name__,
                            "tool.error.message": str(e),
                            "tool.name": tool_name,
                        }
                    )
                raise

            finally:
                client.finish_span(span)

                # Finish auto-created trace
                if auto_created_trace and current_trace:
                    client.finish_trace(current_trace)

        wrapper._noveum_traced = True  # type: ignore
        wrapper._noveum_trace_type = "tool"  # type: ignore
        wrapper._noveum_tool_name = actual_tool_name  # type: ignore

        return wrapper

    # Handle both @trace_tool and @trace_tool() usage
    if func is None:
        # Called as @trace_tool() with arguments
        return decorator
    else:
        # Called as @trace_tool without arguments
        return decorator(func)


def _extract_tool_output(result: Any) -> dict[str, Any]:
    """Extract tool output metadata."""
    output_data = {
        "tool.output.result": _serialize_value(result),
        "tool.output.type": type(result).__name__,
    }

    if isinstance(result, dict):
        # Extract common tool output fields
        output_fields = {
            "status": "tool.output.status",
            "data": "tool.output.data",
            "error": "tool.output.error",
            "metadata": "tool.output.metadata",
        }

        for field, attr_name in output_fields.items():
            if field in result:
                output_data[attr_name] = _serialize_value(result[field])

    return output_data
