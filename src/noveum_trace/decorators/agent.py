"""
Agent-specific trace decorator for Noveum Trace SDK.

This module provides the @trace_agent decorator specifically designed
for tracing agent operations in multi-agent systems.
"""

import functools
import inspect
from typing import Any, Callable, Optional

from noveum_trace.core.context import (
    attach_context_to_span,
    get_context_attribute,
    get_current_span,
    set_context_attribute,
)
from noveum_trace.core.span import SpanStatus
from noveum_trace.decorators.base import _serialize_value


def trace_agent(
    agent_id: str,
    *,
    name: Optional[str] = None,
    role: Optional[str] = None,
    agent_type: Optional[str] = None,
    capabilities: Optional[list[str]] = None,
    capture_inputs: bool = True,
    capture_outputs: bool = True,
    capture_reasoning: bool = True,
    capture_tools: bool = True,
    metadata: Optional[dict[str, Any]] = None,
    tags: Optional[dict[str, str]] = None,
) -> Callable[..., Any]:
    """
    Decorator to add agent-specific tracing to functions.

    This decorator automatically captures agent-specific metadata including
    agent identity, role, capabilities, reasoning, and tool usage.

    Args:
        agent_id: Unique identifier for the agent
        name: Custom span name (defaults to function name)
        role: Agent role (e.g., "researcher", "analyst", "coordinator")
        agent_type: Type of agent (e.g., "llm_agent", "rule_based", "hybrid")
        capabilities: List of agent capabilities
        capture_inputs: Whether to capture agent inputs
        capture_outputs: Whether to capture agent outputs
        capture_reasoning: Whether to capture reasoning steps
        capture_tools: Whether to capture tool usage
        metadata: Additional metadata to attach to the span
        tags: Tags to add to the span

    Returns:
        Decorated function

    Example:
        >>> @trace_agent(
        ...     agent_id="researcher",
        ...     role="information_gatherer",
        ...     capabilities=["web_search", "document_analysis"]
        ... )
        >>> def research_task(query: str) -> dict:
        ...     # Agent implementation
        ...     return {"findings": "...", "sources": [...]}

        >>> @trace_agent(
        ...     agent_id="coordinator",
        ...     role="workflow_manager",
        ...     agent_type="orchestrator"
        ... )
        >>> def coordinate_agents(task: str) -> dict:
        ...     # Coordination logic
        ...     return {"status": "completed", "results": {...}}
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Get function metadata
        func_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Import here to avoid circular imports
            from noveum_trace import get_client, is_initialized
            from noveum_trace.core.context import get_current_trace

            # Check if SDK is initialized
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
                        "type": "agent_operation",
                        "agent_id": agent_id,
                    },
                )

            # Create span attributes
            attributes = {
                "function.name": func_name,
                "function.module": func.__module__,
                "function.qualname": func.__qualname__,
                "function.type": "agent_operation",
                "agent.id": agent_id,
                "agent.operation": func_name,
            }

            # Add agent metadata
            if role:
                attributes["agent.role"] = role
            if agent_type:
                attributes["agent.type"] = agent_type
            if capabilities:
                attributes["agent.capabilities"] = ",".join(capabilities)

            # Add custom metadata
            if metadata:
                for key, value in metadata.items():
                    attributes[f"metadata.{key}"] = value

            # Add tags
            if tags:
                for key, value in tags.items():
                    attributes[f"tag.{key}"] = value

            # Set agent context
            set_context_attribute("agent_id", agent_id)
            if role:
                set_context_attribute("agent_role", role)

            # Extract agent inputs
            if capture_inputs:
                try:
                    sig = inspect.signature(func)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()

                    # Capture agent inputs
                    for param_name, param_value in bound_args.arguments.items():
                        serialized_value = _serialize_value(param_value)
                        attributes[f"agent.input.{param_name}"] = serialized_value

                except Exception as e:
                    attributes["agent.input.error"] = str(e)

            # Start the span
            span = client.start_span(
                name=f"agent:{agent_id}:{func_name}",
                attributes=attributes,
            )

            # Attach context to span
            attach_context_to_span(span)

            # Track agent workflow context
            workflow_id = get_context_attribute("workflow_id")
            if workflow_id:
                span.set_attribute("agent.workflow_id", workflow_id)

            # Track parent agent if in nested call
            parent_agent_id = get_context_attribute("agent_id")
            if parent_agent_id and parent_agent_id != agent_id:
                span.set_attribute("agent.parent_agent_id", parent_agent_id)

            try:
                # Execute the function
                result = func(*args, **kwargs)

                # Capture agent outputs
                if capture_outputs:
                    try:
                        output_data = _extract_agent_output(result)
                        span.set_attributes(output_data)
                    except Exception as e:
                        span.set_attribute("agent.output.error", str(e))

                # Capture reasoning if present in result
                if capture_reasoning:
                    reasoning_data = _extract_reasoning(result)
                    if reasoning_data:
                        span.set_attributes(reasoning_data)

                # Capture tool usage if present
                if capture_tools:
                    tool_data = _extract_tool_usage(result)
                    if tool_data:
                        span.set_attributes(tool_data)

                # Set success status
                span.set_status(SpanStatus.OK)

                return result

            except Exception as e:
                # Handle agent-specific errors
                span.record_exception(e)
                span.set_status(SpanStatus.ERROR, str(e))

                # Add agent-specific error attributes
                span.set_attributes(
                    {
                        "agent.error.type": type(e).__name__,
                        "agent.error.message": str(e),
                        "agent.error.agent_id": agent_id,
                    }
                )

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
        wrapper._noveum_trace_type = "agent"  # type: ignore
        wrapper._noveum_agent_id = agent_id  # type: ignore
        wrapper._noveum_trace_config = {  # type: ignore
            "agent_id": agent_id,
            "name": func_name,
            "role": role,
            "agent_type": agent_type,
            "capabilities": capabilities,
            "capture_inputs": capture_inputs,
            "capture_outputs": capture_outputs,
            "capture_reasoning": capture_reasoning,
            "capture_tools": capture_tools,
            "metadata": metadata,
            "tags": tags,
        }

        return wrapper

    return decorator


def _extract_agent_output(result: Any) -> dict[str, Any]:
    """
    Extract agent output metadata from function result.

    Args:
        result: Function result

    Returns:
        Dictionary of agent output metadata
    """
    output_data = {}

    # Serialize the main result
    output_data["agent.output.result"] = _serialize_value(result)
    output_data["agent.output.type"] = type(result).__name__

    # Extract structured output if result is a dictionary
    if isinstance(result, dict):
        # Look for common agent output fields
        output_fields = {
            "status": "agent.output.status",
            "confidence": "agent.output.confidence",
            "reasoning": "agent.output.reasoning",
            "sources": "agent.output.sources",
            "tools_used": "agent.output.tools_used",
            "next_action": "agent.output.next_action",
            "metadata": "agent.output.metadata",
        }

        for field, attr_name in output_fields.items():
            if field in result:
                output_data[attr_name] = _serialize_value(result[field])

    return output_data


def _extract_reasoning(result: Any) -> Optional[dict[str, Any]]:
    """
    Extract reasoning information from agent result.

    Args:
        result: Function result

    Returns:
        Dictionary of reasoning metadata or None
    """
    reasoning_data = {}

    if isinstance(result, dict):
        # Look for reasoning-related fields
        reasoning_fields = {
            "reasoning": "agent.reasoning.steps",
            "thought_process": "agent.reasoning.process",
            "decision_factors": "agent.reasoning.factors",
            "alternatives_considered": "agent.reasoning.alternatives",
            "confidence_score": "agent.reasoning.confidence",
        }

        for field, attr_name in reasoning_fields.items():
            if field in result:
                reasoning_data[attr_name] = _serialize_value(result[field])

    return reasoning_data if reasoning_data else None


def _extract_tool_usage(result: Any) -> Optional[dict[str, Any]]:
    """
    Extract tool usage information from agent result.

    Args:
        result: Function result

    Returns:
        Dictionary of tool usage metadata or None
    """
    tool_data = {}

    if isinstance(result, dict):
        # Look for tool-related fields
        tool_fields = {
            "tools_used": "agent.tools.used",
            "tool_calls": "agent.tools.calls",
            "tool_results": "agent.tools.results",
            "tool_errors": "agent.tools.errors",
        }

        for field, attr_name in tool_fields.items():
            if field in result:
                tool_data[attr_name] = _serialize_value(result[field])

    return tool_data if tool_data else None


def get_current_agent_id() -> Optional[str]:
    """
    Get the current agent ID from context.

    Returns:
        Current agent ID or None if not in agent context
    """
    return get_context_attribute("agent_id")


def set_agent_workflow(workflow_id: str) -> None:
    """
    Set the current workflow ID for agent coordination.

    Args:
        workflow_id: Unique workflow identifier
    """
    set_context_attribute("workflow_id", workflow_id)


def add_agent_communication(
    from_agent: str, to_agent: str, message: Any, message_type: str = "data"
) -> None:
    """
    Record communication between agents.

    Args:
        from_agent: Source agent ID
        to_agent: Target agent ID
        message: Message content
        message_type: Type of message (data, command, query, etc.)
    """
    current_span = get_current_span()
    if current_span:
        current_span.add_event(
            "agent_communication",
            {
                "from_agent": from_agent,
                "to_agent": to_agent,
                "message_type": message_type,
                "message": _serialize_value(message),
            },
        )


def record_agent_decision(
    decision: str,
    reasoning: Optional[str] = None,
    confidence: Optional[float] = None,
    alternatives: Optional[list[str]] = None,
) -> None:
    """
    Record an agent decision with reasoning.

    Args:
        decision: The decision made
        reasoning: Reasoning behind the decision
        confidence: Confidence score (0.0 to 1.0)
        alternatives: Alternative options considered
    """
    current_span = get_current_span()
    if current_span:
        event_attributes: dict[str, Any] = {
            "decision": decision,
        }

        if reasoning:
            event_attributes["reasoning"] = reasoning
        if confidence is not None:
            event_attributes["confidence"] = confidence
        if alternatives:
            event_attributes["alternatives"] = alternatives

        current_span.add_event("agent_decision", event_attributes)
