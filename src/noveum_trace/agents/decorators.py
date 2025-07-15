"""
Simplified decorators for multi-agent tracing.

This module provides a single, unified @trace decorator that handles all tracing
scenarios through parameters, eliminating the need for multiple decorators.
"""

import asyncio
import functools
import inspect
import time
from typing import Any, Callable, Dict, List, Optional, Union, Set
from contextlib import contextmanager

from ..core.tracer import get_current_tracer
from ..core.span import Span
from ..types import SpanKind, SpanData, LLMRequest, LLMResponse, OperationType, AISystem
from .context import get_current_agent


def trace(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
    type: Optional[str] = None,  # "llm", "component", "function"
    operation: Optional[str] = None,  # "chat", "completion", "embedding"
    model: Optional[str] = None,
    ai_system: Optional[AISystem] = None,
    metrics: Optional[List[str]] = None,
    capture_args: bool = False,
    capture_result: bool = False,
    capture_input: bool = False,
    capture_output: bool = False,
    agent: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None
) -> Union[Callable, Any]:
    """
    Unified tracing decorator that handles all tracing scenarios.
    
    This single decorator replaces @trace, @observe, and @llm_trace by using
    parameters to specify the type of tracing needed.
    
    Args:
        func: The function to decorate (automatically provided)
        name: Custom span name (defaults to function name)
        kind: Span kind (INTERNAL, CLIENT, SERVER, etc.)
        type: Type of operation ("llm", "component", "function")
        operation: Specific operation type ("chat", "completion", "embedding")
        model: Model name for LLM operations
        ai_system: AI system for LLM operations (openai, anthropic, etc.)
        metrics: List of metrics to track
        capture_args: Whether to capture function arguments
        capture_result: Whether to capture function result
        capture_input: Whether to capture input (alias for capture_args)
        capture_output: Whether to capture output (alias for capture_result)
        agent: Specific agent name to use (overrides current agent)
        attributes: Additional span attributes
    
    Examples:
        # Basic function tracing
        @trace
        def process_data(data):
            return transform(data)
        
        # LLM operation tracing
        @trace(type="llm", model="gpt-4", operation="chat", ai_system="openai")
        def chat_completion(messages):
            return openai_client.chat.completions.create(messages=messages)
        
        # Component-level tracing with metrics
        @trace(type="component", metrics=["accuracy", "latency"], capture_input=True)
        def llm_component(prompt):
            return call_llm(prompt)
    """
    
    def decorator(f: Callable) -> Callable:
        # Determine span name
        span_name = name or f.__name__
        
        # Resolve capture flags (input/output are aliases)
        should_capture_args = capture_args or capture_input
        should_capture_result = capture_result or capture_output
        
        # Determine span type and attributes
        span_attributes = attributes or {}
        if type:
            span_attributes["operation.type"] = type
        if operation:
            span_attributes["operation.name"] = operation
        if model:
            span_attributes["llm.model"] = model
        if ai_system:
            span_attributes["llm.system"] = ai_system.value if hasattr(ai_system, 'value') else str(ai_system)
        if metrics:
            span_attributes["metrics.enabled"] = metrics
        
        @functools.wraps(f)
        def sync_wrapper(*args, **kwargs):
            # Get tracer (from agent or global)
            tracer = _get_tracer_for_function(f.__name__, agent)
            if not tracer:
                # No tracer available, execute function normally
                return f(*args, **kwargs)
            
            # Create span
            with tracer.start_span(
                name=span_name,
                kind=kind,
                attributes=span_attributes
            ) as span:
                try:
                    # Capture input if requested
                    if should_capture_args:
                        _capture_function_input(span, f, args, kwargs)
                    
                    # Add agent information
                    _add_agent_context(span)
                    
                    # Execute function
                    start_time = time.time()
                    result = f(*args, **kwargs)
                    end_time = time.time()
                    
                    # Add timing
                    span.set_attribute("duration_ms", (end_time - start_time) * 1000)
                    
                    # Capture output if requested
                    if should_capture_result:
                        _capture_function_output(span, result, type)
                    
                    # Handle LLM-specific processing
                    if type == "llm":
                        _process_llm_result(span, result, model, operation, ai_system)
                    
                    # Mark as successful
                    span.set_status("OK")
                    
                    return result
                    
                except Exception as e:
                    # Record error
                    span.set_status("ERROR", str(e))
                    span.set_attribute("error.type", e.__class__.__name__)
                    span.set_attribute("error.message", str(e))
                    raise
        
        @functools.wraps(f)
        async def async_wrapper(*args, **kwargs):
            # Get tracer (from agent or global)
            tracer = _get_tracer_for_function(f.__name__, agent)
            if not tracer:
                # No tracer available, execute function normally
                return await f(*args, **kwargs)
            
            # Create span
            with tracer.start_span(
                name=span_name,
                kind=kind,
                attributes=span_attributes
            ) as span:
                try:
                    # Capture input if requested
                    if should_capture_args:
                        _capture_function_input(span, f, args, kwargs)
                    
                    # Add agent information
                    _add_agent_context(span)
                    
                    # Execute function
                    start_time = time.time()
                    result = await f(*args, **kwargs)
                    end_time = time.time()
                    
                    # Add timing
                    span.set_attribute("duration_ms", (end_time - start_time) * 1000)
                    
                    # Capture output if requested
                    if should_capture_result:
                        _capture_function_output(span, result, type)
                    
                    # Handle LLM-specific processing
                    if type == "llm":
                        _process_llm_result(span, result, model, operation, ai_system)
                    
                    # Mark as successful
                    span.set_status("OK")
                    
                    return result
                    
                except Exception as e:
                    # Record error
                    span.set_status("ERROR", str(e))
                    span.set_attribute("error.type", e.__class__.__name__)
                    span.set_attribute("error.message", str(e))
                    raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(f):
            return async_wrapper
        else:
            return sync_wrapper
    
    # Handle both @trace and @trace() usage
    if func is None:
        # Called as @trace() with parameters
        return decorator
    else:
        # Called as @trace without parameters
        return decorator(func)


def update_current_span(
    attributes: Optional[Dict[str, Any]] = None,
    events: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs
) -> None:
    """
    Update the current active span with additional information.
    
    This function allows updating the current span with new attributes,
    events, or metadata during execution.
    
    Args:
        attributes: Additional attributes to add to the span
        events: List of events to add to the span
        metadata: Metadata to add to the span
        **kwargs: Additional attributes passed as keyword arguments
    """
    from ..core.context import get_current_span
    
    try:
        current_span = get_current_span()
        if current_span and current_span.is_recording:
            # Add attributes
            if attributes:
                for key, value in attributes.items():
                    current_span.set_attribute(key, value)
            
            # Add kwargs as attributes
            for key, value in kwargs.items():
                current_span.set_attribute(key, value)
            
            # Add events
            if events:
                for event in events:
                    event_name = event.get("name", "custom_event")
                    event_attrs = event.get("attributes", {})
                    current_span.add_event(event_name, event_attrs)
            
            # Add metadata as attributes with metadata prefix
            if metadata:
                for key, value in metadata.items():
                    current_span.set_attribute(f"metadata.{key}", value)
                    
    except Exception as e:
        # Silently ignore errors in tracing to avoid breaking application flow
        pass


# Backward compatibility aliases
def observe(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    **kwargs
) -> Union[Callable, Any]:
    """
    Backward compatibility alias for @trace decorator.
    
    This function provides the same functionality as @trace but with
    a different name for compatibility with existing code.
    """
    return trace(func, name=name, **kwargs)


def llm_trace(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    model: Optional[str] = None,
    operation: Optional[str] = None,
    **kwargs
) -> Union[Callable, Any]:
    """
    Backward compatibility alias for @trace decorator with LLM-specific defaults.
    
    This function provides the same functionality as @trace but with
    defaults suitable for LLM operations.
    """
    return trace(
        func,
        name=name,
        type="llm",
        model=model,
        operation=operation,
        **kwargs
    )


# Helper functions

def _get_tracer_for_function(func_name: str, agent_name: Optional[str] = None):
    """Get the appropriate tracer for a function."""
    if agent_name:
        # Use specific agent's tracer
        from .registry import get_agent_registry
        registry = get_agent_registry()
        agent = registry.get_agent(agent_name)
        if agent:
            return agent.tracer
    
    # Use current agent's tracer
    current_agent = get_current_agent()
    if current_agent:
        return current_agent.tracer
    
    # Fall back to global tracer
    return get_current_tracer()


def _capture_function_input(span: Span, func: Callable, args: tuple, kwargs: dict):
    """Capture function input arguments."""
    try:
        # Get function signature
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        # Serialize arguments
        input_data = {}
        for param_name, value in bound_args.arguments.items():
            input_data[param_name] = _serialize_data(value)
        
        span.set_attribute("input", input_data)
        span.set_attribute("input.args_count", len(args))
        span.set_attribute("input.kwargs_count", len(kwargs))
        
    except Exception as e:
        # If we can't capture input, just note the attempt
        span.set_attribute("input.capture_error", str(e))


def _capture_function_output(span: Span, result: Any, operation_type: Optional[str]):
    """Capture function output."""
    try:
        span.set_attribute("output", _serialize_data(result))
        span.set_attribute("output.type", type(result).__name__)
        
        # Add type-specific output processing
        if operation_type == "llm":
            _extract_llm_output_info(span, result)
        
    except Exception as e:
        span.set_attribute("output.capture_error", str(e))


def _process_llm_result(
    span: Span, 
    result: Any, 
    model: Optional[str], 
    operation: Optional[str],
    ai_system: Optional[AISystem]
):
    """Process LLM-specific result information."""
    try:
        # Try to extract standard LLM response information
        if hasattr(result, 'choices') and hasattr(result, 'usage'):
            # OpenAI-style response
            if result.choices:
                choice = result.choices[0]
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    span.set_attribute("llm.response.content", result.choices[0].message.content)
                if hasattr(choice, 'finish_reason'):
                    span.set_attribute("llm.response.finish_reason", choice.finish_reason)
            
            if hasattr(result.usage, 'prompt_tokens'):
                span.set_attribute("llm.usage.prompt_tokens", result.usage.prompt_tokens)
            if hasattr(result.usage, 'completion_tokens'):
                span.set_attribute("llm.usage.completion_tokens", result.usage.completion_tokens)
            if hasattr(result.usage, 'total_tokens'):
                span.set_attribute("llm.usage.total_tokens", result.usage.total_tokens)
        
        # Add model information
        if model:
            span.set_attribute("llm.request.model", model)
        if hasattr(result, 'model'):
            span.set_attribute("llm.response.model", result.model)
        
        # Add operation information
        if operation:
            span.set_attribute("gen_ai.operation.name", operation)
        
        # Add AI system information
        if ai_system:
            span.set_attribute("gen_ai.system", ai_system.value if hasattr(ai_system, 'value') else str(ai_system))
        
    except Exception as e:
        span.set_attribute("llm.processing_error", str(e))


def _extract_llm_output_info(span: Span, result: Any):
    """Extract additional information from LLM output."""
    try:
        if isinstance(result, str):
            span.set_attribute("llm.response.length", len(result))
        elif hasattr(result, 'choices'):
            if result.choices and hasattr(result.choices[0], 'message'):
                content = result.choices[0].message.content
                if content:
                    span.set_attribute("llm.response.length", len(content))
    except Exception:
        pass


def _add_agent_context(span: Span):
    """Add current agent context to span."""
    try:
        current_agent = get_current_agent()
        if current_agent:
            span.set_attribute("agent.name", current_agent.name)
            span.set_attribute("agent.type", current_agent.agent_type)
            span.set_attribute("agent.id", current_agent.agent_id)
            
            # Add agent capabilities
            if current_agent.config.capabilities:
                span.set_attribute("agent.capabilities", list(current_agent.config.capabilities))
            
            # Add agent tags
            if current_agent.config.tags:
                span.set_attribute("agent.tags", list(current_agent.config.tags))
    except Exception:
        pass


def _serialize_data(data: Any, max_length: int = 1000) -> str:
    """Serialize data for span attributes."""
    try:
        if data is None:
            return "null"
        elif isinstance(data, (str, int, float, bool)):
            result = str(data)
        elif isinstance(data, (list, tuple)):
            result = f"[{len(data)} items]"
        elif isinstance(data, dict):
            result = f"{{...}} ({len(data)} keys)"
        else:
            result = f"{type(data).__name__} object"
        
        # Truncate if too long
        if len(result) > max_length:
            result = result[:max_length] + "..."
        
        return result
    except Exception:
        return f"<serialization error: {type(data).__name__}>"

