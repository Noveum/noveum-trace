"""
Decorators for automatic instrumentation of functions and LLM calls.
"""

import asyncio
import functools
import inspect
import time
from typing import Any, Callable, Dict, Optional, Union, TypeVar, cast
import logging

from ..core.tracer import get_current_tracer
from ..core.span import Span
from ..core.context import SpanContext, AsyncSpanContext
from ..types import SpanKind, LLMRequest, LLMResponse, OperationType, AISystem
from ..utils.exceptions import InstrumentationError

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


def trace_function(
    name: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None,
    tracer=None,
    capture_args: bool = False,
    capture_result: bool = False,
    capture_exceptions: bool = True,
) -> Callable[[F], F]:
    """Decorator to automatically trace function execution.
    
    Args:
        name: Span name (defaults to function name)
        kind: Span kind
        attributes: Additional span attributes
        tracer: Tracer to use (defaults to current tracer)
        capture_args: Whether to capture function arguments
        capture_result: Whether to capture function result
        capture_exceptions: Whether to capture exceptions
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        span_name = name or func.__name__
        span_attributes = attributes or {}
        
        # Add function metadata
        span_attributes.update({
            "function.name": func.__name__,
            "function.module": func.__module__,
        })
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                current_tracer = tracer or get_current_tracer()
                if not current_tracer:
                    logger.warning("No tracer available for function tracing")
                    return await func(*args, **kwargs)
                
                span = current_tracer.start_span(
                    name=span_name,
                    kind=kind,
                    attributes=span_attributes.copy()
                )
                
                # Capture arguments if requested
                if capture_args:
                    _capture_function_args(span, func, args, kwargs)
                
                async with AsyncSpanContext(span):
                    try:
                        result = await func(*args, **kwargs)
                        
                        # Capture result if requested
                        if capture_result:
                            _capture_function_result(span, result)
                        
                        span.set_status("ok")
                        return result
                        
                    except Exception as e:
                        if capture_exceptions:
                            span.record_exception(e)
                        raise
            
            return cast(F, async_wrapper)
        
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                current_tracer = tracer or get_current_tracer()
                if not current_tracer:
                    logger.warning("No tracer available for function tracing")
                    return func(*args, **kwargs)
                
                span = current_tracer.start_span(
                    name=span_name,
                    kind=kind,
                    attributes=span_attributes.copy()
                )
                
                # Capture arguments if requested
                if capture_args:
                    _capture_function_args(span, func, args, kwargs)
                
                with SpanContext(span):
                    try:
                        result = func(*args, **kwargs)
                        
                        # Capture result if requested
                        if capture_result:
                            _capture_function_result(span, result)
                        
                        span.set_status("ok")
                        return result
                        
                    except Exception as e:
                        if capture_exceptions:
                            span.record_exception(e)
                        raise
            
            return cast(F, sync_wrapper)
    
    return decorator


def trace_llm_call(
    operation_type: Optional[OperationType] = None,
    ai_system: Optional[AISystem] = None,
    model: Optional[str] = None,
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    tracer=None,
    capture_content: bool = False,
) -> Callable[[F], F]:
    """Decorator to automatically trace LLM function calls.
    
    Args:
        operation_type: Type of LLM operation
        ai_system: AI system being used
        model: Model name
        name: Span name (defaults to function name)
        attributes: Additional span attributes
        tracer: Tracer to use (defaults to current tracer)
        capture_content: Whether to capture LLM content (prompts/responses)
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        span_name = name or f"llm.{func.__name__}"
        span_attributes = attributes or {}
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                current_tracer = tracer or get_current_tracer()
                if not current_tracer:
                    logger.warning("No tracer available for LLM tracing")
                    return await func(*args, **kwargs)
                
                # Extract LLM parameters from function arguments
                llm_request = _extract_llm_request(
                    func, args, kwargs, operation_type, ai_system, model
                )
                
                span = current_tracer.start_llm_span(
                    operation_name=span_name,
                    request=llm_request,
                    attributes=span_attributes
                )
                
                async with AsyncSpanContext(span):
                    try:
                        start_time = time.time()
                        result = await func(*args, **kwargs)
                        end_time = time.time()
                        
                        # Extract and set response information
                        llm_response = _extract_llm_response(result)
                        span.set_llm_response(llm_response)
                        
                        # Set timing attributes
                        span.set_attribute("llm.latency_ms", (end_time - start_time) * 1000)
                        
                        span.set_status("ok")
                        return result
                        
                    except Exception as e:
                        span.record_exception(e)
                        raise
            
            return cast(F, async_wrapper)
        
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                current_tracer = tracer or get_current_tracer()
                if not current_tracer:
                    logger.warning("No tracer available for LLM tracing")
                    return func(*args, **kwargs)
                
                # Extract LLM parameters from function arguments
                llm_request = _extract_llm_request(
                    func, args, kwargs, operation_type, ai_system, model
                )
                
                span = current_tracer.start_llm_span(
                    operation_name=span_name,
                    request=llm_request,
                    attributes=span_attributes
                )
                
                with SpanContext(span):
                    try:
                        start_time = time.time()
                        result = func(*args, **kwargs)
                        end_time = time.time()
                        
                        # Extract and set response information
                        llm_response = _extract_llm_response(result)
                        span.set_llm_response(llm_response)
                        
                        # Set timing attributes
                        span.set_attribute("llm.latency_ms", (end_time - start_time) * 1000)
                        
                        span.set_status("ok")
                        return result
                        
                    except Exception as e:
                        span.record_exception(e)
                        raise
            
            return cast(F, sync_wrapper)
    
    return decorator


def trace_streaming_llm_call(
    operation_type: Optional[OperationType] = None,
    ai_system: Optional[AISystem] = None,
    model: Optional[str] = None,
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    tracer=None,
) -> Callable[[F], F]:
    """Decorator to automatically trace streaming LLM function calls.
    
    Args:
        operation_type: Type of LLM operation
        ai_system: AI system being used
        model: Model name
        name: Span name (defaults to function name)
        attributes: Additional span attributes
        tracer: Tracer to use (defaults to current tracer)
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        span_name = name or f"llm.streaming.{func.__name__}"
        span_attributes = attributes or {}
        span_attributes["llm.streaming"] = True
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                current_tracer = tracer or get_current_tracer()
                if not current_tracer:
                    logger.warning("No tracer available for streaming LLM tracing")
                    async for item in func(*args, **kwargs):
                        yield item
                    return
                
                # Extract LLM parameters from function arguments
                llm_request = _extract_llm_request(
                    func, args, kwargs, operation_type, ai_system, model
                )
                
                span = current_tracer.start_llm_span(
                    operation_name=span_name,
                    request=llm_request,
                    attributes=span_attributes
                )
                
                async with AsyncSpanContext(span):
                    try:
                        start_time = time.time()
                        first_token_time = None
                        token_count = 0
                        accumulated_content = ""
                        
                        async for chunk in func(*args, **kwargs):
                            if first_token_time is None:
                                first_token_time = time.time()
                                ttft_ms = (first_token_time - start_time) * 1000
                                span.set_attribute("llm.time_to_first_token_ms", ttft_ms)
                            
                            # Accumulate streaming content
                            if hasattr(chunk, 'choices') and chunk.choices:
                                delta = chunk.choices[0].delta
                                if hasattr(delta, 'content') and delta.content:
                                    accumulated_content += delta.content
                                    token_count += 1
                            
                            yield chunk
                        
                        end_time = time.time()
                        
                        # Set final metrics
                        span.set_attribute("llm.latency_ms", (end_time - start_time) * 1000)
                        span.set_attribute("llm.streaming.token_count", token_count)
                        
                        if token_count > 0:
                            tokens_per_second = token_count / (end_time - start_time)
                            span.set_attribute("llm.streaming.tokens_per_second", tokens_per_second)
                        
                        # Create response object
                        llm_response = LLMResponse(
                            content=accumulated_content,
                            finish_reason="stop"  # Assume successful completion
                        )
                        span.set_llm_response(llm_response)
                        
                        span.set_status("ok")
                        
                    except Exception as e:
                        span.record_exception(e)
                        raise
            
            return cast(F, async_wrapper)
        
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                current_tracer = tracer or get_current_tracer()
                if not current_tracer:
                    logger.warning("No tracer available for streaming LLM tracing")
                    for item in func(*args, **kwargs):
                        yield item
                    return
                
                # Extract LLM parameters from function arguments
                llm_request = _extract_llm_request(
                    func, args, kwargs, operation_type, ai_system, model
                )
                
                span = current_tracer.start_llm_span(
                    operation_name=span_name,
                    request=llm_request,
                    attributes=span_attributes
                )
                
                with SpanContext(span):
                    try:
                        start_time = time.time()
                        first_token_time = None
                        token_count = 0
                        accumulated_content = ""
                        
                        for chunk in func(*args, **kwargs):
                            if first_token_time is None:
                                first_token_time = time.time()
                                ttft_ms = (first_token_time - start_time) * 1000
                                span.set_attribute("llm.time_to_first_token_ms", ttft_ms)
                            
                            # Accumulate streaming content
                            if hasattr(chunk, 'choices') and chunk.choices:
                                delta = chunk.choices[0].delta
                                if hasattr(delta, 'content') and delta.content:
                                    accumulated_content += delta.content
                                    token_count += 1
                            
                            yield chunk
                        
                        end_time = time.time()
                        
                        # Set final metrics
                        span.set_attribute("llm.latency_ms", (end_time - start_time) * 1000)
                        span.set_attribute("llm.streaming.token_count", token_count)
                        
                        if token_count > 0:
                            tokens_per_second = token_count / (end_time - start_time)
                            span.set_attribute("llm.streaming.tokens_per_second", tokens_per_second)
                        
                        # Create response object
                        llm_response = LLMResponse(
                            content=accumulated_content,
                            finish_reason="stop"  # Assume successful completion
                        )
                        span.set_llm_response(llm_response)
                        
                        span.set_status("ok")
                        
                    except Exception as e:
                        span.record_exception(e)
                        raise
            
            return cast(F, sync_wrapper)
    
    return decorator


def _capture_function_args(span: Span, func: Callable, args: tuple, kwargs: dict) -> None:
    """Capture function arguments as span attributes."""
    try:
        # Get function signature
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        # Add arguments as attributes (with size limits)
        for name, value in bound_args.arguments.items():
            attr_name = f"function.arg.{name}"
            attr_value = _safe_serialize_value(value)
            span.set_attribute(attr_name, attr_value)
            
    except Exception as e:
        logger.debug(f"Failed to capture function arguments: {e}")


def _capture_function_result(span: Span, result: Any) -> None:
    """Capture function result as span attribute."""
    try:
        attr_value = _safe_serialize_value(result)
        span.set_attribute("function.result", attr_value)
    except Exception as e:
        logger.debug(f"Failed to capture function result: {e}")


def _safe_serialize_value(value: Any, max_length: int = 1024) -> str:
    """Safely serialize a value for span attributes."""
    try:
        if isinstance(value, (str, int, float, bool)):
            result = str(value)
        elif isinstance(value, (list, tuple)):
            result = f"[{len(value)} items]"
        elif isinstance(value, dict):
            result = f"{{dict with {len(value)} keys}}"
        else:
            result = f"<{type(value).__name__}>"
        
        # Truncate if too long
        if len(result) > max_length:
            result = result[:max_length-3] + "..."
        
        return result
    except Exception:
        return "<serialization_error>"


def _extract_llm_request(
    func: Callable,
    args: tuple,
    kwargs: dict,
    operation_type: Optional[OperationType],
    ai_system: Optional[AISystem],
    model: Optional[str]
) -> LLMRequest:
    """Extract LLM request information from function arguments."""
    # Try to extract from common parameter names
    extracted_model = model or kwargs.get('model') or kwargs.get('engine')
    extracted_operation = operation_type or OperationType.CUSTOM
    extracted_system = ai_system or AISystem.CUSTOM
    
    # Try to infer from function name
    if not model and hasattr(func, '__name__'):
        func_name = func.__name__.lower()
        if 'chat' in func_name:
            extracted_operation = OperationType.CHAT
        elif 'completion' in func_name:
            extracted_operation = OperationType.COMPLETION
        elif 'embedding' in func_name:
            extracted_operation = OperationType.EMBEDDING
    
    return LLMRequest(
        operation_type=extracted_operation,
        ai_system=extracted_system,
        model=extracted_model or "unknown",
        prompt=kwargs.get('prompt'),
        messages=kwargs.get('messages'),
    )


def _extract_llm_response(result: Any) -> LLMResponse:
    """Extract LLM response information from function result."""
    response = LLMResponse()
    
    try:
        # Handle OpenAI-style responses
        if hasattr(result, 'choices') and result.choices:
            choice = result.choices[0]
            if hasattr(choice, 'message'):
                response.content = choice.message.content
            elif hasattr(choice, 'text'):
                response.content = choice.text
            
            if hasattr(choice, 'finish_reason'):
                response.finish_reason = choice.finish_reason
        
        # Handle usage information
        if hasattr(result, 'usage'):
            from ..types import TokenUsage
            response.token_usage = TokenUsage(
                input_tokens=getattr(result.usage, 'prompt_tokens', None),
                output_tokens=getattr(result.usage, 'completion_tokens', None),
                total_tokens=getattr(result.usage, 'total_tokens', None),
            )
        
        # Handle model information
        if hasattr(result, 'model'):
            response.model = result.model
        
        # Handle response ID
        if hasattr(result, 'id'):
            response.response_id = result.id
            
    except Exception as e:
        logger.debug(f"Failed to extract LLM response: {e}")
    
    return response

