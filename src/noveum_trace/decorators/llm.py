"""
LLM-specific trace decorator for Noveum Trace SDK.

This module provides the @trace_llm decorator specifically designed
for tracing LLM interactions with automatic metadata capture.
"""

import functools
import inspect
import json
from typing import Any, Callable, Optional, Union

from noveum_trace.core.context import attach_context_to_span
from noveum_trace.core.span import SpanStatus
from noveum_trace.decorators.base import _serialize_value
from noveum_trace.utils.llm_utils import (
    detect_llm_provider,
    estimate_cost,
    estimate_token_count,
)


def trace_llm(
    func: Optional[Callable[..., Any]] = None,
    *,
    name: Optional[str] = None,
    provider: Optional[str] = None,
    capture_prompts: bool = True,
    capture_completions: bool = True,
    capture_tokens: bool = True,
    estimate_costs: bool = True,
    redact_pii: bool = False,
    metadata: Optional[dict[str, Any]] = None,
    tags: Optional[dict[str, str]] = None,
) -> Union[Callable[..., Any], Callable[[Callable[..., Any]], Callable[..., Any]]]:
    """
    Decorator to add LLM-specific tracing to functions.

    This decorator automatically captures LLM-specific metadata including
    prompts, completions, token usage, costs, and model information.

    Args:
        func: Function to decorate (when used as @trace_llm)
        name: Custom span name (defaults to function name)
        provider: LLM provider name (e.g., 'openai', 'anthropic')
        capture_prompts: Whether to capture input prompts
        capture_completions: Whether to capture LLM responses
        capture_tokens: Whether to capture token usage information
        estimate_costs: Whether to estimate API costs
        redact_pii: Whether to redact personally identifiable information
        metadata: Additional metadata to attach to the span
        tags: Tags to add to the span

    Returns:
        Decorated function

    Example:
        >>> @trace_llm
        >>> def call_openai(prompt: str) -> str:
        ...     response = openai.chat.completions.create(
        ...         model="gpt-4",
        ...         messages=[{"role": "user", "content": prompt}]
        ...     )
        ...     return response.choices[0].message.content

        >>> @trace_llm(capture_tokens=True, estimate_costs=True)
        >>> def call_anthropic(prompt: str) -> str:
        ...     response = anthropic.messages.create(
        ...         model="claude-3-opus-20240229",
        ...         messages=[{"role": "user", "content": prompt}]
        ...     )
        ...     return response.content[0].text
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
                        "type": "llm_call",
                    },
                )

            # Create span attributes
            attributes: dict[str, Any] = {
                "function.name": func_name,
                "function.module": func.__module__,
                "function.qualname": func.__qualname__,
                "function.type": "llm_call",
                "llm.operation_type": "completion",
            }

            # Add metadata
            if metadata:
                for key, value in metadata.items():
                    attributes[f"metadata.{key}"] = value

            # Add tags
            if tags:
                for key, value in tags.items():
                    attributes[f"tag.{key}"] = value

            # Extract LLM parameters from function arguments
            try:
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()

                # Detect LLM provider and extract metadata
                llm_metadata = _extract_llm_call_metadata(bound_args.arguments)
                attributes.update(llm_metadata)

                # Capture prompts if enabled
                if capture_prompts:
                    prompts = _extract_prompts(bound_args.arguments)
                    if prompts:
                        if redact_pii:
                            prompts = _redact_pii(prompts)
                        attributes["llm.prompts"] = prompts

                        # Estimate input tokens
                        if capture_tokens:
                            input_tokens = estimate_token_count(prompts)
                            attributes["llm.usage.input_tokens"] = input_tokens

            except Exception as e:
                attributes["llm.extraction_error"] = str(e)

            # Start the span
            span = client.start_span(
                name=func_name,
                attributes=attributes,
            )

            # Attach context to span
            attach_context_to_span(span)

            try:
                # Execute the function
                result = func(*args, **kwargs)

                # Capture completions if enabled
                if capture_completions:
                    completion = _extract_completion(result)
                    if completion:
                        if redact_pii:
                            completion = _redact_pii(completion)
                        span.set_attribute("llm.completion", completion)

                        # Estimate output tokens
                        if capture_tokens:
                            output_tokens = estimate_token_count(completion)
                            span.set_attribute("llm.usage.output_tokens", output_tokens)

                            # Calculate total tokens
                            input_tokens = span.attributes.get(
                                "llm.usage.input_tokens", 0
                            )
                            total_tokens = input_tokens + output_tokens
                            span.set_attribute("llm.usage.total_tokens", total_tokens)

                # Estimate costs if enabled
                if estimate_costs and capture_tokens:
                    cost_info = _estimate_llm_cost(span.attributes)
                    if cost_info:
                        span.set_attributes(cost_info)

                # Set success status
                span.set_status(SpanStatus.OK)

                return result

            except Exception as e:
                # Handle LLM-specific errors
                span.record_exception(e)
                span.set_status(SpanStatus.ERROR, str(e))

                # Add LLM-specific error attributes
                span.set_attributes(
                    {
                        "llm.error.type": type(e).__name__,
                        "llm.error.message": str(e),
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
        wrapper._noveum_trace_type = "llm"  # type: ignore
        wrapper._noveum_trace_config = {  # type: ignore
            "name": func_name,
            "capture_prompts": capture_prompts,
            "capture_completions": capture_completions,
            "capture_tokens": capture_tokens,
            "estimate_costs": estimate_costs,
            "redact_pii": redact_pii,
            "metadata": metadata,
            "tags": tags,
        }

        return wrapper

    # Handle both @trace_llm and @trace_llm() usage
    if func is None:
        # Called as @trace_llm() with arguments
        return decorator
    else:
        # Called as @trace_llm without arguments
        return decorator(func)


def _extract_llm_call_metadata(arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Extract LLM metadata from function arguments.

    Args:
        arguments: Function arguments

    Returns:
        Dictionary of LLM metadata
    """
    metadata = {}

    # Common parameter names across LLM providers
    param_mappings = {
        "model": "llm.model",
        "temperature": "llm.temperature",
        "max_tokens": "llm.max_tokens",
        "top_p": "llm.top_p",
        "frequency_penalty": "llm.frequency_penalty",
        "presence_penalty": "llm.presence_penalty",
        "stop": "llm.stop_sequences",
    }

    for arg_name, arg_value in arguments.items():
        # Direct parameter mapping
        if arg_name in param_mappings:
            metadata[param_mappings[arg_name]] = arg_value

        # Handle nested parameters (e.g., in kwargs or config objects)
        elif isinstance(arg_value, dict):
            for key, value in arg_value.items():
                if key in param_mappings:
                    metadata[param_mappings[key]] = value

    # Detect provider from arguments
    provider = detect_llm_provider(arguments)
    if provider:
        metadata["llm.provider"] = provider

    return metadata


def _extract_prompts(arguments: dict[str, Any]) -> Optional[str]:
    """
    Extract prompts from function arguments.

    Args:
        arguments: Function arguments

    Returns:
        Extracted prompt text or None
    """
    # Common prompt parameter names
    prompt_params = ["prompt", "messages", "input", "text", "query"]

    for param_name, param_value in arguments.items():
        if param_name.lower() in prompt_params:
            return _serialize_prompt(param_value)

    return None


def _extract_completion(result: Any) -> Optional[str]:
    """
    Extract completion text from function result.

    Args:
        result: Function result

    Returns:
        Extracted completion text or None
    """
    if isinstance(result, str):
        return result
    elif hasattr(result, "choices") and result.choices:
        # OpenAI-style response
        choice = result.choices[0]
        if hasattr(choice, "message"):
            return choice.message.content
        elif hasattr(choice, "text"):
            return choice.text
    elif hasattr(result, "content"):
        # Anthropic-style response
        if isinstance(result.content, list) and result.content:
            return result.content[0].text
        else:
            return str(result.content)
    elif isinstance(result, dict):
        # Dictionary response - look for common keys
        for key in ["text", "content", "response", "output"]:
            if key in result:
                return str(result[key])

    return _serialize_value(result)


def _serialize_prompt(prompt: Any) -> str:
    """
    Serialize prompt data to string.

    Args:
        prompt: Prompt data

    Returns:
        Serialized prompt string
    """
    if isinstance(prompt, str):
        return prompt
    elif isinstance(prompt, list):
        # Handle messages format
        if all(isinstance(msg, dict) for msg in prompt):
            return json.dumps(prompt, indent=2)
        else:
            return str(prompt)
    elif isinstance(prompt, dict):
        return json.dumps(prompt, indent=2)
    else:
        return str(prompt)


def _estimate_llm_cost(attributes: dict[str, Any]) -> Optional[dict[str, Any]]:
    """
    Estimate LLM API cost based on usage.

    Args:
        attributes: Span attributes containing usage information

    Returns:
        Cost information dictionary or None
    """
    model = attributes.get("llm.model")
    input_tokens = attributes.get("llm.usage.input_tokens", 0)
    output_tokens = attributes.get("llm.usage.output_tokens", 0)

    if not model or not (input_tokens or output_tokens):
        return None

    try:
        cost_info = estimate_cost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        return {
            "llm.cost.input": cost_info.get("input_cost", 0),
            "llm.cost.output": cost_info.get("output_cost", 0),
            "llm.cost.total": cost_info.get("total_cost", 0),
            "llm.cost.currency": cost_info.get("currency", "USD"),
        }

    except Exception:
        return None


def _redact_pii(text: str) -> str:
    """
    Redact personally identifiable information from text.

    Args:
        text: Text to redact

    Returns:
        Text with PII redacted
    """
    # Import here to avoid circular imports
    from noveum_trace.utils.pii_redaction import redact_pii

    return redact_pii(text)
