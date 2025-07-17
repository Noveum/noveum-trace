"""
OpenAI integration for Noveum Trace SDK.

This module provides automatic instrumentation for OpenAI API calls with
comprehensive model support, environment-specific configuration handling,
and robust error handling for proxy services and alternative endpoints.
"""

import logging
import time
from typing import Any

from noveum_trace.core.span import SpanStatus
from noveum_trace.utils.exceptions import TracingError
from noveum_trace.utils.llm_utils import (
    estimate_cost,
    estimate_token_count,
    extract_llm_metadata,
    get_model_info,
    get_supported_models,
    validate_model_compatibility,
)

logger = logging.getLogger(__name__)

_original_create = None
_original_async_create = None
_patched = False


def patch_openai() -> None:
    """
    Patch OpenAI client to automatically trace API calls.

    This function monkey-patches the OpenAI client to automatically
    add tracing to all completion calls with enhanced error handling
    for environment-specific configurations and proxy services.

    Raises:
        TracingError: If patching fails
    """
    global _patched, _original_create, _original_async_create

    if _patched:
        logger.warning("OpenAI is already patched")
        return

    try:
        import openai
    except ImportError as e:
        raise TracingError(
            "OpenAI package not found. Install with: pip install openai"
        ) from e

    try:
        # Patch the chat completions create method
        if hasattr(openai, "OpenAI"):
            # OpenAI v1.x
            _patch_openai_v1(openai)
        else:
            # OpenAI v0.x (legacy)
            _patch_openai_legacy(openai)

        _patched = True
        logger.info("OpenAI integration patched successfully")

    except Exception as e:
        raise TracingError(f"Failed to patch OpenAI: {e}") from e


def unpatch_openai() -> None:
    """
    Remove OpenAI patching and restore original functionality.
    """
    global _patched, _original_create, _original_async_create

    if not _patched:
        return

    try:
        import openai

        # Restore original methods
        if _original_create and hasattr(openai, "OpenAI"):
            openai.OpenAI.chat.completions.create = _original_create

        if _original_async_create and hasattr(openai, "AsyncOpenAI"):
            openai.AsyncOpenAI.chat.completions.create = _original_async_create

        _patched = False
        _original_create = None
        _original_async_create = None
        logger.info("OpenAI integration unpatched")

    except Exception as e:
        logger.error(f"Failed to unpatch OpenAI: {e}")


def _patch_openai_v1(openai_module: Any) -> None:
    """Patch OpenAI v1.x client."""
    global _original_create, _original_async_create

    # Store original methods
    _original_create = openai_module.OpenAI.chat.completions.create

    if hasattr(openai_module, "AsyncOpenAI"):
        _original_async_create = openai_module.AsyncOpenAI.chat.completions.create

    # Create traced wrapper for sync client
    def traced_create(self: Any, **kwargs: Any) -> Any:
        """Traced version of OpenAI chat completions create."""
        return _create_traced_completion(self, _original_create, **kwargs)

    # Create traced wrapper for async client
    async def traced_async_create(self: Any, **kwargs: Any) -> Any:
        """Traced version of OpenAI async chat completions create."""
        return await _create_traced_async_completion(
            self, _original_async_create, **kwargs
        )

    # Apply patches
    openai_module.OpenAI.chat.completions.create = traced_create

    if hasattr(openai_module, "AsyncOpenAI"):
        openai_module.AsyncOpenAI.chat.completions.create = traced_async_create


def _patch_openai_legacy(openai_module: Any) -> None:
    """Patch OpenAI v0.x (legacy) client."""
    # Implementation for legacy OpenAI client
    # This would patch the older API format if needed
    logger.warning("Legacy OpenAI client detected - limited tracing support")


def _create_traced_completion(client: Any, original_func: Any, **kwargs: Any) -> Any:
    """Create a traced completion with enhanced error handling."""

    # Import here to avoid circular imports
    from noveum_trace import get_client, is_initialized
    from noveum_trace.core.context import get_current_trace

    # Check if SDK is initialized
    if not is_initialized():
        return original_func(client, **kwargs)

    # Extract and validate model information
    model = kwargs.get("model", "unknown")
    messages = kwargs.get("messages", [])

    # Validate model compatibility and get suggestions
    validation = validate_model_compatibility(model, messages)

    noveum_client = get_client()

    # Auto-create trace if none exists
    auto_created_trace = False
    current_trace = get_current_trace()
    if current_trace is None:
        auto_created_trace = True
        current_trace = noveum_client.start_trace(
            name=f"openai_completion_{model}",
            attributes={
                "auto_created": True,
                "function": "openai.chat.completions.create",
                "type": "llm_call",
                "provider": "openai",
            },
        )

    # Create span attributes
    attributes = _build_span_attributes(kwargs, validation)

    # Start the span
    span = noveum_client.start_span(
        name="openai_chat_completion",
        attributes=attributes,
    )

    try:
        start_time = time.time()

        # Execute the function
        result = original_func(client, **kwargs)

        end_time = time.time()
        duration = end_time - start_time

        # Extract metadata from response
        response_metadata = extract_llm_metadata(result)
        span.set_attributes(response_metadata)

        # Calculate and set cost information
        _set_cost_attributes(span, result, model)

        # Set performance metrics
        span.set_attribute("llm.duration_ms", round(duration * 1000, 2))

        # Set success status
        span.set_status(SpanStatus.OK)

        return result

    except Exception as e:
        # Enhanced error handling for environment-specific issues
        _handle_openai_error(span, e, model, validation)
        raise

    finally:
        # Always finish the span
        noveum_client.finish_span(span)

        # Finish auto-created trace
        if auto_created_trace and current_trace:
            noveum_client.finish_trace(current_trace)


async def _create_traced_async_completion(
    client: Any, original_func: Any, **kwargs: Any
) -> Any:
    """Create a traced async completion with enhanced error handling."""

    # Import here to avoid circular imports
    from noveum_trace import get_client, is_initialized
    from noveum_trace.core.context import get_current_trace

    # Check if SDK is initialized
    if not is_initialized():
        return await original_func(client, **kwargs)

    # Extract and validate model information
    model = kwargs.get("model", "unknown")
    messages = kwargs.get("messages", [])

    # Validate model compatibility and get suggestions
    validation = validate_model_compatibility(model, messages)

    noveum_client = get_client()

    # Auto-create trace if none exists
    auto_created_trace = False
    current_trace = get_current_trace()
    if current_trace is None:
        auto_created_trace = True
        current_trace = noveum_client.start_trace(
            name=f"openai_async_completion_{model}",
            attributes={
                "auto_created": True,
                "function": "openai.chat.completions.acreate",
                "type": "llm_call",
                "provider": "openai",
            },
        )

    # Create span attributes
    attributes = _build_span_attributes(kwargs, validation)
    attributes["llm.async"] = True

    # Start the span
    span = noveum_client.start_span(
        name="openai_async_chat_completion",
        attributes=attributes,
    )

    try:
        start_time = time.time()

        # Execute the function
        result = await original_func(client, **kwargs)

        end_time = time.time()
        duration = end_time - start_time

        # Extract metadata from response
        response_metadata = extract_llm_metadata(result)
        span.set_attributes(response_metadata)

        # Calculate and set cost information
        _set_cost_attributes(span, result, model)

        # Set performance metrics
        span.set_attribute("llm.duration_ms", round(duration * 1000, 2))

        # Set success status
        span.set_status(SpanStatus.OK)

        return result

    except Exception as e:
        # Enhanced error handling for environment-specific issues
        _handle_openai_error(span, e, model, validation)
        raise

    finally:
        # Always finish the span
        noveum_client.finish_span(span)

        # Finish auto-created trace
        if auto_created_trace and current_trace:
            noveum_client.finish_trace(current_trace)


def _build_span_attributes(
    kwargs: dict[str, Any], validation: dict[str, Any]
) -> dict[str, Any]:
    """Build comprehensive span attributes from request parameters and validation."""

    attributes = {
        "llm.provider": "openai",
        "llm.operation_type": "completion",
        "llm.request_type": "chat",
    }

    # Extract basic parameters
    model = kwargs.get("model", "unknown")
    attributes["llm.model"] = model

    # Add model information if available
    if validation.get("model_info"):
        model_info = validation["model_info"]
        attributes.update(
            {
                "llm.context_window": model_info.context_window,
                "llm.max_output_tokens": model_info.max_output_tokens,
                "llm.supports_vision": model_info.supports_vision,
                "llm.supports_function_calling": model_info.supports_function_calling,
            }
        )
        if model_info.training_cutoff:
            attributes["llm.training_cutoff"] = model_info.training_cutoff

    # Extract standard OpenAI parameters
    param_mappings = {
        "temperature": "llm.temperature",
        "max_tokens": "llm.max_tokens",
        "top_p": "llm.top_p",
        "frequency_penalty": "llm.frequency_penalty",
        "presence_penalty": "llm.presence_penalty",
        "stop": "llm.stop_sequences",
        "stream": "llm.stream",
        "n": "llm.n_choices",
        "user": "llm.user_id",
    }

    for param, attr_name in param_mappings.items():
        if param in kwargs and kwargs[param] is not None:
            attributes[attr_name] = kwargs[param]

    # Handle messages and prompts
    messages = kwargs.get("messages", [])
    if messages:
        # Estimate input tokens
        input_tokens = estimate_token_count(messages)
        attributes["llm.usage.estimated_input_tokens"] = str(input_tokens)

        # Extract message roles and count
        roles = [
            msg.get("role", "unknown") for msg in messages if isinstance(msg, dict)
        ]
        attributes["llm.message_count"] = str(len(messages))
        attributes["llm.message_roles"] = ",".join(roles)

        # Check for vision content
        has_vision = any(
            isinstance(msg.get("content"), list)
            and any(
                isinstance(item, dict) and item.get("type") == "image_url"
                for item in msg["content"]
            )
            for msg in messages
            if isinstance(msg, dict)
        )
        if has_vision:
            attributes["llm.has_vision_content"] = str(True)

    # Handle function calling
    if "functions" in kwargs or "tools" in kwargs:
        attributes["llm.has_function_calls"] = str(True)
        if "functions" in kwargs:
            attributes["llm.function_count"] = str(len(kwargs["functions"]))
        if "tools" in kwargs:
            attributes["llm.tool_count"] = str(len(kwargs["tools"]))

    # Add validation warnings as attributes
    if validation.get("warnings"):
        attributes["llm.validation_warnings"] = "; ".join(validation["warnings"])

    if validation.get("suggestions"):
        attributes["llm.model_suggestions"] = ",".join(validation["suggestions"])

    return attributes


def _set_cost_attributes(span: Any, response: Any, model: str) -> None:
    """Set cost-related attributes on the span."""

    # Extract token usage from response
    input_tokens = 0
    output_tokens = 0

    if hasattr(response, "usage"):
        usage = response.usage
        input_tokens = getattr(usage, "prompt_tokens", 0) or getattr(
            usage, "input_tokens", 0
        )
        output_tokens = getattr(usage, "completion_tokens", 0) or getattr(
            usage, "output_tokens", 0
        )

    if input_tokens > 0 or output_tokens > 0:
        # Calculate cost
        cost_info = estimate_cost(model, input_tokens, output_tokens)

        span.set_attributes(
            {
                "llm.cost.input": cost_info["input_cost"],
                "llm.cost.output": cost_info["output_cost"],
                "llm.cost.total": cost_info["total_cost"],
                "llm.cost.currency": cost_info["currency"],
                "llm.cost.input_rate_per_1m": cost_info["input_cost_per_1m"],
                "llm.cost.output_rate_per_1m": cost_info["output_cost_per_1m"],
            }
        )


def _handle_openai_error(
    span: Any, error: Exception, model: str, validation: dict[str, Any]
) -> None:
    """Handle OpenAI-specific errors with enhanced diagnostics."""

    error_type = type(error).__name__
    error_message = str(error)

    # Standard error attributes
    span.set_attributes(
        {
            "llm.error.type": error_type,
            "llm.error.message": error_message,
        }
    )

    # Enhanced error handling for specific cases
    if "400" in error_message and "Unsupported model" in error_message:
        # Handle the specific environment compatibility issue
        span.set_attributes(
            {
                "llm.error.category": "model_availability",
                "llm.error.is_environment_specific": True,
            }
        )

        # Extract allowed models from error message if available
        if "Only the following models are allowed:" in error_message:
            allowed_models_text = error_message.split(
                "Only the following models are allowed:"
            )[-1]
            # Parse allowed models (they're typically comma-separated)
            allowed_models = [
                m.strip().strip("'\"") for m in allowed_models_text.split(",")
            ]
            allowed_models = [m for m in allowed_models if m and not m.startswith("{")]

            if allowed_models:
                span.set_attribute("llm.error.allowed_models", ",".join(allowed_models))

                # Check if any allowed models are in our registry
                supported_allowed = []
                for allowed_model in allowed_models:
                    if get_model_info(allowed_model):
                        supported_allowed.append(allowed_model)

                if supported_allowed:
                    span.set_attribute(
                        "llm.error.compatible_alternatives", ",".join(supported_allowed)
                    )

            # Add guidance for proxy/alternative endpoint usage
            if any(
                provider in allowed_models_text.lower()
                for provider in ["gemini", "claude", "llama"]
            ):
                span.set_attribute("llm.error.detected_proxy_service", True)
                span.set_attribute(
                    "llm.error.guidance",
                    "API key appears to be configured for a proxy service with multi-provider access",
                )

    elif "401" in error_message or "authentication" in error_message.lower():
        span.set_attributes(
            {
                "llm.error.category": "authentication",
                "llm.error.guidance": "Check API key configuration and permissions",
            }
        )

    elif "429" in error_message or "rate limit" in error_message.lower():
        span.set_attributes(
            {
                "llm.error.category": "rate_limit",
                "llm.error.guidance": "Request rate exceeded, implement backoff strategy",
            }
        )

    elif "403" in error_message or "forbidden" in error_message.lower():
        span.set_attributes(
            {
                "llm.error.category": "authorization",
                "llm.error.guidance": "API key lacks permissions for this model or operation",
            }
        )

    elif "500" in error_message or "502" in error_message or "503" in error_message:
        span.set_attributes(
            {
                "llm.error.category": "server_error",
                "llm.error.guidance": "OpenAI service issue, retry with exponential backoff",
            }
        )

    # Add model validation context to error
    if not validation.get("valid"):
        span.set_attribute("llm.error.model_validation_failed", True)
        if validation.get("suggestions"):
            span.set_attribute(
                "llm.error.suggested_models", ",".join(validation["suggestions"])
            )

    # Set error status
    span.set_status(SpanStatus.ERROR, error_message)

    # Record the exception
    span.record_exception(error)


def is_patched() -> bool:
    """
    Check if OpenAI integration is currently patched.

    Returns:
        True if patched, False otherwise
    """
    return _patched


def get_integration_info() -> dict[str, Any]:
    """
    Get comprehensive information about the OpenAI integration.

    Returns:
        Dictionary with integration information
    """
    try:
        import openai

        openai_version = getattr(openai, "__version__", "unknown")
    except ImportError:
        openai_version = None

    # Get supported OpenAI models
    supported_models = get_supported_models("openai")

    info = {
        "name": "openai",
        "patched": _patched,
        "openai_version": openai_version,
        "supported": openai_version is not None,
        "supported_models_count": len(supported_models),
        "latest_models": supported_models[:10] if supported_models else [],
        "features": {
            "sync_tracing": True,
            "async_tracing": (
                hasattr(openai, "AsyncOpenAI") if openai_version else False
            ),
            "cost_estimation": True,
            "token_counting": True,
            "error_diagnostics": True,
            "model_validation": True,
            "proxy_detection": True,
        },
    }

    # Add environment diagnostics
    if openai_version:
        try:
            # Try to detect client configuration without making API calls
            info["environment"] = {
                "can_import_openai": True,
                "has_async_support": hasattr(openai, "AsyncOpenAI"),
                "client_classes": [cls for cls in dir(openai) if "OpenAI" in cls],
            }
        except Exception as e:
            info["environment"] = {
                "can_import_openai": True,
                "error": str(e),
            }

    return info


def diagnose_environment() -> dict[str, Any]:
    """
    Diagnose the current environment for OpenAI compatibility issues.

    Returns:
        Dictionary with diagnostic information
    """
    diagnosis: dict[str, Any] = {
        "timestamp": time.time(),
        "checks": {},
        "recommendations": [],
    }

    # Check OpenAI package
    try:
        import openai

        diagnosis["checks"]["openai_import"] = {
            "status": "success",
            "version": getattr(openai, "__version__", "unknown"),
        }
    except ImportError as e:
        diagnosis["checks"]["openai_import"] = {
            "status": "error",
            "error": str(e),
        }
        diagnosis["recommendations"].append(
            "Install OpenAI package: pip install openai"
        )
        return diagnosis

    # Check for supported models in registry
    supported_models = get_supported_models("openai")
    diagnosis["checks"]["model_registry"] = {
        "status": "success",
        "count": len(supported_models),
        "sample_models": supported_models[:5],
    }

    # Check client initialization (without API call)
    try:
        client = openai.OpenAI(api_key="test-key-for-init-check")
        diagnosis["checks"]["client_init"] = {
            "status": "success",
            "client_type": type(client).__name__,
        }
    except Exception as e:
        diagnosis["checks"]["client_init"] = {
            "status": "error",
            "error": str(e),
        }

    # Check async client if available
    if hasattr(openai, "AsyncOpenAI"):
        try:
            async_client = openai.AsyncOpenAI(api_key="test-key-for-init-check")
            diagnosis["checks"]["async_client_init"] = {
                "status": "success",
                "client_type": type(async_client).__name__,
            }
        except Exception as e:
            diagnosis["checks"]["async_client_init"] = {
                "status": "error",
                "error": str(e),
            }

    # Add recommendations based on checks
    if all(check.get("status") == "success" for check in diagnosis["checks"].values()):
        diagnosis["recommendations"].append(
            "Environment appears compatible with OpenAI integration"
        )
    else:
        failed_checks = [
            name
            for name, check in diagnosis["checks"].items()
            if check.get("status") == "error"
        ]
        diagnosis["recommendations"].append(
            f"Address failed checks: {', '.join(failed_checks)}"
        )

    return diagnosis
