"""
Auto-instrumentation for Anthropic SDK.
"""

import functools
import logging
import time
from typing import Any

from noveum_trace.core.tracer import get_current_tracer
from noveum_trace.types import (
    AISystem,
    LLMRequest,
    LLMResponse,
    Message,
    TokenUsage,
)

logger = logging.getLogger(__name__)

# Track if instrumentation is enabled
_instrumentation_enabled = False
_original_methods = {}


def is_instrumented() -> bool:
    """Check if Anthropic instrumentation is enabled."""
    return _instrumentation_enabled


def instrument_anthropic() -> None:
    """Enable Anthropic auto-instrumentation."""
    global _instrumentation_enabled

    if _instrumentation_enabled:
        return

    try:
        import anthropic

        # Store original methods
        _original_methods["messages_create"] = (
            anthropic.resources.messages.Messages.create
        )

        # Patch methods
        anthropic.resources.messages.Messages.create = _wrap_messages_create(  # type: ignore[method-assign]
            anthropic.resources.messages.Messages.create
        )

        _instrumentation_enabled = True
        logger.info("Anthropic auto-instrumentation enabled")

    except ImportError:
        logger.warning("Anthropic not installed, skipping auto-instrumentation")
    except Exception as e:
        logger.error(f"Failed to instrument Anthropic: {e}")


def uninstrument_anthropic() -> None:
    """Disable Anthropic auto-instrumentation."""
    global _instrumentation_enabled

    if not _instrumentation_enabled:
        return

    try:
        import anthropic

        # Restore original methods
        if "messages_create" in _original_methods:
            anthropic.resources.messages.Messages.create = _original_methods[  # type: ignore[method-assign]
                "messages_create"
            ]

        _instrumentation_enabled = False
        logger.info("Anthropic auto-instrumentation disabled")

    except ImportError:
        pass
    except Exception as e:
        logger.error(f"Failed to uninstrument Anthropic: {e}")


def _wrap_messages_create(original_method: Any) -> Any:
    """Wrap Anthropic messages create method."""

    @functools.wraps(original_method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        tracer = get_current_tracer()
        if not tracer:
            return original_method(self, *args, **kwargs)

        # Extract model and messages
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])

        # Create span
        span = tracer.create_llm_span(
            name="anthropic.messages.create",
            model=model,
            operation="chat",
            ai_system=AISystem.ANTHROPIC,
        )

        try:
            # Add request attributes
            span.set_attribute("llm.request.model", model)
            span.set_attribute("llm.request.messages", len(messages))

            # Add optional parameters
            for param in ["max_tokens", "temperature", "top_p", "top_k"]:
                if param in kwargs:
                    span.set_attribute(f"llm.request.{param}", kwargs[param])

            # Add OpenTelemetry semantic attributes
            span.set_attribute("gen_ai.system", "anthropic")
            span.set_attribute("gen_ai.operation.name", "chat")
            span.set_attribute("gen_ai.request.model", model)

            # Create LLM request
            if messages:
                llm_messages = [
                    Message(
                        role=msg.get("role", "user"), content=msg.get("content", "")
                    )
                    for msg in messages
                ]

                llm_request = LLMRequest(
                    model=model,
                    messages=llm_messages,
                    temperature=kwargs.get("temperature"),
                    max_tokens=kwargs.get("max_tokens"),
                    top_p=kwargs.get("top_p"),
                    stop=kwargs.get("stop_sequences"),
                )

                span.set_llm_request(llm_request)

            # Execute the original method
            start_time = time.time()
            response = original_method(self, *args, **kwargs)
            end_time = time.time()

            # Add timing
            span.set_attribute("llm.latency_ms", (end_time - start_time) * 1000)

            # Process response
            if hasattr(response, "content") and response.content:
                # Add response attributes
                span.set_attribute(
                    "gen_ai.response.model", getattr(response, "model", model)
                )
                span.set_attribute("gen_ai.response.id", getattr(response, "id", ""))

                if hasattr(response, "stop_reason"):
                    span.set_attribute(
                        "gen_ai.response.finish_reasons", [response.stop_reason]
                    )

                # Add usage information
                if hasattr(response, "usage") and response.usage:
                    usage = response.usage
                    span.set_attribute(
                        "gen_ai.usage.input_tokens", getattr(usage, "input_tokens", 0)
                    )
                    span.set_attribute(
                        "gen_ai.usage.output_tokens", getattr(usage, "output_tokens", 0)
                    )

                    # Create token usage
                    token_usage = TokenUsage(
                        prompt_tokens=getattr(usage, "input_tokens", 0),
                        completion_tokens=getattr(usage, "output_tokens", 0),
                        total_tokens=getattr(usage, "input_tokens", 0)
                        + getattr(usage, "output_tokens", 0),
                    )

                    # Extract content
                    content = ""
                    if (
                        response.content
                        and len(response.content) > 0
                        and hasattr(response.content[0], "text")
                    ):
                        content = response.content[0].text

                    # Create LLM response
                    llm_response = LLMResponse(
                        id=getattr(response, "id", ""),
                        model=getattr(response, "model", model),
                        choices=[
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": content},
                                "finish_reason": getattr(response, "stop_reason", None),
                            }
                        ],
                        usage=token_usage,
                    )

                    span.set_llm_response(llm_response)

            span.set_status("OK")
            return response

        except Exception as e:
            span.record_exception(e)
            raise
        finally:
            span.end()

    return wrapper


# Export functions
__all__ = [
    "instrument_anthropic",
    "uninstrument_anthropic",
]
