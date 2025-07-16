"""
Auto-instrumentation for OpenAI SDK.
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
    """Check if OpenAI instrumentation is enabled."""
    return _instrumentation_enabled


def instrument_openai() -> None:
    """Enable OpenAI auto-instrumentation."""
    global _instrumentation_enabled

    if _instrumentation_enabled:
        return

    try:
        import openai

        # Store original methods
        _original_methods["chat_completions_create"] = (
            openai.resources.chat.completions.Completions.create
        )

        # Patch methods
        openai.resources.chat.completions.Completions.create = (  # type: ignore[method-assign]
            _wrap_chat_completions_create(
                openai.resources.chat.completions.Completions.create
            )
        )

        _instrumentation_enabled = True
        logger.info("OpenAI auto-instrumentation enabled")

    except ImportError:
        logger.warning("OpenAI not installed, skipping auto-instrumentation")
    except Exception as e:
        logger.error(f"Failed to instrument OpenAI: {e}")


def uninstrument_openai() -> None:
    """Disable OpenAI auto-instrumentation."""
    global _instrumentation_enabled

    if not _instrumentation_enabled:
        return

    try:
        import openai

        # Restore original methods
        if "chat_completions_create" in _original_methods:
            openai.resources.chat.completions.Completions.create = _original_methods[  # type: ignore[method-assign]
                "chat_completions_create"
            ]

        _instrumentation_enabled = False
        logger.info("OpenAI auto-instrumentation disabled")

    except ImportError:
        pass
    except Exception as e:
        logger.error(f"Failed to uninstrument OpenAI: {e}")


def _wrap_chat_completions_create(original_method: Any) -> Any:
    """Wrap OpenAI chat completions create method."""

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
            name="openai.chat.completions.create",
            model=model,
            operation="chat",
            ai_system=AISystem.OPENAI,
        )

        try:
            # Add request attributes
            span.set_attribute("llm.request.model", model)
            span.set_attribute("llm.request.messages", len(messages))

            # Add optional parameters
            for param in [
                "temperature",
                "max_tokens",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
            ]:
                if param in kwargs:
                    span.set_attribute(f"llm.request.{param}", kwargs[param])

            # Add OpenTelemetry semantic attributes
            span.set_attribute("gen_ai.system", "openai")
            span.set_attribute("gen_ai.operation.name", "chat")
            span.set_attribute("gen_ai.request.model", model)

            # Extract message data
            messages = kwargs.get("messages", [])
            if messages:
                llm_messages = [
                    Message(
                        role=msg.get("role", "user"),
                        content=msg.get("content", ""),
                        name=msg.get("name"),
                        tool_calls=msg.get("tool_calls"),
                        tool_call_id=msg.get("tool_call_id"),
                    )
                    for msg in messages
                ]

                llm_request = LLMRequest(
                    model=model,
                    messages=llm_messages,
                    temperature=kwargs.get("temperature"),
                    max_tokens=kwargs.get("max_tokens"),
                    top_p=kwargs.get("top_p"),
                    frequency_penalty=kwargs.get("frequency_penalty"),
                    presence_penalty=kwargs.get("presence_penalty"),
                    stop=kwargs.get("stop"),
                    stream=kwargs.get("stream", False),
                    tools=kwargs.get("tools"),
                    tool_choice=kwargs.get("tool_choice"),
                )

                span.set_llm_request(llm_request)

            # Execute the original method
            start_time = time.time()
            response = original_method(self, *args, **kwargs)
            end_time = time.time()

            # Add timing
            span.set_attribute("llm.latency_ms", (end_time - start_time) * 1000)

            # Process response
            if hasattr(response, "choices") and response.choices:
                choice = response.choices[0]

                # Add response attributes
                span.set_attribute(
                    "gen_ai.response.model", getattr(response, "model", model)
                )
                span.set_attribute("gen_ai.response.id", getattr(response, "id", ""))

                if hasattr(choice, "finish_reason"):
                    span.set_attribute(
                        "gen_ai.response.finish_reasons", [choice.finish_reason]
                    )

                # Add usage information
                if hasattr(response, "usage") and response.usage:
                    usage = response.usage
                    span.set_attribute(
                        "gen_ai.usage.input_tokens", getattr(usage, "prompt_tokens", 0)
                    )
                    span.set_attribute(
                        "gen_ai.usage.output_tokens",
                        getattr(usage, "completion_tokens", 0),
                    )

                    # Create token usage
                    token_usage = TokenUsage(
                        prompt_tokens=getattr(usage, "prompt_tokens", 0),
                        completion_tokens=getattr(usage, "completion_tokens", 0),
                        total_tokens=getattr(usage, "total_tokens", 0),
                    )

                    # Create LLM response
                    llm_response = LLMResponse(
                        id=getattr(response, "id", ""),
                        model=getattr(response, "model", model),
                        choices=[
                            {
                                "index": getattr(choice, "index", 0),
                                "message": {
                                    "role": getattr(
                                        choice.message, "role", "assistant"
                                    ),
                                    "content": getattr(choice.message, "content", ""),
                                    "tool_calls": getattr(
                                        choice.message, "tool_calls", None
                                    ),
                                },
                                "finish_reason": getattr(choice, "finish_reason", None),
                            }
                        ],
                        usage=token_usage,
                        created=getattr(response, "created", None),
                        system_fingerprint=getattr(
                            response, "system_fingerprint", None
                        ),
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
    "instrument_openai",
    "uninstrument_openai",
]
