"""
LiveKit LLM integration for noveum-trace.

This module provides wrapper classes that automatically trace LiveKit LLM
operations, capturing comprehensive metadata including inputs, outputs, tokens,
costs, timings, and all sampling parameters.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional, Union

from noveum_trace.core.context import get_current_trace
from noveum_trace.core.span import SpanStatus
from noveum_trace.integrations.livekit.livekit_constants import (
    LLM_MODEL_DEFAULT_VALUE,
    LLM_PROVIDER_DEFAULT_VALUE,
    LLM_REQUEST_ID_DEFAULT_VALUE,
    LLM_RESPONSE_DEFAULT_VALUE,
    MAX_CONVERSATION_HISTORY,
)
from noveum_trace.integrations.livekit.livekit_utils import (
    create_constants_metadata,
    serialize_chat_history,
)

logger = logging.getLogger(__name__)

try:
    from livekit.agents.llm import LLM as BaseLLM
    from livekit.agents.llm import (
        ChatChunk,
        ChatContext,
        CompletionUsage,
        FunctionToolCall,
    )
    from livekit.agents.types import NOT_GIVEN

    LIVEKIT_AVAILABLE = True
except ImportError as e:
    LIVEKIT_AVAILABLE = False
    BaseLLM = object  # Fallback for when LiveKit is not available
    NOT_GIVEN = None
    logger.debug(
        "LiveKit is not importable. LiveKit LLM integration features will not be available. "
        "Install it with: pip install livekit livekit-agents",
        exc_info=e,
    )


class LiveKitLLMWrapper(BaseLLM):
    """
    Wrapper for LiveKit LLM providers that automatically creates spans for chat completions.

    This wrapper inherits from the base LLM class to ensure proper event forwarding
    and type compatibility. It captures comprehensive data including:
    - Full chat context (messages, system prompt)
    - All sampling parameters (temperature, top_p, etc.)
    - Complete response text and tool calls
    - Token usage (including cached tokens)
    - Timing metrics (TTFT, duration, tokens/sec)
    - Cost calculations
    - All metadata

    The wrapper forwards metrics_collected and error events from the base LLM instance
    to ensure that downstream listeners receive all events.

    Example:
        >>> import noveum_trace
        >>> from livekit.plugins import openai
        >>> from noveum_trace.integrations.livekit import LiveKitLLMWrapper
        >>>
        >>> # Initialize noveum-trace (done elsewhere)
        >>> noveum_trace.init(project="livekit-agents")
        >>>
        >>> # Wrap LLM provider
        >>> base_llm = openai.LLM(model="gpt-4", temperature=0.7)
        >>> traced_llm = LiveKitLLMWrapper(
        ...     llm=base_llm,
        ...     session_id="session_123",
        ...     job_context={"job_id": "job_abc"}
        ... )
        >>>
        >>> # Use normally
        >>> stream = traced_llm.chat(chat_ctx=my_context, tools=my_tools)
        >>> async for chunk in stream:
        ...     if chunk.delta and chunk.delta.content:
        ...         print(chunk.delta.content, end="")
    """

    def __init__(
        self,
        llm: Any,  # noqa: F811 - parameter shadows import
        session_id: str,
        job_context: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize LLM wrapper.

        Args:
            llm: Base LiveKit LLM provider instance
            session_id: Session identifier for organizing traces
            job_context: Dictionary of job context information to attach to spans
        """
        if not LIVEKIT_AVAILABLE:
            logger.error(
                "Cannot initialize LiveKitLLMWrapper: LiveKit is not available. "
                "Install it with: pip install livekit livekit-agents"
            )
            # Initialize with minimal state for graceful degradation
            self._base_llm = llm
            self._session_id = session_id
            self._job_context = job_context or {}
            return

        # Initialize base class (LLM has no required __init__ parameters)
        super().__init__()

        # Store wrapper-specific state
        self._base_llm = llm
        self._session_id = session_id
        self._job_context = job_context or {}

        # Forward metrics_collected and error events from base LLM to this wrapper
        # This ensures that agent_activity and other listeners receive the events
        self._base_llm.on("metrics_collected", self._forward_metrics)
        self._base_llm.on("error", self._forward_error)

    def _forward_metrics(self, metrics: Any) -> None:
        """Forward metrics_collected events from base LLM to wrapper listeners."""
        self.emit("metrics_collected", metrics)

    def _forward_error(self, error: Any) -> None:
        """Forward error events from base LLM to wrapper listeners."""
        self.emit("error", error)

    @property
    def model(self) -> str:
        """Get model name from base provider."""
        if not LIVEKIT_AVAILABLE or self._base_llm is None:
            return LLM_MODEL_DEFAULT_VALUE
        return getattr(self._base_llm, "model", LLM_MODEL_DEFAULT_VALUE)

    @property
    def provider(self) -> str:
        """Get provider name from base provider."""
        if not LIVEKIT_AVAILABLE or self._base_llm is None:
            return LLM_PROVIDER_DEFAULT_VALUE
        return getattr(self._base_llm, "provider", LLM_PROVIDER_DEFAULT_VALUE)

    @property
    def label(self) -> str:
        """Get label from base provider."""
        if not LIVEKIT_AVAILABLE or self._base_llm is None:
            return "LiveKitLLMWrapper"
        return getattr(self._base_llm, "label", self._base_llm.__class__.__name__)

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: Optional[list[Any]] = None,
        **kwargs: Any,
    ) -> _WrappedLLMStream:
        """
        Create a chat completion stream with tracing.

        Args:
            chat_ctx: Chat context with conversation history
            tools: Optional list of tools available to the LLM
            **kwargs: Additional arguments passed to base LLM

        Returns:
            Wrapped LLM stream that captures all data
        """
        base_stream = self._base_llm.chat(
            chat_ctx=chat_ctx, tools=tools or [], **kwargs
        )
        return _WrappedLLMStream(
            base_stream=base_stream,
            llm_wrapper=self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            kwargs=kwargs,
        )

    def prewarm(self) -> None:
        """Pre-warm connection to the LLM service."""
        if hasattr(self._base_llm, "prewarm"):
            self._base_llm.prewarm()

    async def aclose(self) -> None:
        """Close the LLM provider and unregister event handlers."""
        # Unregister event handlers to prevent memory leaks
        if LIVEKIT_AVAILABLE and hasattr(self._base_llm, "off"):
            try:
                self._base_llm.off("metrics_collected", self._forward_metrics)
                self._base_llm.off("error", self._forward_error)
            except Exception:
                pass  # Ignore errors during cleanup

        if hasattr(self._base_llm, "aclose"):
            await self._base_llm.aclose()

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to base LLM."""
        return getattr(self._base_llm, name)


class _WrappedLLMStream:
    """Wrapper for LLM streaming that captures all data and creates comprehensive spans."""

    def __init__(
        self,
        base_stream: Any,
        llm_wrapper: LiveKitLLMWrapper,
        chat_ctx: ChatContext,
        tools: list[Any],
        kwargs: dict[str, Any],
    ):
        self._base_stream = base_stream
        self._llm_wrapper = llm_wrapper
        self._chat_ctx = chat_ctx
        self._tools = tools
        self._kwargs = kwargs

        # Buffers for aggregating data
        self._buffered_chunks: list[ChatChunk] = []
        self._response_content = ""
        self._tool_calls: list[FunctionToolCall] = []
        self._usage: Optional[CompletionUsage] = None
        self._request_id: Optional[str] = None
        self._response_role: Optional[str] = None

        # Timing
        self._start_time = time.perf_counter()
        self._ttft: Optional[float] = None

        # State
        self._cancelled = False
        self._had_error = False
        self._span_created = False

    async def __anext__(self) -> ChatChunk:
        """
        Get next chat chunk from the stream.

        Returns:
            ChatChunk from the base stream
        """
        try:
            chunk = await self._base_stream.__anext__()

            # Record TTFT on first chunk with content
            if self._ttft is None and chunk.delta and chunk.delta.content:
                self._ttft = time.perf_counter() - self._start_time

            # Buffer chunk
            self._buffered_chunks.append(chunk)

            # Aggregate data
            if chunk.delta:
                if chunk.delta.content:
                    self._response_content += chunk.delta.content
                if chunk.delta.role:
                    self._response_role = str(chunk.delta.role)
                if chunk.delta.tool_calls:
                    self._tool_calls.extend(chunk.delta.tool_calls)

            # Store usage and request_id
            if chunk.usage:
                self._usage = chunk.usage
            if chunk.id:
                self._request_id = chunk.id

            return chunk  # Pass through unchanged

        except StopAsyncIteration:
            # Stream ended naturally - create span before raising
            await self._create_span()
            raise
        except Exception:
            self._had_error = True
            # Still create span on error to capture what we got
            await self._create_span()
            raise

    def __aiter__(self) -> _WrappedLLMStream:
        """Return self as async iterator."""
        return self

    async def __aenter__(self) -> _WrappedLLMStream:
        """Enter async context manager."""
        # If base stream is an async context manager, enter it
        if hasattr(self._base_stream, "__aenter__"):
            await self._base_stream.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: Union[type[BaseException], None],
        exc: Union[BaseException, None],
        exc_tb: Any,
    ) -> None:
        """Exit async context manager."""
        # If base stream is an async context manager, exit it
        if hasattr(self._base_stream, "__aexit__"):
            await self._base_stream.__aexit__(exc_type, exc, exc_tb)
        else:
            # Fallback to aclose if no context manager support
            await self.aclose()

    async def aclose(self) -> None:
        """Close the stream and create span if not already created."""
        # Create span if not already created
        if not self._span_created:
            await self._create_span()

        if hasattr(self._base_stream, "aclose"):
            await self._base_stream.aclose()

    async def _create_span(self) -> None:
        """Create comprehensive span with all captured data."""
        if self._span_created:
            return

        self._span_created = True

        # Calculate duration
        duration = time.perf_counter() - self._start_time

        # Get current trace
        trace = get_current_trace()
        if not trace:
            logger.debug("No active trace for LLM chat, skipping span creation")
            return

        try:
            from noveum_trace import get_client

            client = get_client()
            if not client:
                logger.debug("No client available, skipping span creation")
                return

            # Build ALL attributes
            attributes: dict[str, Any] = {}

            # 1. Model/Provider/Label
            attributes["llm.model"] = self._llm_wrapper.model
            attributes["llm.provider"] = self._llm_wrapper.provider
            attributes["llm.label"] = self._llm_wrapper.label

            # 2. Request metadata
            if self._request_id:
                attributes["llm.request_id"] = self._request_id
            else:
                attributes["llm.request_id"] = LLM_REQUEST_ID_DEFAULT_VALUE

            # 3. Input - Chat Context (FULL, with truncation)
            self._add_chat_context_attributes(attributes)

            # 4. Input - Available Tools
            self._add_tools_attributes(attributes)

            # 5. Sampling Parameters (extract from base LLM)
            self._add_sampling_params(attributes)

            # 6. Output - Response
            attributes["llm.response"] = (
                self._response_content
                if self._response_content
                else LLM_RESPONSE_DEFAULT_VALUE
            )
            if self._response_role:
                attributes["llm.response_role"] = self._response_role

            # 7. Output - Tool Calls
            self._add_tool_calls_attributes(attributes)

            # 8. Token Metrics
            self._add_token_metrics(attributes)

            # 9. Timing Metrics
            self._add_timing_metrics(attributes, duration)

            # 10. Cost Metrics
            self._add_cost_metrics(attributes)

            # 11. Stream Metadata
            attributes["llm.mode"] = "streaming"
            attributes["llm.chunk_count"] = len(self._buffered_chunks)
            attributes["llm.cancelled"] = self._cancelled
            attributes["llm.had_error"] = self._had_error

            # 12. Job Context
            for key, value in self._llm_wrapper._job_context.items():
                if key.startswith("job."):
                    attributes[key] = value
                elif key.startswith("job_"):
                    attributes[f"job.{key[4:]}"] = value
                else:
                    attributes[f"job.{key}"] = value

            # 13. Constants Metadata
            attributes["metadata"] = create_constants_metadata()

            # Create and finish span
            span = client.start_span(name="llm.chat", attributes=attributes)
            span.set_status(SpanStatus.ERROR if self._had_error else SpanStatus.OK)
            client.finish_span(span)

            logger.debug(
                f"Created LLM span: model={attributes.get('llm.model')}, "
                f"tokens={attributes.get('llm.total_tokens')}, "
                f"duration={duration:.2f}s"
            )

        except Exception as e:
            logger.warning(f"Failed to create span for LLM chat: {e}", exc_info=True)

    def _add_chat_context_attributes(self, attributes: dict[str, Any]) -> None:
        """Add chat context attributes including messages and system prompt."""
        try:
            chat_items = (
                self._chat_ctx.items if hasattr(self._chat_ctx, "items") else []
            )

            # Truncate if too large
            if len(chat_items) > MAX_CONVERSATION_HISTORY:
                chat_items = chat_items[-MAX_CONVERSATION_HISTORY:]

            attributes["llm.message_count"] = len(chat_items)

            # Serialize full chat context
            if chat_items:
                serialized_history = serialize_chat_history(chat_items)
                attributes["llm.chat_ctx"] = json.dumps(serialized_history, default=str)

                # Extract system prompt (first system message)
                for item in chat_items:
                    role = str(getattr(item, "role", ""))
                    if role == "system":
                        content = getattr(item, "content", None)
                        if content:
                            # Handle both string and list content
                            if isinstance(content, str):
                                attributes["llm.system_prompt"] = content
                            elif isinstance(content, list):
                                text_parts = []
                                for part in content:
                                    if isinstance(part, str):
                                        text_parts.append(part)
                                    elif hasattr(part, "text"):
                                        text_parts.append(str(part.text))
                                if text_parts:
                                    attributes["llm.system_prompt"] = "\n".join(
                                        text_parts
                                    )
                        break
            else:
                attributes["llm.chat_ctx"] = "[]"

        except Exception as e:
            logger.warning(f"Failed to serialize chat context: {e}", exc_info=True)
            attributes["llm.message_count"] = 0
            attributes["llm.chat_ctx"] = "[]"

    def _add_tools_attributes(self, attributes: dict[str, Any]) -> None:
        """Add available tools attributes."""
        if not self._tools:
            return

        try:
            tool_schemas = []
            for tool in self._tools:
                tool_info = self._extract_tool_info(tool)
                tool_schemas.append(tool_info)

            attributes["llm.available_tools.count"] = len(tool_schemas)
            attributes["llm.available_tools.names"] = [
                t.get("name", "unknown") for t in tool_schemas
            ]
            attributes["llm.available_tools.schemas"] = json.dumps(
                tool_schemas, default=str
            )
        except Exception as e:
            logger.warning(f"Failed to serialize tools: {e}", exc_info=True)

    def _extract_tool_info(self, tool: Any) -> dict[str, Any]:
        """Extract complete tool information including name, description, and schema."""
        tool_info: dict[str, Any] = {}

        try:
            # Get tool name
            if hasattr(tool, "name"):
                tool_info["name"] = str(tool.name)
            elif hasattr(tool, "__name__"):
                tool_info["name"] = str(tool.__name__)
            elif hasattr(tool, "func") and hasattr(tool.func, "__name__"):
                tool_info["name"] = str(tool.func.__name__)
            else:
                tool_info["name"] = "unknown_tool"

            # Get tool description
            if hasattr(tool, "description"):
                tool_info["description"] = str(tool.description)
            elif hasattr(tool, "__doc__") and tool.__doc__:
                tool_info["description"] = str(tool.__doc__).strip()
            else:
                tool_info["description"] = ""

            # Get args schema if available
            if hasattr(tool, "args_schema"):
                args_schema = tool.args_schema
                if hasattr(args_schema, "model_json_schema"):
                    tool_info["args_schema"] = args_schema.model_json_schema()
                elif hasattr(args_schema, "schema"):
                    tool_info["args_schema"] = args_schema.schema()
                elif isinstance(args_schema, dict):
                    tool_info["args_schema"] = args_schema
            elif hasattr(tool, "parameters"):
                tool_info["args_schema"] = tool.parameters

        except Exception as e:
            logger.debug(f"Failed to extract tool info: {e}")

        return tool_info

    def _add_sampling_params(self, attributes: dict[str, Any]) -> None:
        """Add sampling parameters from LLM configuration."""
        llm = self._llm_wrapper._base_llm

        # Try to access _opts (OpenAI pattern)
        if hasattr(llm, "_opts"):
            opts = llm._opts
            param_names = [
                "temperature",
                "top_p",
                "max_completion_tokens",
                "max_tokens",
                "presence_penalty",
                "frequency_penalty",
                "stop",
                "parallel_tool_calls",
                "tool_choice",
            ]

            for param in param_names:
                if hasattr(opts, param):
                    val = getattr(opts, param)
                    # Check if value is set (not NOT_GIVEN or None)
                    if val is not None and (NOT_GIVEN is None or val is not NOT_GIVEN):
                        attributes[f"llm.{param}"] = val

        # Try direct attributes (fallback for other providers)
        direct_params = ["temperature", "top_p", "max_tokens"]
        for param in direct_params:
            if hasattr(llm, param) and param not in [
                k.replace("llm.", "") for k in attributes.keys()
            ]:
                val = getattr(llm, param)
                if val is not None:
                    attributes[f"llm.{param}"] = val

        # Try to get tool_choice and parallel_tool_calls from kwargs
        if "tool_choice" in self._kwargs:
            attributes["llm.tool_choice"] = str(self._kwargs["tool_choice"])
        if "parallel_tool_calls" in self._kwargs:
            attributes["llm.parallel_tool_calls"] = self._kwargs["parallel_tool_calls"]

    def _add_tool_calls_attributes(self, attributes: dict[str, Any]) -> None:
        """Add tool calls attributes."""
        if not self._tool_calls:
            return

        try:
            serialized_calls = []
            for tc in self._tool_calls:
                serialized_calls.append(
                    {
                        "name": tc.name,
                        "arguments": tc.arguments,
                        "call_id": tc.call_id,
                        "type": tc.type,
                    }
                )

            attributes["llm.tool_calls.count"] = len(serialized_calls)
            attributes["llm.tool_calls.names"] = [c["name"] for c in serialized_calls]
            attributes["llm.tool_calls"] = json.dumps(serialized_calls, default=str)
        except Exception as e:
            logger.warning(f"Failed to serialize tool calls: {e}", exc_info=True)

    def _add_token_metrics(self, attributes: dict[str, Any]) -> None:
        """Add token usage metrics."""
        if not self._usage:
            return

        attributes["llm.completion_tokens"] = self._usage.completion_tokens
        attributes["llm.prompt_tokens"] = self._usage.prompt_tokens
        attributes["llm.total_tokens"] = self._usage.total_tokens

        # Add optional cached token metrics
        if hasattr(self._usage, "prompt_cached_tokens"):
            attributes["llm.prompt_cached_tokens"] = self._usage.prompt_cached_tokens
        if hasattr(self._usage, "cache_creation_tokens"):
            attributes["llm.cache_creation_tokens"] = self._usage.cache_creation_tokens
        if hasattr(self._usage, "cache_read_tokens"):
            attributes["llm.cache_read_tokens"] = self._usage.cache_read_tokens

    def _add_timing_metrics(self, attributes: dict[str, Any], duration: float) -> None:
        """Add timing metrics."""
        # TTFT (Time to First Token)
        if self._ttft is not None:
            attributes["llm.ttft"] = self._ttft
            attributes["llm.ttft_ms"] = self._ttft * 1000

        # Duration
        attributes["llm.duration"] = duration
        attributes["llm.duration_ms"] = duration * 1000

        # Tokens per second
        if self._usage and duration > 0:
            attributes["llm.tokens_per_second"] = (
                self._usage.completion_tokens / duration
            )

    def _add_cost_metrics(self, attributes: dict[str, Any]) -> None:
        """Add cost metrics using estimate_cost utility."""
        if not self._usage:
            return

        try:
            from noveum_trace.utils.llm_utils import estimate_cost

            model = attributes.get("llm.model")
            if not model or not isinstance(model, str):
                model = LLM_MODEL_DEFAULT_VALUE

            cost_info = estimate_cost(
                model=model,
                input_tokens=self._usage.prompt_tokens,
                output_tokens=self._usage.completion_tokens,
            )

            attributes["llm.cost.input"] = cost_info.get("input_cost", 0)
            attributes["llm.cost.output"] = cost_info.get("output_cost", 0)
            attributes["llm.cost.total"] = cost_info.get("total_cost", 0)
            attributes["llm.cost.currency"] = cost_info.get("currency", "USD")
        except Exception as e:
            logger.debug(f"Failed to calculate LLM cost: {e}")

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to base stream."""
        return getattr(self._base_stream, name)
