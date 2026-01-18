"""
LiveKit LLM integration for noveum-trace.

This module provides wrapper classes that automatically trace LiveKit LLM
operations, capturing chat history, function calls, and tools in spans.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from noveum_trace.core.context import get_current_trace
from noveum_trace.core.span import SpanStatus
from noveum_trace.integrations.livekit.livekit_constants import (
    MAX_CONVERSATION_HISTORY,
    MAX_PENDING_FUNCTION_CALLS,
    MAX_PENDING_FUNCTION_OUTPUTS,
)
from noveum_trace.integrations.livekit.livekit_utils import (
    create_constants_metadata,
)
from noveum_trace.utils.llm_utils import (
    estimate_cost,
    estimate_token_count,
    get_model_info,
    normalize_model_name,
)

logger = logging.getLogger(__name__)

try:
    import livekit.agents.llm  # noqa: F401

    LIVEKIT_AVAILABLE = True
except ImportError as e:
    LIVEKIT_AVAILABLE = False
    logger.debug(
        "LiveKit is not importable. LiveKit LLM integration features will not be available. "
        "Install it with: pip install livekit livekit-agents or check the documentation",
        exc_info=e,
    )


def extract_available_tools(agent: Any) -> list[dict[str, Any]]:
    """
    Extract available tools from a LiveKit Agent.

    Args:
        agent: LiveKit Agent instance

    Returns:
        List of tool dictionaries with name, description, and args_schema
    """
    tools: list[dict[str, Any]] = []

    if not agent:
        return tools

    # Try to get tools from agent
    agent_tools = None
    if hasattr(agent, "tools"):
        agent_tools = agent.tools
    elif hasattr(agent, "_tools"):
        agent_tools = agent._tools

    if not agent_tools:
        return tools

    for tool in agent_tools:
        try:
            tool_info: dict[str, Any] = {}

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

            tools.append(tool_info)

        except Exception as e:
            logger.debug(f"Failed to extract tool info: {e}")
            continue

    return tools


def serialize_tools_for_attributes(tools: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Serialize tools list into span attributes format.

    Args:
        tools: List of tool dictionaries

    Returns:
        Dictionary of span attributes for tools
    """
    if not tools:
        return {}

    attributes: dict[str, Any] = {
        "llm.available_tools.count": len(tools),
        "llm.available_tools.names": [t.get("name", "unknown") for t in tools],
    }

    # Add full schemas as JSON
    try:
        attributes["llm.available_tools.schemas"] = json.dumps(tools, default=str)
    except Exception as e:
        logger.debug(f"Failed to serialize tools schemas: {e}")

    return attributes


def serialize_chat_history(messages: list[Any]) -> list[dict[str, Any]]:
    """
    Serialize chat messages into a list of dictionaries.

    Args:
        messages: List of chat messages (ChatMessage, dict, or similar)

    Returns:
        List of serialized message dictionaries
    """
    serialized: list[dict[str, Any]] = []

    for msg in messages:
        try:
            msg_dict: dict[str, Any] = {}

            # Handle dict messages
            if isinstance(msg, dict):
                msg_dict = {
                    "role": msg.get("role", "unknown"),
                    "content": str(msg.get("content", "")),
                }
                if msg.get("name"):
                    msg_dict["name"] = msg["name"]
                serialized.append(msg_dict)
                continue

            # Handle ChatMessage or similar objects
            if hasattr(msg, "role"):
                role = msg.role
                # Handle enum roles
                if hasattr(role, "value"):
                    msg_dict["role"] = str(role.value)
                else:
                    msg_dict["role"] = str(role)

            # Extract content
            if hasattr(msg, "text_content"):
                msg_dict["content"] = str(msg.text_content)
            elif hasattr(msg, "content"):
                content = msg.content
                if isinstance(content, str):
                    msg_dict["content"] = content
                elif isinstance(content, list):
                    # Handle list of content parts
                    text_parts = []
                    for part in content:
                        if isinstance(part, str):
                            text_parts.append(part)
                        elif hasattr(part, "text"):
                            text_parts.append(str(part.text))
                        elif isinstance(part, dict) and "text" in part:
                            text_parts.append(str(part["text"]))
                    msg_dict["content"] = "\n".join(text_parts)
                else:
                    msg_dict["content"] = str(content)

            # Extract name if present
            if hasattr(msg, "name") and msg.name:
                msg_dict["name"] = str(msg.name)

            if msg_dict:
                serialized.append(msg_dict)

        except Exception as e:
            logger.debug(f"Failed to serialize message: {e}")
            continue

    return serialized


def serialize_function_calls(function_calls: list[Any]) -> list[dict[str, Any]]:
    """
    Serialize function calls into a list of dictionaries.

    Args:
        function_calls: List of function call objects

    Returns:
        List of serialized function call dictionaries
    """
    serialized: list[dict[str, Any]] = []

    for call in function_calls:
        try:
            call_dict: dict[str, Any] = {}

            if isinstance(call, dict):
                call_dict = {
                    "name": call.get("name", "unknown"),
                    "arguments": call.get("arguments", ""),
                }
                if call.get("call_id"):
                    call_dict["call_id"] = call["call_id"]
                serialized.append(call_dict)
                continue

            # Handle FunctionCall objects
            if hasattr(call, "name"):
                call_dict["name"] = str(call.name)
            if hasattr(call, "arguments"):
                args = call.arguments
                if isinstance(args, str):
                    call_dict["arguments"] = args
                else:
                    call_dict["arguments"] = json.dumps(args, default=str)
            if hasattr(call, "call_id"):
                call_dict["call_id"] = str(call.call_id)

            if call_dict:
                serialized.append(call_dict)

        except Exception as e:
            logger.debug(f"Failed to serialize function call: {e}")
            continue

    return serialized


class LiveKitLLMWrapper:
    """
    Wrapper for LiveKit LLM providers that automatically creates spans for completions.

    This wrapper captures chat history, function calls, available tools, and creates
    spans with metadata for each LLM operation.

    Example:
        >>> import noveum_trace
        >>> from livekit.plugins import openai as openai_plugin
        >>> from noveum_trace.integrations.livekit import LiveKitLLMWrapper
        >>>
        >>> # Initialize noveum-trace (done elsewhere)
        >>> noveum_trace.init(project="livekit-agents")
        >>>
        >>> # Wrap LLM provider
        >>> base_llm = openai_plugin.LLM(model="gpt-4o-mini")
        >>> traced_llm = LiveKitLLMWrapper(
        ...     llm=base_llm,
        ...     session_id="session_123",
        ...     job_context={"job_id": "job_abc"}
        ... )
        >>>
        >>> # Use in agent session
        >>> session = AgentSession(llm=traced_llm, ...)
    """

    def __init__(
        self,
        llm: Any,
        session_id: str,
        job_context: Optional[dict[str, Any]] = None,
        available_tools: Optional[list[dict[str, Any]]] = None,
    ):
        """
        Initialize LLM wrapper.

        Args:
            llm: Base LiveKit LLM provider instance
            session_id: Session identifier for organizing spans
            job_context: Dictionary of job context information to attach to spans
            available_tools: List of available tools (extracted from agent)
        """
        self._base_llm = llm
        self._session_id = session_id
        self._job_context = job_context or {}
        self._available_tools = available_tools or []
        self._counter = 0

        # Conversation history tracking
        self._conversation_history: list[dict[str, Any]] = []

        # Pending function call data (to merge with generation span)
        self._pending_function_calls: list[dict[str, Any]] = []
        self._pending_function_outputs: list[dict[str, Any]] = []

        if not LIVEKIT_AVAILABLE:
            logger.error(
                "Cannot initialize LiveKitLLMWrapper: LiveKit is not available. "
                "Install it with: pip install livekit livekit-agents"
            )

    @property
    def model(self) -> str:
        """Get model name from base provider."""
        if self._base_llm is None:
            return "unknown"
        return getattr(self._base_llm, "model", "unknown")

    @property
    def provider(self) -> str:
        """Get provider name (inferred from model)."""
        model_info = get_model_info(self.model)
        if model_info:
            return model_info.provider
        # Fallback: check class name
        if self._base_llm is not None:
            class_name = self._base_llm.__class__.__name__.lower()
            if "openai" in class_name:
                return "openai"
            elif "anthropic" in class_name:
                return "anthropic"
            elif "google" in class_name or "gemini" in class_name:
                return "google"
        return "unknown"

    @property
    def label(self) -> str:
        """Get label from base provider."""
        if self._base_llm is None:
            return "LiveKitLLMWrapper"
        return getattr(self._base_llm, "label", self._base_llm.__class__.__name__)

    @property
    def conversation_history(self) -> list[dict[str, Any]]:
        """Get the accumulated conversation history."""
        return self._conversation_history.copy()

    def set_available_tools(self, tools: list[dict[str, Any]]) -> None:
        """
        Set available tools for this LLM wrapper.

        Args:
            tools: List of tool dictionaries
        """
        self._available_tools = tools

    def add_to_history(
        self, role: str, content: str, name: Optional[str] = None
    ) -> None:
        """
        Add a message to conversation history.

        Args:
            role: Message role (user, assistant, system, tool)
            content: Message content
            name: Optional name (for tool messages)
        """
        msg: dict[str, Any] = {"role": role, "content": content}
        if name:
            msg["name"] = name
        self._conversation_history.append(msg)
        # Enforce cap to prevent memory growth
        if len(self._conversation_history) > MAX_CONVERSATION_HISTORY:
            self._conversation_history = self._conversation_history[
                -MAX_CONVERSATION_HISTORY:
            ]

    def record_function_call(
        self,
        name: str,
        arguments: str,
        call_id: Optional[str] = None,
    ) -> None:
        """
        Record a function call (to be merged with LLM span).

        Args:
            name: Function name
            arguments: Function arguments (JSON string)
            call_id: Optional call ID
        """
        call: dict[str, Any] = {"name": name, "arguments": arguments}
        if call_id:
            call["call_id"] = call_id
        self._pending_function_calls.append(call)
        # Enforce cap to prevent memory growth
        if len(self._pending_function_calls) > MAX_PENDING_FUNCTION_CALLS:
            self._pending_function_calls = self._pending_function_calls[
                -MAX_PENDING_FUNCTION_CALLS:
            ]

    def record_function_output(
        self,
        name: str,
        output: str,
        is_error: bool = False,
    ) -> None:
        """
        Record a function output (to be merged with LLM span).

        Args:
            name: Function name
            output: Function output
            is_error: Whether the output is an error
        """
        self._pending_function_outputs.append(
            {
                "name": name,
                "output": output,
                "is_error": is_error,
            }
        )
        # Enforce cap to prevent memory growth
        if len(self._pending_function_outputs) > MAX_PENDING_FUNCTION_OUTPUTS:
            self._pending_function_outputs = self._pending_function_outputs[
                -MAX_PENDING_FUNCTION_OUTPUTS:
            ]

    def _flush_pending_function_data(
        self,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Flush and return pending function call data."""
        calls = self._pending_function_calls.copy()
        outputs = self._pending_function_outputs.copy()
        self._pending_function_calls.clear()
        self._pending_function_outputs.clear()
        return calls, outputs

    def chat(
        self,
        chat_ctx: Any,
        **kwargs: Any,
    ) -> _WrappedLLMStream:
        """
        Start a chat completion with tracing.

        Args:
            chat_ctx: Chat context containing messages
            **kwargs: Additional arguments passed to base LLM

        Returns:
            Wrapped LLM stream
        """
        # Extract messages from chat context
        messages: list[Any] = []
        if hasattr(chat_ctx, "messages"):
            messages = list(chat_ctx.messages)
        elif hasattr(chat_ctx, "items"):
            messages = list(chat_ctx.items)
        elif isinstance(chat_ctx, list):
            messages = chat_ctx

        # Update conversation history
        serialized_messages = serialize_chat_history(messages)
        for msg in serialized_messages:
            if msg not in self._conversation_history:
                self._conversation_history.append(msg)
        # Enforce cap to prevent memory growth
        if len(self._conversation_history) > MAX_CONVERSATION_HISTORY:
            self._conversation_history = self._conversation_history[
                -MAX_CONVERSATION_HISTORY:
            ]

        # Get base stream
        base_stream = self._base_llm.chat(chat_ctx, **kwargs)

        # Increment counter
        self._counter += 1

        return _WrappedLLMStream(
            base_stream=base_stream,
            wrapper=self,
            input_messages=serialized_messages,
            counter=self._counter,
        )

    async def aclose(self) -> None:
        """Close the LLM provider."""
        if hasattr(self._base_llm, "aclose"):
            await self._base_llm.aclose()

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to base LLM."""
        return getattr(self._base_llm, name)


class _WrappedLLMStream:
    """Wrapper for LLM streaming that captures responses and creates spans."""

    def __init__(
        self,
        base_stream: Any,
        wrapper: LiveKitLLMWrapper,
        input_messages: list[dict[str, Any]],
        counter: int,
    ):
        self._base_stream = base_stream
        self._wrapper = wrapper
        self._input_messages = input_messages
        self._counter = counter

        # Response accumulation
        self._response_text = ""
        # Tool calls accumulated by call_id to merge streamed fragments
        self._function_calls_by_id: dict[str, dict[str, Any]] = {}
        self._no_id_counter = 0  # Counter for tool calls without an id
        self._usage: dict[str, Any] = {}
        self._span_created = False

    async def __anext__(self) -> Any:
        """Get next chunk from the stream."""
        chunk = await self._base_stream.__anext__()

        # Accumulate response text
        if hasattr(chunk, "choices") and chunk.choices:
            choice = chunk.choices[0]
            if hasattr(choice, "delta"):
                delta = choice.delta
                if hasattr(delta, "content") and delta.content:
                    self._response_text += delta.content
                # Capture function calls from delta
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        self._capture_tool_call(tool_call)

        # Capture usage if available
        if hasattr(chunk, "usage") and chunk.usage:
            self._usage = {
                "prompt_tokens": getattr(chunk.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(chunk.usage, "completion_tokens", 0),
                "total_tokens": getattr(chunk.usage, "total_tokens", 0),
            }

        return chunk

    def _capture_tool_call(self, tool_call: Any) -> None:
        """
        Capture and merge a tool call fragment from the stream.

        LiveKit Agents emits delta.tool_calls incrementally across multiple
        stream chunks. This method merges fragments by call_id, concatenating
        arguments as they arrive.
        """
        try:
            # Extract call_id - use it as the key for merging
            call_id: str | None = None
            if hasattr(tool_call, "id") and tool_call.id:
                call_id = str(tool_call.id)

            # Extract function name and arguments from delta
            name: str | None = None
            arguments: str = ""
            if hasattr(tool_call, "function") and tool_call.function:
                func = tool_call.function
                # Name may only appear in the first chunk
                if hasattr(func, "name") and func.name:
                    name = str(func.name)
                # Arguments are streamed incrementally
                if hasattr(func, "arguments") and func.arguments:
                    arguments = str(func.arguments)

            # Determine the key for this tool call
            if call_id:
                key = call_id
            else:
                # For tool calls without id, create a unique key
                key = f"_no_id_{self._no_id_counter}"
                self._no_id_counter += 1

            # Check if we already have an entry for this call_id
            if key in self._function_calls_by_id:
                # Merge: concatenate arguments, update name if provided
                existing = self._function_calls_by_id[key]
                if name:
                    existing["name"] = name
                existing["arguments"] = existing.get("arguments", "") + arguments
            else:
                # Create new entry
                call_dict: dict[str, Any] = {
                    "name": name or "unknown",
                    "arguments": arguments,
                }
                if call_id:
                    call_dict["call_id"] = call_id
                self._function_calls_by_id[key] = call_dict

        except Exception as e:
            logger.debug(f"Failed to capture tool call: {e}")

    @property
    def _function_calls(self) -> list[dict[str, Any]]:
        """Get accumulated function calls as a list."""
        return list(self._function_calls_by_id.values())

    def __aiter__(self) -> _WrappedLLMStream:
        """Return self as async iterator."""
        return self

    async def __aenter__(self) -> _WrappedLLMStream:
        """Enter async context manager."""
        if hasattr(self._base_stream, "__aenter__"):
            await self._base_stream.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit async context manager and create span."""
        # Create span before exiting
        if not self._span_created:
            await self._create_span()

        if hasattr(self._base_stream, "__aexit__"):
            await self._base_stream.__aexit__(exc_type, exc, exc_tb)
        else:
            await self.aclose()

    async def aclose(self) -> None:
        """Close the stream and create span."""
        # Create span if not already created
        if not self._span_created:
            await self._create_span()

        if hasattr(self._base_stream, "aclose"):
            await self._base_stream.aclose()

    async def _create_span(self) -> None:
        """Create the LLM span with all collected data."""
        if self._span_created:
            return

        self._span_created = True

        trace = get_current_trace()
        if not trace:
            return

        try:
            from noveum_trace import get_client

            client = get_client()
            if not client:
                return

            # Get pending function data from wrapper
            pending_calls, pending_outputs = (
                self._wrapper._flush_pending_function_data()
            )

            # Merge function calls
            all_function_calls = self._function_calls + pending_calls

            # Build span attributes
            model = self._wrapper.model
            provider = self._wrapper.provider
            normalized_model = normalize_model_name(model)

            attributes: dict[str, Any] = {
                "llm.provider": provider,
                "llm.model": model,
                "llm.model_normalized": normalized_model,
                "llm.operation": "chat",
                "llm.session_id": self._wrapper._session_id,
            }

            # Add job context
            for key, value in self._wrapper._job_context.items():
                if key.startswith("job."):
                    attributes[key] = value
                elif key.startswith("job_"):
                    attributes[f"job.{key[4:]}"] = value
                else:
                    attributes[f"job.{key}"] = value

            # Add input messages
            attributes["llm.input.message_count"] = len(self._input_messages)
            try:
                attributes["llm.input.messages"] = json.dumps(
                    self._input_messages, default=str
                )
            except Exception:
                pass

            # Add conversation history (bounded snapshot to prevent oversized attributes)
            history = self._wrapper.conversation_history[-MAX_CONVERSATION_HISTORY:]
            attributes["llm.conversation.message_count"] = len(history)
            try:
                attributes["llm.conversation.history"] = json.dumps(
                    history, default=str
                )
            except Exception:
                pass

            # Add output
            if self._response_text:
                attributes["llm.output.response"] = self._response_text
                # Add to wrapper's history
                self._wrapper.add_to_history("assistant", self._response_text)

            # Add available tools
            tool_attrs = serialize_tools_for_attributes(self._wrapper._available_tools)
            attributes.update(tool_attrs)

            # Add function calls (merged from stream and pending)
            if all_function_calls:
                attributes["llm.function_calls.count"] = len(all_function_calls)
                try:
                    attributes["llm.function_calls"] = json.dumps(
                        all_function_calls, default=str
                    )
                except Exception:
                    pass

            # Add function outputs
            if pending_outputs:
                attributes["llm.function_outputs.count"] = len(pending_outputs)
                try:
                    attributes["llm.function_outputs"] = json.dumps(
                        pending_outputs, default=str
                    )
                except Exception:
                    pass

            # Add token usage
            input_tokens = self._usage.get("prompt_tokens", 0)
            output_tokens = self._usage.get("completion_tokens", 0)

            if not input_tokens and self._input_messages:
                # Estimate input tokens
                input_text = json.dumps(self._input_messages, default=str)
                input_tokens = estimate_token_count(
                    input_text, model=model, provider=provider
                )

            if not output_tokens and self._response_text:
                # Estimate output tokens
                output_tokens = estimate_token_count(
                    self._response_text, model=model, provider=provider
                )

            attributes["llm.input_tokens"] = input_tokens
            attributes["llm.output_tokens"] = output_tokens
            attributes["llm.total_tokens"] = input_tokens + output_tokens

            # Add cost estimation
            cost_info = estimate_cost(model, input_tokens, output_tokens)
            attributes["llm.cost.input"] = cost_info.get("input_cost", 0)
            attributes["llm.cost.output"] = cost_info.get("output_cost", 0)
            attributes["llm.cost.total"] = cost_info.get("total_cost", 0)
            attributes["llm.cost.currency"] = cost_info.get("currency", "USD")

            # Add constants metadata
            attributes["metadata"] = create_constants_metadata()

            # Create and finish span
            span = client.start_span(name="llm.chat", attributes=attributes)
            span.set_status(SpanStatus.OK)
            client.finish_span(span)

        except Exception as e:
            logger.warning(f"Failed to create LLM span: {e}", exc_info=True)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to base stream."""
        return getattr(self._base_stream, name)
