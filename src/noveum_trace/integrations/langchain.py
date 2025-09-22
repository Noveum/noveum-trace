"""
LangChain integration for Noveum Trace SDK.

This module provides a callback handler that automatically traces LangChain
operations including LLM calls, chains, agents, tools, and retrieval operations.
"""

import logging
import threading
from collections.abc import Sequence
from typing import Any, Optional
from uuid import UUID

# Import LangChain dependencies
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.outputs import LLMResult

from noveum_trace.core.span import SpanStatus

logger = logging.getLogger(__name__)


class NoveumTraceCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler for Noveum Trace integration."""

    def __init__(self) -> None:
        """Initialize the callback handler."""
        super().__init__()

        # Thread-safe runs dictionary for span tracking
        # Maps run_id -> span (for backward compatibility)
        self.runs: dict[UUID, Any] = {}
        self._runs_lock = threading.Lock()

        # Track if we're managing a trace lifecycle
        self._trace_managed_by_langchain: Optional[Any] = None

        # Import here to avoid circular imports
        from noveum_trace import get_client

        try:
            self._client = get_client()
        except Exception as e:
            logger.warning("Failed to get Noveum Trace client: %s", e)
            self._client = None  # type: ignore[assignment]

    def _set_run(self, run_id: UUID, span: Any) -> None:
        """Thread-safe method to set a run span."""
        with self._runs_lock:
            self.runs[run_id] = span

    def _pop_run(self, run_id: UUID) -> Any:
        """Thread-safe method to pop and return a run span."""
        with self._runs_lock:
            return self.runs.pop(run_id, None)

    def _active_runs(self) -> int:
        """Thread-safe method to get the number of active runs."""
        with self._runs_lock:
            return len(self.runs)

    def _get_run(self, run_id: UUID) -> Any:
        """Thread-safe method to get a run span without removing it."""
        with self._runs_lock:
            return self.runs.get(run_id)

    def _get_or_create_trace_context(self, operation_name: str) -> tuple[Any, bool]:
        """
        Get existing trace from global context or create new one.

        Args:
            operation_name: Name for the operation

        Returns:
            (trace, should_manage_lifecycle) tuple
        """
        from noveum_trace.core.context import get_current_trace, set_current_trace

        existing_trace = get_current_trace()

        if existing_trace is not None:
            # Use existing trace - don't manage its lifecycle
            return existing_trace, False
        else:
            # Create new trace in global context
            new_trace = self._client.start_trace(operation_name)
            set_current_trace(new_trace)
            return new_trace, True

    def _get_operation_name(self, event_type: str, serialized: dict[str, Any]) -> str:
        """Generate standardized operation names."""
        if serialized is None:
            return f"{event_type}.unknown"
        name = serialized.get("name", "unknown")

        if event_type == "llm_start":
            # Use model name instead of class name for better readability
            model_name = self._extract_model_name(serialized)
            return f"llm.{model_name}"
        elif event_type == "chain_start":
            return f"chain.{name}"
        elif event_type == "agent_start":
            return f"agent.{name}"
        elif event_type == "retriever_start":
            return f"retrieval.{name}"
        elif event_type == "tool_start":
            return f"tool.{name}"

        return f"{event_type}.{name}"

    def _extract_model_name(self, serialized: dict[str, Any]) -> str:
        """Extract model name from serialized LLM data."""
        if not serialized:
            return "unknown"

        # Try to get model name from kwargs
        kwargs = serialized.get("kwargs", {})
        model = kwargs.get("model")
        if model:
            return model

        # Fallback to provider name
        id_path = serialized.get("id", [])
        if len(id_path) >= 2:
            # e.g., "openai" from ["langchain", "chat_models", "openai", "ChatOpenAI"]
            return id_path[-2]

        # Final fallback to class name
        return serialized.get("name", "unknown")

    def _extract_agent_type(self, serialized: dict[str, Any]) -> str:
        """Extract agent type from serialized agent data."""
        if not serialized:
            return "unknown"

        # Get agent category from ID path
        id_path = serialized.get("id", [])
        if len(id_path) >= 2:
            # e.g., "react" from ["langchain", "agents", "react", "ReActAgent"]
            return id_path[-2]

        return "unknown"

    def _extract_agent_capabilities(self, serialized: dict[str, Any]) -> str:
        """Extract agent capabilities from tools in serialized data."""
        if not serialized:
            return "unknown"

        capabilities = []
        kwargs = serialized.get("kwargs", {})
        tools = kwargs.get("tools", [])

        if tools:
            capabilities.append("tool_usage")

            # Extract specific tool types
            tool_types = set()
            for tool in tools:
                if isinstance(tool, dict):
                    tool_name = tool.get("name", "").lower()
                    if "search" in tool_name or "web" in tool_name:
                        tool_types.add("web_search")
                    elif "calc" in tool_name or "math" in tool_name:
                        tool_types.add("calculation")
                    elif "file" in tool_name or "read" in tool_name:
                        tool_types.add("file_operations")
                    elif "api" in tool_name or "request" in tool_name:
                        tool_types.add("api_calls")
                    else:
                        tool_types.add(
                            tool.get("name", "other") if tool.get("name") else "other"
                        )

            if tool_types:
                capabilities.extend(tool_types)

        # Add default capabilities
        if not capabilities:
            capabilities = ["reasoning"]

        return ",".join(capabilities)

    def _extract_tool_function_name(self, serialized: dict[str, Any]) -> str:
        """Extract function name from serialized tool data."""
        if not serialized:
            return "unknown"

        kwargs = serialized.get("kwargs", {})
        func_name = kwargs.get("name")
        if func_name:
            return func_name

        # Fallback to class name
        return serialized.get("name", "unknown")

    def _create_tool_span_from_action(
        self, action: "AgentAction", run_id: UUID
    ) -> None:
        """Create a tool span from an agent action (when on_tool_start/on_tool_end aren't triggered)."""
        try:
            tool_name = action.tool
            tool_input = str(action.tool_input)

            # Create a tool span similar to on_tool_start
            span = self._client.start_span(
                name=f"tool:{tool_name}:{tool_name}",
                attributes={
                    "langchain.run_id": str(run_id),
                    "tool.name": tool_name,
                    "tool.operation": tool_name,
                    "tool.input.input_str": tool_input,
                    "tool.input.argument_count": 1,
                    "tool.input.expression": tool_input,  # For calculator tools
                },
            )
            # Store in runs dict with agent run_id as prefix to associate with parent agent
            import uuid

            tool_run_id = f"{run_id}_tool_{uuid.uuid4()}"
            self._set_run(tool_run_id, span)

        except Exception as e:
            logger.error("Error creating tool span from action: %s", e)

    def _complete_tool_spans_from_finish(
        self, finish: "AgentFinish", agent_run_id: UUID
    ) -> None:
        """Complete any pending tool spans when agent finishes."""
        try:
            # Look for tool spans in runs dict that belong to this specific agent
            tool_spans_to_complete = []
            with self._runs_lock:
                for run_id, span in list(self.runs.items()):
                    # Only complete tool spans that belong to this agent (prefixed with agent_run_id)
                    if str(run_id).startswith(f"{agent_run_id}_tool_"):
                        tool_spans_to_complete.append((run_id, span))

            # Complete tool spans with the final result
            for run_id, tool_span in tool_spans_to_complete:
                # Remove from runs dict
                self._pop_run(run_id)

                # Extract result from the finish log
                result = "Tool execution completed"
                if hasattr(finish, "log") and finish.log:
                    # Try to extract the result from the log
                    log_lines = finish.log.split("\n")
                    for line in log_lines:
                        if "Observation:" in line:
                            result = line.replace("Observation:", "").strip()
                            break
                        elif "Final Answer:" in line:
                            result = line.replace("Final Answer:", "").strip()
                            break

                tool_span.set_attributes(
                    {
                        "tool.output.output": result,
                    }
                )
                tool_span.set_status(SpanStatus.OK)
                self._client.finish_span(tool_span)

        except Exception as e:
            logger.error("Error completing tool spans from finish: %s", e)

    def _ensure_client(self) -> bool:
        """Ensure we have a valid client."""
        if self._client is None:
            try:
                from noveum_trace import get_client

                self._client = get_client()
                return True
            except Exception as e:
                logger.warning("Noveum Trace client not available: %s", e)
                return False
        return True

    # LLM Events
    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle LLM start event."""
        if not self._ensure_client():
            return

        operation_name = self._get_operation_name("llm_start", serialized)

        try:
            # Get or create trace context
            trace, should_manage = self._get_or_create_trace_context(operation_name)

            # Create span
            span = self._client.start_span(
                name=operation_name,
                attributes={
                    "langchain.run_id": str(run_id),
                    "llm.model": self._extract_model_name(serialized),
                    "llm.provider": (
                        serialized.get("id", ["unknown"])[-1]
                        if serialized and isinstance(serialized.get("id"), list)
                        else (
                            serialized.get("id", "unknown") if serialized else "unknown"
                        )
                    ),
                    "llm.operation": "completion",
                    # Input attributes
                    "llm.input.prompts": prompts[:5] if len(prompts) > 5 else prompts,
                    "llm.input.prompt_count": len(prompts),
                    **{
                        k: v
                        for k, v in kwargs.items()
                        if k not in ["tags", "metadata"]
                        and isinstance(v, (str, int, float, bool))
                    },
                },
            )

            # Store span for later cleanup
            self._set_run(run_id, span)

            # Track if we need to manage trace lifecycle
            if should_manage:
                self._trace_managed_by_langchain = trace

        except Exception as e:
            logger.error("Error handling LLM start event: %s", e)

    def on_llm_end(
        self,
        response: "LLMResult",
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Handle LLM end event."""
        if not self._ensure_client():
            return

        # Get and remove span from runs dict
        span = self._pop_run(run_id)
        if span is None:
            return

        try:
            # Add response data
            generations = []
            token_usage = {}

            if hasattr(response, "generations") and response.generations:
                generations = [
                    gen.text
                    for generation_list in response.generations
                    for gen in generation_list
                ][
                    :10
                ]  # Limit number of generations

            if hasattr(response, "llm_output") and response.llm_output:
                token_usage = response.llm_output.get("token_usage", {})

            # Flatten usage attributes to match ContextManager format
            usage_attrs = {}
            if token_usage:
                usage_attrs.update(
                    {
                        "llm.input_tokens": token_usage.get("prompt_tokens", 0),
                        "llm.output_tokens": token_usage.get("completion_tokens", 0),
                        "llm.total_tokens": token_usage.get("total_tokens", 0),
                    }
                )

            span.set_attributes(
                {
                    # Output attributes
                    "llm.output.response": generations,
                    "llm.output.response_count": len(generations),
                    "llm.output.finish_reason": (
                        response.llm_output.get("finish_reason")
                        if hasattr(response, "llm_output") and response.llm_output
                        else None
                    ),
                    # Flattened usage attributes
                    **usage_attrs,
                }
            )

            span.set_status(SpanStatus.OK)
            self._client.finish_span(span)

            # Check if we should finish the trace
            if (
                self._trace_managed_by_langchain and self._active_runs() == 0
            ):  # No more active spans
                self._client.finish_trace(self._trace_managed_by_langchain)
                from noveum_trace.core.context import set_current_trace

                set_current_trace(None)
                self._trace_managed_by_langchain = None

        except Exception as e:
            logger.error("Error handling LLM end event: %s", e)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Handle LLM error event."""
        if not self._ensure_client():
            return None

        # Get and remove span from runs dict
        span = self._pop_run(run_id)
        if span is None:
            return None

        try:
            span.record_exception(error)
            span.set_status(SpanStatus.ERROR, str(error))
            self._client.finish_span(span)

            # Check if we should finish the trace
            if (
                self._trace_managed_by_langchain and self._active_runs() == 0
            ):  # No more active spans
                self._client.finish_trace(self._trace_managed_by_langchain)
                from noveum_trace.core.context import set_current_trace

                set_current_trace(None)
                self._trace_managed_by_langchain = None

        except Exception as e:
            logger.error("Error handling LLM error event: %s", e)

        return None

    # Chain Events
    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle chain start event."""
        if not self._ensure_client():
            return

        operation_name = self._get_operation_name("chain_start", serialized)

        try:
            # Get or create trace context
            trace, should_manage = self._get_or_create_trace_context(operation_name)

            # Create span for chain
            span = self._client.start_span(
                name=operation_name,
                attributes={
                    "langchain.run_id": str(run_id),
                    "chain.name": (
                        serialized.get("name", "unknown") if serialized else "unknown"
                    ),
                    "chain.operation": "execution",
                    # Input attributes
                    "chain.inputs": {k: str(v) for k, v in inputs.items()},
                    **{
                        k: v
                        for k, v in kwargs.items()
                        if k not in ["tags", "metadata"]
                        and isinstance(v, (str, int, float, bool))
                    },
                },
            )

            # Store span for later cleanup
            self._set_run(run_id, span)

            # Track if we need to manage trace lifecycle
            if should_manage:
                self._trace_managed_by_langchain = trace

        except Exception as e:
            logger.error("Error handling chain start event: %s", e)

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Handle chain end event."""
        if not self._ensure_client():
            return

        # Get and remove span from runs dict
        span = self._pop_run(run_id)
        if span is None:
            return

        try:
            span.set_attributes(
                {
                    # Output attributes
                    "chain.output.outputs": {k: str(v) for k, v in outputs.items()}
                }
            )

            span.set_status(SpanStatus.OK)
            self._client.finish_span(span)

            # Check if we should finish the trace
            if (
                self._trace_managed_by_langchain and self._active_runs() == 0
            ):  # No more active spans
                self._client.finish_trace(self._trace_managed_by_langchain)
                from noveum_trace.core.context import set_current_trace

                set_current_trace(None)
                self._trace_managed_by_langchain = None

        except Exception as e:
            logger.error("Error handling chain end event: %s", e)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Handle chain error event."""
        if not self._ensure_client():
            return None

        # Get and remove span from runs dict
        span = self._pop_run(run_id)
        if span is None:
            return None

        try:
            span.record_exception(error)
            span.set_status(SpanStatus.ERROR, str(error))
            self._client.finish_span(span)

            # Check if we should finish the trace
            if (
                self._trace_managed_by_langchain and self._active_runs() == 0
            ):  # No more active spans
                self._client.finish_trace(self._trace_managed_by_langchain)
                from noveum_trace.core.context import set_current_trace

                set_current_trace(None)
                self._trace_managed_by_langchain = None

        except Exception as e:
            logger.error("Error handling chain error event: %s", e)

        return None

    # Tool Events
    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        inputs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle tool start event."""
        if not self._ensure_client():
            return

        operation_name = self._get_operation_name("tool_start", serialized)

        try:
            # Get or create trace context
            trace, should_manage = self._get_or_create_trace_context(operation_name)

            tool_name = serialized.get("name", "unknown") if serialized else "unknown"

            # Extract actual function name from serialized data
            func_name = self._extract_tool_function_name(serialized)

            # Prepare input attributes
            input_attrs = {
                "tool.input.input_str": input_str,  # String representation for compatibility
            }

            # Add structured inputs if available
            if inputs:
                for key, value in inputs.items():
                    # Convert values to strings for attribute storage
                    input_attrs[f"tool.input.{key}"] = str(value)
                input_attrs["tool.input.argument_count"] = str(len(inputs))
            else:
                input_attrs["tool.input.argument_count"] = "0"

            span = self._client.start_span(
                name=f"tool:{tool_name}:{func_name}",
                attributes={
                    "langchain.run_id": str(run_id),
                    "tool.name": tool_name,
                    "tool.operation": func_name,
                    # Input attributes
                    **input_attrs,
                    **{
                        k: v
                        for k, v in kwargs.items()
                        if k not in ["tags", "metadata", "inputs"]
                        and isinstance(v, (str, int, float, bool))
                    },
                },
            )

            # Store span for later cleanup
            self._set_run(run_id, span)

            # Track if we need to manage trace lifecycle
            if should_manage:
                self._trace_managed_by_langchain = trace

        except Exception as e:
            logger.error("Error handling tool start event: %s", e)

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Handle tool end event."""
        if not self._ensure_client():
            return

        # Get and remove span from runs dict
        span = self._pop_run(run_id)
        if span is None:
            return

        try:
            span.set_attributes({"tool.output.output": output})
            span.set_status(SpanStatus.OK)
            self._client.finish_span(span)

            # Check if we should finish the trace
            if (
                self._trace_managed_by_langchain and self._active_runs() == 0
            ):  # No more active spans
                self._client.finish_trace(self._trace_managed_by_langchain)
                from noveum_trace.core.context import set_current_trace

                set_current_trace(None)
                self._trace_managed_by_langchain = None

        except Exception as e:
            logger.error("Error handling tool end event: %s", e)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Handle tool error event."""
        if not self._ensure_client():
            return None

        # Get and remove span from runs dict
        span = self._pop_run(run_id)
        if span is None:
            return None

        try:
            span.record_exception(error)
            span.set_status(SpanStatus.ERROR, str(error))
            self._client.finish_span(span)

            # Check if we should finish the trace
            if (
                self._trace_managed_by_langchain and self._active_runs() == 0
            ):  # No more active spans
                self._client.finish_trace(self._trace_managed_by_langchain)
                from noveum_trace.core.context import set_current_trace

                set_current_trace(None)
                self._trace_managed_by_langchain = None

        except Exception as e:
            logger.error("Error handling tool error event: %s", e)

        return None

    # Agent Events
    def on_agent_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle agent start event."""
        if not self._ensure_client():
            return

        operation_name = self._get_operation_name("agent_start", serialized)

        try:
            # Get or create trace context
            trace, should_manage = self._get_or_create_trace_context(operation_name)

            # Create span for agent
            agent_name = serialized.get("name", "unknown") if serialized else "unknown"

            # Extract actual agent information from serialized data
            agent_type = self._extract_agent_type(serialized)
            agent_capabilities = self._extract_agent_capabilities(serialized)

            span = self._client.start_span(
                name=operation_name,
                attributes={
                    "langchain.run_id": str(run_id),
                    "agent.name": agent_name,
                    "agent.type": agent_type,
                    "agent.operation": "execution",
                    "agent.capabilities": agent_capabilities,
                    # Input attributes
                    "agent.input.inputs": {k: str(v) for k, v in inputs.items()},
                    **{
                        k: v
                        for k, v in kwargs.items()
                        if k not in ["tags", "metadata"]
                        and isinstance(v, (str, int, float, bool))
                    },
                },
            )

            # Store span for later cleanup
            self._set_run(run_id, span)

            # Track if we need to manage trace lifecycle
            if should_manage:
                self._trace_managed_by_langchain = trace

        except Exception as e:
            logger.error("Error handling agent start event: %s", e)

    def on_agent_action(
        self,
        action: "AgentAction",
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Handle agent action event."""
        if not self._ensure_client():
            return

        try:
            # Get the current agent span
            span = self._get_run(run_id)
            if span is None:
                return

            # Add agent output attributes
            span.set_attributes(
                {
                    "agent.output.action.tool": action.tool,
                    "agent.output.action.tool_input": str(action.tool_input),
                    "agent.output.action.log": action.log,
                }
            )

            # Add event for agent action (tool call decision)
            span.add_event(
                "agent_action",
                {
                    "action.tool": action.tool,
                    "action.tool_input": str(action.tool_input),
                    "action.log": action.log,
                },
            )

            # Also create a tool span for the tool execution
            # This handles cases where LangChain doesn't trigger on_tool_start/on_tool_end
            self._create_tool_span_from_action(action, run_id)

        except Exception as e:
            logger.error("Error handling agent action event: %s", e)

    def on_agent_finish(
        self,
        finish: "AgentFinish",
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Handle agent finish event."""
        if not self._ensure_client():
            return

        # Get and remove span from runs dict
        span = self._pop_run(run_id)
        if span is None:
            return

        try:
            # Complete any pending tool spans first
            self._complete_tool_spans_from_finish(finish, run_id)

            # Add agent output attributes
            span.set_attributes(
                {
                    "agent.output.finish.return_values": {
                        k: str(v) for k, v in finish.return_values.items()
                    },
                    "agent.output.finish.log": finish.log,
                }
            )

            # Add event for agent finish
            span.add_event(
                "agent_finish",
                {
                    "finish.return_values": {
                        k: str(v) for k, v in finish.return_values.items()
                    },
                    "finish.log": finish.log,
                },
            )

            span.set_status(SpanStatus.OK)
            self._client.finish_span(span)

            # Check if we should finish the trace
            if (
                self._trace_managed_by_langchain and self._active_runs() == 0
            ):  # No more active spans
                self._client.finish_trace(self._trace_managed_by_langchain)
                from noveum_trace.core.context import set_current_trace

                set_current_trace(None)
                self._trace_managed_by_langchain = None

        except Exception as e:
            logger.error("Error handling agent finish event: %s", e)

    # Retrieval Events
    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle retriever start event."""
        if not self._ensure_client():
            return

        operation_name = self._get_operation_name("retriever_start", serialized)

        try:
            # Get or create trace context
            trace, should_manage = self._get_or_create_trace_context(operation_name)

            # Create span
            span = self._client.start_span(
                name=operation_name,
                attributes={
                    "langchain.run_id": str(run_id),
                    "retrieval.type": "search",
                    "retrieval.operation": (
                        serialized.get("name", "unknown") if serialized else "unknown"
                    ),
                    # Input attributes
                    "retrieval.query": query,
                    **{
                        k: v
                        for k, v in kwargs.items()
                        if k not in ["tags", "metadata"]
                        and isinstance(v, (str, int, float, bool))
                    },
                },
            )

            # Store span for later cleanup
            self._set_run(run_id, span)

            # Track if we need to manage trace lifecycle
            if should_manage:
                self._trace_managed_by_langchain = trace

        except Exception as e:
            logger.error("Error handling retriever start event: %s", e)

    def on_retriever_end(
        self,
        documents: Sequence["Document"],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Handle retriever end event."""
        if not self._ensure_client():
            return None

        # Get and remove span from runs dict
        span = self._pop_run(run_id)
        if span is None:
            return None

        try:
            # Extract document content safely
            doc_previews = []
            for doc in documents[:10]:  # Limit to first 10 documents
                if hasattr(doc, "page_content"):
                    doc_previews.append(doc.page_content)

            span.set_attributes(
                {
                    # Output attributes
                    "retrieval.result_count": len(documents),
                    "retrieval.sample_results": doc_previews,
                    "retrieval.results_truncated": len(documents) > 10,
                }
            )

            span.set_status(SpanStatus.OK)
            self._client.finish_span(span)

            # Check if we should finish the trace
            if (
                self._trace_managed_by_langchain and self._active_runs() == 0
            ):  # No more active spans
                self._client.finish_trace(self._trace_managed_by_langchain)
                from noveum_trace.core.context import set_current_trace

                set_current_trace(None)
                self._trace_managed_by_langchain = None

        except Exception as e:
            logger.error("Error handling retriever end event: %s", e)

        return None

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Handle retriever error event."""
        if not self._ensure_client():
            return None

        # Get and remove span from runs dict
        span = self._pop_run(run_id)
        if span is None:
            return None

        try:
            span.record_exception(error)
            span.set_status(SpanStatus.ERROR, str(error))
            self._client.finish_span(span)

            # Check if we should finish the trace
            if (
                self._trace_managed_by_langchain and self._active_runs() == 0
            ):  # No more active spans
                self._client.finish_trace(self._trace_managed_by_langchain)
                from noveum_trace.core.context import set_current_trace

                set_current_trace(None)
                self._trace_managed_by_langchain = None

        except Exception as e:
            logger.error("Error handling retriever error event: %s", e)

        return None

    def on_text(
        self,
        text: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Handle text event (optional, for debugging)."""
        if not self._ensure_client():
            return

        try:
            span = self._get_run(run_id)
            if span is not None:
                span.add_event("text_output", {"text": text})
        except Exception as e:
            logger.error("Error handling text event: %s", e)

    def __repr__(self) -> str:
        """String representation of the callback handler."""
        return (
            f"NoveumTraceCallbackHandler("
            f"active_runs={self._active_runs()}, "
            f"managing_trace={self._trace_managed_by_langchain is not None})"
        )


# For backwards compatibility and ease of import
__all__ = ["NoveumTraceCallbackHandler"]
