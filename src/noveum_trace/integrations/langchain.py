"""
LangChain integration for Noveum Trace SDK.

This module provides a callback handler that automatically traces LangChain
operations including LLM calls, chains, agents, tools, and retrieval operations.
"""

import logging
from typing import Any, Optional
from collections.abc import Sequence
from uuid import UUID

from noveum_trace.core.span import SpanStatus

logger = logging.getLogger(__name__)

# Try to import LangChain dependencies
try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.documents import Document

    LANGCHAIN_AVAILABLE = True
except ImportError:
    logger.warning("LangChain not available. Install with: pip install langchain-core")
    LANGCHAIN_AVAILABLE = False
    # Create stub base class
    from typing import Protocol

    class BaseCallbackHandler(Protocol):  # type: ignore[no-redef]
        def __init__(self) -> None: ...


class NoveumTraceCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler for Noveum Trace integration."""

    def __init__(self) -> None:
        """Initialize the callback handler."""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is not installed. Install with: pip install langchain-core"
            )

        super().__init__()
        self._trace_stack: list[Any] = []  # Active traces
        self._span_stack: list[Any] = []  # Active spans
        self._current_trace: Optional[Any] = None  # Current trace context

        # Import here to avoid circular imports
        from noveum_trace import get_client

        try:
            self._client = get_client()
        except Exception as e:
            logger.warning("Failed to get Noveum Trace client: %s", e)
            self._client = None  # type: ignore[assignment]

    def _should_create_trace(
        self, event_type: str, _serialized: dict[str, Any]
    ) -> bool:
        """Determine if event should create new trace or just span."""
        if event_type in ["chain_start", "agent_start"]:
            return True  # Always create trace for chains/agents

        if event_type in ["llm_start", "retriever_start"]:
            return len(self._trace_stack) == 0  # Only if not nested

        return False

    def _get_operation_name(self, event_type: str, serialized: dict[str, Any]) -> str:
        """Generate standardized operation names."""
        if serialized is None:
            return f"{event_type}.unknown"
        name = serialized.get("name", "unknown")

        if event_type == "llm_start":
            return f"llm.{name}"
        elif event_type == "chain_start":
            return f"chain.{name}"
        elif event_type == "agent_start":
            return f"agent.{name}"
        elif event_type == "retriever_start":
            return f"retrieval.{name}"
        elif event_type == "tool_start":
            return f"tool.{name}"

        return f"{event_type}.{name}"

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
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Handle LLM start event."""
        if not self._ensure_client():
            return

        operation_name = self._get_operation_name("llm_start", serialized)

        try:
            if self._should_create_trace("llm_start", serialized):
                # Standalone LLM call - create new trace
                self._current_trace = self._client.start_trace(operation_name)
                self._trace_stack.append(self._current_trace)

            # Create span (either in new trace or existing trace)
            span = self._client.start_span(
                name=operation_name,
                attributes={
                    "langchain.run_id": str(run_id),
                    "llm.model": (
                        serialized.get("name", "unknown") if serialized else "unknown"
                    ),
                    "llm.provider": (
                        serialized.get("id", ["unknown"])[-1]
                        if serialized and isinstance(serialized.get("id"), list)
                        else (
                            serialized.get("id", "unknown") if serialized else "unknown"
                        )
                    ),
                    # Limit to avoid large payloads
                    "llm.prompts": prompts[:5] if len(prompts) > 5 else prompts,
                    "llm.prompt_count": len(prompts),
                    **{
                        k: v
                        for k, v in kwargs.items()
                        if k not in ["tags", "metadata"]
                        and isinstance(v, (str, int, float, bool))
                    },
                },
            )
            self._span_stack.append(span)

        except Exception as e:
            logger.error("Error handling LLM start event: %s", e)

    def on_llm_end(self, response: "LLMResult", run_id: UUID, **kwargs: Any) -> None:
        """Handle LLM end event."""
        if not self._ensure_client() or not self._span_stack:
            return

        try:
            span = self._span_stack.pop()

            # Add response data
            generations = []
            token_usage = {}

            if hasattr(response, "generations") and response.generations:
                generations = [
                    gen.text[:500] + "..." if len(gen.text) > 500 else gen.text
                    for generation_list in response.generations
                    for gen in generation_list
                ][
                    :10
                ]  # Limit number of generations

            if hasattr(response, "llm_output") and response.llm_output:
                token_usage = response.llm_output.get("token_usage", {})

            span.set_attributes(
                {
                    "llm.response": generations,
                    "llm.response_count": len(generations),
                    "llm.usage": token_usage,
                    "llm.finish_reason": (
                        response.llm_output.get("finish_reason")
                        if hasattr(response, "llm_output") and response.llm_output
                        else None
                    ),
                }
            )

            span.set_status(SpanStatus.OK)
            self._client.finish_span(span)

            # Finish trace if this was a standalone LLM call
            if self._current_trace and len(self._span_stack) == 0:
                self._client.finish_trace(self._current_trace)
                self._trace_stack.pop()
                self._current_trace = None

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
        if not self._ensure_client() or not self._span_stack:
            return None

        try:
            span = self._span_stack.pop()
            span.record_exception(error)
            span.set_status(SpanStatus.ERROR, str(error))
            self._client.finish_span(span)

            # Finish trace if this was a standalone LLM call
            if self._current_trace and len(self._span_stack) == 0:
                self._client.finish_trace(self._current_trace)
                self._trace_stack.pop()
                self._current_trace = None

        except Exception as e:
            logger.error("Error handling LLM error event: %s", e)

        return None

    # Chain Events
    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Handle chain start event."""
        if not self._ensure_client():
            return

        operation_name = self._get_operation_name("chain_start", serialized)

        try:
            if self._should_create_trace("chain_start", serialized):
                # Create new trace for chain
                self._current_trace = self._client.start_trace(operation_name)
                self._trace_stack.append(self._current_trace)

            # Create span for chain
            span = self._client.start_span(
                name=operation_name,
                attributes={
                    "langchain.run_id": str(run_id),
                    "chain.name": (
                        serialized.get("name", "unknown") if serialized else "unknown"
                    ),
                    "chain.inputs": {
                        k: str(v)[:200] + "..." if len(str(v)) > 200 else str(v)
                        for k, v in inputs.items()
                    },
                    **{
                        k: v
                        for k, v in kwargs.items()
                        if k not in ["tags", "metadata"]
                        and isinstance(v, (str, int, float, bool))
                    },
                },
            )
            self._span_stack.append(span)

        except Exception as e:
            logger.error("Error handling chain start event: %s", e)

    def on_chain_end(
        self, outputs: dict[str, Any], run_id: UUID, **kwargs: Any
    ) -> None:
        """Handle chain end event."""
        if not self._ensure_client() or not self._span_stack:
            return

        try:
            span = self._span_stack.pop()

            span.set_attributes(
                {
                    "chain.outputs": {
                        k: str(v)[:200] + "..." if len(str(v)) > 200 else str(v)
                        for k, v in outputs.items()
                    }
                }
            )

            span.set_status(SpanStatus.OK)
            self._client.finish_span(span)

            # Finish trace if this was the top-level chain
            if self._current_trace and len(self._span_stack) == 0:
                self._client.finish_trace(self._current_trace)
                self._trace_stack.pop()
                self._current_trace = None

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
        if not self._ensure_client() or not self._span_stack:
            return None

        try:
            span = self._span_stack.pop()
            span.record_exception(error)
            span.set_status(SpanStatus.ERROR, str(error))
            self._client.finish_span(span)

            # Finish trace if this was the top-level chain
            if self._current_trace and len(self._span_stack) == 0:
                self._client.finish_trace(self._current_trace)
                self._trace_stack.pop()
                self._current_trace = None

        except Exception as e:
            logger.error("Error handling chain error event: %s", e)

        return None

    # Tool Events
    def on_tool_start(
        self, serialized: dict[str, Any], input_str: str, run_id: UUID, **kwargs: Any
    ) -> None:
        """Handle tool start event."""
        if not self._ensure_client():
            return

        operation_name = self._get_operation_name("tool_start", serialized)

        try:
            # Tools always create spans (never standalone traces)
            span = self._client.start_span(
                name=operation_name,
                attributes={
                    "langchain.run_id": str(run_id),
                    "tool.name": (
                        serialized.get("name", "unknown") if serialized else "unknown"
                    ),
                    "tool.input": (
                        input_str[:500] + "..." if len(input_str) > 500 else input_str
                    ),
                    **{
                        k: v
                        for k, v in kwargs.items()
                        if k not in ["tags", "metadata"]
                        and isinstance(v, (str, int, float, bool))
                    },
                },
            )
            self._span_stack.append(span)

        except Exception as e:
            logger.error("Error handling tool start event: %s", e)

    def on_tool_end(self, output: str, run_id: UUID, **kwargs: Any) -> None:
        """Handle tool end event."""
        if not self._ensure_client() or not self._span_stack:
            return

        try:
            span = self._span_stack.pop()

            span.set_attributes(
                {"tool.output": output[:500] + "..." if len(output) > 500 else output}
            )

            span.set_status(SpanStatus.OK)
            self._client.finish_span(span)

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
        if not self._ensure_client() or not self._span_stack:
            return None

        try:
            span = self._span_stack.pop()
            span.record_exception(error)
            span.set_status(SpanStatus.ERROR, str(error))
            self._client.finish_span(span)

        except Exception as e:
            logger.error("Error handling tool error event: %s", e)

        return None

    # Agent Events
    def on_agent_action(
        self, action: "AgentAction", run_id: UUID, **kwargs: Any
    ) -> None:
        """Handle agent action event."""
        if not self._ensure_client() or not self._span_stack:
            return

        try:
            span = self._span_stack[-1]  # Add to current span
            span.add_event(
                "agent_action",
                {
                    "action.tool": action.tool,
                    "action.tool_input": (
                        str(action.tool_input)[:200] + "..."
                        if len(str(action.tool_input)) > 200
                        else str(action.tool_input)
                    ),
                    "action.log": (
                        action.log[:300] + "..."
                        if len(action.log) > 300
                        else action.log
                    ),
                },
            )

        except Exception as e:
            logger.error("Error handling agent action event: %s", e)

    def on_agent_finish(
        self, finish: "AgentFinish", run_id: UUID, **kwargs: Any
    ) -> None:
        """Handle agent finish event."""
        if not self._ensure_client() or not self._span_stack:
            return

        try:
            span = self._span_stack[-1]  # Add to current span
            span.add_event(
                "agent_finish",
                {
                    "finish.return_values": {
                        k: str(v)[:200] + "..." if len(str(v)) > 200 else str(v)
                        for k, v in finish.return_values.items()
                    },
                    "finish.log": (
                        finish.log[:300] + "..."
                        if len(finish.log) > 300
                        else finish.log
                    ),
                },
            )

        except Exception as e:
            logger.error("Error handling agent finish event: %s", e)

    # Retrieval Events
    def on_retriever_start(
        self, serialized: dict[str, Any], query: str, run_id: UUID, **kwargs: Any
    ) -> None:
        """Handle retriever start event."""
        if not self._ensure_client():
            return

        operation_name = self._get_operation_name("retriever_start", serialized)

        try:
            if self._should_create_trace("retriever_start", serialized):
                # Standalone retrieval - create new trace
                self._current_trace = self._client.start_trace(operation_name)
                self._trace_stack.append(self._current_trace)

            # Create span
            span = self._client.start_span(
                name=operation_name,
                attributes={
                    "langchain.run_id": str(run_id),
                    "retrieval.query": (
                        query[:300] + "..." if len(query) > 300 else query
                    ),
                    "retrieval.source": (
                        serialized.get("name", "unknown") if serialized else "unknown"
                    ),
                    **{
                        k: v
                        for k, v in kwargs.items()
                        if k not in ["tags", "metadata"]
                        and isinstance(v, (str, int, float, bool))
                    },
                },
            )
            self._span_stack.append(span)

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
        if not self._ensure_client() or not self._span_stack:
            return None

        try:
            span = self._span_stack.pop()

            # Extract document content safely
            doc_previews = []
            for doc in documents[:10]:  # Limit to first 10 documents
                if hasattr(doc, "page_content"):
                    content = (
                        doc.page_content[:200] + "..."
                        if len(doc.page_content) > 200
                        else doc.page_content
                    )
                    doc_previews.append(content)

            span.set_attributes(
                {
                    "retrieval.documents_count": len(documents),
                    "retrieval.documents": doc_previews,
                }
            )

            span.set_status(SpanStatus.OK)
            self._client.finish_span(span)

            # Finish trace if this was a standalone retrieval
            if self._current_trace and len(self._span_stack) == 0:
                self._client.finish_trace(self._current_trace)
                self._trace_stack.pop()
                self._current_trace = None

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
        if not self._ensure_client() or not self._span_stack:
            return None

        try:
            span = self._span_stack.pop()
            span.record_exception(error)
            span.set_status(SpanStatus.ERROR, str(error))
            self._client.finish_span(span)

            # Finish trace if this was a standalone retrieval
            if self._current_trace and len(self._span_stack) == 0:
                self._client.finish_trace(self._current_trace)
                self._trace_stack.pop()
                self._current_trace = None

        except Exception as e:
            logger.error("Error handling retriever error event: %s", e)

        return None

    def on_text(self, text: str, run_id: UUID, **kwargs: Any) -> None:
        """Handle text event (optional, for debugging)."""
        if not self._ensure_client() or not self._span_stack:
            return

        try:
            span = self._span_stack[-1]
            span.add_event(
                "text_output", {"text": text[:200] + "..." if len(text) > 200 else text}
            )
        except Exception as e:
            logger.error("Error handling text event: %s", e)

    def __repr__(self) -> str:
        """String representation of the callback handler."""
        return (
            f"NoveumTraceCallbackHandler("
            f"active_traces={len(self._trace_stack)}, "
            f"active_spans={len(self._span_stack)})"
        )


# For backwards compatibility and ease of import
__all__ = ["NoveumTraceCallbackHandler"]
