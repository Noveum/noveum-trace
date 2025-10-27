"""
LangChain integration for Noveum Trace SDK.

This module provides a callback handler that automatically traces LangChain
operations including LLM calls, chains, agents, tools, and retrieval operations.
"""

import logging
import threading
from collections.abc import Sequence
from typing import Any, Optional, Union
from uuid import UUID

# Import LangChain dependencies
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.outputs import LLMResult

from noveum_trace.core.span import SpanStatus
from noveum_trace.integrations.langchain_utils import (
    build_langgraph_attributes,
    build_routing_attributes,
    extract_agent_capabilities,
    extract_agent_type,
    extract_langgraph_metadata,
    extract_model_name,
    extract_noveum_metadata,
    extract_tool_function_name,
    get_operation_name,
)

logger = logging.getLogger(__name__)


class NoveumTraceCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler for Noveum Trace integration."""

    def __init__(self, use_langchain_assigned_parent: bool = False) -> None:
        """Initialize the callback handler.

        Args:
            use_langchain_assigned_parent: If True, use LangChain's parent_run_id
                to determine parent span relationships instead of context-based
                parent assignment. Falls back to context-based with warning if
                parent_run_id lookup fails.
        """
        super().__init__()

        # Thread-safe runs dictionary for span tracking
        # Maps run_id -> span (for backward compatibility)
        self.runs: dict[Union[UUID, str], Any] = {}
        self._runs_lock = threading.Lock()

        # Custom name mapping for explicit parent relationships
        # Maps custom name -> span_id (kept for handler's lifetime)
        self.names: dict[str, str] = {}
        self._names_lock = threading.Lock()

        # Track if we're managing a trace lifecycle
        self._trace_managed_by_langchain: Optional[Any] = None

        # Track if trace is manually controlled (started via start_trace())
        self._manual_trace_control: bool = False

        # Parent assignment mode
        self._use_langchain_assigned_parent = use_langchain_assigned_parent

        # Import here to avoid circular imports
        from noveum_trace import get_client

        try:
            self._client = get_client()
        except Exception as e:
            logger.warning("Failed to get Noveum Trace client: %s", e)
            self._client = None  # type: ignore[assignment]

    def _set_run(self, run_id: Union[UUID, str], span: Any) -> None:
        """Thread-safe method to set a run span."""
        with self._runs_lock:
            self.runs[run_id] = span

    def _pop_run(self, run_id: Union[UUID, str]) -> Any:
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

    def _set_name(self, name: str, span_id: str) -> None:
        """Thread-safe method to set a custom name mapping."""
        with self._names_lock:
            self.names[name] = span_id

    def _get_span_id_by_name(self, name: str) -> Optional[str]:
        """Thread-safe method to get a span_id by custom name."""
        with self._names_lock:
            return self.names.get(name)

    def _get_parent_span_id_from_name(self, parent_name: str) -> Optional[str]:
        """
        Get parent span ID from custom parent name.

        Args:
            parent_name: Custom name of parent span

        Returns:
            Parent span ID if found, None otherwise
        """
        span_id = self._get_span_id_by_name(parent_name)
        if span_id is None:
            logger.warning(
                f"Parent span with name '{parent_name}' not found. "
                "Falling back to auto-discovery."
            )
            return None

        return span_id

    def _resolve_parent_span_id(
        self, parent_run_id: Optional[UUID], parent_name: Optional[str]
    ) -> Optional[str]:
        """
        Resolve parent span ID based on mode.

        When use_langchain_assigned_parent=True:
        - Use parent_run_id to look up parent span
        - Fallback to parent_name if parent_run_id lookup fails
        - Fallback to context-based parent with WARNING if both fail

        When use_langchain_assigned_parent=False:
        - Use parent_name if provided
        - Otherwise return None (uses context-based parent normally)

        Args:
            parent_run_id: LangChain's parent run ID
            parent_name: Custom parent name from metadata

        Returns:
            Parent span ID if resolved, None otherwise
        """
        if self._use_langchain_assigned_parent:
            # Try parent_run_id first
            if parent_run_id:
                parent_span = self._get_run(parent_run_id)
                if parent_span:
                    return parent_span.span_id

            # Fallback to parent_name
            if parent_name:
                span_id = self._get_parent_span_id_from_name(parent_name)
                if span_id:
                    return span_id

            # Final fallback: context-based parent with WARNING
            from noveum_trace.core.context import get_current_span

            current_span = get_current_span()
            if current_span:
                logger.warning(
                    f"Could not resolve parent from parent_run_id ({parent_run_id}) "
                    f"or parent_name ({parent_name}). Auto-assigning parent span "
                    f"from context: {current_span.span_id}"
                )
                return current_span.span_id

            # No parent found at all
            return None
        else:
            # Legacy behavior: only use parent_name
            if parent_name:
                return self._get_parent_span_id_from_name(parent_name)
            return None

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
            # Use existing trace - don't manage its lifecycle if manually controlled
            should_manage = False
            return existing_trace, should_manage
        else:
            # Only create auto trace if not manually controlled
            if self._manual_trace_control:
                # Manual control but no trace - this shouldn't happen
                # but fallback gracefully
                logger.warning(
                    "Manual trace control enabled but no trace found. "
                    "Call start_trace() first."
                )

            # Create new trace in global context
            new_trace = self._client.start_trace(operation_name)
            set_current_trace(new_trace)
            return new_trace, True

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

    def start_trace(self, name: str) -> None:
        """
        Manually start a trace.

        This disables auto-finishing behavior - you must call end_trace()
        to finish the trace.

        Args:
            name: Name for the trace

        """
        if not self._ensure_client():
            logger.error(
                "Noveum Trace client is not available. Tracing functionality will be disabled."
            )

        from noveum_trace.core.context import get_current_trace, set_current_trace

        # Check if trace already exists
        existing_trace = get_current_trace()
        if existing_trace is not None:
            logger.warning(
                f"A trace is already active: {existing_trace.trace_id}. "
                "Calling end_trace() prematurely may cause unexpected trace structure."
            )

        # Create new trace
        trace = self._client.start_trace(name)
        set_current_trace(trace)

        # Enable manual control - disables auto-finishing
        self._manual_trace_control = True
        self._trace_managed_by_langchain = trace

        logger.debug(f"Manually started trace: {trace.trace_id}")

    def end_trace(self) -> None:
        """
        Manually end the current trace.

        This replicates the auto-finishing behavior but is called explicitly.
        Clears the trace from context and re-enables auto-management for
        future traces.

        """
        if not self._ensure_client():
            logger.warning(
                "Noveum Trace client is not available; unable to end the trace."
            )
            return

        from noveum_trace.core.context import get_current_trace, set_current_trace

        # Get current trace
        trace = get_current_trace()
        if trace is None:
            logger.error("No active trace to end")
            return

        # Finish the trace
        self._client.finish_trace(trace)

        # Clear context
        set_current_trace(None)
        self._trace_managed_by_langchain = None
        self._manual_trace_control = False

        logger.debug(f"Manually ended trace: {trace.trace_id}")

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

    def _finish_trace_if_needed(self) -> None:
        """Finish the trace if we're managing it and no active spans remain."""
        # Don't auto-finish manually controlled traces
        if self._manual_trace_control:
            return

        if (
            self._trace_managed_by_langchain and self._active_runs() == 0
        ):  # No more active spans
            self._client.finish_trace(self._trace_managed_by_langchain)
            from noveum_trace.core.context import set_current_trace

            set_current_trace(None)
            self._trace_managed_by_langchain = None

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

        operation_name = get_operation_name("llm_start", serialized)

        try:
            # Extract Noveum-specific metadata
            noveum_config = extract_noveum_metadata(metadata)
            custom_name = noveum_config.get("name")
            parent_name = noveum_config.get("parent_name")

            # Use custom name if provided, otherwise use operation name
            span_name = custom_name if custom_name else operation_name

            # Resolve parent span ID based on mode
            parent_span_id = self._resolve_parent_span_id(parent_run_id, parent_name)

            # Get or create trace context
            trace, should_manage = self._get_or_create_trace_context(span_name)

            # Create span
            span = self._client.start_span(
                name=span_name,
                parent_span_id=parent_span_id,
                attributes={
                    "langchain.run_id": str(run_id),
                    "llm.model": extract_model_name(serialized),
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

            # Store custom name mapping if provided
            if custom_name:
                self._set_name(custom_name, span.span_id)

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
            self._finish_trace_if_needed()

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
            self._finish_trace_if_needed()

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

        try:
            # Extract LangGraph-specific metadata (with safe fallbacks)
            langgraph_metadata = extract_langgraph_metadata(
                metadata=metadata, tags=tags, serialized=serialized
            )

            # Generate operation name with LangGraph support
            operation_name = get_operation_name(
                "chain_start", serialized, langgraph_metadata=langgraph_metadata
            )

            # Extract Noveum-specific metadata
            noveum_config = extract_noveum_metadata(metadata)
            custom_name = noveum_config.get("name")
            parent_name = noveum_config.get("parent_name")

            # Use custom name if provided, otherwise use operation name
            span_name = custom_name if custom_name else operation_name

            # Resolve parent span ID based on mode
            parent_span_id = self._resolve_parent_span_id(parent_run_id, parent_name)

            # Get or create trace context
            trace, should_manage = self._get_or_create_trace_context(span_name)

            # Build base attributes
            attributes = {
                "langchain.run_id": str(run_id),
                "chain.name": (
                    serialized.get("name", "unknown") if serialized else "unknown"
                ),
                "chain.operation": "execution",
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k not in ["tags", "metadata"]
                    and isinstance(v, (str, int, float, bool))
                },
            }
            
            # Handle inputs based on type
            if isinstance(inputs, list):
                # LangGraph prebuilt agents send list of tool calls
                for index, input_dict in enumerate(inputs):
                    if isinstance(input_dict, dict):
                        attributes[f"chain.inputs.{index}"] = {k: str(v) for k, v in input_dict.items()}
                    else:
                        attributes[f"chain.inputs.{index}"] = str(input_dict)
            elif isinstance(inputs, dict):
                # Standard dict inputs
                attributes["chain.inputs"] = {k: str(v) for k, v in inputs.items()}
            else:
                # Fallback for other types
                attributes["chain.inputs"] = str(inputs)

            # Add LangGraph-specific attributes if available
            langgraph_attrs = build_langgraph_attributes(langgraph_metadata)
            if langgraph_attrs:
                attributes.update(langgraph_attrs)

            # Create span for chain
            span = self._client.start_span(
                name=span_name,
                parent_span_id=parent_span_id,
                attributes=attributes,
            )

            # Store span for later cleanup
            self._set_run(run_id, span)

            # Store custom name mapping if provided
            if custom_name:
                self._set_name(custom_name, span.span_id)

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
            # Handle both dict and non-dict outputs
            if isinstance(outputs, dict):
                output_attrs = {k: str(v) for k, v in outputs.items()}
            else:
                # Handle string or other primitive types
                output_attrs = str(outputs)

            span.set_attributes(
                {
                    # Output attributes
                    "chain.output.outputs": output_attrs
                }
            )

            span.set_status(SpanStatus.OK)
            self._client.finish_span(span)

            # Check if we should finish the trace
            self._finish_trace_if_needed()

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
            self._finish_trace_if_needed()

        except Exception as e:
            logger.error("Error handling chain error event: %s", e)

        return None

    # Custom Events, primarily used for routing decisions
    def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: UUID,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Handle custom events including routing decisions.

        Args:
            name: Event name (e.g., "langgraph.routing_decision")
            data: Event data/payload
            run_id: Run ID of the parent span (source node)
            tags: Optional tags
            metadata: Optional metadata
        """
        if name == "langgraph.routing_decision":
            self._handle_routing_decision(data, run_id)

    def _handle_routing_decision(
        self,
        payload: dict[str, Any],
        run_id: UUID,
    ) -> None:
        """
        Handle routing decision by creating a separate span.

        Routing spans follow the same structure as LLM/Chain/Tool spans:
        1. Create span with create_span()
        2. Set attributes
        3. Set status
        4. Finish span with finish_span()

        Note: The run_id parameter is the PARENT's run_id (source node).
        The routing span is created and finished immediately without being
        stored in self.runs since it has no lifecycle to manage.

        Args:
            payload: Routing decision data from user
            run_id: Parent node's run_id (used to find parent span)
        """
        if not self._ensure_client():
            return

        try:
            # Extract routing information
            source_node = payload.get("source_node", "unknown")
            target_node = payload.get("target_node", "unknown")

            # Create span name following pattern: routing.{source}_to_{target}
            span_name = f"routing.{source_node}_to_{target_node}"

            # Get current trace
            from noveum_trace.core.context import get_current_trace

            trace = get_current_trace()

            if not trace:
                logger.warning("No trace context for routing decision")
                return

            # Determine parent span
            # run_id is the PARENT's run_id (the node making the routing decision)
            parent_span = self._get_run(run_id)
            parent_span_id = parent_span.span_id if parent_span else None

            # If no parent span, routing span becomes root-level span under trace
            if not parent_span_id:
                logger.debug(
                    "No parent span for routing decision, creating as root-level span"
                )

            # Create routing span (same method as LLM/Chain/Tool spans)
            routing_span = self._client.start_span(
                name=span_name,
                parent_span_id=parent_span_id,  # None if no parent = root span
            )

            # Build attributes from payload
            attributes = build_routing_attributes(payload)

            # Set all attributes
            routing_span.set_attributes(attributes)

            # Set status to OK (routing decisions are successful operations)
            routing_span.set_status(SpanStatus.OK)

            # Finish span immediately (routing is instant operation)
            self._client.finish_span(routing_span)

            # Note: We do NOT store routing_span in self.runs because:
            # 1. It's already finished
            # 2. It has no lifecycle events to track
            # 3. It won't receive any future callbacks
            # 4. The run_id we have is the parent's, not this span's

            logger.debug(
                f"Created routing span: {span_name} "
                f"(parent: {parent_span_id or 'root'})"
            )

        except Exception as e:
            logger.error(f"Error handling routing decision: {e}", exc_info=True)

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

        get_operation_name("tool_start", serialized)

        try:
            # Extract Noveum-specific metadata
            noveum_config = extract_noveum_metadata(metadata)
            custom_name = noveum_config.get("name")
            parent_name = noveum_config.get("parent_name")

            tool_name = serialized.get("name", "unknown") if serialized else "unknown"

            # Extract actual function name from serialized data
            func_name = extract_tool_function_name(serialized)

            # Use custom name if provided, otherwise use standard format
            span_name = custom_name if custom_name else f"tool:{tool_name}:{func_name}"

            # Resolve parent span ID based on mode
            parent_span_id = self._resolve_parent_span_id(parent_run_id, parent_name)

            # Get or create trace context
            trace, should_manage = self._get_or_create_trace_context(span_name)

            # Prepare input attributes
            input_attrs = {
                "tool.input.input_str": input_str,  # String representation for compatibility
            }

            # Add structured inputs if available - handle different input types
            if inputs is not None:
                if isinstance(inputs, dict):
                    # Case 1: Dict input (current/standard behavior)
                    for key, value in inputs.items():
                        # Convert values to strings for attribute storage
                        input_attrs[f"tool.input.{key}"] = str(value)
                    input_attrs["tool.input.argument_count"] = str(len(inputs))
                    
                elif isinstance(inputs, list):
                    # Check if all elements are dicts
                    if inputs and all(isinstance(item, dict) for item in inputs):
                        # Case 2: List of dicts (similar to chain handling)
                        for index, input_dict in enumerate(inputs):
                            input_attrs[f"tool.input.{index}"] = {k: str(v) for k, v in input_dict.items()}
                        input_attrs["tool.input.argument_count"] = str(len(inputs))
                    else:
                        # Case 3: List but not all elements are dicts
                        for index, item in enumerate(inputs):
                            input_attrs[f"tool.input.{index}"] = str(item)
                        input_attrs["tool.input.argument_count"] = str(len(inputs))
                        
                elif isinstance(inputs, tuple):
                    # Case 4: Tuple input (treat like list case 3)
                    for index, item in enumerate(inputs):
                        input_attrs[f"tool.input.{index}"] = str(item)
                    input_attrs["tool.input.argument_count"] = str(len(inputs))
                    
                else:
                    # Case 5: Single value (str, int, etc.)
                    input_attrs["tool.input.arg"] = str(inputs)
                    input_attrs["tool.input.argument_count"] = "1"
            else:
                input_attrs["tool.input.argument_count"] = "0"

            span = self._client.start_span(
                name=span_name,
                parent_span_id=parent_span_id,
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

            # Store custom name mapping if provided
            if custom_name:
                self._set_name(custom_name, span.span_id)

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
            self._finish_trace_if_needed()

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
            self._finish_trace_if_needed()

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

        operation_name = get_operation_name("agent_start", serialized)

        try:
            # Extract Noveum-specific metadata
            noveum_config = extract_noveum_metadata(metadata)
            custom_name = noveum_config.get("name")
            parent_name = noveum_config.get("parent_name")

            # Use custom name if provided, otherwise use operation name
            span_name = custom_name if custom_name else operation_name

            # Resolve parent span ID based on mode
            parent_span_id = self._resolve_parent_span_id(parent_run_id, parent_name)

            # Get or create trace context
            trace, should_manage = self._get_or_create_trace_context(span_name)

            # Create span for agent
            agent_name = serialized.get("name", "unknown") if serialized else "unknown"

            # Extract actual agent information from serialized data
            agent_type = extract_agent_type(serialized)
            agent_capabilities = extract_agent_capabilities(serialized)

            span = self._client.start_span(
                name=span_name,
                parent_span_id=parent_span_id,
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

            # Store custom name mapping if provided
            if custom_name:
                self._set_name(custom_name, span.span_id)

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
            self._finish_trace_if_needed()

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

        operation_name = get_operation_name("retriever_start", serialized)

        try:
            # Extract Noveum-specific metadata
            noveum_config = extract_noveum_metadata(metadata)
            custom_name = noveum_config.get("name")
            parent_name = noveum_config.get("parent_name")

            # Use custom name if provided, otherwise use operation name
            span_name = custom_name if custom_name else operation_name

            # Resolve parent span ID based on mode
            parent_span_id = self._resolve_parent_span_id(parent_run_id, parent_name)

            # Get or create trace context
            trace, should_manage = self._get_or_create_trace_context(span_name)

            # Create span
            span = self._client.start_span(
                name=span_name,
                parent_span_id=parent_span_id,
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

            # Store custom name mapping if provided
            if custom_name:
                self._set_name(custom_name, span.span_id)

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
            self._finish_trace_if_needed()

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
            self._finish_trace_if_needed()

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
        with self._names_lock:
            named_spans = len(self.names)
        return (
            f"NoveumTraceCallbackHandler("
            f"active_runs={self._active_runs()}, "
            f"named_spans={named_spans}, "
            f"managing_trace={self._trace_managed_by_langchain is not None}, "
            f"manual_control={self._manual_trace_control}, "
            f"use_langchain_parent={self._use_langchain_assigned_parent})"
        )


# For backwards compatibility and ease of import
__all__ = ["NoveumTraceCallbackHandler"]
