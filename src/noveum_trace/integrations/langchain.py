"""
LangChain integration for Noveum Trace SDK.

This module provides a callback handler that automatically traces LangChain
operations including LLM calls, chains, agents, tools, and retrieval operations.
"""

import inspect
import logging
import threading
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any, Optional, Protocol
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
from noveum_trace.utils.llm_utils import estimate_cost, estimate_token_count

logger = logging.getLogger(__name__)


# Helper function for safe input conversion
def safe_inputs_to_dict(inputs: Any, prefix: str = "item") -> dict[str, str]:
    """Safely convert inputs to dict for span attributes."""
    if isinstance(inputs, dict):
        return {k: str(v) for k, v in inputs.items()}
    elif isinstance(inputs, (list, tuple)):
        return {f"{prefix}_{i}": str(v) for i, v in enumerate(inputs)}
    else:
        return {prefix: str(inputs)}


def get_code_location(skip_frames: int = 2) -> dict[str, Any]:
    """
    Extract code location information from the current call stack.

    This function inspects the call stack to extract file path, function name,
    and line number information. It's useful for adding code context to trace
    attributes in async/parallel execution scenarios.

    Args:
        skip_frames: Number of frames to skip (default: 2 to skip get_code_location
                     and the calling frame)

    Returns:
        Dictionary with keys:
        - code.filepath: Path to the source file
        - code.function: Name of the function
        - code.lineno: Line number in the file
        Returns empty dict if frame inspection fails.
    """
    try:
        # Get the current frame
        frame = inspect.currentframe()
        if frame is None:
            return {}

        # Traverse up the call stack to skip the specified number of frames
        current_frame = frame
        for _ in range(skip_frames):
            if current_frame is None:
                return {}
            current_frame = current_frame.f_back

        if current_frame is None:
            return {}

        # Get frame info
        frame_info = inspect.getframeinfo(current_frame)

        return {
            "code.filepath": frame_info.filename,
            "code.function": frame_info.function,
            "code.lineno": frame_info.lineno,
        }
    except Exception:
        # Return empty dict on any exception
        return {}


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
        self.runs: "dict[Union[UUID, str], Any]" = {}
        self._runs_lock = threading.Lock()

        # Track root traces by root run_id
        # Maps root_run_id -> trace (for LangGraph workflow grouping)
        self.root_traces: "dict[Union[UUID, str], Any]" = {}
        self._root_traces_lock = threading.Lock()

        # Track parent relationships
        # Maps run_id -> parent_run_id (self.parent_map)
        self.parent_map: "dict[Union[UUID, str], Optional[Union[UUID, str]]]" = {}
        self._parent_map_lock = threading.Lock()

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

    def _set_run(self, run_id: "Union[UUID, str]", span: Any) -> None:
        """Thread-safe method to set a run span."""
        with self._runs_lock:
            self.runs[run_id] = span

    def _pop_run(self, run_id: "Union[UUID, str]") -> Any:
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

    def _set_root_trace(self, root_run_id: "Union[UUID, str]", trace: Any) -> None:
        """Thread-safe method to set a root trace."""
        with self._root_traces_lock:
            self.root_traces[root_run_id] = trace

    def _get_root_trace(self, root_run_id: "Union[UUID, str]") -> Any:
        """Thread-safe method to get a root trace."""
        with self._root_traces_lock:
            return self.root_traces.get(root_run_id)

    def _set_parent(
        self, run_id: "Union[UUID, str]", parent_run_id: "Optional[Union[UUID, str]]"
    ) -> None:
        """Thread-safe method to set parent relationship."""
        with self._parent_map_lock:
            self.parent_map[run_id] = parent_run_id

    def _get_parent(self, run_id: "Union[UUID, str]") -> "Optional[Union[UUID, str]]":
        """Thread-safe method to get parent run_id."""
        with self._parent_map_lock:
            return self.parent_map.get(run_id)

    def _find_root_run_id_for_trace(
        self, target_trace: Any
    ) -> "Optional[Union[UUID, str]]":
        """Find the root_run_id associated with a specific trace object."""
        with self._root_traces_lock:
            for root_run_id, trace in self.root_traces.items():
                if trace is target_trace:
                    return root_run_id
        return None

    def _is_descendant_of(
        self, run_id: "Union[UUID, str]", potential_ancestor: "Union[UUID, str]"
    ) -> bool:
        """Check if run_id is a descendant of potential_ancestor in the parent chain."""
        current = self._get_parent(run_id)
        visited = {run_id}  # Avoid cycles

        while current is not None and current not in visited:
            if current == potential_ancestor:
                return True
            visited.add(current)
            current = self._get_parent(current)

        return False

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

    def _extract_invocation_param(
        self,
        serialized: Optional[dict[str, Any]],
        kwargs: dict[str, Any],
        key: str,
    ) -> Any:
        """Extract a specific invocation parameter from LangChain metadata."""

        sources: list[dict[str, Any]] = []

        if kwargs:
            invocation_params = kwargs.get("invocation_params")
            if isinstance(invocation_params, dict):
                sources.append(invocation_params)
            sources.append(kwargs)

        if serialized:
            serialized_kwargs = serialized.get("kwargs")
            if isinstance(serialized_kwargs, dict):
                sources.append(serialized_kwargs)

        for source in sources:
            if key in source and source[key] is not None:
                return source[key]

        return None

    def _extract_model_name(self, serialized: dict[str, Any]) -> str:
        """Extract and normalize model name from serialized LLM data."""
        if not serialized:
            return "unknown"

        # Try to get model name from kwargs
        kwargs = serialized.get("kwargs", {})
        model = kwargs.get("model")
        if model:
            # Clean up model name (remove common prefixes)
            model_str = str(model).strip()
            # Remove "models/" prefix if present (e.g., "models/gemini-2.0-flash" -> "gemini-2.0-flash")
            if model_str.startswith("models/"):
                model_str = model_str[7:]  # Remove "models/" prefix
            # Use normalize_model_name utility instead of hardcoding prefixes
            try:
                from noveum_trace.utils.llm_utils import normalize_model_name

                normalized = normalize_model_name(model_str)
                return normalized
            except Exception:
                # Fallback to basic prefix removal
                for prefix in [
                    "openai/",
                    "anthropic/",
                    "google/",
                    "meta/",
                    "microsoft/",
                    "gemini/",
                ]:
                    if model_str.startswith(prefix):
                        model_str = model_str[len(prefix) :]
                return model_str

        # Fallback to provider name from id path
        id_path = serialized.get("id", [])
        if len(id_path) >= 2:
            # e.g., "openai" from ["langchain", "chat_models", "openai", "ChatOpenAI"]
            return id_path[-2]

        # Final fallback to class name
        return serialized.get("name", "unknown")

    def _extract_provider_name(self, serialized: dict[str, Any]) -> str:
        """Extract provider name from serialized LLM data using model registry."""
        if not serialized:
            return "unknown"

        # Strategy 1: Get provider from model name using the model registry
        # This is the most reliable way as it uses the actual model name
        model_name = self._extract_model_name(serialized)
        if model_name and model_name != "unknown":
            try:
                from noveum_trace.utils.llm_utils import get_model_info

                model_info = get_model_info(model_name)
                if model_info and model_info.provider:
                    return model_info.provider
            except Exception:
                pass  # Fall through to other methods if registry lookup fails

        # Strategy 2: Try to find provider by matching model patterns in registry
        # Instead of hardcoding patterns, we search the registry for matching models
        if model_name and model_name != "unknown":
            try:
                from noveum_trace.utils.llm_utils import MODEL_REGISTRY

                model_lower = model_name.lower()
                # Search registry for models that match or are prefixes of the model name
                # This catches variations like "gpt-4", "gpt-4o", "gpt-3.5", etc.
                # We check if the model name starts with any registry model name (at least 3 chars)
                best_match = None
                best_match_length = 0
                for registry_model_name, model_info in MODEL_REGISTRY.items():
                    registry_lower = registry_model_name.lower()
                    # Check if model name starts with registry model, or vice versa
                    # Use at least 3 characters to avoid false matches
                    if len(registry_lower) >= 3:
                        if model_lower.startswith(
                            registry_lower
                        ) or registry_lower.startswith(model_lower):
                            # Prefer longer matches for better accuracy
                            if len(registry_lower) > best_match_length:
                                best_match = model_info.provider
                                best_match_length = len(registry_lower)

                if best_match:
                    return best_match
            except Exception:
                pass

        # Strategy 3: Try to find provider in id path by checking against registry providers
        id_path = serialized.get("id", [])
        if id_path:
            try:
                from noveum_trace.utils.llm_utils import MODEL_REGISTRY

                # Get all unique providers from the registry dynamically
                valid_providers = {info.provider for info in MODEL_REGISTRY.values()}

                # Check id path elements against valid providers from registry
                for path_element in id_path:
                    if isinstance(path_element, str) and path_element.lower() in {
                        p.lower() for p in valid_providers
                    }:
                        # Find the matching provider with correct case from registry
                        for provider in valid_providers:
                            if provider.lower() == path_element.lower():
                                return provider
            except Exception:
                pass

        # Strategy 4: Use second-to-last element of id path as fallback
        # (This might be "chat_models" or similar, but it's better than "unknown")
        if len(id_path) >= 2:
            return id_path[-2]

        # Final fallback
        return "unknown"

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

    def _is_descendant_of_unlocked(
        self,
        run_id: "Union[UUID, str]",
        potential_ancestor: "Union[UUID, str]",
        parent_map: "dict[Union[UUID, str], Optional[Union[UUID, str]]]",
    ) -> bool:
        """
        Check if run_id is a descendant of potential_ancestor (without acquiring locks).

        This version is called from within locked sections and operates directly on the parent_map.

        Args:
            run_id: The run ID to check
            potential_ancestor: The potential ancestor run ID
            parent_map: The parent_map dictionary (already locked by caller)
        """
        current = parent_map.get(run_id)
        visited = {run_id}  # Avoid cycles

        while current is not None and current not in visited:
            if current == potential_ancestor:
                return True
            visited.add(current)
            current = parent_map.get(current)

        return False

    def _cleanup_trace_tracking(self, root_run_id: "Union[UUID, str]") -> None:
        """
        Clean up tracking data for a finished trace.

        This prevents memory leaks by removing entries from root_traces and parent_map
        when traces complete.

        Args:
            root_run_id: The root run ID of the finished trace
        """
        try:
            # Remove from root_traces
            with self._root_traces_lock:
                removed_trace = self.root_traces.pop(root_run_id, None)

            # Clean up all parent_map entries associated with this trace
            # This includes the root_run_id itself and all its descendants
            with self._parent_map_lock:
                to_remove = []

                # Find all run_ids that are descendants of this root
                # Use unlocked version since we already hold the lock
                for run_id in list(self.parent_map.keys()):
                    if run_id == root_run_id or self._is_descendant_of_unlocked(
                        run_id, root_run_id, self.parent_map
                    ):
                        to_remove.append(run_id)

                # Remove all identified entries
                for run_id in to_remove:
                    self.parent_map.pop(run_id, None)

            if removed_trace:
                logger.debug(
                    f"Cleaned up tracking data for trace {getattr(removed_trace, 'trace_id', 'unknown')} "
                    f"(root_run_id: {root_run_id}, cleaned {len(to_remove)} parent_map entries)"
                )

        except Exception as e:
            logger.error(f"Error cleaning up trace tracking data: {e}")

    def _find_root_run_id(
        self, run_id: "Union[UUID, str]", parent_run_id: "Optional[Union[UUID, str]]"
    ) -> "Union[UUID, str]":
        """Find the root run_id by traversing parent relationships."""
        # Store this parent relationship
        self._set_parent(run_id, parent_run_id)

        # If no parent, this is the root
        if parent_run_id is None:
            return run_id

        # Traverse up the parent chain to find the root
        current: "Optional[Union[UUID, str]]" = parent_run_id
        visited: "set[Union[UUID, str]]" = {run_id}  # Avoid cycles

        while current is not None and current not in visited:
            visited.add(current)

            # Check if this run_id has a root trace stored
            trace = self._get_root_trace(current)
            if trace is not None:
                # Found the root!
                return current

            # Get the parent of current
            parent = self._get_parent(current)
            if parent is None:
                # current has no parent, so it's the root
                return current

            # Move up the chain
            current = parent

        # If we exit the loop, current is the root
        # Here, parent_run_id cannot be None because of the early return above
        assert parent_run_id is not None
        return current if current is not None else parent_run_id

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

    def _get_or_create_trace_context(
        self,
        operation_name: str,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
    ) -> tuple[Any, bool]:
        """
        Get existing trace from global context or create new one.

        For LangGraph workflows:
        - When parent_run_id is None (root call), create ONE trace for entire workflow
        - When parent_run_id exists (child calls), reuse the root trace

        Args:
            operation_name: Name for the operation
            run_id: Current run ID
            parent_run_id: LangChain parent run ID (None for root calls)

        Returns:
            (trace, should_manage_lifecycle) tuple
        """
        from noveum_trace.core.context import get_current_trace, set_current_trace

        # Handle case where run_id is None (for testing)
        if run_id is None:
            # Check global context for existing trace
            existing_trace = get_current_trace()
            if existing_trace is not None:
                return existing_trace, False

            # Create new trace
            new_trace = self._client.start_trace(operation_name)
            set_current_trace(new_trace)
            return new_trace, True

        # Find the root run_id for this operation
        root_run_id = self._find_root_run_id(run_id, parent_run_id)

        # Check if we already have a trace for this root
        existing_root_trace = self._get_root_trace(root_run_id)
        if existing_root_trace is not None:
            # Reuse the root trace
            set_current_trace(existing_root_trace)
            return existing_root_trace, False

        # Check global context as fallback
        existing_trace = get_current_trace()
        if existing_trace is not None:
            # Use existing trace from context
            return existing_trace, False

        # Create new trace only for root calls
        if parent_run_id is None:
            # This is a root call - create trace and store it
            if self._manual_trace_control:
                logger.warning(
                    "Manual trace control enabled but no trace found. "
                    "Call start_trace() first."
                )

            new_trace = self._client.start_trace(operation_name)
            set_current_trace(new_trace)
            self._set_root_trace(root_run_id, new_trace)
            return new_trace, True
        else:
            # Child call with no root trace in our map
            # Try to find the parent's trace by looking up the parent_run_id
            parent_trace = None
            if parent_run_id:
                parent_span = self._get_run(parent_run_id)
                if parent_span:
                    # Get the trace that this parent span belongs to
                    parent_trace = (
                        parent_span.trace if hasattr(parent_span, "trace") else None
                    )

                # If we couldn't get trace from span, try looking up parent's root trace
                if not parent_trace:
                    parent_root_run_id = self._find_root_run_id(parent_run_id, None)
                    parent_trace = self._get_root_trace(parent_root_run_id)

            if parent_trace:
                # Reuse parent's trace
                set_current_trace(parent_trace)
                # Store this trace under current root_run_id for future lookups
                self._set_root_trace(root_run_id, parent_trace)
                return parent_trace, False
            else:
                # Last resort: create fallback trace
                logger.warning(
                    f"Child operation '{operation_name}' has no parent trace. "
                    "Creating new trace as fallback."
                )
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

        # Find the root_run_id for cleanup before finishing
        root_run_id = self._find_root_run_id_for_trace(trace)

        # Finish the trace
        self._client.finish_trace(trace)

        # Clear context
        set_current_trace(None)
        self._trace_managed_by_langchain = None
        self._manual_trace_control = False

        # Clean up tracking dictionaries to prevent memory leaks
        if root_run_id is not None:
            self._cleanup_trace_tracking(root_run_id)

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
            # Find the root_run_id for cleanup before finishing
            root_run_id = self._find_root_run_id_for_trace(
                self._trace_managed_by_langchain
            )

            # Finish the trace
            self._client.finish_trace(self._trace_managed_by_langchain)
            from noveum_trace.core.context import set_current_trace

            set_current_trace(None)
            self._trace_managed_by_langchain = None

            # Clean up tracking dictionaries to prevent memory leaks
            if root_run_id is not None:
                self._cleanup_trace_tracking(root_run_id)

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
            trace, should_manage = self._get_or_create_trace_context(
                span_name, run_id, parent_run_id
            )

            # Extract the actual model name and provider
            # Try extracting model from kwargs passed to this function first (LangChain may pass it here)
            # Then fall back to serialized kwargs
            model_from_kwargs = kwargs.get("invocation_params", {}).get(
                "model"
            ) or kwargs.get("model")

            # Create a temporary serialized dict with model if found in kwargs
            if model_from_kwargs and not serialized.get("kwargs", {}).get("model"):
                serialized_with_model = serialized.copy()
                if "kwargs" not in serialized_with_model:
                    serialized_with_model["kwargs"] = {}
                serialized_with_model["kwargs"]["model"] = model_from_kwargs
                extracted_model_name = self._extract_model_name(serialized_with_model)
            else:
                extracted_model_name = self._extract_model_name(serialized)

            extracted_provider = self._extract_provider_name(serialized)

            temperature = self._extract_invocation_param(
                serialized, kwargs, "temperature"
            )

            attribute_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in ["tags", "metadata", "temperature"]
                and isinstance(v, (str, int, float, bool))
            }

            span_attributes: dict[str, Any] = {
                "langchain.run_id": str(run_id),
                "llm.model": extracted_model_name,
                "llm.provider": extracted_provider,
                "llm.operation": "completion",
                # Input attributes
                "llm.input.prompts": prompts[:5] if len(prompts) > 5 else prompts,
                "llm.input.prompt_count": len(prompts),
                **attribute_kwargs,
            }

            if temperature is not None:
                if isinstance(temperature, (int, float)) and not isinstance(
                    temperature, bool
                ):
                    span_attributes["llm.input.temperature"] = float(temperature)
                elif isinstance(temperature, str):
                    try:
                        span_attributes["llm.input.temperature"] = float(temperature)
                    except ValueError:
                        span_attributes["llm.input.temperature"] = temperature
                else:
                    span_attributes["llm.input.temperature"] = temperature

            # Create span (either in new trace or existing trace)
            span = self._client.start_span(
                name=span_name,
                parent_span_id=parent_span_id,
                attributes=span_attributes,
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
            end_time = datetime.now(timezone.utc)
            latency_ms = None
            if getattr(span, "start_time", None) is not None:
                latency_ms = (end_time - span.start_time).total_seconds() * 1000

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

            provider = span.attributes.get("llm.provider")
            model = span.attributes.get("llm.model")

            if model and not usage_attrs.get("llm.input_tokens"):
                prompts = span.attributes.get("llm.input.prompts")
                if prompts:
                    usage_attrs["llm.input_tokens"] = estimate_token_count(
                        prompts, model=model, provider=provider
                    )

            if model and not usage_attrs.get("llm.output_tokens") and generations:
                usage_attrs["llm.output_tokens"] = estimate_token_count(
                    generations, model=model, provider=provider
                )

            if (
                usage_attrs.get("llm.input_tokens") is not None
                and usage_attrs.get("llm.output_tokens") is not None
            ):
                usage_attrs["llm.total_tokens"] = (
                    usage_attrs.get("llm.total_tokens")
                    or usage_attrs["llm.input_tokens"]
                    + usage_attrs["llm.output_tokens"]
                )

            cost_attrs = {}
            if model and usage_attrs.get("llm.input_tokens") is not None:
                cost_info = estimate_cost(
                    model,
                    input_tokens=usage_attrs.get("llm.input_tokens", 0),
                    output_tokens=usage_attrs.get("llm.output_tokens", 0),
                )
                cost_attrs = {
                    "llm.cost.input": cost_info.get("input_cost", 0),
                    "llm.cost.output": cost_info.get("output_cost", 0),
                    "llm.cost.total": cost_info.get("total_cost", 0),
                    "llm.cost.currency": cost_info.get("currency", "USD"),
                }

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
                    **cost_attrs,
                    **(
                        {"llm.latency_ms": latency_ms} if latency_ms is not None else {}
                    ),
                }
            )

            span.set_status(SpanStatus.OK)
            self._client.finish_span(span, end_time=end_time)

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
            trace, should_manage = self._get_or_create_trace_context(
                span_name, run_id, parent_run_id
            )

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
            if isinstance(inputs, dict):
                # Standard dict input - use as-is
                attributes["chain.inputs"] = {k: str(v) for k, v in inputs.items()}
            elif isinstance(inputs, list):
                # List input - use chain.inputs.0, chain.inputs.1 format
                for i, item in enumerate(inputs):
                    if isinstance(item, dict):
                        for k, v in item.items():
                            attributes[f"chain.inputs.{i}.{k}"] = v
                    else:
                        attributes[f"chain.inputs.{i}"] = str(item)
            else:
                # Non-dict, non-list input (e.g., string) - store as raw value
                attributes["chain.inputs"] = inputs

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
                # Dict outputs - convert values to strings
                chain_outputs = {k: str(v) for k, v in outputs.items()}
            else:
                # Non-dict outputs (e.g., string) - store as raw value
                chain_outputs = outputs

            span.set_attributes(
                {
                    # Output attributes
                    "chain.output.outputs": chain_outputs
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
            trace, should_manage = self._get_or_create_trace_context(
                span_name, run_id, parent_run_id
            )

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
                            for k, v in input_dict.items():
                                input_attrs[f"tool.input.{index}.{k}"] = v
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
            trace, should_manage = self._get_or_create_trace_context(
                span_name, run_id, parent_run_id
            )

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
                    "agent.input.inputs": safe_inputs_to_dict(inputs, "input"),
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
                    "agent.output.finish.return_values": safe_inputs_to_dict(
                        finish.return_values, "return"
                    ),
                    "agent.output.finish.log": finish.log,
                }
            )

            # Add event for agent finish
            span.add_event(
                "agent_finish",
                {
                    "finish.return_values": safe_inputs_to_dict(
                        finish.return_values, "return"
                    ),
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
            trace, should_manage = self._get_or_create_trace_context(
                span_name, run_id, parent_run_id
            )

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
