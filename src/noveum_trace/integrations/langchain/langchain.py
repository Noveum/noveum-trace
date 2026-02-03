"""
LangChain integration for Noveum Trace SDK.

This module provides a callback handler that automatically traces LangChain
operations including LLM calls, chains, agents, tools, and retrieval operations.
"""

import json
import logging
import threading
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any, Optional, Union
from uuid import UUID

# Import LangChain dependencies
from langchain_core.agents import (
    AgentAction,
    AgentFinish,
)
from langchain_core.callbacks import (
    BaseCallbackHandler,
)
from langchain_core.documents import Document
from langchain_core.outputs import LLMResult

from noveum_trace.core.span import SpanStatus
from noveum_trace.integrations.langchain.langchain_utils import (
    build_langgraph_attributes,
    build_routing_attributes,
    extract_agent_capabilities,
    extract_agent_type,
    extract_available_tools,
    extract_code_location_info,
    extract_function_definition_info,
    extract_langgraph_metadata,
    extract_noveum_metadata,
    extract_tool_function_name,
    get_operation_name,
)
from noveum_trace.integrations.langchain.message_utils import (
    message_to_dict,
    process_chain_inputs_outputs,
)
from noveum_trace.utils.llm_utils import (
    estimate_cost,
    estimate_token_count,
    parse_usage_from_response,
)

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


class NoveumTraceCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler for Noveum Trace integration."""

    def __init__(
        self,
        use_langchain_assigned_parent: bool = True,
        prioritize_manually_assigned_parents: bool = False,
    ) -> None:
        """Initialize the callback handler.

        Args:
            use_langchain_assigned_parent: If True, use LangChain's parent_run_id
                to determine parent span relationships instead of context-based
                parent assignment. Falls back to context-based with warning if
                parent_run_id lookup fails. Default is True.
            prioritize_manually_assigned_parents: If True (and use_langchain_assigned_parent
                is also True), prioritize manually assigned parent_name over LangChain's
                parent_run_id. When False, parent_run_id takes priority. Default is False.
                This flag is ignored when use_langchain_assigned_parent is False.
        """
        super().__init__()

        # Thread-safe runs dictionary for span tracking
        # Maps run_id -> span (for backward compatibility)
        self.runs: "dict[Union[UUID, str], Any]" = {}  # noqa: UP037, F821
        self._runs_lock = threading.Lock()

        # Track root traces by root run_id
        # Maps root_run_id -> trace (for LangGraph workflow grouping)
        self.root_traces: "dict[Union[UUID, str], Any]" = {}  # noqa: UP037, F821
        self._root_traces_lock = threading.Lock()

        # Track parent relationships
        # Maps run_id -> parent_run_id (self.parent_map)
        self.parent_map: dict[Union[UUID, str], Optional[Union[UUID, str]]] = (
            {}
        )  # noqa: UP037, F821
        self._parent_map_lock = threading.Lock()

        # Custom name mapping for explicit parent relationships
        # Maps custom name -> span_id (kept for handler's lifetime)
        self.names: dict[str, str] = {}
        self._names_lock = threading.Lock()

        # Track available tools for each agent run
        # Maps run_id -> list of tool dicts with schema info
        self._run_tools: dict[Union[UUID, str], list[dict[str, Any]]] = (
            {}
        )  # noqa: UP037, F821
        self._run_tools_lock = threading.Lock()

        # Track if we're managing a trace lifecycle
        self._trace_managed_by_langchain: Optional[Any] = None

        # Track if trace is manually controlled (started via start_trace())
        self._manual_trace_control: bool = False

        # Parent assignment mode
        self._use_langchain_assigned_parent = use_langchain_assigned_parent
        self._prioritize_manually_assigned_parents = (
            prioritize_manually_assigned_parents
        )

        # Import here to avoid circular imports
        from noveum_trace import get_client
        from noveum_trace.core.client import NoveumClient

        try:
            self._client: Optional[NoveumClient] = get_client()
        except Exception as e:
            logger.warning("Failed to get Noveum Trace client: %s", e)
            self._client = None

        # Track first token received for TTFT (time-to-first-token) calculation
        # Used by on_llm_new_token to record streaming metrics
        self._first_token_received: set[Union[UUID, str]] = set()
        self._first_token_lock = threading.Lock()

        # Track pending tool calls (in-flight tool data before completion)
        # Maps run_id -> tool_call_data dict
        self._pending_tool_calls: dict[Union[UUID, str], dict[str, Any]] = (
            {}
        )  # noqa: UP037, F821
        self._pending_tool_calls_lock = threading.Lock()

        # Track which LLM generated each tool call by ID
        # Maps tool_call_id (str) -> llm_run_id (UUID)
        self._tool_call_id_to_llm: dict[str, Union[UUID, str]] = {}
        self._tool_call_id_to_llm_lock = threading.Lock()

        # Track expected and completed tool counts per LLM
        # Maps llm_run_id -> {"expected": int, "completed": int}
        self._llm_tool_counts: dict[Union[UUID, str], dict[str, int]] = {}
        self._llm_tool_counts_lock = threading.Lock()

        # Track active tools per trace (global count, not per-LLM)
        # Maps root_run_id -> active_tool_count
        self._trace_active_tool_counts: dict[Union[UUID, str], int] = {}
        self._trace_active_tool_counts_lock = threading.Lock()

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

    def _get_run(self, run_id: "Union[UUID, str]") -> Any:
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

    def _set_run_tools(
        self, run_id: "Union[UUID, str]", tools: "list[dict[str, Any]]"
    ) -> None:
        """Thread-safe method to set available tools for a run."""
        with self._run_tools_lock:
            self._run_tools[run_id] = tools

    def _get_run_tools(
        self, run_id: "Union[UUID, str]"
    ) -> "Optional[list[dict[str, Any]]]":
        """Thread-safe method to get available tools for a run."""
        with self._run_tools_lock:
            return self._run_tools.get(run_id)

    def _pop_run_tools(
        self, run_id: "Union[UUID, str]"
    ) -> "Optional[list[dict[str, Any]]]":
        """Thread-safe method to pop and return available tools for a run."""
        with self._run_tools_lock:
            return self._run_tools.pop(run_id, None)

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

    def _set_pending_tool_call(
        self, run_id: "Union[UUID, str]", tool_data: dict[str, Any]
    ) -> None:
        """Thread-safe method to set pending tool call data."""
        with self._pending_tool_calls_lock:
            self._pending_tool_calls[run_id] = tool_data

    def _pop_pending_tool_call(
        self, run_id: "Union[UUID, str]"
    ) -> "Optional[dict[str, Any]]":
        """Thread-safe method to pop and return pending tool call data."""
        with self._pending_tool_calls_lock:
            return self._pending_tool_calls.pop(run_id, None)

    def _set_tool_call_id_to_llm(
        self, tool_call_id: str, llm_run_id: "Union[UUID, str]"
    ) -> None:
        """Map a tool_call_id to the LLM run_id that generated it."""
        with self._tool_call_id_to_llm_lock:
            self._tool_call_id_to_llm[tool_call_id] = llm_run_id
            logger.debug(
                f"Mapped tool_call_id '{tool_call_id}' to LLM run_id '{llm_run_id}'"
            )

    def _get_llm_from_tool_call_id(
        self, tool_call_id: str
    ) -> "Optional[Union[UUID, str]]":
        """Get the LLM run_id that generated this tool_call_id."""
        with self._tool_call_id_to_llm_lock:
            return self._tool_call_id_to_llm.get(tool_call_id)

    def _increment_trace_tool_count(self, root_run_id: Union[UUID, str]) -> None:
        """Increment active tool count for a trace."""
        with self._trace_active_tool_counts_lock:
            current = self._trace_active_tool_counts.get(root_run_id, 0)
            self._trace_active_tool_counts[root_run_id] = current + 1
            logger.debug(f"Trace {root_run_id} tool count: {current + 1}")

    def _decrement_trace_tool_count(self, root_run_id: Union[UUID, str]) -> None:
        """Decrement active tool count for a trace."""
        with self._trace_active_tool_counts_lock:
            current = self._trace_active_tool_counts.get(root_run_id, 0)
            if current > 0:
                self._trace_active_tool_counts[root_run_id] = current - 1
                logger.debug(f"Trace {root_run_id} tool count: {current - 1}")
            else:
                logger.warning(
                    f"Attempted to decrement tool count for trace {root_run_id} but it was already 0"
                )

    def _get_trace_tool_count(self, root_run_id: Union[UUID, str]) -> int:
        """Get active tool count for a trace."""
        with self._trace_active_tool_counts_lock:
            return self._trace_active_tool_counts.get(root_run_id, 0)

    def _is_llm_span(self, run_id: Union[UUID, str]) -> bool:
        """Check if a run_id corresponds to an LLM span."""
        span = self._get_run(run_id)
        if not span:
            return False
        span_attrs = getattr(span, "attributes", {})
        return "llm.model" in span_attrs

    def _find_fallback_llm(
        self, parent_run_id: Optional[UUID]
    ) -> Optional[Union[UUID, str]]:
        """Find fallback LLM for tool attachment.

        Logic:
        1. If parent_run_id is an LLM span, use it
        2. Otherwise, find the last LLM sibling (same parent_run_id)
        """
        if not parent_run_id:
            return None

        # Check if parent is an LLM
        if self._is_llm_span(parent_run_id):
            return parent_run_id

        # Find last LLM sibling with same parent
        with self._runs_lock:
            for run_id in reversed(list(self.runs.keys())):
                span = self.runs[run_id]
                span_attrs = getattr(span, "attributes", {})
                span_parent = span_attrs.get("langchain.parent_run_id")
                if span_parent == str(parent_run_id) and "llm.model" in span_attrs:
                    return run_id

        return None

    def _append_tool_call_to_span(
        self,
        span: Any,
        tool_call_data: dict[str, Any],
        llm_run_id: Optional[Union[UUID, str]] = None,
    ) -> None:
        """
        Append a tool call to the LLM span's executed tool_calls list.

        Creates the list if it doesn't exist, then appends the tool call.

        Args:
            span: The LLM span to append to
            tool_call_data: The tool call data dict to append
            llm_run_id: Optional LLM run ID to track tool completion and finish span
        """
        if not span:
            return

        try:
            # Get current tool_calls list from span attributes
            span_attrs = getattr(span, "attributes", {})
            if not isinstance(span_attrs, dict):
                span_attrs = {}

            # Get existing tool_calls list or create new one
            tool_calls_value = span_attrs.get("llm.executed_tool_calls", [])
            if isinstance(tool_calls_value, str):
                # If stored as JSON string, parse it
                try:
                    tool_calls = json.loads(tool_calls_value)
                except (json.JSONDecodeError, TypeError):
                    tool_calls = []
            elif isinstance(tool_calls_value, list):
                # Already a list (native format)
                tool_calls = tool_calls_value
            else:
                # Not a list or string, start fresh
                tool_calls = []

            # Append new tool call
            tool_calls.append(tool_call_data)

            # Update span attribute with updated list (store as native list, not JSON string)
            span.set_attributes({"llm.executed_tool_calls": tool_calls})

            # Check if all tools are complete and finish span if needed
            if llm_run_id:
                self._check_and_finish_llm_span(llm_run_id)

        except Exception as e:
            logger.error("Error appending tool call to span: %s", e)

    def _check_and_finish_llm_span(self, llm_run_id: Union[UUID, str]) -> None:
        """
        Check if all tools for an LLM have completed and finish the span if so.

        Args:
            llm_run_id: The LLM run ID to check
        """
        with self._llm_tool_counts_lock:
            if llm_run_id not in self._llm_tool_counts:
                return

            counts = self._llm_tool_counts[llm_run_id]
            counts["completed"] += 1

            logger.debug(
                f"Tool completion: {counts['completed']}/{counts['expected']} for LLM {llm_run_id}"
            )

            # Check if all tools are complete
            if counts["completed"] >= counts["expected"]:
                # All tools complete, finish the LLM span
                llm_span = self._get_run(llm_run_id)
                if llm_span:
                    # Get stored end_time
                    end_time = getattr(llm_span, "_end_time", None)

                    # Log the executed tool calls before finishing
                    executed_tools = llm_span.attributes.get(
                        "llm.executed_tool_calls", []
                    )
                    logger.debug(
                        f"Finishing LLM span {llm_run_id} with {len(executed_tools)} executed tool calls"
                    )

                    assert self._client is not None
                    self._client.finish_span(llm_span, end_time=end_time)

                    # Remove from tracking
                    self._pop_run(llm_run_id)
                    del self._llm_tool_counts[llm_run_id]

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
            return f"{event_type}.node"
        name = serialized.get("name", "node")

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

            # Clean up tool tracking entries for this trace
            tools_cleaned = 0
            with self._run_tools_lock:
                for run_id in to_remove:
                    if self._run_tools.pop(run_id, None) is not None:
                        tools_cleaned += 1

            # Clean up pending tool calls for this trace
            pending_tools_cleaned = 0
            with self._pending_tool_calls_lock:
                for run_id in list(self._pending_tool_calls.keys()):
                    # Check if this tool call belongs to any run_id in to_remove
                    if any(str(run_id).startswith(f"{r}_tool_") for r in to_remove):
                        self._pending_tool_calls.pop(run_id, None)
                        pending_tools_cleaned += 1

            # Clean up tool_call_id -> llm mappings for this trace
            # Need to identify which tool_call_ids belong to this trace
            # They were generated by LLMs in this trace (whose run_ids are in to_remove)
            tool_call_id_cleaned = 0
            with self._tool_call_id_to_llm_lock:
                for tool_call_id, llm_run_id in list(self._tool_call_id_to_llm.items()):
                    if llm_run_id in to_remove:
                        self._tool_call_id_to_llm.pop(tool_call_id, None)
                        tool_call_id_cleaned += 1

            # Clean up LLM tool count tracking for this trace
            llm_tool_counts_cleaned = 0
            with self._llm_tool_counts_lock:
                for llm_run_id in to_remove:
                    if self._llm_tool_counts.pop(llm_run_id, None) is not None:
                        llm_tool_counts_cleaned += 1

            # Clean up trace tool counts
            with self._trace_active_tool_counts_lock:
                self._trace_active_tool_counts.pop(root_run_id, None)

            if removed_trace:
                logger.debug(
                    f"Cleaned up tracking data for trace {getattr(removed_trace, 'trace_id', 'unknown')} "
                    f"(root_run_id: {root_run_id}, cleaned {len(to_remove)} parent_map entries, "
                    f"{tools_cleaned} tool entries, {pending_tools_cleaned} pending tool calls, "
                    f"{tool_call_id_cleaned} tool_call_id mappings)"
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
        current: Optional[Union[UUID, str]] = parent_run_id
        visited: set[Union[UUID, str]] = {run_id}  # Avoid cycles

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
        - If prioritize_manually_assigned_parents=False (default):
          * Use parent_run_id to look up parent span
          * Fallback to parent_name if parent_run_id lookup fails
          * Fallback to context-based parent with WARNING if both fail
        - If prioritize_manually_assigned_parents=True:
          * Use parent_name first (manual override)
          * Fallback to parent_run_id if parent_name lookup fails
          * Fallback to context-based parent with WARNING if both fail

        When use_langchain_assigned_parent=False (legacy):
        - Use parent_name if provided
        - Otherwise return None (uses context-based parent normally)

        Args:
            parent_run_id: LangChain's parent run ID
            parent_name: Custom parent name from metadata

        Returns:
            Parent span ID if resolved, None otherwise
        """
        if self._use_langchain_assigned_parent:
            # Determine priority order based on manual override flag
            if self._prioritize_manually_assigned_parents:
                # Priority 1: Try parent_name first (manual override)
                if parent_name:
                    span_id = self._get_parent_span_id_from_name(parent_name)
                    if span_id:
                        return span_id

                # Priority 2: Fallback to parent_run_id
                if parent_run_id:
                    parent_span = self._get_run(parent_run_id)
                    if parent_span:
                        return parent_span.span_id
            else:
                # Priority 1: Try parent_run_id first (default)
                if parent_run_id:
                    parent_span = self._get_run(parent_run_id)
                    if parent_span:
                        return parent_span.span_id

                # Priority 2: Fallback to parent_name
                if parent_name:
                    span_id = self._get_parent_span_id_from_name(parent_name)
                    if span_id:
                        return span_id

            # Final fallback: context-based parent with WARNING
            from noveum_trace.core.context import get_current_span

            current_span = get_current_span()
            if current_span:
                override_mode = (
                    " (manual override mode)"
                    if self._prioritize_manually_assigned_parents
                    else ""
                )
                logger.warning(
                    f"Could not resolve parent from parent_run_id ({parent_run_id}) "
                    f"or parent_name ({parent_name}){override_mode}. Auto-assigning parent span "
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
            if not self._ensure_client():
                return None, False
            assert self._client is not None  # Type guard after _ensure_client
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

            if not self._ensure_client():
                return None, False
            assert self._client is not None  # Type guard after _ensure_client
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
                if not self._ensure_client():
                    return None, False
                assert self._client is not None  # Type guard after _ensure_client
                new_trace = self._client.start_trace(operation_name)
                set_current_trace(new_trace)
                return new_trace, True

    def _create_tool_span_from_action(
        self, action: "AgentAction", run_id: UUID
    ) -> None:
        """Create tool call data from an agent action (when on_tool_start/on_tool_end aren't triggered)."""
        try:
            tool_name = action.tool
            tool_input = str(action.tool_input)

            # Create tool call data similar to on_tool_start
            import uuid

            tool_run_id = f"{run_id}_tool_{uuid.uuid4()}"

            # Try to extract tool_call_id from action
            tool_call_id = None
            if hasattr(action, "tool_call_id"):
                tool_call_id = action.tool_call_id

            tool_call_data: dict[str, Any] = {
                "name": tool_name,
                "operation": tool_name,
                "langchain.run_id": str(tool_run_id),
                "start_time": datetime.now(timezone.utc).isoformat(),
                "input": {
                    "input_str": tool_input,
                    "expression": tool_input,  # For calculator tools
                    "argument_count": 1,
                },
            }

            # Add tool_call_id if found
            if tool_call_id:
                tool_call_data["tool_call_id"] = tool_call_id

            # Store pending tool call data (will be completed in _complete_tool_spans_from_finish)
            self._set_pending_tool_call(tool_run_id, tool_call_data)

        except Exception as e:
            logger.error("Error creating tool call data from action: %s", e)

    def _complete_tool_spans_from_finish(
        self, finish: "AgentFinish", agent_run_id: UUID
    ) -> None:
        """Complete any pending tool calls when agent finishes and append to LLM span."""
        try:
            # Look for pending tool calls that belong to this specific agent
            tool_calls_to_complete = []
            with self._pending_tool_calls_lock:
                for run_id, tool_call_data in list(self._pending_tool_calls.items()):
                    # Only complete tool calls that belong to this agent (prefixed with agent_run_id)
                    if str(run_id).startswith(f"{agent_run_id}_tool_"):
                        tool_calls_to_complete.append((run_id, tool_call_data))

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

            # Complete and append each tool call using tool_call_id lookup
            for run_id, tool_call_data in tool_calls_to_complete:
                # Remove from pending dict
                self._pop_pending_tool_call(run_id)

                # Complete the tool call data with output
                tool_call_data["output"] = result
                tool_call_data["status"] = "ok"
                tool_call_data["end_time"] = datetime.now(timezone.utc).isoformat()

                # Look up LLM using tool_call_id
                tool_call_id = tool_call_data.get("tool_call_id")
                if tool_call_id:
                    llm_run_id = self._get_llm_from_tool_call_id(tool_call_id)
                    if llm_run_id:
                        llm_span = self._get_run(llm_run_id)
                        if llm_span:
                            self._append_tool_call_to_span(
                                llm_span, tool_call_data, llm_run_id
                            )
                        else:
                            logger.debug(f"LLM span {llm_run_id} not found")
                    else:
                        logger.debug(f"No LLM found for tool_call_id {tool_call_id}")
                else:
                    logger.debug("No tool_call_id in agent tool data")

        except Exception as e:
            logger.error("Error completing tool calls from finish: %s", e)

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
        if not self._ensure_client():
            return
        assert self._client is not None  # Type guard after _ensure_client
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
        if not self._ensure_client():
            return
        assert self._client is not None  # Type guard after _ensure_client
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

        if not self._trace_managed_by_langchain:
            return

        root_run_id = self._find_root_run_id_for_trace(self._trace_managed_by_langchain)
        if not root_run_id:
            return

        # Check trace-level tool count
        active_tool_count = self._get_trace_tool_count(root_run_id)
        if active_tool_count > 0:
            # Tools still running
            return

        # Count remaining runs, excluding stuck LLMs
        remaining_non_stuck_runs = 0
        with self._runs_lock:
            for run_id in list(self.runs.keys()):
                # If not an LLM span, count it as non-stuck
                if not self._is_llm_span(run_id):
                    remaining_non_stuck_runs += 1
                    continue

                # Check if this LLM span is stuck waiting for tools
                with self._llm_tool_counts_lock:
                    if run_id not in self._llm_tool_counts:
                        # No tool count tracking, not stuck
                        remaining_non_stuck_runs += 1
                        continue

                    counts = self._llm_tool_counts[run_id]
                    if counts["completed"] >= counts["expected"]:
                        # All tools complete, not stuck
                        remaining_non_stuck_runs += 1
                        continue

                    # This LLM is stuck
                    logger.debug(
                        f"LLM {run_id} is stuck: {counts['completed']}/{counts['expected']}"
                    )

        # If only stuck LLMs or no runs at all, finish the trace
        if remaining_non_stuck_runs == 0:
            logger.debug(
                f"Finishing trace {root_run_id}: all tools complete, no non-stuck spans remaining"
            )

            if not self._ensure_client():
                return
            assert self._client is not None
            self._client.finish_trace(self._trace_managed_by_langchain)
            from noveum_trace.core.context import set_current_trace

            set_current_trace(None)
            self._trace_managed_by_langchain = None

            # Clean up tracking dictionaries
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
            custom_metadata = noveum_config.get("metadata", {})

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

            # Extract code location information
            code_location_info = extract_code_location_info(
                skip_frames=1
            )  # Skip this frame

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

            # Add code location information if available
            if code_location_info:
                span_attributes.update(code_location_info)

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

            # Extract tools from invocation_params (for bind_tools pattern)
            # This works for:
            # 1. Standalone LLM with bind_tools (invocation_params['tools'])
            # 2. LLM within LangGraph agent (invocation_params['tools'])
            # 3. OpenAI Functions Agent (invocation_params['functions'])
            if kwargs:
                invocation_params = kwargs.get("invocation_params", {})
                # Check both 'tools' (bind_tools) and 'functions' (OpenAI Functions Agent)
                tools_in_params = invocation_params.get(
                    "tools"
                ) or invocation_params.get("functions", [])

                logger.debug(
                    f"🔍 Checking for tools in invocation_params: {len(tools_in_params) if tools_in_params else 0} tools found"
                )

                if tools_in_params:
                    # Convert tools from invocation_params to our standard format
                    from noveum_trace.integrations.langchain.langchain_utils import (
                        _convert_tools_to_dict_list,
                    )

                    converted_tools = _convert_tools_to_dict_list(tools_in_params)

                    logger.debug(
                        f"🔧 Converted {len(converted_tools) if converted_tools else 0} tools"
                    )

                    if converted_tools:
                        # Store tools with appropriate run_id for later retrieval
                        if parent_run_id:
                            # LLM inside agent - store with parent's run_id
                            self._set_run_tools(parent_run_id, converted_tools)
                        else:
                            # Standalone LLM - store with its own run_id
                            self._set_run_tools(run_id, converted_tools)

                        # Add tool tracking attributes to the LLM span so they're visible in traces
                        span_attributes["llm.available_tools.count"] = len(
                            converted_tools
                        )
                        span_attributes["llm.available_tools.names"] = [
                            t["name"] for t in converted_tools
                        ]
                        # Add complete tool schemas for better trace visibility
                        span_attributes["llm.available_tools.schemas"] = json.dumps(
                            converted_tools, default=str
                        )

                        logger.debug(
                            f"✅ Added tool attributes to span: count={len(converted_tools)}, names={[t['name'] for t in converted_tools]}"
                        )

                        # If LLM is inside an agent, also update parent span with tool info
                        # This is critical for LangGraph agents where serialized=None in on_chain_start
                        # so tools can't be detected there - we detect them here and propagate up
                        if parent_run_id:
                            parent_span = self._get_run(parent_run_id)
                            if parent_span:
                                try:
                                    parent_span.set_attributes(
                                        {
                                            "agent.available_tools.count": len(
                                                converted_tools
                                            ),
                                            "agent.available_tools.names": [
                                                t["name"] for t in converted_tools
                                            ],
                                            "agent.available_tools.schemas": json.dumps(
                                                converted_tools, default=str
                                            ),
                                        }
                                    )
                                except Exception as e:
                                    logger.debug(
                                        f"Could not update parent span with tools: {e}"
                                    )

            # Add custom noveum metadata to span attributes
            if custom_metadata:
                span_attributes["noveum.additional_attributes"] = json.dumps(
                    custom_metadata, default=str
                )

            # Create span (either in new trace or existing trace)
            if not self._ensure_client():
                return None
            assert self._client is not None  # Type guard after _ensure_client
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

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle chat model start event.

        This is called instead of on_llm_start when using chat models like
        ChatOpenAI, ChatAnthropic, etc. It receives structured messages
        instead of flat string prompts.
        """
        if not self._ensure_client():
            return

        operation_name = get_operation_name("llm_start", serialized)

        try:
            # Extract Noveum-specific metadata
            noveum_config = extract_noveum_metadata(metadata)
            custom_name = noveum_config.get("name")
            parent_name = noveum_config.get("parent_name")
            custom_metadata = noveum_config.get("metadata", {})

            # Use custom name if provided, otherwise use operation name
            span_name = custom_name if custom_name else operation_name

            # Resolve parent span ID based on mode
            parent_span_id = self._resolve_parent_span_id(parent_run_id, parent_name)

            # Get or create trace context
            trace, should_manage = self._get_or_create_trace_context(
                span_name, run_id, parent_run_id
            )

            # Convert messages to dicts using existing message_to_dict
            message_dicts = [[message_to_dict(m) for m in batch] for batch in messages]

            # Flatten for analysis (messages is List[List[BaseMessage]])
            flat_messages = message_dicts[0] if message_dicts else []

            # Analyze message content
            has_system_prompt = any(m.get("type") == "system" for m in flat_messages)
            has_tool_calls = any(m.get("tool_calls") for m in flat_messages)

            # Extract the actual model name and provider
            model_from_kwargs = kwargs.get("invocation_params", {}).get(
                "model"
            ) or kwargs.get("model")

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

            # Extract code location information
            code_location_info = extract_code_location_info(skip_frames=1)

            span_attributes: dict[str, Any] = {
                "langchain.run_id": str(run_id),
                "llm.model": extracted_model_name,
                "llm.provider": extracted_provider,
                "llm.operation": "chat",
                # Chat-specific input attributes
                "llm.input.type": "chat",
                "llm.input.messages": json.dumps(message_dicts, default=str),
                "llm.input.message_count": len(flat_messages),
                "llm.input.has_system_prompt": has_system_prompt,
                "llm.input.has_tool_calls": has_tool_calls,
                **attribute_kwargs,
            }

            # Add code location information if available
            if code_location_info:
                span_attributes.update(code_location_info)

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

            # Extract tools from invocation_params (for bind_tools pattern)
            if kwargs:
                invocation_params = kwargs.get("invocation_params", {})
                tools_in_params = invocation_params.get(
                    "tools"
                ) or invocation_params.get("functions", [])

                if tools_in_params:
                    from noveum_trace.integrations.langchain.langchain_utils import (
                        _convert_tools_to_dict_list,
                    )

                    converted_tools = _convert_tools_to_dict_list(tools_in_params)

                    if converted_tools:
                        if parent_run_id:
                            self._set_run_tools(parent_run_id, converted_tools)
                        else:
                            self._set_run_tools(run_id, converted_tools)

                        span_attributes["llm.available_tools.count"] = len(
                            converted_tools
                        )
                        span_attributes["llm.available_tools.names"] = [
                            t["name"] for t in converted_tools
                        ]
                        span_attributes["llm.available_tools.schemas"] = json.dumps(
                            converted_tools, default=str
                        )

                        if parent_run_id:
                            parent_span = self._get_run(parent_run_id)
                            if parent_span:
                                try:
                                    parent_span.set_attributes(
                                        {
                                            "agent.available_tools.count": len(
                                                converted_tools
                                            ),
                                            "agent.available_tools.names": [
                                                t["name"] for t in converted_tools
                                            ],
                                            "agent.available_tools.schemas": json.dumps(
                                                converted_tools, default=str
                                            ),
                                        }
                                    )
                                except Exception as e:
                                    logger.debug(
                                        f"Could not update parent span with tools: {e}"
                                    )

            # Add custom noveum metadata to span attributes
            if custom_metadata:
                span_attributes["noveum.additional_attributes"] = json.dumps(
                    custom_metadata, default=str
                )

            # Create span
            if not self._ensure_client():
                return None
            assert self._client is not None
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
            logger.error("Error handling chat model start event: %s", e)

    def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Handle streaming token event - capture time-to-first-token (TTFT).

        This callback fires for each token during streaming responses.
        We only record metrics on the first token to calculate TTFT.
        """
        # Quick check without lock first (optimization for common case)
        if run_id in self._first_token_received:
            return

        # Thread-safe check and record
        with self._first_token_lock:
            if run_id in self._first_token_received:
                return  # Already recorded first token
            self._first_token_received.add(run_id)

        # Get the span and record TTFT
        span = self._get_run(run_id)
        if span:
            try:
                now = datetime.now(timezone.utc)
                # Calculate TTFT from span start time
                if hasattr(span, "start_time") and span.start_time:
                    ttft_ms = (now - span.start_time).total_seconds() * 1000
                    span.set_attribute("llm.time_to_first_token_ms", ttft_ms)
                span.set_attribute("llm.first_token_time", now.isoformat())
                span.set_attribute("llm.streaming", True)
            except Exception as e:
                logger.debug(f"Error recording TTFT metrics: {e}")
        else:
            with self._first_token_lock:
                self._first_token_received.discard(run_id)

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

        with self._first_token_lock:
            self._first_token_received.discard(run_id)

        # Get span from runs dict
        # Note: We DON'T remove it yet, as tools may need to append results later
        # It will be cleaned up during trace finalization or when tools complete
        span = self._get_run(run_id)
        if span is None:
            return

        try:
            # Add response data
            generations = []
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

            # Extract tool calls from ChatGeneration messages
            tool_calls = []
            if hasattr(response, "generations") and response.generations:
                for generation_list in response.generations:
                    for gen in generation_list:
                        # Skip if not a ChatGeneration with a message
                        if not hasattr(gen, "message"):
                            continue

                        message = gen.message

                        # Extract tool_calls from AIMessage (modern format)
                        if hasattr(message, "tool_calls") and message.tool_calls:
                            for tc in message.tool_calls:
                                try:
                                    # Handle both dict and object tool calls
                                    if isinstance(tc, dict):
                                        tool_call_id = tc.get("id")
                                        tool_calls.append(
                                            {
                                                "name": tc.get("name"),
                                                "args": tc.get("args"),
                                                "id": tool_call_id,
                                            }
                                        )
                                    else:
                                        tool_call_id = getattr(tc, "id", None)
                                        tool_calls.append(
                                            {
                                                "name": getattr(tc, "name", None),
                                                "args": getattr(tc, "args", None),
                                                "id": tool_call_id,
                                            }
                                        )

                                    # Store the mapping for later lookup
                                    if tool_call_id:
                                        self._set_tool_call_id_to_llm(
                                            tool_call_id, run_id
                                        )

                                except Exception as e:
                                    logger.debug(f"Error extracting tool call: {e}")

                        # Extract function_call from additional_kwargs (legacy format)
                        if hasattr(message, "additional_kwargs"):
                            additional_kwargs = message.additional_kwargs
                            if isinstance(additional_kwargs, dict):
                                function_call = additional_kwargs.get("function_call")
                                if function_call:
                                    try:
                                        # Parse arguments from JSON string to dict
                                        args_str = function_call.get("arguments", "{}")
                                        args_dict = (
                                            json.loads(args_str) if args_str else {}
                                        )

                                        tool_calls.append(
                                            {
                                                "name": function_call.get("name"),
                                                "args": args_dict,
                                                "id": None,  # Legacy format has no ID
                                            }
                                        )
                                    except json.JSONDecodeError as e:
                                        logger.debug(
                                            f"Failed to parse function_call arguments: {e}"
                                        )
                                    except Exception as e:
                                        logger.debug(
                                            f"Error extracting function_call: {e}"
                                        )

            # Flatten usage attributes to match ContextManager format
            usage_attrs = parse_usage_from_response(
                response,
                provider=span.attributes.get("llm.provider"),
                model=span.attributes.get("llm.model"),
            )

            provider = span.attributes.get("llm.provider")
            model = span.attributes.get("llm.model")

            if model and not usage_attrs.get("llm.input_tokens"):
                prompts = span.attributes.get("llm.input.prompts")
                if not prompts:
                    prompts = span.attributes.get("llm.input.messages")
                if prompts:
                    parsed_prompts = prompts
                    if isinstance(prompts, str):
                        try:
                            parsed_prompts = json.loads(prompts)
                        except json.JSONDecodeError:
                            parsed_prompts = prompts
                    usage_attrs["llm.input_tokens"] = estimate_token_count(
                        parsed_prompts, model=model, provider=provider
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

            # Build output attributes
            output_attrs = {
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
                **({"llm.latency_ms": latency_ms} if latency_ms is not None else {}),
            }

            # Add tool call attributes if present
            if tool_calls:
                output_attrs["llm.output.tool_calls"] = json.dumps(
                    tool_calls, default=str
                )
                output_attrs["llm.output.tool_calls.count"] = len(tool_calls)
                output_attrs["llm.output.tool_calls.names"] = [
                    tc["name"] for tc in tool_calls if tc.get("name")
                ]

            span.set_attributes(output_attrs)

            span.set_status(SpanStatus.OK)

            # Don't finish the span yet if it has tool calls - tools will append results later
            # The span will be finished when tools complete or during cleanup
            if not tool_calls:
                assert self._client is not None  # Type guard after _ensure_client
                self._client.finish_span(span, end_time=end_time)
                # Remove from runs since it's finished
                self._pop_run(run_id)
            else:
                # Keep span in self.runs for tools to append results
                # Store end_time for later use
                span._end_time = end_time

                # Track expected tool count
                with self._llm_tool_counts_lock:
                    self._llm_tool_counts[run_id] = {
                        "expected": len(tool_calls),
                        "completed": 0,
                    }

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

        with self._first_token_lock:
            self._first_token_received.discard(run_id)

        # Get and remove span from runs dict
        span = self._pop_run(run_id)
        if span is None:
            return None

        try:
            span.record_exception(error)
            span.set_status(SpanStatus.ERROR, str(error))
            assert self._client is not None  # Type guard after _ensure_client
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
            custom_metadata = noveum_config.get("metadata", {})

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
                # Use new message parsing for dicts
                parsed_inputs = process_chain_inputs_outputs(inputs)
                for key, value in parsed_inputs.items():
                    attributes[f"chain.inputs.{key}"] = value
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

            # Extract tools from chains (including AgentExecutor and LangGraph nodes)
            # This works for:
            # 1. LangGraph agent nodes (langgraph_metadata.node == "agent")
            # 2. AgentExecutor chains (manual injection via metadata)
            # 3. Any chain with tools in metadata or serialized data
            available_tools = extract_available_tools(serialized, metadata)

            # Store tools for this chain run (for later retrieval by tool spans)
            if available_tools:
                self._set_run_tools(run_id, available_tools)

                # Add tool tracking attributes to span
                attributes["agent.available_tools.count"] = len(available_tools)
                attributes["agent.available_tools.names"] = [
                    t["name"] for t in available_tools
                ]
                attributes["agent.available_tools.schemas"] = json.dumps(
                    available_tools, default=str
                )

            # Add custom noveum metadata to span attributes
            if custom_metadata:
                attributes["noveum.additional_attributes"] = json.dumps(
                    custom_metadata, default=str
                )

            # Create span for chain
            assert self._client is not None  # Type guard after _ensure_client
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

        # Clean up tool tracking if this was an agent chain
        self._pop_run_tools(run_id)

        try:
            # Handle both dict and non-dict outputs
            if isinstance(outputs, dict):
                # Use new message parsing for dicts
                parsed_outputs = process_chain_inputs_outputs(outputs)
                for key, value in parsed_outputs.items():
                    span.set_attributes({f"chain.output.{key}": value})
            else:
                # Non-dict outputs (e.g., string) - store as raw value
                span.set_attributes({"chain.output.outputs": str(outputs)})

            span.set_status(SpanStatus.OK)
            assert self._client is not None  # Type guard after _ensure_client
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

        # Clean up tool tracking if this was an agent chain
        self._pop_run_tools(run_id)

        try:
            span.record_exception(error)
            span.set_status(SpanStatus.ERROR, str(error))
            assert self._client is not None  # Type guard after _ensure_client
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
            assert self._client is not None  # Type guard after _ensure_client
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
            assert self._client is not None  # Type guard after _ensure_client
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
        """Handle tool start event - collect tool data for inline storage."""
        if not self._ensure_client():
            return

        get_operation_name("tool_start", serialized)

        try:
            # Extract Noveum-specific metadata
            noveum_config = extract_noveum_metadata(metadata)
            custom_name = noveum_config.get("name")
            custom_metadata = noveum_config.get("metadata", {})

            tool_name = serialized.get("name", "unknown") if serialized else "unknown"

            # Extract actual function name from serialized data
            func_name = extract_tool_function_name(serialized)

            # Get available tools from parent agent run
            available_tools = []
            if parent_run_id:
                available_tools = self._get_run_tools(parent_run_id) or []

            # Prepare input data
            tool_input_data: dict[str, Any] = {
                "input_str": input_str,
            }

            # Add structured inputs if available - handle different input types
            if inputs is not None:
                if isinstance(inputs, dict):
                    tool_input_data["inputs"] = inputs
                    tool_input_data["argument_count"] = len(inputs)
                elif isinstance(inputs, list):
                    tool_input_data["inputs"] = inputs
                    tool_input_data["argument_count"] = len(inputs)
                elif isinstance(inputs, tuple):
                    tool_input_data["inputs"] = list(inputs)
                    tool_input_data["argument_count"] = len(inputs)
                else:
                    tool_input_data["input"] = inputs
                    tool_input_data["argument_count"] = 1
            else:
                tool_input_data["argument_count"] = 0

            # Extract code location information (includes function definition if available)
            code_location_info = extract_code_location_info(
                skip_frames=1
            )  # Skip this frame

            # Try to extract function definition information from tool object
            # For LangChain tools created with @tool, the function is in tool.func
            function_def_info = None
            try:
                # Method 1: Try to get function from kwargs (if tool object is passed)
                tool_obj = kwargs.get("tool") or kwargs.get("tool_instance")
                if tool_obj:
                    if hasattr(tool_obj, "func"):
                        # Tool has a func attribute (for @tool decorated functions)
                        func = tool_obj.func
                        function_def_info = extract_function_definition_info(func)
                    elif callable(tool_obj):
                        # Tool itself is callable
                        function_def_info = extract_function_definition_info(tool_obj)

                # Method 2: If code_location_info has function definition, use it
                if not function_def_info and code_location_info:
                    # Check if code_location_info already has function definition info
                    if "function.definition.file" in code_location_info:
                        function_def_info = {
                            "function.definition.file": code_location_info.get(
                                "function.definition.file"
                            ),
                            "function.definition.start_line": code_location_info.get(
                                "function.definition.start_line"
                            ),
                            "function.definition.end_line": code_location_info.get(
                                "function.definition.end_line"
                            ),
                        }
            except Exception:
                # If extraction fails, continue without function definition info
                pass

            # Find the called tool in available tools
            called_tool_info = None
            if available_tools:
                for tool in available_tools:
                    if tool.get("name") == tool_name:
                        called_tool_info = tool
                        break

            # Note: tool_call_id is NOT available in on_tool_start callback
            # It will be extracted from the ToolMessage output in on_tool_end

            # Build tool call data dict
            tool_call_data: dict[str, Any] = {
                "name": tool_name,
                "operation": func_name,
                "langchain.run_id": str(run_id),
                "start_time": datetime.now(timezone.utc).isoformat(),
                "input": tool_input_data,
            }
            # tool_call_id will be added later in on_tool_end when it's available

            # Add tool tracking information
            if available_tools:
                tool_call_data["available_tools"] = {
                    "count": len(available_tools),
                    "names": [t["name"] for t in available_tools],
                }

            # Add called tool information
            tool_call_data["called_tool"] = {
                "name": tool_name,
            }
            if called_tool_info:
                tool_call_data["called_tool"]["description"] = called_tool_info.get(
                    "description", "No description"
                )
                if called_tool_info.get("args_schema"):
                    tool_call_data["called_tool"]["args_schema"] = str(
                        called_tool_info["args_schema"]
                    )

            # Add code location information if available
            if code_location_info:
                tool_call_data["code_location"] = code_location_info

            # Add function definition information if available
            if function_def_info:
                tool_call_data["function_definition"] = function_def_info

            # Add custom noveum name if provided
            if custom_name:
                tool_call_data["custom_name"] = custom_name

            # Add custom noveum metadata
            if custom_metadata:
                tool_call_data["noveum_metadata"] = custom_metadata

            # Identify fallback LLM for potential error attachment
            fallback_llm_run_id = self._find_fallback_llm(parent_run_id)
            if fallback_llm_run_id:
                tool_call_data["fallback_llm_run_id"] = fallback_llm_run_id
                logger.debug(
                    f"Identified fallback LLM {fallback_llm_run_id} for tool {run_id}"
                )

            # Increment trace-level tool count
            root_run_id = self._find_root_run_id(run_id, parent_run_id)
            self._increment_trace_tool_count(root_run_id)

            # Store pending tool call data (will be completed in on_tool_end/on_tool_error)
            self._set_pending_tool_call(run_id, tool_call_data)

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
        """Handle tool end event - append completed tool call to LLM span via tool_call_id lookup."""
        if not self._ensure_client():
            return

        # Get and remove pending tool call data
        tool_call_data = self._pop_pending_tool_call(run_id)
        if not tool_call_data:
            return

        try:
            # Decrement trace-level tool count
            root_run_id = self._find_root_run_id(run_id, parent_run_id)
            self._decrement_trace_tool_count(root_run_id)

            # Extract tool_call_id from output (ToolMessage)
            tool_call_id = None
            if hasattr(output, "tool_call_id"):
                tool_call_id = output.tool_call_id
                logger.debug(f"Extracted tool_call_id from output: {tool_call_id}")

            # Complete the tool call data with output
            tool_call_data["output"] = (
                str(output) if not isinstance(output, str) else output
            )
            tool_call_data["status"] = "ok"
            tool_call_data["end_time"] = datetime.now(timezone.utc).isoformat()

            # Store tool_call_id in tool_call_data
            if tool_call_id:
                tool_call_data["tool_call_id"] = tool_call_id

            # Find LLM span using tool_call_id
            if not tool_call_id:
                tool_call_id = tool_call_data.get("tool_call_id")

            if tool_call_id:
                llm_run_id = self._get_llm_from_tool_call_id(tool_call_id)
                if llm_run_id:
                    # Attach to correct LLM
                    llm_span = self._get_run(llm_run_id)
                    if llm_span:
                        self._append_tool_call_to_span(
                            llm_span, tool_call_data, llm_run_id
                        )
                        logger.debug(f"Attached tool to correct LLM {llm_run_id}")
            else:
                # No tool_call_id available, attach to fallback if available
                fallback_llm_run_id = tool_call_data.get("fallback_llm_run_id")
                if fallback_llm_run_id:
                    llm_span = self._get_run(fallback_llm_run_id)
                    if llm_span:
                        self._append_tool_call_to_span(
                            llm_span, tool_call_data, fallback_llm_run_id
                        )

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
        """Handle tool error event - append failed tool call to LLM span via tool_call_id lookup."""
        if not self._ensure_client():
            return None

        # Get and remove pending tool call data
        tool_call_data = self._pop_pending_tool_call(run_id)
        if not tool_call_data:
            return None

        try:
            # Decrement trace-level tool count
            root_run_id = self._find_root_run_id(run_id, parent_run_id)
            self._decrement_trace_tool_count(root_run_id)

            # Complete the tool call data with error information
            tool_call_data["status"] = "error"
            tool_call_data["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            tool_call_data["end_time"] = datetime.now(timezone.utc).isoformat()

            # Attach to fallback LLM (identified during on_tool_start)
            fallback_llm_run_id = tool_call_data.get("fallback_llm_run_id")
            if fallback_llm_run_id:
                llm_span = self._get_run(fallback_llm_run_id)
                if llm_span:
                    self._append_tool_call_to_span(
                        llm_span,
                        tool_call_data,
                        None,  # Don't increment LLM count for error fallback
                    )
                    logger.debug(
                        f"Attached error tool to fallback LLM {fallback_llm_run_id}"
                    )
                else:
                    logger.debug(f"Fallback LLM span {fallback_llm_run_id} not found")
            else:
                logger.debug(f"No fallback LLM identified for tool error {run_id}")

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
            custom_metadata = noveum_config.get("metadata", {})

            # Extract available tools from metadata or serialized data
            available_tools = extract_available_tools(serialized, metadata)

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

            # Store tools for this agent run (for later retrieval by tool spans)
            if available_tools:
                self._set_run_tools(run_id, available_tools)

            # Build span attributes with tool information
            span_attributes = {
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
            }

            # Add tool tracking attributes if tools are available
            if available_tools:
                span_attributes["agent.available_tools.count"] = len(available_tools)
                span_attributes["agent.available_tools.names"] = [
                    t["name"] for t in available_tools
                ]
                span_attributes["agent.available_tools.schemas"] = json.dumps(
                    available_tools, default=str
                )

            # Add custom noveum metadata to span attributes
            if custom_metadata:
                span_attributes["noveum.additional_attributes"] = json.dumps(
                    custom_metadata, default=str
                )

            assert self._client is not None  # Type guard after _ensure_client
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

            # Clean up tool tracking for this agent run
            self._pop_run_tools(run_id)

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
            assert self._client is not None  # Type guard after _ensure_client
            self._client.finish_span(span)

            # Check if we should finish the trace
            self._finish_trace_if_needed()

        except Exception as e:
            logger.error("Error handling agent finish event: %s", e)

    def on_agent_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Handle agent error event."""
        if not self._ensure_client():
            return None

        # Get and remove span from runs dict
        span = self._pop_run(run_id)
        if span is None:
            return None

        try:
            # Clean up tool tracking for this agent run
            self._pop_run_tools(run_id)

            span.record_exception(error)
            span.set_status(SpanStatus.ERROR, str(error))
            assert self._client is not None  # Type guard after _ensure_client
            self._client.finish_span(span)

            # Check if we should finish the trace
            self._finish_trace_if_needed()

        except Exception as e:
            logger.error("Error handling agent error event: %s", e)

        return None

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
            custom_metadata = noveum_config.get("metadata", {})

            # Use custom name if provided, otherwise use operation name
            span_name = custom_name if custom_name else operation_name

            # Resolve parent span ID based on mode
            parent_span_id = self._resolve_parent_span_id(parent_run_id, parent_name)

            # Get or create trace context
            trace, should_manage = self._get_or_create_trace_context(
                span_name, run_id, parent_run_id
            )

            # Build span attributes
            span_attributes = {
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
            }

            # Add custom noveum metadata to span attributes
            if custom_metadata:
                span_attributes["noveum.additional_attributes"] = json.dumps(
                    custom_metadata, default=str
                )

            # Create span
            assert self._client is not None  # Type guard after _ensure_client
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
            assert self._client is not None  # Type guard after _ensure_client
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
            assert self._client is not None  # Type guard after _ensure_client
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
            f"use_langchain_parent={self._use_langchain_assigned_parent}, "
            f"prioritize_manual_parents={self._prioritize_manually_assigned_parents})"
        )


# For backwards compatibility and ease of import
__all__ = ["NoveumTraceCallbackHandler"]
