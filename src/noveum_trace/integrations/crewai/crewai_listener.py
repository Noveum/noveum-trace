"""
Noveum CrewAI Integration — Main Listener Orchestrator.

Combines all handler mixins into one unified listener class that registers
with CrewAI crews via BaseEventListener interface.

Usage
-----
>>> from noveum_trace import get_client
>>> from noveum_trace.integrations.crewai import NoveumCrewAIListener
>>>
>>> client = get_client()
>>> listener = NoveumCrewAIListener(client)
>>> crew = Crew(agents=[...], tasks=[...])
>>> crew.callback_function = listener
>>> crew.kickoff()  # Events are traced automatically
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Optional
from weakref import WeakSet

from crewai.events import BaseEventListener

from noveum_trace.integrations.crewai._handlers_a2a import _A2AHandlersMixin
from noveum_trace.integrations.crewai._handlers_agent import _AgentHandlersMixin
from noveum_trace.integrations.crewai._handlers_crew import _CrewHandlersMixin
from noveum_trace.integrations.crewai._handlers_flow import _FlowHandlersMixin
from noveum_trace.integrations.crewai._handlers_guardrail import _GuardrailHandlersMixin
from noveum_trace.integrations.crewai._handlers_knowledge import _KnowledgeHandlersMixin
from noveum_trace.integrations.crewai._handlers_llm import _LLMHandlersMixin
from noveum_trace.integrations.crewai._handlers_mcp import _MCPHandlersMixin
from noveum_trace.integrations.crewai._handlers_memory import _MemoryHandlersMixin
from noveum_trace.integrations.crewai._handlers_reasoning import _ReasoningHandlersMixin
from noveum_trace.integrations.crewai._handlers_task import _TaskHandlersMixin
from noveum_trace.integrations.crewai._handlers_tool import _ToolHandlersMixin
from noveum_trace.integrations.crewai.crewai_state import _CrewAIObserverMixinBase

if TYPE_CHECKING:
    from noveum_trace.core import NoveumClient

logger = logging.getLogger(__name__)

# ============================================================================
# Class-level token patch management (shared across all listener instances)
# ============================================================================

_patch_applied: bool = False
# Class-level WeakSet: add/discard only under ``_patch_lock`` (see ``__init__``,
# ``shutdown``, and the BaseLLM monkey-patch) so iteration cannot race mutation.
_active_listeners: WeakSet[Any] = WeakSet()
_original_track_token_usage: Optional[Any] = None
_patch_lock = threading.RLock()


# ============================================================================
# Main Listener Class
# ============================================================================


class NoveumCrewAIListener(
    _CrewHandlersMixin,
    _TaskHandlersMixin,
    _AgentHandlersMixin,
    _LLMHandlersMixin,
    _ToolHandlersMixin,
    _MemoryHandlersMixin,
    _FlowHandlersMixin,
    _KnowledgeHandlersMixin,
    _ReasoningHandlersMixin,
    _GuardrailHandlersMixin,
    _MCPHandlersMixin,
    _A2AHandlersMixin,
    _CrewAIObserverMixinBase,
    BaseEventListener,
):
    """
    Unified CrewAI event listener for Noveum tracing.

    Inherits from all handler mixins (which provide event handler methods)
    and from CrewAI's BaseEventListener (which defines the interface).

    When a crew calls kickoff(), CrewAI fires events through BaseEventListener,
    which routes them to the appropriate handler methods (on_crew_kickoff_started,
    on_llm_call_started, etc.).

    Supports configurable capture flags to let users control what gets traced.
    """

    # Class-level constants
    _MAX_TOKEN_BUFFER_ENTRIES: int = 512

    def __init__(
        self,
        client: Optional[NoveumClient] = None,
        *,
        capture_inputs: bool = True,
        capture_outputs: bool = True,
        capture_llm_messages: bool = True,
        capture_tool_schemas: bool = True,
        capture_agent_snapshot: bool = True,
        capture_crew_snapshot: bool = True,
        capture_memory: bool = True,
        capture_knowledge: bool = True,
        capture_a2a: bool = True,
        capture_mcp: bool = True,
        capture_flow: bool = True,
        capture_reasoning: bool = True,
        capture_guardrails: bool = True,
        capture_streaming: bool = True,
        capture_thinking: bool = True,
        trace_name_prefix: str = "crewai",
        verbose: bool = False,
    ) -> None:
        """
        Initialize the CrewAI listener with configuration flags.

        Args:
            client: Noveum tracing client. If None, will attempt to get the current client.
            capture_inputs: Capture input messages, tool args, task prompts.
            capture_outputs: Capture LLM responses, tool outputs, task results.
            capture_llm_messages: Capture full message history (system prompt + RAG).
            capture_tool_schemas: Capture tool definitions and available functions.
            capture_agent_snapshot: Capture agent goal/backstory at execution start.
            capture_crew_snapshot: Capture crew agents/tasks at kickoff.
            capture_memory: Capture memory operations (query/save/retrieval).
            capture_knowledge: Capture knowledge integration events.
            capture_a2a: Capture agent-to-agent delegation and communication.
            capture_mcp: Capture MCP (Model Context Protocol) server calls.
            capture_flow: Capture CrewAI Flow execution events.
            capture_reasoning: Capture agent reasoning/thinking steps.
            capture_guardrails: Capture guardrail violations/checks.
            capture_streaming: Accumulate streaming chunks (LLM responses).
            capture_thinking: Capture extended thinking / chain-of-thought tokens.
            trace_name_prefix: Prefix for trace names (default: "crewai").
            verbose: Enable debug logging.
        """
        # Everything that setup_listeners() reads must be set before super().__init__()
        # because BaseEventListener.__init__ calls setup_listeners() immediately.
        self._lock = threading.RLock()
        self._handlers: list[tuple[type, Any]] = []
        self._crewai_event_bus: Optional[Any] = None

        # Flags that setup_listeners() reads — must be set before super().__init__()
        # because BaseEventListener.__init__ calls setup_listeners() immediately.
        self.capture_memory = capture_memory
        self.capture_knowledge = capture_knowledge
        self.capture_a2a = capture_a2a
        self.capture_mcp = capture_mcp
        self.capture_flow = capture_flow
        self.capture_reasoning = capture_reasoning
        self.capture_guardrails = capture_guardrails

        super().__init__()

        # Client reference (may be None until initialized)
        self._client = client
        self._verbose = verbose

        # Lifecycle flags
        self._is_shutdown = False

        # Remaining capture flags (only read inside handler methods, not setup_listeners)
        self.capture_inputs = capture_inputs
        self.capture_outputs = capture_outputs
        self.capture_llm_messages = capture_llm_messages
        self.capture_tool_schemas = capture_tool_schemas
        self.capture_agent_snapshot = capture_agent_snapshot
        self.capture_crew_snapshot = capture_crew_snapshot
        self.capture_streaming = capture_streaming
        self.capture_thinking = capture_thinking
        self.trace_name_prefix = trace_name_prefix

        # =====================================================================
        # Initialize all span correlation maps (empty dicts)
        # =====================================================================

        self._crew_spans: dict[str, Any] = {}
        self._task_to_crew_id: dict[str, str] = {}
        self._task_spans: dict[str, Any] = {}
        self._agent_spans: dict[str, Any] = {}
        self._llm_call_spans: dict[str, Any] = {}
        self._tool_spans: dict[str, Any] = {}
        self._tool_run_id_to_agent_id: dict[str, str] = {}
        self._flow_spans: dict[str, Any] = {}
        self._flow_method_spans: dict[str, Any] = {}
        self._memory_op_spans: dict[str, Any] = {}
        self._reasoning_spans: dict[str, Any] = {}
        self._observation_spans: dict[str, Any] = {}
        self._guardrail_spans: dict[str, Any] = {}
        self._a2a_spans: dict[str, Any] = {}
        self._mcp_spans: dict[str, Any] = {}

        # =====================================================================
        # Token & cost accumulators
        # =====================================================================

        self._total_tokens_by_crew: dict[str, int] = {}
        self._total_cost_by_crew: dict[str, float] = {}

        # =====================================================================
        # Token buffer for late-arriving token counts (from monkey-patch)
        # =====================================================================

        self._token_buffer: OrderedDict[str, bool] = OrderedDict()
        self._llm_usage_by_call_id: dict[str, dict[str, Any]] = {}

        # =====================================================================
        # Streaming buffers for LLM responses
        # =====================================================================

        self._llm_stream_chunks: dict[str, list[str]] = {}
        self._llm_thinking_chunks: dict[str, list[str]] = {}

        # =====================================================================
        # A2A streaming buffers
        # =====================================================================

        self._a2a_stream_buffers: dict[str, list[Any]] = {}

        # =====================================================================
        # Pending metadata stash
        # =====================================================================

        self._pending_llm_metadata: dict[str, dict[str, Any]] = {}

        # =====================================================================
        # Start time trackers (for duration calculation using monotonic clock)
        # =====================================================================

        self._task_start_times: dict[str, float] = {}
        self._agent_start_times: dict[str, float] = {}
        self._llm_call_start_times: dict[str, float] = {}
        self._tool_start_times: dict[str, float] = {}
        self._flow_start_times: dict[str, float] = {}
        self._memory_op_start_times: dict[str, float] = {}
        self._a2a_start_times: dict[str, float] = {}

        # =====================================================================
        # Register with class-level listener set and apply token patch
        # =====================================================================

        with _patch_lock:
            _active_listeners.add(self)
            NoveumCrewAIListener._patch_token_tracking()

        logger.info(
            "NoveumCrewAIListener initialized (verbose=%s, prefix=%s)",
            self._verbose,
            self.trace_name_prefix,
        )

        self._migrate_legacy_prefixed_span_maps()

    def _migrate_legacy_prefixed_span_maps(self) -> None:
        """
        Move pre-dedicated-dict namespaced keys out of shared maps.

        Older releases stored reasoning, observation, and guardrail spans under
        ``rsn::``, ``obs::``, and ``grail::`` prefixes inside ``_flow_method_spans``
        / ``_memory_op_spans``.  Keys with more than one ``::`` are left alone so
        legitimate flow keys like ``rsn::method_name::method_id`` are not stolen.
        """
        rsn_p = "rsn::"
        obs_p = "obs::"
        grail_p = "grail::"
        with self._lock:
            for k in list(self._flow_method_spans.keys()):
                if not k.startswith(rsn_p) or k.count("::") != 1:
                    continue
                entry = self._flow_method_spans.pop(k, None)
                if entry is None:
                    continue
                rid = k[len(rsn_p) :]
                self._reasoning_spans.setdefault(rid, entry)
            for k in list(self._memory_op_spans.keys()):
                if k.startswith(obs_p) and k.count("::") == 1:
                    span = self._memory_op_spans.pop(k, None)
                    if span is None:
                        continue
                    oid = k[len(obs_p) :]
                    start_t = self._memory_op_start_times.pop(k, None)
                    self._observation_spans.setdefault(
                        oid, {"span": span, "start_t": start_t}
                    )
                elif k.startswith(grail_p) and k.count("::") == 1:
                    span = self._memory_op_spans.pop(k, None)
                    if span is None:
                        continue
                    gid = k[len(grail_p) :]
                    start_t = self._memory_op_start_times.pop(k, None)
                    self._guardrail_spans.setdefault(
                        gid, {"span": span, "start_t": start_t}
                    )

    def __enter__(self) -> NoveumCrewAIListener:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit — ensures cleanup."""
        self.shutdown()

    # =========================================================================
    # Lifecycle management
    # =========================================================================

    def shutdown(self) -> None:
        """
        Gracefully shut down the listener.

        - Marks as shutdown to block new spans
        - Deregisters all event handlers from CrewAI
        - Cleans up any pending resources
        - Restores token patch (only when last listener shuts down)
        """
        with self._lock:
            if self._is_shutdown:
                logger.debug("NoveumCrewAIListener already shutdown")
                return

            # Resolve client before ``_is_shutdown`` makes ``_get_client()`` return
            # None, so force-close paths can still call ``finish_trace`` on traces.
            shutdown_finish_client = self._get_client()

            self._is_shutdown = True

            # Unsubscribe all event handlers from the CrewAI event bus
            if self._crewai_event_bus is not None:
                for event_cls, handler in self._handlers:
                    try:
                        self._crewai_event_bus.off(event_cls, handler)
                    except Exception:
                        pass
            self._handlers.clear()

            # Snapshot dangling ids BEFORE clearing state so we can
            # force-close them after releasing the lock.  (_finish_crew_span
            # and _finish_flow_span both acquire self._lock internally, so
            # they must be called outside this with-block.)
            dangling_crew_ids = list(self._crew_spans.keys())
            dangling_flow_ids = list(self._flow_spans.keys())

        with _patch_lock:
            _active_listeners.discard(self)
            if len(_active_listeners) == 0:
                NoveumCrewAIListener._restore_token_tracking()

        # --- Force-close any crew/flow spans still open at shutdown time -----
        # This happens when shutdown() is called immediately after crew.kickoff()
        # returns (e.g. zero-impact check) and the CrewKickoffCompletedEvent
        # fires just as the handler is being deregistered.  Writing best-effort
        # completion attributes here ensures crew.status / crew.total_tokens /
        # crew.total_cost / crew.output are not silently dropped.
        for crew_id in dangling_crew_ids:
            try:
                self._finish_crew_span(
                    crew_id,
                    status="ok",
                    output=None,
                    error=None,
                    extra_attrs={"crew.shutdown_closed": True},
                    finish_trace_client=shutdown_finish_client,
                )
            except Exception:
                pass

        for flow_id in dangling_flow_ids:
            try:
                self._finish_flow_span(
                    flow_id,
                    status="ok",
                    error=None,
                    finish_trace_client=shutdown_finish_client,
                )
            except Exception:
                pass

        with self._lock:
            # Clear remaining state (anything not already removed by force-close above)
            self._crew_spans.clear()
            self._task_to_crew_id.clear()
            self._task_spans.clear()
            self._agent_spans.clear()
            self._llm_call_spans.clear()
            self._tool_spans.clear()
            self._tool_run_id_to_agent_id.clear()
            self._flow_spans.clear()
            self._flow_method_spans.clear()
            self._memory_op_spans.clear()
            self._reasoning_spans.clear()
            self._observation_spans.clear()
            self._guardrail_spans.clear()
            self._a2a_spans.clear()
            self._mcp_spans.clear()

            self._llm_stream_chunks.clear()
            self._llm_thinking_chunks.clear()
            self._a2a_stream_buffers.clear()
            self._token_buffer.clear()
            self._llm_usage_by_call_id.clear()
            self._pending_llm_metadata.clear()

            logger.info("NoveumCrewAIListener shutdown complete")

    # =========================================================================
    # Token patch management (class-level)
    # =========================================================================

    @classmethod
    def _patch_token_tracking(cls) -> None:
        """
        Class-level: patch BaseLLM._track_token_usage_internal once for all listeners.

        Correct import path: ``crewai.llms.base_llm`` (plural ``llms``).
        Correct signature:  ``_track_token_usage_internal(self, usage_data: dict)``.
        The call-id is retrieved via the module-level ``get_current_call_id()``
        context-var accessor rather than being passed as an argument.
        """
        global _patch_applied, _original_track_token_usage

        with _patch_lock:
            if _patch_applied:
                return

            try:
                from crewai.llms.base_llm import (
                    BaseLLM,
                )
                from crewai.llms.base_llm import get_current_call_id as _get_call_id
            except ImportError:
                logger.warning(
                    "Could not import BaseLLM from crewai.llms.base_llm — "
                    "token monkey-patch skipped"
                )
                return

            if not hasattr(BaseLLM, "_track_token_usage_internal"):
                logger.debug(
                    "BaseLLM._track_token_usage_internal not found — skipping patch"
                )
                return

            _original_track_token_usage = BaseLLM._track_token_usage_internal

            def _noveum_track_token_usage(
                self_llm: Any,
                usage_data: dict[str, Any],
            ) -> Any:
                """Intercept token tracking and forward to all active listeners."""
                result = _original_track_token_usage(self_llm, usage_data)
                try:
                    call_id = _get_call_id()
                    if usage_data and call_id:
                        with _patch_lock:
                            listeners_snapshot = list(_active_listeners)
                        for listener in listeners_snapshot:
                            listener._buffer_token_usage(call_id, usage_data)
                except Exception as exc:
                    logger.debug("Error buffering tokens in monkey-patch: %s", exc)
                return result

            BaseLLM._track_token_usage_internal = _noveum_track_token_usage
            _patch_applied = True
            logger.info("CrewAI BaseLLM token tracking monkey-patch applied")

    @classmethod
    def _restore_token_tracking(cls) -> None:
        """
        Class-level: restore original method.

        Only called when the last listener shuts down.
        """
        global _patch_applied, _original_track_token_usage

        with _patch_lock:
            if not _patch_applied or _original_track_token_usage is None:
                return

            try:
                from crewai.llms.base_llm import BaseLLM
            except ImportError:
                return

            BaseLLM._track_token_usage_internal = _original_track_token_usage
            _patch_applied = False
            _original_track_token_usage = None
            logger.info("CrewAI BaseLLM token tracking monkey-patch removed")

    def _buffer_token_usage(self, call_id: str, usage: dict[str, Any]) -> None:
        """
        Buffer token usage for a call_id (called from the class-level monkey-patch).

        Writes into ``_llm_usage_by_call_id`` — the same dict that
        ``_handlers_llm._finish_llm_span`` drains — so patched token counts
        flow through to LLM span attributes automatically.

        Uses an :class:`~collections.OrderedDict` for LRU eviction when the
        buffer exceeds ``_MAX_TOKEN_BUFFER_ENTRIES``.
        """
        with self._lock:
            if self._is_shutdown:
                return

            agg = self._llm_usage_by_call_id.setdefault(
                call_id,
                {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cache_read_tokens": 0,
                    "cache_creation_tokens": 0,
                    "reasoning_tokens": 0,
                },
            )
            agg["prompt_tokens"] += int(
                usage.get("prompt_tokens") or usage.get("input_tokens") or 0
            )
            agg["completion_tokens"] += int(
                usage.get("completion_tokens") or usage.get("output_tokens") or 0
            )
            agg["total_tokens"] += int(usage.get("total_tokens") or 0)
            agg["cache_read_tokens"] += int(
                usage.get("cached_tokens")
                or usage.get("cache_read_tokens")
                or usage.get("cache_read_input_tokens")
                or 0
            )
            agg["cache_creation_tokens"] += int(
                usage.get("cache_creation_tokens")
                or usage.get("cache_creation_input_tokens")
                or 0
            )
            agg["reasoning_tokens"] += int(usage.get("reasoning_tokens") or 0)

            # LRU sentinel: bump ``call_id`` to most-recently-used before evicting
            # oldest entries (``popitem(last=False)``).
            if call_id in self._token_buffer:
                self._token_buffer.move_to_end(call_id, last=True)
            else:
                self._token_buffer[call_id] = True
            while len(self._token_buffer) > self._MAX_TOKEN_BUFFER_ENTRIES:
                evicted_id, _ = self._token_buffer.popitem(last=False)
                self._llm_usage_by_call_id.pop(evicted_id, None)
                logger.debug(
                    "Token buffer evicted call_id=%s (limit %d exceeded)",
                    evicted_id,
                    self._MAX_TOKEN_BUFFER_ENTRIES,
                )

    def _accumulate_tokens(self, crew_id: str, tokens: int, cost: float) -> None:
        """
        Add *tokens* and *cost* to the per-crew accumulators.

        Called by ``_handlers_llm._finish_llm_span`` after each LLM call.
        The accumulated totals are written to ``crew.total_tokens`` and
        ``crew.total_cost`` when the crew trace closes.
        """
        with self._lock:
            self._total_tokens_by_crew[crew_id] = (
                self._total_tokens_by_crew.get(crew_id, 0) + tokens
            )
            self._total_cost_by_crew[crew_id] = (
                self._total_cost_by_crew.get(crew_id, 0.0) + cost
            )

    # =========================================================================
    # Helper methods (used by all mixins)
    # =========================================================================

    def _is_active(self) -> bool:
        """Return True if listener is not shutdown and client is available."""
        if self._is_shutdown:
            return False
        if self._get_client() is None:
            return False
        return True

    def _get_client(self) -> Optional[NoveumClient]:
        """
        Return the tracing client, or None if shutdown or not initialized.

        Attempts to:
        1. Return cached client if available
        2. Get the current initialized client from noveum_trace
        3. Return None if not ready
        """
        if self._is_shutdown:
            return None

        if self._client is not None:
            return self._client

        try:
            from noveum_trace import get_client, is_initialized

            if is_initialized():
                self._client = get_client()
                return self._client
        except Exception:
            pass

        return None

    def _create_child_span(
        self,
        span_name: str,
        parent_span: Optional[Any] = None,
        attributes: Optional[dict[str, Any]] = None,
        crew_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Any:
        """
        Create a child span using ``trace.create_span()`` so no thread-local
        current-trace state is required or polluted.

        Looks up the owning ``Trace`` object via the parent span's ``trace_id``
        attribute (which every :class:`~noveum_trace.core.span.Span` carries).
        Falls back to scanning ``_crew_spans`` / ``_flow_spans`` entries when
        the parent span is ``None``.

        ``crew_id`` or ``task_id`` can be provided as hints to unambiguously
        select the correct trace when multiple crews are running concurrently.
        The ``_task_to_crew_id`` reverse-map (populated at ``CrewKickoffStartedEvent``)
        is consulted first, so task/agent/llm spans are always routed to the
        correct crew trace even when crew lifetimes overlap.
        """
        client = self._get_client()
        if client is None:
            return None

        # --- Resolve trace via parent span's trace_id -----------------------
        parent_span_id: Optional[str] = None
        trace: Any = None

        if parent_span is not None:
            parent_span_id = getattr(parent_span, "span_id", None)
            trace_id = getattr(parent_span, "trace_id", None)
            if trace_id:
                # Look in client's registry (no new internal API needed)
                try:
                    with client._lock:
                        trace = client._active_traces.get(trace_id)
                except Exception:
                    pass

        if trace is None:
            # --- Resolve trace by crew_id hint or task→crew reverse map ------
            # This prevents cross-crew contamination when two crews' lifetimes
            # overlap: we use the task's pre-registered crew_id rather than
            # blindly picking the first open crew trace.
            resolved_crew_id: Optional[str] = crew_id
            if resolved_crew_id is None and task_id is not None:
                with self._lock:
                    resolved_crew_id = self._task_to_crew_id.get(str(task_id))

            if resolved_crew_id is not None:
                with self._lock:
                    entry = self._crew_spans.get(resolved_crew_id)
                if isinstance(entry, dict):
                    t = entry.get("trace")
                    if t is not None:
                        trace = t
                        if parent_span_id is None:
                            fallback_span = entry.get("span")
                            if fallback_span is not None:
                                parent_span_id = getattr(fallback_span, "span_id", None)

        if trace is None:
            # --- Last-resort fallback: scan all open crew / flow entries ------
            # ONLY apply this when exactly one crew trace is open.  With two or
            # more open traces we cannot safely determine the owner and must
            # leave the span unparented rather than cross-contaminate traces.
            with self._lock:
                open_crew_entries = [
                    e
                    for e in list(self._crew_spans.values())
                    + list(self._flow_spans.values())
                    if isinstance(e, dict) and e.get("trace") is not None
                ]

            if len(open_crew_entries) == 1:
                entry = open_crew_entries[0]
                trace = entry.get("trace")
                if parent_span_id is None:
                    fallback_span = entry.get("span")
                    if fallback_span is not None:
                        parent_span_id = getattr(fallback_span, "span_id", None)

        if trace is None:
            logger.debug(
                "_create_child_span: no active trace found for span '%s'", span_name
            )
            return None

        try:
            return trace.create_span(
                name=span_name,
                parent_span_id=parent_span_id,
                attributes=attributes or {},
            )
        except Exception as exc:
            logger.debug("Error creating span %s: %s", span_name, exc)
            return None

    def _get_agent_or_crew_span(
        self, agent_id: Optional[str], crew_id: Optional[str] = None
    ) -> Optional[Any]:
        """Return agent span if available, else crew span, else None."""
        if agent_id:
            span = self._get_agent_span(agent_id)
            if span:
                return span

        if crew_id:
            return self._get_crew_span(crew_id)

        return None

    def _get_agent_or_task_span(
        self, agent_id: Optional[str], task_id: Optional[str]
    ) -> Optional[Any]:
        """Return agent span if available, else task span, else None."""
        if agent_id:
            span = self._get_agent_span(agent_id)
            if span:
                return span

        if task_id:
            with self._lock:
                return self._task_spans.get(task_id)

        return None

    def _get_tool_or_agent_span(
        self, run_id: Optional[str], agent_id: Optional[str]
    ) -> Optional[Any]:
        """Return tool span if available, else agent span, else None."""
        if run_id:
            with self._lock:
                span = self._tool_spans.get(run_id)
                if span:
                    return span

        if agent_id:
            return self._get_agent_span(agent_id)

        return None

    def _get_crew_span(self, crew_id: str) -> Optional[Any]:
        """Retrieve open crew span, or None."""
        with self._lock:
            entry = self._crew_spans.get(crew_id)
        return entry.get("span") if entry else None

    def _get_agent_span(self, agent_id: Optional[str]) -> Optional[Any]:
        """Retrieve open agent span, or None."""
        if not agent_id:
            return None
        with self._lock:
            return self._agent_spans.get(agent_id)

    def _get_task_span(self, task_id: str) -> Optional[Any]:
        """Retrieve open task span, or None."""
        with self._lock:
            return self._task_spans.get(task_id)

    def _get_tool_span(self, run_id: str) -> Optional[Any]:
        """Retrieve open tool span, or None."""
        with self._lock:
            return self._tool_spans.get(run_id)

    def _get_memory_span(self, op_id: str) -> Optional[Any]:
        """Retrieve open memory span, or None."""
        with self._lock:
            return self._memory_op_spans.get(op_id)

    # =========================================================================
    # CrewAI BaseEventListener contract
    # =========================================================================

    def setup_listeners(self, crewai_event_bus: Any) -> None:
        """
        Subscribe all handler methods to the CrewAI event bus.

        Called automatically by ``BaseEventListener.__init__`` with the
        global ``crewai_event_bus`` singleton.  Each event type is imported
        from ``crewai.events.types.*`` inside its own ``try/except ImportError``
        block so that older CrewAI versions that lack certain event classes
        continue to work — only the missing event types are silently skipped.

        The ``(event_class, handler)`` pairs are stored in ``self._handlers``
        so that ``shutdown()`` can call ``crewai_event_bus.off()`` for each.
        """
        self._crewai_event_bus = crewai_event_bus

        # ------------------------------------------------------------------
        # Helper: subscribe one handler using the direct registration API.
        # crewai_event_bus.on() is a decorator-factory (returns a decorator),
        # not a direct subscribe call — use register_handler() instead.
        # ------------------------------------------------------------------
        def _sub(event_cls: Any, handler: Any) -> None:
            crewai_event_bus.register_handler(event_cls, handler)
            with self._lock:
                self._handlers.append((event_cls, handler))

        # ── Crew ──────────────────────────────────────────────────────────
        try:
            from crewai.events.types.crew_events import (
                CrewKickoffCompletedEvent,
                CrewKickoffFailedEvent,
                CrewKickoffStartedEvent,
            )

            _sub(CrewKickoffStartedEvent, self.on_crew_kickoff_started)
            _sub(CrewKickoffCompletedEvent, self.on_crew_kickoff_completed)
            _sub(CrewKickoffFailedEvent, self.on_crew_kickoff_failed)
        except ImportError:
            logger.debug("Crew kickoff events not available in this CrewAI version")

        try:
            from crewai.events.types.crew_events import (
                CrewTestCompletedEvent,
                CrewTestStartedEvent,
            )

            _sub(CrewTestStartedEvent, self.on_crew_test_started)
            _sub(CrewTestCompletedEvent, self.on_crew_test_completed)
        except ImportError:
            pass

        try:
            from crewai.events.types.crew_events import CrewTestResultEvent

            _sub(CrewTestResultEvent, self.on_crew_test_result)
        except ImportError:
            pass

        try:
            from crewai.events.types.crew_events import (
                CrewTrainCompletedEvent,
                CrewTrainFailedEvent,
                CrewTrainStartedEvent,
            )

            _sub(CrewTrainStartedEvent, self.on_crew_train_started)
            _sub(CrewTrainCompletedEvent, self.on_crew_train_completed)
            _sub(CrewTrainFailedEvent, self.on_crew_train_failed)
        except ImportError:
            pass

        # ── Task ──────────────────────────────────────────────────────────
        try:
            from crewai.events.types.task_events import (
                TaskCompletedEvent,
                TaskFailedEvent,
                TaskStartedEvent,
            )

            _sub(TaskStartedEvent, self.on_task_started)
            _sub(TaskCompletedEvent, self.on_task_completed)
            _sub(TaskFailedEvent, self.on_task_failed)
        except ImportError:
            logger.debug("Task events not available in this CrewAI version")

        try:
            from crewai.events.types.task_events import TaskEvaluationEvent

            _sub(TaskEvaluationEvent, self.on_task_evaluation)
        except ImportError:
            pass

        # ── Agent ─────────────────────────────────────────────────────────
        try:
            from crewai.events.types.agent_events import (
                AgentExecutionCompletedEvent,
                AgentExecutionErrorEvent,
                AgentExecutionStartedEvent,
            )

            _sub(AgentExecutionStartedEvent, self.on_agent_execution_started)
            _sub(AgentExecutionCompletedEvent, self.on_agent_execution_completed)
            _sub(AgentExecutionErrorEvent, self.on_agent_execution_error)
        except ImportError:
            logger.debug("Agent execution events not available in this CrewAI version")

        try:
            from crewai.events.types.agent_events import (
                LiteAgentExecutionCompletedEvent,
                LiteAgentExecutionErrorEvent,
                LiteAgentExecutionStartedEvent,
            )

            _sub(LiteAgentExecutionStartedEvent, self.on_lite_agent_started)
            _sub(LiteAgentExecutionCompletedEvent, self.on_lite_agent_completed)
            _sub(LiteAgentExecutionErrorEvent, self.on_lite_agent_error)
        except ImportError:
            pass

        try:
            from crewai.events.types.agent_events import (
                AgentEvaluationCompletedEvent,
                AgentEvaluationFailedEvent,
                AgentEvaluationStartedEvent,
            )

            _sub(AgentEvaluationStartedEvent, self.on_agent_evaluation_started)
            _sub(AgentEvaluationCompletedEvent, self.on_agent_evaluation_completed)
            _sub(AgentEvaluationFailedEvent, self.on_agent_evaluation_error)
        except ImportError:
            pass

        # ── LLM ───────────────────────────────────────────────────────────
        try:
            from crewai.events.types.llm_events import (
                LLMCallCompletedEvent,
                LLMCallFailedEvent,
                LLMCallStartedEvent,
            )

            _sub(LLMCallStartedEvent, self.on_llm_call_started)
            _sub(LLMCallCompletedEvent, self.on_llm_call_completed)
            _sub(LLMCallFailedEvent, self.on_llm_call_failed)
        except ImportError:
            logger.debug("LLM call events not available in this CrewAI version")

        try:
            from crewai.events.types.llm_events import LLMStreamChunkEvent

            _sub(LLMStreamChunkEvent, self.on_llm_stream_chunk)
        except ImportError:
            pass

        try:
            from crewai.events.types.llm_events import LLMThinkingChunkEvent

            _sub(LLMThinkingChunkEvent, self.on_llm_thinking_chunk)
        except ImportError:
            pass

        # ── Tool ──────────────────────────────────────────────────────────
        try:
            from crewai.events.types.tool_usage_events import (
                ToolUsageErrorEvent,
                ToolUsageFinishedEvent,
                ToolUsageStartedEvent,
            )

            _sub(ToolUsageStartedEvent, self.on_tool_usage_started)
            _sub(ToolUsageFinishedEvent, self.on_tool_usage_finished)
            _sub(ToolUsageErrorEvent, self.on_tool_usage_error)
        except ImportError:
            logger.debug("Tool usage events not available in this CrewAI version")

        try:
            from crewai.events.types.tool_usage_events import (
                ToolValidateInputErrorEvent,
            )

            _sub(ToolValidateInputErrorEvent, self.on_tool_validate_input_error)
        except ImportError:
            pass

        try:
            from crewai.events.types.tool_usage_events import ToolSelectionErrorEvent

            _sub(ToolSelectionErrorEvent, self.on_tool_selection_error)
        except ImportError:
            pass

        try:
            from crewai.events.types.tool_usage_events import ToolExecutionErrorEvent

            _sub(ToolExecutionErrorEvent, self.on_tool_execution_error)
        except ImportError:
            pass

        # ── Memory ────────────────────────────────────────────────────────
        if self.capture_memory:
            try:
                from crewai.events.types.memory_events import (
                    MemoryQueryCompletedEvent,
                    MemoryQueryFailedEvent,
                    MemoryQueryStartedEvent,
                    MemorySaveCompletedEvent,
                    MemorySaveFailedEvent,
                    MemorySaveStartedEvent,
                )

                _sub(MemoryQueryStartedEvent, self.on_memory_query_started)
                _sub(MemoryQueryCompletedEvent, self.on_memory_query_completed)
                _sub(MemoryQueryFailedEvent, self.on_memory_query_failed)
                _sub(MemorySaveStartedEvent, self.on_memory_save_started)
                _sub(MemorySaveCompletedEvent, self.on_memory_save_completed)
                _sub(MemorySaveFailedEvent, self.on_memory_save_failed)
            except ImportError:
                pass

            try:
                from crewai.events.types.memory_events import (
                    MemoryRetrievalCompletedEvent,
                    MemoryRetrievalFailedEvent,
                    MemoryRetrievalStartedEvent,
                )

                _sub(MemoryRetrievalStartedEvent, self.on_memory_retrieval_started)
                _sub(MemoryRetrievalCompletedEvent, self.on_memory_retrieval_completed)
                _sub(MemoryRetrievalFailedEvent, self.on_memory_retrieval_failed)
            except ImportError:
                pass

        # ── Knowledge ─────────────────────────────────────────────────────
        if self.capture_knowledge:
            try:
                from crewai.events.types.knowledge_events import (
                    KnowledgeQueryCompletedEvent,
                    KnowledgeQueryFailedEvent,
                    KnowledgeQueryStartedEvent,
                    KnowledgeRetrievalCompletedEvent,
                    KnowledgeRetrievalStartedEvent,
                    KnowledgeSearchQueryFailedEvent,
                )

                _sub(
                    KnowledgeRetrievalStartedEvent, self.on_knowledge_retrieval_started
                )
                _sub(
                    KnowledgeRetrievalCompletedEvent,
                    self.on_knowledge_retrieval_completed,
                )
                _sub(KnowledgeQueryStartedEvent, self.on_knowledge_query_started)
                _sub(KnowledgeQueryCompletedEvent, self.on_knowledge_query_completed)
                _sub(KnowledgeQueryFailedEvent, self.on_knowledge_query_failed)
                _sub(
                    KnowledgeSearchQueryFailedEvent,
                    self.on_knowledge_search_query_failed,
                )
            except ImportError:
                pass

        # ── Flow ──────────────────────────────────────────────────────────
        # NOTE: FlowFailedEvent does not exist in CrewAI 1.x — it is kept in
        # its own try/except so missing it never blocks the core flow events.
        if self.capture_flow:
            try:
                from crewai.events.types.flow_events import (
                    FlowFinishedEvent,
                    FlowStartedEvent,
                    MethodExecutionFailedEvent,
                    MethodExecutionFinishedEvent,
                    MethodExecutionStartedEvent,
                )

                _sub(FlowStartedEvent, self.on_flow_started)
                _sub(FlowFinishedEvent, self.on_flow_finished)
                _sub(MethodExecutionStartedEvent, self.on_method_execution_started)
                _sub(MethodExecutionFinishedEvent, self.on_method_execution_finished)
                _sub(MethodExecutionFailedEvent, self.on_method_execution_failed)
            except ImportError:
                pass

            try:
                from crewai.events.types.flow_events import FlowFailedEvent

                _sub(FlowFailedEvent, self.on_flow_failed)
            except ImportError:
                pass

            try:
                from crewai.events.types.flow_events import FlowPausedEvent

                _sub(FlowPausedEvent, self.on_flow_paused)
            except ImportError:
                pass

            try:
                from crewai.events.types.flow_events import (
                    FlowInputReceivedEvent,
                    FlowInputRequestedEvent,
                )

                _sub(FlowInputRequestedEvent, self.on_flow_input_requested)
                _sub(FlowInputReceivedEvent, self.on_flow_input_received)
            except ImportError:
                pass

            try:
                from crewai.events.types.flow_events import (
                    HumanFeedbackReceivedEvent,
                    HumanFeedbackRequestedEvent,
                )

                _sub(HumanFeedbackRequestedEvent, self.on_human_feedback_requested)
                _sub(HumanFeedbackReceivedEvent, self.on_human_feedback_received)
            except ImportError:
                pass

        # ── Reasoning ─────────────────────────────────────────────────────
        # AgentReasoning* is in crewai.events.types.reasoning_events;
        # Observation/Planning events are in crewai.events.types.observation_events.
        if self.capture_reasoning:
            try:
                from crewai.events.types.reasoning_events import (
                    AgentReasoningCompletedEvent,
                    AgentReasoningFailedEvent,
                    AgentReasoningStartedEvent,
                )

                _sub(AgentReasoningStartedEvent, self.on_agent_reasoning_started)
                _sub(AgentReasoningCompletedEvent, self.on_agent_reasoning_completed)
                _sub(AgentReasoningFailedEvent, self.on_agent_reasoning_failed)
            except ImportError:
                pass

            try:
                from crewai.events.types.observation_events import (
                    StepObservationCompletedEvent,
                    StepObservationFailedEvent,
                    StepObservationStartedEvent,
                )

                _sub(StepObservationStartedEvent, self.on_step_observation_started)
                _sub(StepObservationCompletedEvent, self.on_step_observation_completed)
                _sub(StepObservationFailedEvent, self.on_step_observation_failed)
            except ImportError:
                pass

            try:
                from crewai.events.types.observation_events import PlanRefinementEvent

                _sub(PlanRefinementEvent, self.on_plan_refinement)
            except ImportError:
                pass

            try:
                from crewai.events.types.observation_events import (
                    PlanReplanTriggeredEvent,
                )

                _sub(PlanReplanTriggeredEvent, self.on_plan_replan_triggered)
            except ImportError:
                pass

            try:
                from crewai.events.types.observation_events import (
                    GoalAchievedEarlyEvent,
                )

                _sub(GoalAchievedEarlyEvent, self.on_goal_achieved_early)
            except ImportError:
                pass

        # ── Guardrail ─────────────────────────────────────────────────────
        if self.capture_guardrails:
            try:
                from crewai.events.types.llm_guardrail_events import (
                    LLMGuardrailCompletedEvent,
                    LLMGuardrailStartedEvent,
                )

                _sub(LLMGuardrailStartedEvent, self.on_llm_guardrail_started)
                _sub(LLMGuardrailCompletedEvent, self.on_llm_guardrail_completed)
            except ImportError:
                pass

            try:
                from crewai.events.types.llm_guardrail_events import (
                    LLMGuardrailFailedEvent,
                )

                _sub(LLMGuardrailFailedEvent, self.on_llm_guardrail_failed)
            except ImportError:
                pass

        # ── MCP ───────────────────────────────────────────────────────────
        if self.capture_mcp:
            try:
                from crewai.events.types.mcp_events import (
                    MCPConnectionCompletedEvent,
                    MCPConnectionFailedEvent,
                    MCPConnectionStartedEvent,
                    MCPToolExecutionCompletedEvent,
                    MCPToolExecutionFailedEvent,
                    MCPToolExecutionStartedEvent,
                )

                _sub(MCPConnectionStartedEvent, self.on_mcp_connection_started)
                _sub(MCPConnectionCompletedEvent, self.on_mcp_connection_completed)
                _sub(MCPConnectionFailedEvent, self.on_mcp_connection_failed)
                _sub(MCPToolExecutionStartedEvent, self.on_mcp_tool_execution_started)
                _sub(
                    MCPToolExecutionCompletedEvent, self.on_mcp_tool_execution_completed
                )
                _sub(MCPToolExecutionFailedEvent, self.on_mcp_tool_execution_failed)
            except ImportError:
                pass

            try:
                from crewai.events.types.mcp_events import MCPConfigFetchFailedEvent

                _sub(MCPConfigFetchFailedEvent, self.on_mcp_config_fetch_failed)
            except ImportError:
                pass

        # ── A2A ───────────────────────────────────────────────────────────
        # Each import is isolated: some classes (e.g. A2ADelegationFailedEvent,
        # A2AConversationFailedEvent, A2AMessageReceivedEvent) do not exist in
        # CrewAI 1.x and must not be bundled with events that do.
        if self.capture_a2a:
            try:
                from crewai.events.types.a2a_events import (
                    A2ADelegationCompletedEvent,
                    A2ADelegationStartedEvent,
                )

                _sub(A2ADelegationStartedEvent, self.on_a2a_delegation_started)
                _sub(A2ADelegationCompletedEvent, self.on_a2a_delegation_completed)
            except ImportError:
                pass

            try:
                from crewai.events.types.a2a_events import A2ADelegationFailedEvent

                _sub(A2ADelegationFailedEvent, self.on_a2a_delegation_failed)
            except ImportError:
                pass

            try:
                from crewai.events.types.a2a_events import (
                    A2AConversationCompletedEvent,
                    A2AConversationStartedEvent,
                )

                _sub(A2AConversationStartedEvent, self.on_a2a_conversation_started)
                _sub(A2AConversationCompletedEvent, self.on_a2a_conversation_completed)
            except ImportError:
                pass

            try:
                from crewai.events.types.a2a_events import A2AMessageSentEvent

                _sub(A2AMessageSentEvent, self.on_a2a_message_sent)
            except ImportError:
                pass

            try:
                from crewai.events.types.a2a_events import (
                    A2AStreamingChunkEvent,
                    A2AStreamingStartedEvent,
                )

                _sub(A2AStreamingStartedEvent, self.on_a2a_streaming_started)
                _sub(A2AStreamingChunkEvent, self.on_a2a_streaming_chunk)
            except ImportError:
                pass

            try:
                from crewai.events.types.a2a_events import (
                    A2APollingStartedEvent,
                    A2APollingStatusEvent,
                )

                _sub(A2APollingStartedEvent, self.on_a2a_polling_started)
                _sub(A2APollingStatusEvent, self.on_a2a_polling_status)
            except ImportError:
                pass

            try:
                from crewai.events.types.a2a_events import A2AArtifactReceivedEvent

                _sub(A2AArtifactReceivedEvent, self.on_a2a_artifact_received)
            except ImportError:
                pass

            try:
                from crewai.events.types.a2a_events import A2AAuthenticationFailedEvent

                _sub(A2AAuthenticationFailedEvent, self.on_a2a_auth_failed)
            except ImportError:
                pass

            try:
                from crewai.events.types.a2a_events import A2AConnectionErrorEvent

                _sub(A2AConnectionErrorEvent, self.on_a2a_connection_error)
            except ImportError:
                pass

        logger.info(
            "NoveumCrewAIListener subscribed to %d event types",
            len(self._handlers),
        )


# ============================================================================
# Factory function
# ============================================================================


def setup_crewai_tracing(**kwargs: Any) -> NoveumCrewAIListener:
    """
    Factory function to set up CrewAI tracing.

    Requires noveum_trace.init() to be called first.

    Args:
        **kwargs: Passed to NoveumCrewAIListener constructor (capture flags, etc.)

    Returns:
        NoveumCrewAIListener instance ready to be attached to crews.

    Raises:
        RuntimeError: If Noveum tracing is not initialized.

    Example
    -------
    >>> from noveum_trace import init
    >>> from noveum_trace.integrations.crewai import setup_crewai_tracing
    >>>
    >>> init(workspace="my-workspace", api_key="...")
    >>> listener = setup_crewai_tracing(capture_reasoning=True)
    >>> crew.callback_function = listener
    """
    try:
        from noveum_trace import get_client, is_initialized
    except ImportError as exc:
        raise RuntimeError(
            "noveum_trace not installed. Install with: pip install noveum-trace"
        ) from exc

    if not is_initialized():
        raise RuntimeError(
            "Noveum tracing not initialized. Call noveum_trace.init() first."
        )

    client = get_client()
    return NoveumCrewAIListener(client, **kwargs)
