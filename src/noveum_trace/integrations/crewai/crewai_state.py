"""
Typing helpers and mixin base for the CrewAI observer.

``_CrewAIObserverState`` holds all instance-attribute annotations that are
actually initialised inside ``NoveumCrewAIListener.__init__``.
``_CrewAIObserverMethods`` is a ``Protocol`` documenting the helper methods
on the listener class that mixins are allowed to call.
Mixins inherit only ``_CrewAIObserverState`` via ``_CrewAIObserverMixinBase``
so that cooperative ``super().__init__`` always reaches the CrewAI
``BaseEventListener`` (or any other base) without the Protocol being inserted
into the MRO.

RLock choice
------------
``threading.RLock`` is used throughout (not ``Lock``) to match the
``NoveumClient`` pattern and to allow re-entrant acquisition from the same
thread — for example when a span-finish callback triggers cost accumulation
that itself needs the same lock.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Optional, Protocol

# ---------------------------------------------------------------------------
# Annotation-only state class
# ---------------------------------------------------------------------------


class _CrewAIObserverState:
    """
    Annotation-only: every field mirrors an initialisation in
    ``NoveumCrewAIListener.__init__``.

    No ``__init__`` here — the annotations exist purely to satisfy static
    type-checkers and to document the complete state surface of the listener.
    """

    # -- Synchronisation -----------------------------------------------------
    #: Re-entrant lock protecting all mutable state below.
    _lock: threading.RLock

    # -- Lifecycle flags ------------------------------------------------------
    #: Set to ``True`` after ``shutdown()`` / ``__exit__`` to block new spans.
    _is_shutdown: bool

    # -- Registered event handlers for deregistration at shutdown -------------
    #: List of ``(event_type, handler_fn)`` pairs registered with CrewAI so
    #: they can be cleanly removed when the listener shuts down.
    _handlers: list[tuple[type, Any]]

    # =========================================================================
    # Span correlation maps
    # =========================================================================
    # Each map is keyed by the domain identifier of the CrewAI entity and
    # holds the *open* Span object (or whatever the NoveumClient returns from
    # ``start_span``).  Spans are removed from the map when they are finished.

    #: crew_id  → root Span for the entire crew execution
    _crew_spans: dict[str, Any]

    #: task_id  → crew_id   (populated at CrewKickoffStartedEvent)
    #: Allows task/agent/llm spans to resolve the correct parent trace even
    #: when multiple crews are running concurrently or their lifetimes overlap.
    _task_to_crew_id: dict[str, str]

    #: task_id  → Span for a single Task execution
    _task_spans: dict[str, Any]

    #: agent_id → Span for an Agent's current execution step
    _agent_spans: dict[str, Any]

    #: call_id  → Span for a single LLM call (``on_llm_start`` … ``on_llm_end``)
    _llm_call_spans: dict[str, Any]

    #: tool_run_id → Span for a single Tool invocation
    _tool_spans: dict[str, Any]

    #: tool run_id → agent_id that opened the span.
    #: Allows _close_orphan_tool_spans to find all tool spans for a given agent
    #: so they can be force-closed when the agent finishes without a FinishedEvent.
    _tool_run_id_to_agent_id: dict[str, str]

    #: flow_id  → Span tracking a CrewAI Flow execution
    _flow_spans: dict[str, Any]

    #: method_id → Span for a ``@start`` / ``@listen`` flow-method execution
    _flow_method_spans: dict[str, Any]

    #: memory_op_id → Span for a memory read/write operation
    _memory_op_spans: dict[str, Any]

    #: reasoning_id → entry dict for an open ``crewai.reasoning`` span
    #: (``span``, ``start_t``, ``agent_id``)
    _reasoning_spans: dict[str, Any]

    #: obs_id → entry dict for an open ``crewai.step_observation`` span
    #: (``span``, ``start_t``, ``agent_id``)
    _observation_spans: dict[str, Any]

    #: guardrail_id → entry dict for an open ``crewai.guardrail`` span
    #: (``span``, ``start_t``)
    _guardrail_spans: dict[str, Any]

    #: (context_id, span_type) → open A2A span entry; span_type is
    #: ``"delegation"`` or ``"conversation"`` so both may coexist per context.
    _a2a_spans: dict[tuple[str, str], Any]

    #: mcp_key → Span for an MCP (Multi-agent Control Protocol) operation.
    #: Keys are typically ``f"{crew_id}:{tool_name}"`` or a UUID assigned at
    #: the start of the MCP call.
    _mcp_spans: dict[str, Any]

    # =========================================================================
    # Per-crew token & cost accumulators
    # =========================================================================
    # These are incremented every time an LLM call span finishes so that the
    # root crew span can be annotated with ``crew.total_tokens`` and
    # ``crew.total_cost`` when the trace closes.

    #: crew_id → cumulative token count across all nested LLM calls
    _total_tokens_by_crew: dict[str, int]

    #: crew_id → cumulative cost (USD) across all nested LLM calls
    _total_cost_by_crew: dict[str, float]

    # =========================================================================
    # LLM streaming buffers
    # =========================================================================

    #: call_id → ordered list of streaming text chunks received so far.
    #: Joined on span close to produce ``llm.output_text``.
    _llm_stream_chunks: dict[str, list[str]]

    #: call_id → ordered list of "thinking" / chain-of-thought chunks (models
    #: that expose reasoning tokens separately, e.g. Claude 3.7 extended
    #: thinking or o1-series).  Joined on span close to produce
    #: ``llm.thinking_text``.
    _llm_thinking_chunks: dict[str, list[str]]

    # =========================================================================
    # LLM call metadata stash
    # =========================================================================
    #: call_id → dict of pending metadata accumulated between ``on_llm_start``
    #: and ``on_llm_end`` (e.g. messages JSON, system prompt, tool schemas).
    _pending_llm_metadata: dict[str, dict[str, Any]]

    # =========================================================================
    # Task / agent start-time stashes (for duration calculation)
    # =========================================================================
    #: task_id  → monotonic timestamp of task start (``time.monotonic()``)
    _task_start_times: dict[str, float]

    #: agent_id → monotonic timestamp of agent-step start
    _agent_start_times: dict[str, float]

    #: call_id  → monotonic timestamp of LLM call start
    _llm_call_start_times: dict[str, float]

    #: tool_run_id → monotonic timestamp of tool-invocation start
    _tool_start_times: dict[str, float]

    #: flow_id  → monotonic timestamp of flow start
    _flow_start_times: dict[str, float]

    #: memory_op_id → monotonic timestamp of memory op start
    _memory_op_start_times: dict[str, float]

    #: (context_id, span_type) → monotonic timestamp of A2A span start
    _a2a_start_times: dict[tuple[str, str], float]

    # =========================================================================
    # A2A streaming buffers
    # =========================================================================
    #: (context_id, span_type) → ordered list of message dicts (sent/received);
    #: ``span_type="conversation"``. Raw streaming strings use ``_a2a_streaming_chunks``
    #: plus ``_a2a_streaming_lengths`` (running character count per key).
    _a2a_stream_buffers: dict[tuple[str, str], list[Any]]

    #: (context_id, span_type) → raw streaming text chunks (``list[str]``).
    _a2a_streaming_chunks: dict[tuple[str, str], list[str]]

    #: (context_id, span_type) → running sum of ``len(s)`` for that key's chunk list
    #: (updated under ``_lock`` with every append / pop; avoids O(n) re-sums).
    _a2a_streaming_lengths: dict[tuple[str, str], int]

    # =========================================================================
    # Token tracking (monkey-patch buffer)
    # =========================================================================
    #: LRU sentinel map (``call_id → True``) paired with ``_llm_usage_by_call_id``
    _token_buffer: OrderedDict[str, bool]

    #: call_id → dict of token counts populated by monkey-patch
    _llm_usage_by_call_id: dict[str, dict[str, Any]]

    # =========================================================================
    # Capture configuration flags
    # =========================================================================
    capture_inputs: bool
    capture_outputs: bool
    capture_llm_messages: bool
    capture_tool_schemas: bool
    capture_agent_snapshot: bool
    capture_crew_snapshot: bool
    capture_memory: bool
    capture_knowledge: bool
    capture_a2a: bool
    capture_mcp: bool
    capture_flow: bool
    capture_reasoning: bool
    capture_guardrails: bool
    capture_streaming: bool
    capture_thinking: bool
    trace_name_prefix: str
    _verbose: bool


# ---------------------------------------------------------------------------
# Protocol — methods mixins may call on the listener
# ---------------------------------------------------------------------------


class _CrewAIObserverMethods(Protocol):
    """
    Methods implemented on ``NoveumCrewAIListener`` that handler mixins call.

    Using a Protocol (rather than an abstract base) means mixins never
    introduce a concrete method that could shadow the real implementation, and
    type-checkers can verify all call sites without a circular import.
    """

    def _is_active(self) -> bool: ...

    def _create_child_span(
        self,
        span_name: str,
        parent_span: Any = None,
        attributes: Optional[dict[str, Any]] = None,
        crew_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Any: ...

    def _get_client(self) -> Any: ...

    def _get_crew_span(self, crew_id: str) -> Optional[Any]: ...

    def _get_agent_span(self, agent_id: Optional[str]) -> Optional[Any]: ...

    def _get_agent_or_crew_span(
        self, agent_id: Optional[str], crew_id: Optional[str] = None
    ) -> Optional[Any]: ...

    def _get_agent_or_task_span(
        self, agent_id: Optional[str], task_id: Optional[str]
    ) -> Optional[Any]: ...

    def _get_tool_or_agent_span(
        self, run_id: Optional[str], agent_id: Optional[str]
    ) -> Optional[Any]: ...

    def _accumulate_tokens(
        self,
        crew_id: str,
        tokens: int,
        cost: float,
    ) -> None: ...

    def _finish_llm_span(self, *args: Any, **kwargs: Any) -> None: ...

    def _finish_tool_span(self, *args: Any, **kwargs: Any) -> None: ...

    def _close_orphan_observation_spans(
        self, agent_id: str, status: str, error: Any
    ) -> None: ...

    def _close_orphan_reasoning_spans(
        self, agent_id: str, status: str, error: Any
    ) -> None: ...


# ---------------------------------------------------------------------------
# Mixin base — runtime inherits state; type-check sees methods too
# ---------------------------------------------------------------------------


class _CrewAIObserverMixinBase(_CrewAIObserverState):
    """
    Runtime base for all CrewAI handler mixins.

    At runtime this only carries ``_CrewAIObserverState`` annotations so that
    cooperative ``super().__init__`` never hits the Protocol.  Under
    ``TYPE_CHECKING`` the Protocol methods are also present, enabling mypy /
    pyright to validate mixin call sites without import cycles.
    """

    if TYPE_CHECKING:

        def _is_active(self) -> bool: ...

        def _create_child_span(
            self,
            span_name: str,
            parent_span: Any = None,
            attributes: Optional[dict[str, Any]] = None,
            crew_id: Optional[str] = None,
            task_id: Optional[str] = None,
        ) -> Any: ...

        def _get_client(self) -> Any: ...

        def _get_crew_span(self, crew_id: str) -> Optional[Any]: ...

        def _get_agent_span(self, agent_id: Optional[str]) -> Optional[Any]: ...

        def _get_agent_or_crew_span(
            self, agent_id: Optional[str], crew_id: Optional[str] = None
        ) -> Optional[Any]: ...

        def _get_agent_or_task_span(
            self, agent_id: Optional[str], task_id: Optional[str]
        ) -> Optional[Any]: ...

        def _get_tool_or_agent_span(
            self, run_id: Optional[str], agent_id: Optional[str]
        ) -> Optional[Any]: ...

        def _accumulate_tokens(
            self,
            crew_id: str,
            tokens: int,
            cost: float,
        ) -> None: ...

        def _finish_llm_span(self, *args: Any, **kwargs: Any) -> None: ...

        def _finish_tool_span(self, *args: Any, **kwargs: Any) -> None: ...

        def _close_orphan_observation_spans(
            self, agent_id: str, status: str, error: Any
        ) -> None: ...

        def _close_orphan_reasoning_spans(
            self, agent_id: str, status: str, error: Any
        ) -> None: ...

    # ------------------------------------------------------------------
    # Convenience guard — delegates to NoveumCrewAIListener._is_active()
    # which also verifies the client is available.  Mixins call it as:
    #
    #     if not self._is_active():
    #         return
    # ------------------------------------------------------------------
