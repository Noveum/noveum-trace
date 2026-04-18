"""
Unit tests for the CrewAI integration (NoveumCrewAIListener).

Coverage
--------
* Listener initialisation — handler count, capture flags, token patch applied
* Token patch lifecycle — class-level WeakSet, idempotent install, restore on last shutdown
* shutdown() idempotency — safe to call multiple times
* Handlers never raise — malformed / fully-None events must not propagate exceptions
* NoOp when is_initialized()=False — all handlers return without creating spans
* Crew handler path — on_crew_kickoff_started / completed / failed
* Task handler path — on_task_started / completed / failed
* Agent handler path — on_agent_execution_started / completed / error
* LLM handler path — on_llm_call_started / completed / failed / stream_chunk
* Tool handler path — on_tool_usage_started / finished / error
* Memory handler path — on_memory_query_started / completed
* Flow handler path — on_flow_started / finished / method_execution_started
* Token buffer — _buffer_token_usage aggregation, LRU eviction at 512 entries
* _accumulate_tokens — increments per-crew totals under lock
* _create_child_span — resolves trace via parent span's trace_id
* Concurrent thread safety — multiple crews run simultaneously without data corruption
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Ensure we can import from src/
# ---------------------------------------------------------------------------
import sys
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

_src = Path(__file__).parents[3] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))


# ---------------------------------------------------------------------------
# Helpers — mock Noveum objects
# ---------------------------------------------------------------------------


def _make_span(span_id: str = "span-1", trace_id: str = "trace-1") -> MagicMock:
    """Return a minimal mock Span."""
    span = MagicMock()
    span.span_id = span_id
    span.trace_id = trace_id
    span.attributes = {}
    span.finish = MagicMock()
    span.set_attribute = MagicMock(
        side_effect=lambda k, v: span.attributes.update({k: v})
    )
    return span


def _make_trace(trace_id: str = "trace-1") -> MagicMock:
    """Return a minimal mock Trace that creates spans on demand."""
    trace = MagicMock()
    trace.trace_id = trace_id
    trace.finish = MagicMock()

    def _create_span(
        name: str, parent_span_id: Any = None, attributes: Any = None, **kw
    ):
        span = _make_span(trace_id=trace_id)
        if attributes:
            span.attributes.update(dict(attributes))
        return span

    trace.create_span = MagicMock(side_effect=_create_span)
    return trace


def _make_client(trace_id: str = "trace-1") -> MagicMock:
    """Return a minimal mock NoveumClient."""
    client = MagicMock()
    client._lock = threading.RLock()
    trace = _make_trace(trace_id)
    client._active_traces = {trace_id: trace}
    client.start_trace = MagicMock(return_value=trace)
    return client


def _make_listener(client: Any = None, **kwargs) -> Any:
    """Instantiate NoveumCrewAIListener with a mocked client."""
    from noveum_trace.integrations.crewai.crewai_listener import NoveumCrewAIListener

    return NoveumCrewAIListener(client or _make_client(), **kwargs)


# ---------------------------------------------------------------------------
# Helpers — mock CrewAI event factories
# (all fields accessed via safe_getattr so MagicMock defaults are fine;
#  we only set the fields the handler actually reads to relevant values)
# ---------------------------------------------------------------------------


def _crew_event(crew_id: str = "crew-1", crew_name: str = "TestCrew") -> MagicMock:
    ev = MagicMock()
    ev.crew_id = None  # crew_id comes from source.id in handlers
    ev.crew_name = crew_name
    ev.crew = MagicMock()
    ev.inputs = {"topic": "AI"}
    ev.output = "final output"
    ev.total_tokens = 100
    ev.error = None
    return ev


def _crew_source(crew_id: str = "crew-1") -> MagicMock:
    src = MagicMock()
    src.id = crew_id
    src.name = "TestCrew"
    src.process = MagicMock()
    src.process.value = "sequential"
    src.agents = []
    src.tasks = []
    src.memory = False
    src.verbose = False
    src.max_rpm = None
    return src


def _task_event(task_id: str = "task-1") -> MagicMock:
    ev = MagicMock()
    # _resolve_task_id() checks event.task_id first — set it explicitly
    ev.task_id = task_id
    ev.task = MagicMock()
    ev.task.id = task_id
    ev.task.name = "research"
    ev.task.description = "Do research"
    ev.task.expected_output = "A report"
    ev.task.human_input = False
    ev.task.async_execution = False
    ev.task.output_file = None
    ev.task.context = []
    ev.task.agent = MagicMock()
    ev.task.agent.role = "Researcher"
    ev.context = []
    ev.output = MagicMock()
    ev.output.raw = "task result"
    ev.error = None
    return ev


def _agent_event(agent_id: str = "agent-1") -> MagicMock:
    ev = MagicMock()
    # _resolve_agent_id() checks event.agent_id first — set it explicitly
    ev.agent_id = agent_id
    ev.agent = MagicMock()
    ev.agent.id = agent_id
    ev.agent.role = "Researcher"
    ev.agent.goal = "Find info"
    ev.agent.backstory = "Expert"
    ev.agent.max_iter = 5
    ev.agent.allow_delegation = False
    ev.agent.tools = []
    ev.task = MagicMock()
    ev.task.id = "task-1"
    ev.task_prompt = "Do the task"
    ev.tools = []
    ev.output = "agent result"
    ev.error = None
    return ev


def _llm_started_event(call_id: str = "call-1", model: str = "gpt-4o") -> MagicMock:
    ev = MagicMock()
    ev.call_id = call_id
    ev.model = model
    ev.messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
    ]
    ev.tools = []
    ev.callbacks = []
    ev.available_functions = {}
    ev.agent_role = "Researcher"
    ev.task_name = "research"
    ev.agent_id = "agent-1"
    ev.task_id = "task-1"
    return ev


def _llm_completed_event(call_id: str = "call-1", model: str = "gpt-4o") -> MagicMock:
    ev = MagicMock()
    ev.call_id = call_id
    ev.model = model
    ev.response = "The answer is 42."
    ev.call_type = "llm_call"
    ev.usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    ev.messages = []
    ev.agent_role = "Researcher"
    ev.task_name = "research"
    ev.agent_id = "agent-1"
    ev.task_id = "task-1"
    return ev


def _llm_failed_event(call_id: str = "call-1") -> MagicMock:
    ev = MagicMock()
    ev.call_id = call_id
    ev.model = "gpt-4o"
    ev.error = "rate limit exceeded"
    ev.agent_role = "Researcher"
    ev.task_id = "task-1"
    return ev


def _tool_event(tool_name: str = "web_search") -> MagicMock:
    ev = MagicMock()
    ev.tool_name = tool_name
    ev.tool_class = "WebSearchTool"
    ev.tool_args = {"query": "AI trends"}
    ev.agent_key = "agent-1"
    ev.run_attempts = 1
    ev.delegations = 0
    ev.agent = MagicMock()
    ev.agent.role = "Researcher"
    ev.from_task = MagicMock()
    ev.from_task.name = "research"
    ev.from_agent = MagicMock()
    ev.from_agent.role = "Researcher"
    ev.output = "search results"
    ev.started_at = None
    ev.finished_at = None
    ev.from_cache = False
    ev.error = "tool error"
    return ev


def _memory_event(op_id: str = "mem-op-1") -> MagicMock:
    ev = MagicMock()
    # _resolve_op_id() checks event.memory_op_id first — set it explicitly
    ev.memory_op_id = op_id
    ev.query = "what did we discuss?"
    ev.limit = 5
    ev.score_threshold = 0.5
    ev.results = [{"content": "previous discussion"}]
    ev.query_time_ms = 10.0
    ev.from_agent = MagicMock()
    ev.from_agent.role = "Researcher"
    ev.from_task = MagicMock()
    ev.from_task.id = "task-1"
    return ev


def _flow_event(
    flow_name: str = "ContentFlow", flow_id: str | None = None
) -> MagicMock:
    ev = MagicMock()
    # _resolve_flow_id() checks event.flow_id first — set it explicitly
    ev.flow_id = flow_id or flow_name
    ev.flow_name = flow_name
    ev.inputs = {"topic": "AI"}
    ev.result = "flow result"
    ev.state = MagicMock()
    return ev


def _method_event(
    flow_name: str = "ContentFlow", method_name: str = "generate"
) -> MagicMock:
    ev = MagicMock()
    ev.flow_name = flow_name
    ev.method_name = method_name
    ev.params = {}
    ev.state = MagicMock()
    ev.result = "method result"
    return ev


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_patch_state():
    """Guarantee the module-level patch state is clean before each test."""
    import noveum_trace.integrations.crewai.crewai_listener as m

    original = m._original_track_token_usage
    was_applied = m._patch_applied

    # If patch is already applied from another test, restore first
    if was_applied and original is not None:
        try:
            from crewai.llms.base_llm import BaseLLM

            BaseLLM._track_token_usage_internal = original
        except Exception:
            pass

    m._patch_applied = False
    m._original_track_token_usage = None
    # Clear active listeners WeakSet
    for lnr in list(m._active_listeners):
        m._active_listeners.discard(lnr)

    yield

    # Cleanup after test
    if m._patch_applied and m._original_track_token_usage is not None:
        try:
            from crewai.llms.base_llm import BaseLLM

            BaseLLM._track_token_usage_internal = m._original_track_token_usage
        except Exception:
            pass
    m._patch_applied = False
    m._original_track_token_usage = None
    for lnr in list(m._active_listeners):
        m._active_listeners.discard(lnr)


# ===========================================================================
# 1. Initialisation
# ===========================================================================


class TestListenerInit:
    def test_registers_handlers_on_construction(self):
        lnr = _make_listener()
        # 87 handlers registered against the live CrewAI 1.14.1 event bus
        assert len(lnr._handlers) > 0, "Expected handlers to be registered"
        lnr.shutdown()

    def test_capture_flags_default_true(self):
        lnr = _make_listener()
        for flag in [
            "capture_inputs",
            "capture_outputs",
            "capture_llm_messages",
            "capture_memory",
            "capture_flow",
            "capture_reasoning",
            "capture_a2a",
            "capture_mcp",
            "capture_guardrails",
        ]:
            assert getattr(lnr, flag) is True, f"{flag} should default to True"
        lnr.shutdown()

    def test_capture_flags_can_be_disabled(self):
        lnr = _make_listener(capture_memory=False, capture_flow=False)
        assert lnr.capture_memory is False
        assert lnr.capture_flow is False
        lnr.shutdown()

    def test_span_correlation_dicts_are_empty(self):
        lnr = _make_listener()
        for attr in [
            "_crew_spans",
            "_task_spans",
            "_agent_spans",
            "_llm_call_spans",
            "_tool_spans",
            "_flow_spans",
            "_flow_method_spans",
            "_memory_op_spans",
            "_reasoning_spans",
            "_observation_spans",
            "_guardrail_spans",
            "_a2a_spans",
            "_mcp_spans",
        ]:
            assert getattr(lnr, attr) == {}, f"{attr} should start empty"
        lnr.shutdown()

    def test_lock_is_rlock(self):
        lnr = _make_listener()
        # RLock is re-entrant — acquire twice on same thread must not deadlock
        with lnr._lock:
            with lnr._lock:
                pass
        lnr.shutdown()

    def test_is_active_true_when_client_ready(self):
        lnr = _make_listener()
        assert lnr._is_active() is True
        lnr.shutdown()

    def test_is_active_false_after_shutdown(self):
        lnr = _make_listener()
        lnr.shutdown()
        assert lnr._is_active() is False

    def test_context_manager_calls_shutdown(self):
        lnr = _make_listener()
        with lnr:
            pass
        assert lnr._is_shutdown is True


# ===========================================================================
# 2. Token patch lifecycle
# ===========================================================================


class TestTokenPatch:
    def test_patch_applied_on_first_listener(self):
        import noveum_trace.integrations.crewai.crewai_listener as m

        assert m._patch_applied is False
        lnr = _make_listener()
        assert m._patch_applied is True
        lnr.shutdown()

    def test_patch_applied_only_once_for_two_listeners(self):
        from crewai.llms.base_llm import BaseLLM

        lnr1 = _make_listener()
        patched_fn = BaseLLM._track_token_usage_internal

        lnr2 = _make_listener()
        # Should still be the same patched function
        assert BaseLLM._track_token_usage_internal is patched_fn

        lnr1.shutdown()
        lnr2.shutdown()

    def test_patch_restored_when_last_listener_shuts_down(self):
        from crewai.llms.base_llm import BaseLLM

        import noveum_trace.integrations.crewai.crewai_listener as m

        original_fn = BaseLLM._track_token_usage_internal
        lnr1 = _make_listener()
        lnr2 = _make_listener()

        lnr1.shutdown()
        # Patch still active — lnr2 still running
        assert m._patch_applied is True

        lnr2.shutdown()
        # Now restored
        assert m._patch_applied is False
        assert BaseLLM._track_token_usage_internal is original_fn

    def test_active_listeners_weakset_tracks_instance(self):
        import noveum_trace.integrations.crewai.crewai_listener as m

        lnr = _make_listener()
        assert lnr in m._active_listeners

        lnr.shutdown()
        assert lnr not in m._active_listeners

    def test_buffer_token_usage_accumulates(self):
        lnr = _make_listener()
        lnr._buffer_token_usage(
            "call-1", {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        )
        lnr._buffer_token_usage(
            "call-1", {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
        )

        agg = lnr._llm_usage_by_call_id["call-1"]
        assert agg["prompt_tokens"] == 30
        assert agg["completion_tokens"] == 15
        assert agg["total_tokens"] == 45
        lnr.shutdown()

    def test_buffer_token_usage_accepts_input_tokens_alias(self):
        """OpenAI-style keys (prompt_tokens) and Anthropic-style (input_tokens) both work."""
        lnr = _make_listener()
        lnr._buffer_token_usage("call-x", {"input_tokens": 7, "output_tokens": 3})
        agg = lnr._llm_usage_by_call_id["call-x"]
        assert agg["prompt_tokens"] == 7
        assert agg["completion_tokens"] == 3
        lnr.shutdown()

    def test_buffer_lru_eviction_at_512(self):
        """_token_buffer LRU eviction drops oldest entries beyond 512."""
        lnr = _make_listener()
        for i in range(515):
            lnr._buffer_token_usage(
                f"call-{i}",
                {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            )

        # Total keys must be capped at 512
        assert len(lnr._llm_usage_by_call_id) <= 512
        # Oldest entries (call-0, call-1, call-2) should have been evicted
        assert "call-0" not in lnr._llm_usage_by_call_id
        # Newest entry should still be present
        assert "call-514" in lnr._llm_usage_by_call_id
        lnr.shutdown()

    def test_buffer_noop_after_shutdown(self):
        lnr = _make_listener()
        lnr.shutdown()
        # Must not raise and must not modify dict
        lnr._buffer_token_usage("call-z", {"prompt_tokens": 5})
        assert "call-z" not in lnr._llm_usage_by_call_id

    def test_accumulate_tokens_adds_to_crew_totals(self):
        lnr = _make_listener()
        lnr._accumulate_tokens("crew-1", 100, 0.002)
        lnr._accumulate_tokens("crew-1", 50, 0.001)
        assert lnr._total_tokens_by_crew["crew-1"] == 150
        assert abs(lnr._total_cost_by_crew["crew-1"] - 0.003) < 1e-9
        lnr.shutdown()

    def test_accumulate_tokens_multiple_crews(self):
        lnr = _make_listener()
        lnr._accumulate_tokens("crew-A", 100, 0.001)
        lnr._accumulate_tokens("crew-B", 200, 0.002)
        assert lnr._total_tokens_by_crew["crew-A"] == 100
        assert lnr._total_tokens_by_crew["crew-B"] == 200
        lnr.shutdown()


# ===========================================================================
# 3. Shutdown idempotency
# ===========================================================================


class TestShutdownIdempotency:
    def test_double_shutdown_does_not_raise(self):
        lnr = _make_listener()
        lnr.shutdown()
        lnr.shutdown()  # must not raise

    def test_triple_shutdown_does_not_raise(self):
        lnr = _make_listener()
        for _ in range(3):
            lnr.shutdown()

    def test_shutdown_clears_span_maps(self):
        lnr = _make_listener()
        lnr._crew_spans["x"] = object()
        lnr._task_spans["y"] = object()
        lnr.shutdown()
        assert lnr._crew_spans == {}
        assert lnr._task_spans == {}

    def test_is_shutdown_flag_set(self):
        lnr = _make_listener()
        assert lnr._is_shutdown is False
        lnr.shutdown()
        assert lnr._is_shutdown is True

    def test_concurrent_shutdown_is_safe(self):
        """Two threads calling shutdown simultaneously must not deadlock or raise."""
        lnr = _make_listener()
        errors: list[Exception] = []

        def _shutdown():
            try:
                lnr.shutdown()
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=_shutdown)
        t2 = threading.Thread(target=_shutdown)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        assert not t1.is_alive(), "shutdown thread 1 did not finish within timeout"
        t2.join(timeout=5)
        assert not t2.is_alive(), "shutdown thread 2 did not finish within timeout"
        assert errors == [], f"Shutdown raised: {errors}"


# ===========================================================================
# 4. Handlers never raise on malformed events
# ===========================================================================


class TestHandlersNeverRaise:
    """Pass a fully-None MagicMock as both source and event to every handler."""

    _HANDLER_NAMES = [
        "on_crew_kickoff_started",
        "on_crew_kickoff_completed",
        "on_crew_kickoff_failed",
        "on_task_started",
        "on_task_completed",
        "on_task_failed",
        "on_agent_execution_started",
        "on_agent_execution_completed",
        "on_agent_execution_error",
        "on_llm_call_started",
        "on_llm_call_completed",
        "on_llm_call_failed",
        "on_llm_stream_chunk",
        "on_llm_thinking_chunk",
        "on_tool_usage_started",
        "on_tool_usage_finished",
        "on_tool_usage_error",
        "on_memory_query_started",
        "on_memory_query_completed",
        "on_memory_query_failed",
        "on_memory_save_started",
        "on_memory_save_completed",
        "on_memory_save_failed",
        "on_flow_started",
        "on_flow_finished",
        "on_method_execution_started",
        "on_method_execution_finished",
        "on_method_execution_failed",
        "on_knowledge_query_started",
        "on_knowledge_query_completed",
        "on_agent_reasoning_started",
        "on_agent_reasoning_completed",
        "on_llm_guardrail_started",
        "on_llm_guardrail_completed",
        "on_mcp_connection_started",
        "on_mcp_connection_completed",
        "on_mcp_tool_execution_started",
        "on_mcp_tool_execution_completed",
        "on_a2a_delegation_started",
        "on_a2a_delegation_completed",
    ]

    @pytest.mark.parametrize("handler_name", _HANDLER_NAMES)
    def test_handler_never_raises_on_none_event(self, handler_name):
        lnr = _make_listener()
        null_obj = MagicMock()
        null_obj.return_value = None
        # MagicMock attribute access returns a new MagicMock (truthy) by default,
        # set spec=None so safe_getattr returns None for anything falsy
        bare = MagicMock(spec=[])  # no spec = raises AttributeError on all access
        handler = getattr(lnr, handler_name, None)
        if handler is None:
            pytest.skip(f"{handler_name} not present")
        handler(bare, bare)  # must not raise
        lnr.shutdown()

    def test_crew_handler_malformed_source_id(self):
        """source.id raises AttributeError — handler must still succeed (or silently skip)."""
        lnr = _make_listener()
        src = MagicMock(spec=[])
        ev = MagicMock(spec=[])
        lnr.on_crew_kickoff_started(src, ev)
        lnr.shutdown()

    def test_llm_handler_missing_call_id(self):
        lnr = _make_listener()
        ev = MagicMock(spec=[])
        lnr.on_llm_call_started(MagicMock(), ev)
        lnr.on_llm_call_completed(MagicMock(), ev)
        lnr.shutdown()

    def test_tool_handler_missing_fields(self):
        lnr = _make_listener()
        ev = MagicMock(spec=[])
        lnr.on_tool_usage_started(MagicMock(), ev)
        lnr.on_tool_usage_finished(MagicMock(), ev)
        lnr.shutdown()


# ===========================================================================
# 5. NoOp when is_initialized() = False
# ===========================================================================


class TestNoOpWhenNotInitialized:
    def test_crew_handler_noop_no_client(self):
        """Listener with no client and is_initialized()=False must not create spans."""
        from noveum_trace.integrations.crewai.crewai_listener import (
            NoveumCrewAIListener,
        )

        with patch("noveum_trace.is_initialized", return_value=False):
            lnr = NoveumCrewAIListener(client=None)
            src = _crew_source()
            ev = _crew_event()
            lnr.on_crew_kickoff_started(src, ev)
            assert lnr._crew_spans == {}, "No span should be created without a client"
            lnr.shutdown()

    def test_task_handler_noop_no_client(self):
        from noveum_trace.integrations.crewai.crewai_listener import (
            NoveumCrewAIListener,
        )

        with patch("noveum_trace.is_initialized", return_value=False):
            lnr = NoveumCrewAIListener(client=None)
            lnr.on_task_started(MagicMock(), _task_event())
            assert lnr._task_spans == {}
            lnr.shutdown()

    def test_llm_handler_noop_no_client(self):
        from noveum_trace.integrations.crewai.crewai_listener import (
            NoveumCrewAIListener,
        )

        with patch("noveum_trace.is_initialized", return_value=False):
            lnr = NoveumCrewAIListener(client=None)
            lnr.on_llm_call_started(MagicMock(), _llm_started_event())
            assert lnr._llm_call_spans == {}
            lnr.shutdown()


# ===========================================================================
# 6. Crew handler path
# ===========================================================================


class TestCrewHandlers:
    def _listener_with_trace(self):
        """Return (listener, mock_client, mock_trace) ready for crew tests."""
        client = _make_client(trace_id="trace-crew")
        lnr = _make_listener(client=client)
        return lnr, client, client._active_traces["trace-crew"]

    def test_kickoff_started_creates_crew_span_entry(self):
        lnr, client, trace = self._listener_with_trace()
        src = _crew_source("crew-42")
        ev = _crew_event("crew-42")

        lnr.on_crew_kickoff_started(src, ev)

        assert (
            "crew-42" in lnr._crew_spans
        ), "crew-42 should be in _crew_spans after kickoff"
        client.start_trace.assert_called_once()
        lnr.shutdown()

    def test_kickoff_started_stores_trace_reference(self):
        lnr, client, trace = self._listener_with_trace()
        lnr.on_crew_kickoff_started(_crew_source("c1"), _crew_event("c1"))
        assert lnr._crew_spans["c1"]["trace"] is not None
        lnr.shutdown()

    def test_kickoff_completed_clears_crew_span(self):
        lnr, client, trace = self._listener_with_trace()
        src = _crew_source("c2")
        lnr.on_crew_kickoff_started(src, _crew_event("c2"))
        assert "c2" in lnr._crew_spans

        lnr.on_crew_kickoff_completed(src, _crew_event("c2"))
        assert "c2" not in lnr._crew_spans, "Span should be removed after completion"
        lnr.shutdown()

    def test_kickoff_failed_clears_crew_span(self):
        lnr, client, trace = self._listener_with_trace()
        src = _crew_source("c3")
        lnr.on_crew_kickoff_started(src, _crew_event("c3"))
        fail_ev = _crew_event("c3")
        fail_ev.error = "Something broke"
        lnr.on_crew_kickoff_failed(src, fail_ev)
        assert "c3" not in lnr._crew_spans
        lnr.shutdown()

    def test_kickoff_completed_without_started_does_not_raise(self):
        lnr, _, __ = self._listener_with_trace()
        lnr.on_crew_kickoff_completed(_crew_source("missing"), _crew_event("missing"))
        lnr.shutdown()

    def test_multiple_concurrent_crews_isolated(self):
        """Two crews started simultaneously must have independent span entries."""
        lnr = _make_listener()
        src_a, src_b = _crew_source("crew-A"), _crew_source("crew-B")
        lnr.on_crew_kickoff_started(src_a, _crew_event("crew-A"))
        lnr.on_crew_kickoff_started(src_b, _crew_event("crew-B"))

        assert "crew-A" in lnr._crew_spans
        assert "crew-B" in lnr._crew_spans
        lnr.shutdown()


# ===========================================================================
# 7. Task handler path
# ===========================================================================


class TestTaskHandlers:
    def _listener_with_crew(self, crew_id: str = "crew-t"):
        lnr = _make_listener()
        # Pre-populate crew span so task handler can find a parent
        span = _make_span(span_id="crew-span", trace_id="trace-t")
        trace = _make_trace("trace-t")
        lnr._client._active_traces["trace-t"] = trace
        lnr._crew_spans[crew_id] = {
            "trace": trace,
            "span": span,
            "start_t": time.monotonic(),
        }
        return lnr

    def test_task_started_creates_span_entry(self):
        lnr = self._listener_with_crew("crew-t")
        ev = _task_event("task-X")
        lnr.on_task_started(MagicMock(), ev)
        assert "task-X" in lnr._task_spans
        lnr.shutdown()

    def test_task_completed_removes_span(self):
        lnr = self._listener_with_crew("crew-t")
        ev = _task_event("task-Y")
        lnr.on_task_started(MagicMock(), ev)
        assert "task-Y" in lnr._task_spans
        lnr.on_task_completed(MagicMock(), ev)
        assert "task-Y" not in lnr._task_spans
        lnr.shutdown()

    def test_task_failed_removes_span(self):
        lnr = self._listener_with_crew("crew-t")
        ev = _task_event("task-Z")
        lnr.on_task_started(MagicMock(), ev)
        lnr.on_task_failed(MagicMock(), ev)
        assert "task-Z" not in lnr._task_spans
        lnr.shutdown()

    def test_task_completed_without_started_safe(self):
        lnr = self._listener_with_crew()
        lnr.on_task_completed(MagicMock(), _task_event("orphan"))
        lnr.shutdown()


# ===========================================================================
# 8. Agent handler path
# ===========================================================================


class TestAgentHandlers:
    def _listener_with_task(self, task_id: str = "task-a"):
        lnr = _make_listener()
        span = _make_span(span_id="task-span", trace_id="trace-a")
        trace = _make_trace("trace-a")
        lnr._client._active_traces["trace-a"] = trace
        lnr._task_spans[task_id] = span
        return lnr

    def test_agent_started_creates_span(self):
        lnr = self._listener_with_task()
        ev = _agent_event("agt-1")
        ev.task_id = "task-a"  # helps _resolve_task_id find the parent span
        lnr.on_agent_execution_started(MagicMock(), ev)
        assert "agt-1" in lnr._agent_spans
        lnr.shutdown()

    def test_agent_completed_removes_span(self):
        lnr = self._listener_with_task()
        ev = _agent_event("agt-2")
        ev.task_id = "task-a"
        lnr.on_agent_execution_started(MagicMock(), ev)
        lnr.on_agent_execution_completed(MagicMock(), ev)
        assert "agt-2" not in lnr._agent_spans
        lnr.shutdown()

    def test_agent_error_removes_span(self):
        lnr = self._listener_with_task()
        ev = _agent_event("agt-3")
        ev.task_id = "task-a"
        lnr.on_agent_execution_started(MagicMock(), ev)
        lnr.on_agent_execution_error(MagicMock(), ev)
        assert "agt-3" not in lnr._agent_spans
        lnr.shutdown()

    def test_agent_started_sets_available_tools(self):
        """agent.available_tools.* mirrors LangChain-style tool metadata."""
        lnr = self._listener_with_task()
        ev = _agent_event("agt-tools")
        ev.task_id = "task-a"
        t = MagicMock()
        t.name = "calculator"
        t.description = "Do math"
        # _build_agent_attributes reads source.tools before event.tools
        src = MagicMock()
        src.tools = [t]
        lnr.on_agent_execution_started(src, ev)
        span = lnr._agent_spans["agt-tools"]
        assert span.attributes["agent.available_tools.count"] == 1
        assert span.attributes["agent.available_tools.names"] == ["calculator"]
        assert span.attributes["agent.available_tools.descriptions"] == ["Do math"]
        assert "calculator" in span.attributes["agent.available_tools.schemas"]
        lnr.shutdown()


# ===========================================================================
# 9. LLM handler path
# ===========================================================================


class TestLLMHandlers:
    def _listener_with_agent(self, agent_id: str = "agt-llm"):
        lnr = _make_listener()
        span = _make_span(span_id="agent-span", trace_id="trace-llm")
        trace = _make_trace("trace-llm")
        lnr._client._active_traces["trace-llm"] = trace
        lnr._agent_spans[agent_id] = span
        return lnr

    def test_llm_started_creates_span(self):
        lnr = self._listener_with_agent()
        ev = _llm_started_event("cid-1")
        ev.agent_id = "agt-llm"
        lnr.on_llm_call_started(MagicMock(), ev)
        assert "cid-1" in lnr._llm_call_spans
        lnr.shutdown()

    def test_llm_started_sets_available_tools(self):
        """llm.available_tools.* is set when the LLM event carries tools."""
        lnr = self._listener_with_agent()
        ev = _llm_started_event("cid-tools")
        ev.agent_id = "agt-llm"
        tool = MagicMock()
        tool.name = "web_search"
        tool.description = "Search the web"
        ev.tools = [tool]
        lnr.on_llm_call_started(MagicMock(), ev)
        span = lnr._llm_call_spans["cid-tools"]["span"]
        assert span.attributes["llm.available_tools.count"] == 1
        assert span.attributes["llm.available_tools.names"] == ["web_search"]
        assert span.attributes["llm.available_tools.descriptions"] == ["Search the web"]
        assert "web_search" in span.attributes["llm.available_tools.schemas"]
        lnr.shutdown()

    def test_llm_started_sets_available_tools_from_event_inputs(self):
        """CrewAI may pass tools only on ``event.inputs['tools']`` (e.g. source is BaseLLM)."""
        lnr = self._listener_with_agent()
        ev = _llm_started_event("cid-tools-inp")
        ev.agent_id = "agt-llm"
        ev.tools = []
        tool = MagicMock()
        tool.name = "lookup_tool"
        tool.description = "Lookup"
        ev.inputs = {
            "messages": ev.messages,
            "tools": [tool],
        }
        src = MagicMock(spec=[])
        lnr.on_llm_call_started(src, ev)
        span = lnr._llm_call_spans["cid-tools-inp"]["span"]
        assert span.attributes["llm.available_tools.names"] == ["lookup_tool"]
        assert span.attributes["llm.available_tools.descriptions"] == ["Lookup"]
        lnr.shutdown()

    def test_llm_completed_removes_span(self):
        lnr = self._listener_with_agent()
        ev_start = _llm_started_event("cid-2")
        ev_start.agent_id = "agt-llm"
        lnr.on_llm_call_started(MagicMock(), ev_start)

        ev_end = _llm_completed_event("cid-2")
        lnr.on_llm_call_completed(MagicMock(), ev_end)
        assert "cid-2" not in lnr._llm_call_spans
        lnr.shutdown()

    def test_llm_completed_consumes_token_buffer(self):
        """Token buffer populated before completion is drained and written to span."""
        lnr = self._listener_with_agent()
        ev_start = _llm_started_event("cid-tok")
        ev_start.agent_id = "agt-llm"
        lnr.on_llm_call_started(MagicMock(), ev_start)

        # Simulate monkey-patch writing tokens before completion event
        lnr._buffer_token_usage(
            "cid-tok",
            {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )

        ev_end = _llm_completed_event("cid-tok")
        lnr.on_llm_call_completed(MagicMock(), ev_end)

        # Token buffer for that call should be consumed
        assert "cid-tok" not in lnr._llm_usage_by_call_id
        lnr.shutdown()

    def test_llm_failed_removes_span(self):
        lnr = self._listener_with_agent()
        ev_start = _llm_started_event("cid-fail")
        ev_start.agent_id = "agt-llm"
        lnr.on_llm_call_started(MagicMock(), ev_start)

        lnr.on_llm_call_failed(MagicMock(), _llm_failed_event("cid-fail"))
        assert "cid-fail" not in lnr._llm_call_spans
        lnr.shutdown()

    def test_stream_chunk_buffered(self):
        lnr = _make_listener()
        ev = MagicMock()
        ev.call_id = "cid-stream"
        ev.chunk = "Hello"
        lnr.on_llm_stream_chunk(MagicMock(), ev)
        ev2 = MagicMock()
        ev2.call_id = "cid-stream"
        ev2.chunk = " world"
        lnr.on_llm_stream_chunk(MagicMock(), ev2)
        assert lnr._llm_stream_chunks.get("cid-stream") == ["Hello", " world"]
        lnr.shutdown()

    def test_thinking_chunk_buffered(self):
        lnr = _make_listener()
        ev = MagicMock()
        ev.call_id = "cid-think"
        ev.chunk = "reasoning step 1"
        lnr.on_llm_thinking_chunk(MagicMock(), ev)
        assert "cid-think" in lnr._llm_thinking_chunks
        lnr.shutdown()


# ===========================================================================
# 10. Tool handler path
# ===========================================================================


class TestToolHandlers:
    def _listener_with_agent(self):
        lnr = _make_listener()
        span = _make_span(span_id="agent-span", trace_id="trace-tool")
        trace = _make_trace("trace-tool")
        lnr._client._active_traces["trace-tool"] = trace
        lnr._agent_spans["Researcher"] = span
        return lnr

    def test_tool_started_creates_span(self):
        lnr = self._listener_with_agent()
        ev = _tool_event("search")
        ev.from_agent.role = "Researcher"
        lnr.on_tool_usage_started(MagicMock(), ev)
        # tool key is derived from tool_name+agent_role+timestamp
        assert len(lnr._tool_spans) == 1
        lnr.shutdown()

    def test_tool_finished_removes_span(self):
        lnr = self._listener_with_agent()
        ev = _tool_event("search")
        ev.from_agent.role = "Researcher"
        lnr.on_tool_usage_started(MagicMock(), ev)
        key = list(lnr._tool_spans.keys())[0]

        lnr.on_tool_usage_finished(MagicMock(), ev)
        assert key not in lnr._tool_spans
        lnr.shutdown()

    def test_tool_error_removes_span(self):
        lnr = self._listener_with_agent()
        ev = _tool_event("search")
        ev.from_agent.role = "Researcher"
        lnr.on_tool_usage_started(MagicMock(), ev)
        key = list(lnr._tool_spans.keys())[0]

        lnr.on_tool_usage_error(MagicMock(), ev)
        assert key not in lnr._tool_spans
        lnr.shutdown()


# ===========================================================================
# 11. Memory handler path
# ===========================================================================


class TestMemoryHandlers:
    def _listener_with_agent(self):
        lnr = _make_listener()
        span = _make_span(span_id="agent-span", trace_id="trace-mem")
        trace = _make_trace("trace-mem")
        lnr._client._active_traces["trace-mem"] = trace
        lnr._agent_spans["Researcher"] = span
        return lnr

    def test_memory_query_started_creates_span(self):
        lnr = self._listener_with_agent()
        ev = _memory_event()
        ev.from_agent.role = "Researcher"
        lnr.on_memory_query_started(MagicMock(), ev)
        assert len(lnr._memory_op_spans) == 1
        lnr.shutdown()

    def test_memory_query_completed_removes_span(self):
        lnr = self._listener_with_agent()
        ev = _memory_event()
        ev.from_agent.role = "Researcher"
        lnr.on_memory_query_started(MagicMock(), ev)
        key = list(lnr._memory_op_spans.keys())[0]
        lnr.on_memory_query_completed(MagicMock(), ev)
        assert key not in lnr._memory_op_spans
        lnr.shutdown()

    def test_memory_query_completed_pairs_via_started_event_id(self):
        """CrewAI links completed → started via ``started_event_id`` / ``event_id``."""
        from types import SimpleNamespace

        lnr = self._listener_with_agent()
        ev_start = SimpleNamespace(
            event_id="crewai-mem-start-1",
            started_event_id=None,
            memory_op_id=None,
            query="find facts",
            limit=5,
            score_threshold=None,
            agent_id="Researcher",
        )
        lnr.on_memory_query_started(MagicMock(), ev_start)
        assert "crewai-mem-start-1" in lnr._memory_op_spans

        ev_done = SimpleNamespace(
            event_id="crewai-mem-done-9",
            started_event_id="crewai-mem-start-1",
            memory_op_id=None,
            query="find facts",
            results=[],
            limit=5,
            score_threshold=None,
            query_time_ms=12.0,
            agent_id="Researcher",
        )
        lnr.on_memory_query_completed(MagicMock(), ev_done)
        assert "crewai-mem-start-1" not in lnr._memory_op_spans
        lnr.shutdown()

    def test_memory_query_failed_removes_span(self):
        lnr = self._listener_with_agent()
        ev = _memory_event("mem-fail-op")
        ev.from_agent.role = "Researcher"
        lnr.on_memory_query_started(MagicMock(), ev)
        assert "mem-fail-op" in lnr._memory_op_spans

        # fail event must carry same op_id so handler can locate the span
        fail_ev = _memory_event("mem-fail-op")
        fail_ev.error = "timeout"
        lnr.on_memory_query_failed(MagicMock(), fail_ev)
        assert "mem-fail-op" not in lnr._memory_op_spans
        lnr.shutdown()


# ===========================================================================
# 12. Flow handler path
# ===========================================================================


class TestFlowHandlers:
    def test_flow_started_creates_flow_span_entry(self):
        lnr = _make_listener()
        ev = _flow_event("MyFlow", flow_id="flow-1")
        lnr.on_flow_started(MagicMock(), ev)
        assert "flow-1" in lnr._flow_spans
        lnr.shutdown()

    def test_flow_finished_removes_flow_span(self):
        lnr = _make_listener()
        ev = _flow_event("MyFlow2", flow_id="flow-2")
        lnr.on_flow_started(MagicMock(), ev)
        lnr.on_flow_finished(MagicMock(), ev)
        assert "flow-2" not in lnr._flow_spans
        lnr.shutdown()

    def test_method_started_creates_method_span(self):
        lnr = _make_listener()
        flow_ev = _flow_event("FlowX", flow_id="flow-x")
        lnr.on_flow_started(MagicMock(), flow_ev)
        assert len(lnr._flow_spans) == 1

        mev = _method_event("FlowX", "generate")
        lnr.on_method_execution_started(MagicMock(), mev)
        assert len(lnr._flow_method_spans) == 1
        lnr.shutdown()

    def test_method_finished_removes_method_span(self):
        lnr = _make_listener()
        flow_ev = _flow_event("FlowY", flow_id="flow-y")
        lnr.on_flow_started(MagicMock(), flow_ev)

        mev = _method_event("FlowY", "write")
        lnr.on_method_execution_started(MagicMock(), mev)
        key = list(lnr._flow_method_spans.keys())[0]
        lnr.on_method_execution_finished(MagicMock(), mev)
        assert key not in lnr._flow_method_spans
        lnr.shutdown()


# ===========================================================================
# 13. _create_child_span
# ===========================================================================


class TestCreateChildSpan:
    def test_returns_span_when_parent_has_trace_id(self):
        lnr = _make_listener()
        parent_span = _make_span(span_id="parent-1", trace_id="trace-child")
        trace = _make_trace("trace-child")
        lnr._client._active_traces["trace-child"] = trace

        child = lnr._create_child_span("crewai.test", parent_span=parent_span)
        assert child is not None
        trace.create_span.assert_called_once()
        lnr.shutdown()

    def test_returns_none_when_no_active_trace(self):
        lnr = _make_listener()
        parent_span = _make_span(span_id="x", trace_id="nonexistent-trace")
        # _active_traces has no matching trace
        lnr._client._active_traces = {}

        child = lnr._create_child_span("crewai.test", parent_span=parent_span)
        assert child is None
        lnr.shutdown()

    def test_returns_none_when_no_client(self):
        from noveum_trace.integrations.crewai.crewai_listener import (
            NoveumCrewAIListener,
        )

        with patch("noveum_trace.is_initialized", return_value=False):
            lnr = NoveumCrewAIListener(client=None)
            child = lnr._create_child_span("crewai.test", parent_span=_make_span())
            assert child is None
            lnr.shutdown()

    def test_falls_back_to_crew_spans_when_parent_is_none(self):
        """If parent_span is None, _create_child_span scans _crew_spans for a live trace."""
        lnr = _make_listener()
        trace = _make_trace("fallback-trace")
        lnr._client._active_traces["fallback-trace"] = trace
        lnr._crew_spans["crew-fb"] = {
            "trace": trace,
            "span": _make_span(trace_id="fallback-trace"),
            "start_t": 0,
        }

        child = lnr._create_child_span("crewai.test", parent_span=None)
        assert child is not None
        lnr.shutdown()


# ===========================================================================
# 14. Concurrent thread safety — multiple crews
# ===========================================================================


class TestConcurrentCrews:
    """Simulate N concurrent crews to verify no data corruption under lock contention."""

    def test_concurrent_crew_kickoff_no_data_corruption(self):
        N = 8
        lnr = _make_listener()
        errors: list[Exception] = []
        started = threading.Barrier(N)

        def _run_crew(i: int):
            try:
                started.wait()
                crew_id = f"concurrent-crew-{i}"
                src = _crew_source(crew_id)
                ev = _crew_event(crew_id)
                lnr.on_crew_kickoff_started(src, ev)
                time.sleep(0.005)
                lnr.on_crew_kickoff_completed(src, ev)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_run_crew, args=(i,)) for i in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
            assert (
                not t.is_alive()
            ), "concurrent crew thread did not finish within timeout"

        assert errors == [], f"Concurrent crews raised: {errors}"
        # All crews should have been cleaned up
        assert lnr._crew_spans == {}
        lnr.shutdown()

    def test_concurrent_token_buffering(self):
        """Multiple threads writing to _buffer_token_usage must not corrupt totals."""
        N = 20
        calls_per_thread = 50
        lnr = _make_listener()
        errors: list[Exception] = []
        barrier = threading.Barrier(N)

        def _write(thread_id: int):
            try:
                barrier.wait()
                for j in range(calls_per_thread):
                    call_id = f"call-{thread_id}-{j}"
                    lnr._buffer_token_usage(
                        call_id,
                        {
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                            "total_tokens": 2,
                        },
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_write, args=(i,)) for i in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
            assert not t.is_alive(), "token buffer thread did not finish within timeout"

        assert errors == [], f"Token buffer raised: {errors}"
        # Buffer size must be within LRU cap
        assert len(lnr._llm_usage_by_call_id) <= lnr._MAX_TOKEN_BUFFER_ENTRIES
        lnr.shutdown()

    def test_concurrent_accumulate_tokens(self):
        """Parallel _accumulate_tokens must produce the exact expected total."""
        N = 10
        tokens_per_thread = 100
        lnr = _make_listener()
        barrier = threading.Barrier(N)
        errors: list[Exception] = []

        def _accumulate():
            try:
                barrier.wait()
                for _ in range(tokens_per_thread):
                    lnr._accumulate_tokens("shared-crew", 1, 0.0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_accumulate) for _ in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
            assert (
                not t.is_alive()
            ), "accumulate_tokens thread did not finish within timeout"

        assert errors == []
        assert lnr._total_tokens_by_crew["shared-crew"] == N * tokens_per_thread
        lnr.shutdown()


# ---------------------------------------------------------------------------
# New attribute coverage tests
#
# Design notes
# ------------
# * The default ``_make_trace`` mock copies ``attributes`` into ``span.attributes``
#   on create.  ``_make_rich_trace`` does the same for tests that need set_attributes.
# * span.set_attributes() (plural) is auto-mocked by MagicMock and does NOT
#   call the set_attribute side-effect.  The richer span mock wires it up.
# * MagicMock auto-generates a truthy value for every attribute access.
#   When a handler reads event.some_id first, we must explicitly set
#   that field to None to force the fallback to source.id.
# ---------------------------------------------------------------------------


def _make_rich_span(span_id: str = "span-x", trace_id: str = "trace-x") -> MagicMock:
    """Span mock where both set_attribute and set_attributes update .attributes."""
    span = MagicMock()
    span.span_id = span_id
    span.trace_id = trace_id
    span.attributes = {}
    span.finish = MagicMock()
    span.set_attribute = MagicMock(
        side_effect=lambda k, v: span.attributes.update({k: v})
    )
    span.set_attributes = MagicMock(side_effect=lambda d: span.attributes.update(d))
    return span


def _make_rich_trace(trace_id: str = "trace-x") -> MagicMock:
    """Trace mock whose create_span writes the attributes arg into the span."""
    trace = MagicMock()
    trace.trace_id = trace_id
    trace.finish = MagicMock()

    def _create_span(
        name: str, parent_span_id: Any = None, attributes: Any = None, **kw
    ):
        s = _make_rich_span(trace_id=trace_id)
        if attributes:
            s.attributes.update(attributes)
        return s

    trace.create_span = MagicMock(side_effect=_create_span)
    return trace


def _make_rich_listener(**kwargs) -> Any:
    """Listener backed by rich mocks so attribute assertions work."""
    from noveum_trace.integrations.crewai.crewai_listener import NoveumCrewAIListener

    trace_id = "rich-trace-1"
    trace = _make_rich_trace(trace_id)
    client = MagicMock()
    client._lock = threading.RLock()
    client._active_traces = {trace_id: trace}
    client.start_trace = MagicMock(return_value=trace)
    return NoveumCrewAIListener(client, **kwargs)


class TestTaskContextTasks:
    """task.context_tasks is populated from source.context (upstream tasks)."""

    def test_task_context_tasks_populated(self):
        """When source.context has upstream tasks, task.context_tasks is a JSON list."""
        import json

        lnr = _make_rich_listener()

        # Prime a crew span so _create_child_span can find it
        crew_src = _crew_source("crew-ctx")
        crew_ev = _crew_event("crew-ctx")
        lnr.on_crew_kickoff_started(crew_src, crew_ev)

        # Upstream task mock with a real description
        upstream = MagicMock()
        upstream.description = "Research the top 3 AI breakthroughs."
        upstream.id = "task-upstream"
        upstream.name = None

        # Task source: context=[upstream]
        src = MagicMock()
        src.id = "task-summary"
        src.description = "Summarise the research."
        src.expected_output = "One paragraph."
        src.agent = MagicMock()
        src.agent.role = "Writer"
        src.human_input = False
        src.async_execution = False
        src.output_file = None
        src.context = [upstream]

        ev = _task_event("task-summary")
        ev.task_id = "task-summary"
        ev.task.id = "task-summary"

        lnr.on_task_started(src, ev)

        with lnr._lock:
            span = lnr._task_spans.get("task-summary")

        assert span is not None, "task span not created"
        attrs = span.attributes

        assert (
            "task.context_tasks" in attrs
        ), f"task.context_tasks missing; span attributes: {dict(attrs)}"
        context_list = json.loads(attrs["task.context_tasks"])
        assert isinstance(context_list, list)
        assert len(context_list) >= 1
        assert "Research" in context_list[0]

        lnr.shutdown()


class TestLLMTemperature:
    """llm.temperature is captured from the source (BaseLLM) object."""

    def test_llm_temperature_captured(self):
        """When source.temperature is set, llm.temperature appears on the span."""
        lnr = _make_rich_listener()

        # Prime crew + agent span
        crew_src = _crew_source("crew-t")
        lnr.on_crew_kickoff_started(crew_src, _crew_event("crew-t"))
        # Inject agent span directly (like the existing _listener_with_agent pattern)
        agent_span = _make_rich_span("agent-t-span", "rich-trace-1")
        with lnr._lock:
            lnr._agent_spans["agent-t"] = agent_span

        ev = _llm_started_event("llm-temp-1", model="gpt-4o")
        ev.call_id = "llm-temp-1"
        ev.agent_id = "agent-t"
        ev.task_id = None  # force fallback to agent lookup
        # Null out sampling-param fields so the handler reads them from source
        ev.temperature = None
        ev.max_tokens = None
        ev.top_p = None
        ev.seed = None

        # Source is the BaseLLM instance — set temperature on it
        src = MagicMock()
        src.temperature = 0.3
        src.model = "gpt-4o"
        src.llm = None

        lnr.on_llm_call_started(src, ev)

        with lnr._lock:
            entry = lnr._llm_call_spans.get("llm-temp-1")

        assert entry is not None, "LLM span not created"
        # _llm_call_spans stores {"span": span, "crew_id": ..., "agent_id": ...}
        span = entry.get("span") if isinstance(entry, dict) else entry
        assert (
            "llm.temperature" in span.attributes
        ), f"llm.temperature missing; span attributes: {dict(span.attributes)}"
        assert abs(float(span.attributes["llm.temperature"]) - 0.3) < 1e-6

        lnr.shutdown()


class TestToolRunAttempts:
    """tool.run_attempts and tool.delegations are captured from the event."""

    def _listener_with_agent(
        self, crew_id: str = "crew-tool", agent_role: str = "Researcher"
    ):
        """Listener pre-loaded with a crew trace + agent span."""
        lnr = _make_rich_listener()
        trace = _make_rich_trace("rich-trace-1")
        lnr._client._active_traces["rich-trace-1"] = trace
        # Crew span entry (dict form matching _crew_spans layout)
        crew_span = _make_rich_span("crew-sp", "rich-trace-1")
        with lnr._lock:
            lnr._crew_spans[crew_id] = {"span": crew_span, "trace": trace}
        # Agent span
        agent_span = _make_rich_span("agent-sp", "rich-trace-1")
        with lnr._lock:
            lnr._agent_spans[agent_role] = agent_span
        return lnr

    def _tool_ev_with_explicit_id(
        self, run_id: str, tool_name: str, run_attempts: int, delegations: Any
    ) -> MagicMock:
        """Tool event where run_id fields are controlled so key == run_id."""
        ev = MagicMock()
        ev.run_id = run_id  # _resolve_started_run_id checks this first
        ev.tool_run_id = None
        ev.event_id = None
        ev.tool_name = tool_name
        ev.tool_args = {"query": "test"}
        ev.tool_class = "MockTool"
        ev.agent_key = "Researcher"
        ev.agent_id = None
        ev.run_attempts = run_attempts
        ev.delegations = delegations
        ev.agent = MagicMock()
        ev.agent.role = "Researcher"
        ev.from_task = MagicMock()
        ev.from_task.name = "research"
        ev.from_agent = MagicMock()
        ev.from_agent.role = "Researcher"
        return ev

    def test_tool_run_attempts_zero_captured(self):
        """run_attempts=0 (first attempt) is captured — not swallowed by falsy or."""
        lnr = self._listener_with_agent()
        ev = self._tool_ev_with_explicit_id(
            "run-0", "search_tool", run_attempts=0, delegations=None
        )

        lnr.on_tool_usage_started(MagicMock(), ev)

        with lnr._lock:
            span = lnr._tool_spans.get("run-0")

        assert span is not None, "tool span not created for run_id='run-0'"
        assert (
            "tool.run_attempts" in span.attributes
        ), f"tool.run_attempts missing (run_attempts=0); attrs: {dict(span.attributes)}"
        assert span.attributes["tool.run_attempts"] == 0

        lnr.shutdown()

    def test_tool_run_attempts_retry_captured(self):
        """run_attempts=2 (retry) is captured correctly."""
        lnr = self._listener_with_agent()
        ev = self._tool_ev_with_explicit_id(
            "run-2", "search_tool", run_attempts=2, delegations=None
        )

        lnr.on_tool_usage_started(MagicMock(), ev)

        with lnr._lock:
            span = lnr._tool_spans.get("run-2")

        assert span is not None
        assert span.attributes.get("tool.run_attempts") == 2

        lnr.shutdown()

    def test_tool_delegations_captured(self):
        """tool.delegations is captured when non-None."""
        lnr = self._listener_with_agent()
        ev = self._tool_ev_with_explicit_id(
            "run-del", "delegate_work_to_coworker", run_attempts=0, delegations=1
        )

        lnr.on_tool_usage_started(MagicMock(), ev)

        with lnr._lock:
            span = lnr._tool_spans.get("run-del")

        assert span is not None
        assert span.attributes.get("tool.delegations") == 1

        lnr.shutdown()


class TestCrewTestResult:
    """crew.quality_score and crew.test_model are set from CrewTestResultEvent."""

    def test_crew_test_result_sets_quality_score(self):
        """on_crew_test_result writes crew.quality_score and crew.test_model."""
        lnr = _make_rich_listener()
        crew_src = _crew_source("crew-test-1")
        lnr.on_crew_kickoff_started(crew_src, _crew_event("crew-test-1"))

        result_ev = MagicMock()
        result_ev.crew_id = None  # force fallback to source.id
        result_ev.quality = 8.5
        result_ev.execution_duration = 12.3
        result_ev.model = "gpt-4o"
        result_ev.crew_name = "TestCrew"

        lnr.on_crew_test_result(crew_src, result_ev)

        with lnr._lock:
            entry = lnr._crew_spans.get("crew-test-1")

        assert entry is not None
        span = entry.get("span") if isinstance(entry, dict) else entry
        assert span.attributes.get("crew.quality_score") == pytest.approx(8.5)
        assert span.attributes.get("crew.test_model") == "gpt-4o"

        lnr.shutdown()

    def test_crew_test_result_execution_duration(self):
        """on_crew_test_result also writes crew.test.execution_duration_s."""
        lnr = _make_rich_listener()
        crew_src = _crew_source("crew-test-2")
        lnr.on_crew_kickoff_started(crew_src, _crew_event("crew-test-2"))

        result_ev = MagicMock()
        result_ev.crew_id = None
        result_ev.quality = 7.0
        result_ev.execution_duration = 25.6
        result_ev.model = "claude-3-haiku"
        result_ev.crew_name = "TestCrew"

        lnr.on_crew_test_result(crew_src, result_ev)

        with lnr._lock:
            entry = lnr._crew_spans.get("crew-test-2")
        span = entry.get("span") if isinstance(entry, dict) else entry
        assert span.attributes.get("crew.test.execution_duration_s") == pytest.approx(
            25.6
        )

        lnr.shutdown()


class TestTaskEvaluationScore:
    """task.evaluation_score is written from TaskEvaluationEvent."""

    def _make_evaluation_event(
        self, score: float, task_id: str = "task-eval-1"
    ) -> MagicMock:
        ev = MagicMock()
        ev.task_id = task_id
        ev.evaluation_type = "score"
        ev.score = score
        ev.feedback = "Good work."
        ev.model = "gpt-4o-mini"
        ev.result = None  # no nested result object
        ev.criteria = None
        task_mock = MagicMock()
        task_mock.id = task_id
        ev.task = task_mock
        return ev

    def _inject_task_span(self, lnr: Any, task_id: str) -> MagicMock:
        """Directly inject a rich task span so on_task_evaluation can find it."""
        span = _make_rich_span(f"task-span-{task_id}", "rich-trace-1")
        with lnr._lock:
            lnr._task_spans[task_id] = span
        return span

    def test_task_evaluation_score_written(self):
        """on_task_evaluation writes task.evaluation_score to the open task span."""
        lnr = _make_rich_listener()
        crew_src = _crew_source("crew-eval")
        lnr.on_crew_kickoff_started(crew_src, _crew_event("crew-eval"))
        span = self._inject_task_span(lnr, "task-eval-1")

        eval_ev = self._make_evaluation_event(score=9.2, task_id="task-eval-1")
        # Prevent event.task_id from auto-resolving to a random MagicMock key
        eval_ev.task_id = "task-eval-1"

        task_src = MagicMock()
        task_src.id = "task-eval-1"
        lnr.on_task_evaluation(task_src, eval_ev)

        assert (
            "task.evaluation_score" in span.attributes
        ), f"task.evaluation_score missing; attrs: {dict(span.attributes)}"
        assert span.attributes["task.evaluation_score"] == pytest.approx(9.2)

        lnr.shutdown()

    def test_task_evaluation_feedback_written(self):
        """on_task_evaluation also writes task.evaluation_feedback."""
        lnr = _make_rich_listener()
        crew_src = _crew_source("crew-eval-fb")
        lnr.on_crew_kickoff_started(crew_src, _crew_event("crew-eval-fb"))
        span = self._inject_task_span(lnr, "task-eval-2")

        eval_ev = self._make_evaluation_event(score=7.5, task_id="task-eval-2")
        eval_ev.task_id = "task-eval-2"
        eval_ev.feedback = "Could be more concise."

        task_src = MagicMock()
        task_src.id = "task-eval-2"
        lnr.on_task_evaluation(task_src, eval_ev)

        assert (
            span.attributes.get("task.evaluation_feedback") == "Could be more concise."
        )

        lnr.shutdown()
