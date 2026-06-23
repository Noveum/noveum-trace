"""
Real-event contract & regression tests — LLM + TOOL + MEMORY families.

Like ``test_crewai_re_guardrail_task.py`` (whose harness this copies verbatim),
these construct the REAL ``crewai.events.types.*`` Pydantic events and assert
what actually lands on the span — catching upstream field drift that MagicMock
tests silently pass.

``# KNOWN BUG`` baselines current (buggy) behavior so the suite stays green and
the defect is documented (see CREWAI_TEST_PLAN.md §2). Flip the assertion when
the handler is fixed.

Findings exercised here (all empirically confirmed against crewAI 1.14.2a2):
  - §2B: ``truncate_str`` is a no-op → tool args/output never truncated.
  - ``memory.type`` always equals the event's ``type`` Literal
    (e.g. ``"memory_query_started"``), NOT the memory subsystem
    ("short_term"/"all"), because ``_resolve_memory_type`` reads ``event.type``
    first and that field is a frozen Literal. (NEW finding, not in §2A.)
  - §2D (REFUTED as a bug): the LLM ``temperature`` fallback to the ``source``
    object works — the event-field absence does not break capture.
"""

from __future__ import annotations

import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("crewai", reason="requires optional 'crewai' extra")

_src = Path(__file__).parents[3] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from crewai.events.types.llm_events import (  # noqa: E402
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallStartedEvent,
    LLMCallType,
    LLMStreamChunkEvent,
    LLMThinkingChunkEvent,
)
from crewai.events.types.memory_events import (  # noqa: E402
    MemoryQueryCompletedEvent,
    MemoryQueryStartedEvent,
    MemoryRetrievalCompletedEvent,
    MemoryRetrievalStartedEvent,
    MemorySaveCompletedEvent,
    MemorySaveStartedEvent,
)
from crewai.events.types.tool_usage_events import (  # noqa: E402
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)

from noveum_trace.core.span import SpanStatus  # noqa: E402
from noveum_trace.integrations.crewai.crewai_constants import (  # noqa: E402
    ATTR_STATUS_ERROR,
    ATTR_STATUS_SUCCESS,
    MAX_TEXT_LENGTH,
)

TRACE_ID = "trace-x"


# ---------------------------------------------------------------------------
# Self-contained harness (rich mocks so attribute assertions work)
# Copied verbatim from test_crewai_re_guardrail_task.py.
# ---------------------------------------------------------------------------


def _make_rich_span(span_id: str = "span-x", trace_id: str = TRACE_ID) -> MagicMock:
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


def _make_listener(**kwargs: Any) -> Any:
    from noveum_trace.integrations.crewai.crewai_listener import NoveumCrewAIListener

    trace = MagicMock()
    trace.trace_id = TRACE_ID
    trace.finish = MagicMock()

    def _create_span(name, parent_span_id=None, attributes=None, **kw):  # type: ignore[no-untyped-def]
        s = _make_rich_span()
        s.name = name
        if attributes:
            s.attributes.update(attributes)
        return s

    trace.create_span = MagicMock(side_effect=_create_span)
    client = MagicMock()
    client._lock = threading.RLock()
    client._active_traces = {TRACE_ID: trace}
    client.start_trace = MagicMock(return_value=trace)
    lnr = NoveumCrewAIListener(client, **kwargs)
    lnr._test_trace = trace
    return lnr


def _prime_agent(lnr: Any, agent_id: str = "agent-1") -> MagicMock:
    span = _make_rich_span(span_id=f"agent-{agent_id}")
    with lnr._lock:
        lnr._agent_spans[agent_id] = span
    return span


def _prime_crew(lnr: Any, crew_id: str = "crew-1") -> MagicMock:
    span = _make_rich_span(span_id=f"crew-{crew_id}")
    with lnr._lock:
        lnr._crew_spans[crew_id] = {
            "span": span,
            "trace": lnr._test_trace,
            "start_t": time.monotonic(),
        }
    return span


def _prime_task(lnr: Any, task_id: str = "t1") -> MagicMock:
    span = _make_rich_span(span_id=f"task-{task_id}")
    with lnr._lock:
        lnr._task_spans[task_id] = span
    return span


@pytest.fixture(autouse=True)
def _reset_patch_state() -> Any:
    """Keep the module-level token-patch state clean across tests."""
    import noveum_trace.integrations.crewai.crewai_listener as m

    def _restore() -> None:
        if getattr(m, "_patch_applied", False) and m._original_track_token_usage:
            try:
                from crewai.llms.base_llm import BaseLLM

                BaseLLM._track_token_usage_internal = m._original_track_token_usage
            except Exception:
                pass
        m._patch_applied = False
        m._original_track_token_usage = None
        for lnr in list(m._active_listeners):
            m._active_listeners.discard(lnr)

    _restore()
    yield
    _restore()


# ===========================================================================
# LLM
# ===========================================================================


class TestLLMRealEvents:
    def test_started_captures_model_prompt_messages(self) -> None:
        """1. LLMCallStartedEvent(messages=[sys,user], tools=[]) → identity captured."""
        lnr = _make_listener()
        _prime_agent(lnr, "agent-1")
        src = SimpleNamespace(role="Researcher")
        ev = LLMCallStartedEvent(
            call_id="c1",
            model="claude-3-5-sonnet",
            messages=[
                {"role": "system", "content": "be helpful"},
                {"role": "user", "content": "what is 2+2?"},
            ],
            tools=[],
            agent_id="agent-1",
        )
        lnr.on_llm_call_started(src, ev)

        with lnr._lock:
            entry = lnr._llm_call_spans.get("c1")
        assert entry is not None, "LLM span not opened"
        attrs = entry["span"].attributes
        assert attrs["llm.call_id"] == "c1"
        assert attrs["llm.model"] == "claude-3-5-sonnet"
        assert attrs["llm.provider"] == "anthropic"
        assert attrs["llm.system_prompt"] == "be helpful"
        # Full messages serialized to JSON.
        assert '"what is 2+2?"' in attrs["llm.input_messages"]
        assert attrs["agent.role"] == "Researcher"
        lnr.shutdown()

    def test_completed_response_tokens_cost(self) -> None:
        """2. LLMCallCompletedEvent(usage=...) → response, tokens, cost; span removed."""
        lnr = _make_listener()
        _prime_agent(lnr, "agent-1")
        st = LLMCallStartedEvent(
            call_id="c1", model="claude-3-5-sonnet", agent_id="agent-1"
        )
        lnr.on_llm_call_started(SimpleNamespace(role="R"), st)
        span = lnr._llm_call_spans["c1"]["span"]

        comp = LLMCallCompletedEvent(
            call_id="c1",
            response="The answer is 4",
            call_type=LLMCallType.LLM_CALL,
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
            model="claude-3-5-sonnet",
        )
        lnr.on_llm_call_completed(SimpleNamespace(), comp)

        assert "c1" not in lnr._llm_call_spans  # span removed
        attrs = span.attributes
        assert attrs["llm.response"] == "The answer is 4"
        # .name (not the full enum repr) is written.
        assert attrs["llm.call_type"] == "LLM_CALL"
        assert attrs["llm.input_tokens"] == 100
        assert attrs["llm.output_tokens"] == 50
        assert attrs["llm.total_tokens"] == 150
        assert attrs["llm.cost.total"] > 0
        assert attrs["llm.cost.currency"] == "USD"
        lnr.shutdown()

    def test_failed_sets_error_status(self) -> None:
        """3. LLMCallFailedEvent(error=...) → error attrs + ERROR status."""
        lnr = _make_listener()
        _prime_agent(lnr, "agent-1")
        st = LLMCallStartedEvent(call_id="c1", model="gpt-4o", agent_id="agent-1")
        lnr.on_llm_call_started(SimpleNamespace(role="R"), st)
        span = lnr._llm_call_spans["c1"]["span"]

        failed = LLMCallFailedEvent(call_id="c1", error="rate_limit exceeded")
        lnr.on_llm_call_failed(SimpleNamespace(), failed)

        assert "c1" not in lnr._llm_call_spans
        attrs = span.attributes
        assert "rate_limit" in str(attrs.get("error.message", ""))
        # Verify the exact status enum, not just that set_status was called --
        # any value other than the SpanStatus members silently no-ops in source.
        span.set_status.assert_called_once()
        assert span.set_status.call_args[0][0] == SpanStatus.ERROR
        lnr.shutdown()

    def test_stream_and_thinking_chunks_join_on_completion(self) -> None:
        """4. stream + thinking chunk buffering → joined onto the span at completion."""
        lnr = _make_listener(capture_streaming=True, capture_thinking=True)
        _prime_agent(lnr, "agent-1")
        st = LLMCallStartedEvent(call_id="c1", model="gpt-4o", agent_id="agent-1")
        lnr.on_llm_call_started(SimpleNamespace(role="R"), st)
        span = lnr._llm_call_spans["c1"]["span"]

        for piece in ("Hello", " ", "world"):
            lnr.on_llm_stream_chunk(
                SimpleNamespace(), LLMStreamChunkEvent(call_id="c1", chunk=piece)
            )
        for piece in ("Let", " me", " think"):
            lnr.on_llm_thinking_chunk(
                SimpleNamespace(), LLMThinkingChunkEvent(call_id="c1", chunk=piece)
            )

        comp = LLMCallCompletedEvent(
            call_id="c1", response="Hello world", call_type=LLMCallType.LLM_CALL
        )
        lnr.on_llm_call_completed(SimpleNamespace(), comp)

        attrs = span.attributes
        assert attrs["llm.streaming_response"] == "Hello world"
        assert attrs["llm.streaming"] is True
        # thinking buffer → ATTR_LLM_THINKING_TEXT == "llm.thinking"
        assert attrs["llm.thinking"] == "Let me think"
        lnr.shutdown()

    def test_temperature_fallback_from_source_refuted(self) -> None:
        """5. FALLBACK (§2D): event has no temperature field; read from source obj.

        REFUTED as a bug — the fallback works. LLMCallStartedEvent defines no
        ``temperature``/``finish_reason`` field, but the handler resolves
        ``temperature`` from the ``source`` (Agent/LLM) object, and the absent
        event field does not break message/identity capture.
        """
        lnr = _make_listener()
        _prime_agent(lnr, "agent-1")
        # temperature lives on source, NOT on the event.
        src = SimpleNamespace(role="R", temperature=0.3)
        ev = LLMCallStartedEvent(
            call_id="c1",
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            agent_id="agent-1",
        )
        # Sanity: the real event genuinely lacks these fields.
        assert "temperature" not in type(ev).model_fields
        assert "finish_reason" not in type(ev).model_fields

        lnr.on_llm_call_started(src, ev)
        attrs = lnr._llm_call_spans["c1"]["span"].attributes
        assert attrs["llm.temperature"] == 0.3  # read from source obj
        assert "hi" in attrs["llm.input_messages"]  # capture not broken by absence
        lnr.shutdown()


# ===========================================================================
# Tool
# ===========================================================================


class TestToolRealEvents:
    def test_started_then_finished_pairs_via_event_id(self) -> None:
        """6. ToolUsageStartedEvent → identity/input + run_attempts=0 preserved;
        ToolUsageFinishedEvent paired via started_event_id → closed with output."""
        lnr = _make_listener()
        _prime_agent(lnr, "agent-1")
        started = ToolUsageStartedEvent(
            tool_name="calculator",
            tool_args={"expression": "2+2"},
            tool_class="Calculator",
            agent_role="Mathematician",
            agent_id="agent-1",
            run_attempts=0,
        )
        lnr.on_tool_usage_started(SimpleNamespace(role="Mathematician"), started)

        run_id = started.event_id
        with lnr._lock:
            span = lnr._tool_spans.get(run_id)
        assert span is not None, "tool span not opened"
        attrs = span.attributes
        assert attrs["tool.name"] == "calculator"
        assert '"expression"' in attrs["tool.input"]
        # run_attempts=0 must be preserved (explicit None-check, not `or`).
        assert attrs["tool.run_attempts"] == 0

        finished = ToolUsageFinishedEvent(
            tool_name="calculator",
            tool_args={"expression": "2+2"},
            started_at=datetime.now(),
            finished_at=datetime.now(),
            output="4",
            started_event_id=run_id,  # equals started.event_id
        )
        lnr.on_tool_usage_finished(SimpleNamespace(), finished)

        assert run_id not in lnr._tool_spans  # span closed/removed
        assert span.attributes["tool.status"] == ATTR_STATUS_SUCCESS
        assert span.attributes["tool.output"] == "4"
        lnr.shutdown()

    def test_error_sets_error_status_no_output(self) -> None:
        """7. ToolUsageErrorEvent → error attrs, ERROR status, no tool.output."""
        lnr = _make_listener()
        _prime_agent(lnr, "agent-1")
        started = ToolUsageStartedEvent(
            tool_name="search",
            tool_args={"q": "x"},
            agent_id="agent-1",
        )
        lnr.on_tool_usage_started(SimpleNamespace(), started)
        run_id = started.event_id
        span = lnr._tool_spans[run_id]

        err = ToolUsageErrorEvent(
            tool_name="search",
            tool_args={"q": "x"},
            error=ValueError("connection refused"),
            started_event_id=run_id,
        )
        lnr.on_tool_usage_error(SimpleNamespace(), err)

        assert run_id not in lnr._tool_spans
        attrs = span.attributes
        assert attrs["tool.status"] == ATTR_STATUS_ERROR
        assert "connection refused" in str(attrs.get("error.message", ""))
        assert "tool.output" not in attrs
        # Verify the exact status enum, not just that set_status was called.
        span.set_status.assert_called_once()
        assert span.set_status.call_args[0][0] == SpanStatus.ERROR
        lnr.shutdown()

    def test_tool_args_string_not_truncated_known_bug(self) -> None:
        """8. KNOWN BUG (§2B): truncate_str is a no-op.

        Feed a >9000-char string ``tool_args`` (MAX_TEXT_LENGTH == 8192). The
        handler routes it through ``truncate_str(args, MAX_TEXT_LENGTH)``, which
        returns the input unchanged, so ``tool.input`` is NOT truncated. Use a
        plain *string* (not a dict) so length is preserved exactly through the
        ``isinstance(args, str)`` branch (a dict would go through json.dumps).
        """
        lnr = _make_listener()
        _prime_agent(lnr, "agent-1")
        big = "x" * 9000
        assert len(big) > MAX_TEXT_LENGTH
        started = ToolUsageStartedEvent(
            tool_name="bulk",
            tool_args=big,  # tool_args accepts dict | str
            agent_id="agent-1",
        )
        lnr.on_tool_usage_started(SimpleNamespace(), started)
        span = lnr._tool_spans[started.event_id]

        # KNOWN BUG (§2B): truncate_str is a no-op → input preserved at full length.
        assert len(span.attributes["tool.input"]) == 9000
        lnr.shutdown()


# ===========================================================================
# Memory
# ===========================================================================


class TestMemoryRealEvents:
    def test_query_lifecycle_count_preview(self) -> None:
        """9. MemoryQueryStartedEvent → MemoryQueryCompletedEvent(results=[...])
        → result_count/preview, status ok; paired via started_event_id."""
        lnr = _make_listener()
        _prime_agent(lnr, "agent-1")
        qs = MemoryQueryStartedEvent(
            query="find docs",
            limit=5,
            score_threshold=0.7,
            agent_role="Researcher",
            agent_id="agent-1",
        )
        lnr.on_memory_query_started(SimpleNamespace(), qs)

        op_id = qs.event_id
        with lnr._lock:
            span = lnr._memory_op_spans.get(op_id)
        assert span is not None, "memory query span not opened"
        attrs = span.attributes
        assert attrs["memory.operation"] == "query"
        assert attrs["memory.query"] == "find docs"
        assert attrs["memory.limit"] == 5
        assert attrs["memory.score_threshold"] == 0.7
        # KNOWN BUG (new, not in §2A): memory.type reflects the event's `type`
        # Literal, NOT the memory subsystem. _resolve_memory_type reads event.type
        # first and that field is a frozen Literal "memory_query_started", so it
        # always wins over source-class inference — the documented
        # "short_term"/"long_term"/"entity" subsystem value is never captured.
        assert attrs["memory.type"] == "memory_query_started"

        qc = MemoryQueryCompletedEvent(
            query="find docs",
            results=["doc-a", "doc-b", "doc-c"],
            limit=5,
            query_time_ms=12.0,
            started_event_id=op_id,  # pairs to the started event's event_id
        )
        lnr.on_memory_query_completed(SimpleNamespace(), qc)

        assert op_id not in lnr._memory_op_spans  # span removed
        attrs = span.attributes
        assert attrs["memory.status"] == ATTR_STATUS_SUCCESS
        assert attrs["memory.result_count"] == 3
        assert "doc-a" in attrs["memory.results_preview"]
        lnr.shutdown()

    def test_save_lifecycle(self) -> None:
        """10. MemorySaveStartedEvent → MemorySaveCompletedEvent → status ok."""
        lnr = _make_listener()
        _prime_agent(lnr, "agent-1")
        ss = MemorySaveStartedEvent(
            value="remember this fact",
            metadata={"source": "user"},
            agent_role="Researcher",
            agent_id="agent-1",
        )
        lnr.on_memory_save_started(SimpleNamespace(), ss)

        op_id = ss.event_id
        with lnr._lock:
            span = lnr._memory_op_spans.get(op_id)
        assert span is not None, "memory save span not opened"
        attrs = span.attributes
        assert attrs["memory.operation"] == "save"
        assert attrs["memory.value"] == "remember this fact"
        assert '"source"' in attrs["memory.metadata"]

        sc = MemorySaveCompletedEvent(
            value="remember this fact",
            save_time_ms=5.0,
            started_event_id=op_id,
        )
        lnr.on_memory_save_completed(SimpleNamespace(), sc)

        assert op_id not in lnr._memory_op_spans
        assert span.attributes["memory.status"] == ATTR_STATUS_SUCCESS
        lnr.shutdown()

    def test_retrieval_completed_reads_memory_content(self) -> None:
        """11. MemoryRetrievalCompletedEvent uses `memory_content` (the only
        payload field) → captured as memory.content_preview (string path)."""
        lnr = _make_listener()
        _prime_agent(lnr, "agent-1")
        rs = MemoryRetrievalStartedEvent(task_id="t1", agent_id="agent-1")
        lnr.on_memory_retrieval_started(SimpleNamespace(), rs)
        op_id = rs.event_id
        span = lnr._memory_op_spans[op_id]
        assert span.attributes["memory.operation"] == "retrieval"

        rc = MemoryRetrievalCompletedEvent(
            task_id="t1",
            memory_content="relevant context from past tasks",
            retrieval_time_ms=8.0,
            started_event_id=op_id,
        )
        lnr.on_memory_retrieval_completed(SimpleNamespace(), rc)

        assert op_id not in lnr._memory_op_spans
        attrs = span.attributes
        assert attrs["memory.status"] == ATTR_STATUS_SUCCESS
        # memory_content is a str → written verbatim to memory.content_preview.
        assert attrs["memory.content_preview"] == "relevant context from past tasks"
        lnr.shutdown()

    def test_capture_memory_false_no_op(self) -> None:
        """12. capture_memory=False → memory handlers no-op (no span created)."""
        lnr = _make_listener(capture_memory=False)
        _prime_agent(lnr, "agent-1")
        qs = MemoryQueryStartedEvent(query="x", limit=1, agent_id="agent-1")
        lnr.on_memory_query_started(SimpleNamespace(), qs)
        ss = MemorySaveStartedEvent(value="y", agent_id="agent-1")
        lnr.on_memory_save_started(SimpleNamespace(), ss)
        rs = MemoryRetrievalStartedEvent(task_id="t1", agent_id="agent-1")
        lnr.on_memory_retrieval_started(SimpleNamespace(), rs)

        with lnr._lock:
            assert lnr._memory_op_spans == {}  # nothing opened
        lnr.shutdown()
