"""
Real-event contract & regression tests — knowledge + mcp + reasoning families.

Unlike ``test_crewai_integration.py`` (which feeds handlers ``MagicMock``
events), these construct the REAL ``crewai.events.types.*`` Pydantic events and
assert what actually lands on the span. They catch upstream field drift and
expose handler/event field mismatches that MagicMock tests silently pass.

``# KNOWN BUG`` baselines current (buggy) behavior so the suite stays green and
the defect is documented (see CREWAI_TEST_PLAN.md §2). Flip the assertion when
the handler is fixed. ``# REFUTED`` marks a §2 finding that did NOT reproduce
against the installed CrewAI (handler reads the right field).

Authoritative field names come from the INSTALLED CrewAI (1.14.2a2). Verified
field facts that shape these tests:
  * Knowledge started↔completed pair correctly (``_resolve_op_id`` reads
    ``event_id``); MCP and reasoning do NOT (``_resolve_mcp_key`` /
    ``_resolve_reasoning_id`` omit ``event_id``) → a separate pairing bug.
  * ``KnowledgeQueryStartedEvent.task_prompt`` (handler reads query/...).
  * ``KnowledgeRetrievalCompletedEvent.retrieved_knowledge`` is a *string*.
  * ``KnowledgeQueryCompletedEvent`` has NO results-like field.
  * ``KnowledgeQueryFailedEvent.error`` is a *string*.
  * ``MCPConnectionCompletedEvent`` has NO tools field.
  * ``MCPConfigFetchFailedEvent`` has ``slug/error/error_type`` — no ``config``.
  * ``AgentReasoningCompletedEvent.ready`` (handler reads ``is_ready``).
  * ``PlanReplanTriggeredEvent.completed_steps_preserved`` (int) +
    ``replan_reason`` (handler reads ``completed_steps`` and ``reason``/``message``).
  * ``GoalAchievedEarlyEvent.step_number`` (handler reads ``step``/``current_step``).
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("crewai", reason="requires optional 'crewai' extra")

_src = Path(__file__).parents[3] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from crewai.events.types.knowledge_events import (  # noqa: E402
    KnowledgeQueryCompletedEvent,
    KnowledgeQueryFailedEvent,
    KnowledgeQueryStartedEvent,
    KnowledgeRetrievalCompletedEvent,
    KnowledgeRetrievalStartedEvent,
    KnowledgeSearchQueryFailedEvent,
)
from crewai.events.types.mcp_events import (  # noqa: E402
    MCPConfigFetchFailedEvent,
    MCPConnectionCompletedEvent,
    MCPConnectionFailedEvent,
    MCPConnectionStartedEvent,
    MCPToolExecutionCompletedEvent,
    MCPToolExecutionFailedEvent,
    MCPToolExecutionStartedEvent,
)
from crewai.events.types.observation_events import (  # noqa: E402
    GoalAchievedEarlyEvent,
    PlanRefinementEvent,
    PlanReplanTriggeredEvent,
    StepObservationCompletedEvent,
    StepObservationFailedEvent,
    StepObservationStartedEvent,
)
from crewai.events.types.reasoning_events import (  # noqa: E402
    AgentReasoningCompletedEvent,
    AgentReasoningFailedEvent,
    AgentReasoningStartedEvent,
)

from noveum_trace.integrations.crewai._handlers_mcp import (  # noqa: E402
    _redact_config,
)
from noveum_trace.integrations.crewai.crewai_constants import (  # noqa: E402
    ATTR_AGENT_ROLE,
    ATTR_AGENT_STEP,
    ATTR_MEMORY_OP_ID,
    ATTR_MEMORY_OPERATION,
    ATTR_MEMORY_QUERY,
    ATTR_MEMORY_RESULT_COUNT,
    ATTR_MEMORY_STATUS,
    ATTR_MEMORY_TYPE,
    ATTR_STATUS_ERROR,
    ATTR_STATUS_SUCCESS,
)

TRACE_ID = "trace-x"


# ---------------------------------------------------------------------------
# Self-contained harness (rich mocks so attribute assertions work)
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
# Knowledge — pairing WORKS (real events pair via started_event_id==event_id)
# ===========================================================================


class TestKnowledgeRealEvents:
    def test_retrieval_started_opens_span(self) -> None:
        lnr = _make_listener()
        _prime_agent(lnr, "agent-1")
        ev = KnowledgeRetrievalStartedEvent(agent_id="agent-1", agent_role="Researcher")
        lnr.on_knowledge_retrieval_started(None, ev)

        with lnr._lock:
            span = lnr._memory_op_spans.get(str(ev.event_id))
        assert span is not None, "knowledge retrieval span not opened"
        attrs = span.attributes
        assert attrs[ATTR_MEMORY_TYPE] == "knowledge"
        assert attrs[ATTR_MEMORY_OPERATION] == "retrieval"
        assert attrs[ATTR_MEMORY_OP_ID] == str(ev.event_id)
        assert attrs.get(ATTR_AGENT_ROLE) == "Researcher"
        # KNOWN BUG (§2 #8): KnowledgeRetrievalStartedEvent has no sources field.
        assert "knowledge.sources" not in attrs
        lnr.shutdown()

    def test_query_started_loses_task_prompt_known_bug(self) -> None:
        lnr = _make_listener()
        _prime_agent(lnr, "agent-1")
        ev = KnowledgeQueryStartedEvent(
            task_prompt="What are the latest AI trends?",
            agent_id="agent-1",
            agent_role="Researcher",
        )
        lnr.on_knowledge_query_started(None, ev)

        with lnr._lock:
            span = lnr._memory_op_spans.get(str(ev.event_id))
        assert span is not None, "knowledge query span not opened"
        attrs = span.attributes
        # Witness: handler body ran (operation tag written).
        assert attrs[ATTR_MEMORY_OPERATION] == "query"

        # KNOWN BUG (§2 #6): real field is ``task_prompt``; handler reads
        # query/search_query/text/input → memory.query never set.
        assert ATTR_MEMORY_QUERY not in attrs
        # KNOWN BUG (§2 #8): no top_k/limit/sources on the real event.
        assert "knowledge.top_k" not in attrs
        assert "knowledge.sources" not in attrs
        # The query text IS recoverable — it lives on ``task_prompt``.
        assert ev.task_prompt == "What are the latest AI trends?"
        lnr.shutdown()

    def test_retrieval_completed_loses_retrieved_knowledge_known_bug(self) -> None:
        lnr = _make_listener()
        _prime_agent(lnr, "agent-1")
        started = KnowledgeRetrievalStartedEvent(agent_id="agent-1")
        lnr.on_knowledge_retrieval_started(None, started)
        span = lnr._memory_op_spans[str(started.event_id)]

        completed = KnowledgeRetrievalCompletedEvent(
            query="latest AI trends",
            retrieved_knowledge="chunk-a content. chunk-b content.",
            agent_id="agent-1",
            started_event_id=started.event_id,
        )
        lnr.on_knowledge_retrieval_completed(None, completed)

        # Pairing works for knowledge → span closed (removed from map).
        assert str(started.event_id) not in lnr._memory_op_spans
        attrs = span.attributes
        # Witness: handler closed the span as success with a duration.
        assert attrs[ATTR_MEMORY_STATUS] == ATTR_STATUS_SUCCESS
        assert "memory.duration_ms" in attrs

        # KNOWN BUG (§2 #7): real field is ``retrieved_knowledge`` (a str);
        # handler reads results/content/chunks → enrichment never runs.
        assert ATTR_MEMORY_RESULT_COUNT not in attrs
        assert "knowledge.content_preview" not in attrs
        assert "knowledge.results_preview" not in attrs
        # The retrieved text IS recoverable on the real event.
        assert completed.retrieved_knowledge == "chunk-a content. chunk-b content."
        lnr.shutdown()

    def test_query_completed_has_no_results_field_known_bug(self) -> None:
        lnr = _make_listener()
        _prime_agent(lnr, "agent-1")
        started = KnowledgeQueryStartedEvent(task_prompt="p", agent_id="agent-1")
        lnr.on_knowledge_query_started(None, started)
        span = lnr._memory_op_spans[str(started.event_id)]

        completed = KnowledgeQueryCompletedEvent(
            query="latest AI trends",
            agent_id="agent-1",
            started_event_id=started.event_id,
        )
        lnr.on_knowledge_query_completed(None, completed)

        assert str(started.event_id) not in lnr._memory_op_spans
        attrs = span.attributes
        assert attrs[ATTR_MEMORY_STATUS] == ATTR_STATUS_SUCCESS

        # KNOWN BUG (§2 #7): KnowledgeQueryCompletedEvent carries no results /
        # chunks / documents field → result_count + previews never written.
        assert ATTR_MEMORY_RESULT_COUNT not in attrs
        assert "knowledge.results_preview" not in attrs
        assert "knowledge.content_preview" not in attrs
        lnr.shutdown()

    def test_query_failed_error_type_is_str_known_bug(self) -> None:
        lnr = _make_listener()
        _prime_agent(lnr, "agent-1")
        started = KnowledgeQueryStartedEvent(task_prompt="p", agent_id="agent-1")
        lnr.on_knowledge_query_started(None, started)
        span = lnr._memory_op_spans[str(started.event_id)]

        failed = KnowledgeQueryFailedEvent(
            error="vector store unavailable",
            agent_id="agent-1",
            started_event_id=started.event_id,
        )
        lnr.on_knowledge_query_failed(None, failed)

        assert str(started.event_id) not in lnr._memory_op_spans
        attrs = span.attributes
        assert attrs[ATTR_MEMORY_STATUS] == ATTR_STATUS_ERROR
        assert "vector store unavailable" in str(attrs.get("error.message", ""))

        # KNOWN BUG (§2 #9): real ``error`` is a *string*, but the finish path
        # treats it as an exception → error.type ends up the str class name and
        # no real traceback is recorded.
        assert attrs.get("error.type") == "str"
        assert "error.stacktrace" not in attrs
        # Confirm the real event field is indeed a plain string.
        assert isinstance(failed.error, str)
        lnr.shutdown()

    def test_search_query_failed_annotates_open_span(self) -> None:
        lnr = _make_listener()
        _prime_agent(lnr, "agent-1")
        started = KnowledgeQueryStartedEvent(task_prompt="p", agent_id="agent-1")
        lnr.on_knowledge_query_started(None, started)
        span = lnr._memory_op_spans[str(started.event_id)]

        ev = KnowledgeSearchQueryFailedEvent(
            query="latest AI trends",
            error="embedding lookup timed out",
            agent_id="agent-1",
            started_event_id=started.event_id,
        )
        lnr.on_knowledge_search_query_failed(None, ev)

        attrs = span.attributes
        # Search query (a real field) IS captured.
        assert attrs.get("knowledge.search_query") == "latest AI trends"
        assert "embedding lookup timed out" in str(
            attrs.get("knowledge.search_error", "")
        )
        # KNOWN BUG (§2 #9): error is a string → search_error.type == 'str'.
        assert attrs.get("knowledge.search_error.type") == "str"
        lnr.shutdown()


# ===========================================================================
# MCP
# ===========================================================================


class TestMCPRealEvents:
    def test_connection_started_captures_server_details(self) -> None:
        lnr = _make_listener()
        _prime_agent(lnr, "agent-1")
        ev = MCPConnectionStartedEvent(
            server_name="weather-mcp",
            server_url="https://mcp.example.com/sse",
            transport_type="sse",
            agent_id="agent-1",
            agent_role="Researcher",
        )
        lnr.on_mcp_connection_started(None, ev)

        # Real started event keys off id(event); fetch the only open span.
        with lnr._lock:
            entries = list(lnr._mcp_spans.values())
        assert len(entries) == 1, "MCP connection span not opened"
        attrs = entries[0]["span"].attributes
        assert attrs["mcp.operation"] == "connection"
        assert attrs.get("mcp.server_name") == "weather-mcp"
        # ``transport`` works: handler reads ``transport_type`` as a fallback.
        assert attrs.get("mcp.transport") == "sse"
        assert attrs.get(ATTR_AGENT_ROLE) == "Researcher"
        # KNOWN BUG (MCP field drift): real URL field is ``server_url``; handler
        # reads url/endpoint → mcp.url never set.
        assert "mcp.url" not in attrs
        # The URL IS recoverable on the real event via ``server_url``.
        assert ev.server_url == "https://mcp.example.com/sse"
        # Real connection-started event has no config → mcp.config absent.
        assert "mcp.config" not in attrs
        lnr.shutdown()

    def test_connection_started_completed_pairing_broken_known_bug(self) -> None:
        # Real started↔completed pairing baseline: started keys off id(event),
        # completed keys off started_event_id (==started.event_id). They never
        # match because _resolve_mcp_key omits ``event_id``.
        lnr = _make_listener()
        _prime_agent(lnr, "agent-1")
        started = MCPConnectionStartedEvent(server_name="srv", agent_id="agent-1")
        lnr.on_mcp_connection_started(None, started)
        with lnr._lock:
            keys_after_start = set(lnr._mcp_spans.keys())
        assert len(keys_after_start) == 1

        completed = MCPConnectionCompletedEvent(
            server_name="srv",
            agent_id="agent-1",
            started_event_id=started.event_id,
        )
        lnr.on_mcp_connection_completed(None, completed)

        # KNOWN BUG (MCP pairing): completed never finds the started span, so it
        # stays open (leaked) and is never marked ok.
        with lnr._lock:
            keys_after_complete = set(lnr._mcp_spans.keys())
        assert keys_after_complete == keys_after_start, "span should leak (unpaired)"
        leaked = lnr._mcp_spans[next(iter(keys_after_complete))]["span"]
        assert "mcp.status" not in leaked.attributes
        # The link IS present on the real event — recoverable if the key
        # resolver read ``event_id``.
        assert str(completed.started_event_id) == str(started.event_id)
        lnr.shutdown()

    def test_connection_completed_no_tools_field_known_bug(self) -> None:
        # Prime the span directly under a known key so the completed handler
        # body actually runs (isolating the field bug from the pairing bug).
        lnr = _make_listener()
        span = _make_rich_span()
        with lnr._lock:
            lnr._mcp_spans["mcp-1"] = {"span": span, "start_t": time.monotonic()}

        completed = MCPConnectionCompletedEvent(
            server_name="srv", agent_id="agent-1", started_event_id="mcp-1"
        )
        lnr.on_mcp_connection_completed(None, completed)

        attrs = span.attributes
        # Witness: handler ran and closed the span.
        assert "mcp-1" not in lnr._mcp_spans
        assert attrs["mcp.status"] == ATTR_STATUS_SUCCESS
        assert "mcp.duration_ms" in attrs

        # KNOWN BUG (§2 #18): MCPConnectionCompletedEvent has no tools /
        # available_tools field → these are never set.
        assert "mcp.available_tools" not in attrs
        assert "mcp.tool_count" not in attrs
        assert not hasattr(completed, "tools")
        assert not hasattr(completed, "available_tools")
        lnr.shutdown()

    def test_connection_failed_error_type(self) -> None:
        lnr = _make_listener()
        span = _make_rich_span()
        with lnr._lock:
            lnr._mcp_spans["mcp-c"] = {"span": span, "start_t": time.monotonic()}

        failed = MCPConnectionFailedEvent(
            server_name="srv",
            error="connection refused",
            error_type="connection_refused",
            started_event_id="mcp-c",
        )
        lnr.on_mcp_connection_failed(None, failed)

        attrs = span.attributes
        assert "mcp-c" not in lnr._mcp_spans
        assert attrs["mcp.status"] == ATTR_STATUS_ERROR
        # error_type IS a real field and IS captured.
        assert attrs.get("mcp.error_type") == "connection_refused"
        assert "connection refused" in str(attrs.get("error.message", ""))
        # KNOWN BUG (§2 #9-style): error is a string → error.type == 'str'.
        assert attrs.get("error.type") == "str"
        lnr.shutdown()

    def test_tool_execution_started_captures_args(self) -> None:
        lnr = _make_listener()
        _prime_agent(lnr, "agent-1")
        ev = MCPToolExecutionStartedEvent(
            server_name="weather-mcp",
            tool_name="get_forecast",
            tool_args={"city": "Paris"},
            agent_id="agent-1",
            agent_role="Researcher",
        )
        lnr.on_mcp_tool_execution_started(None, ev)

        with lnr._lock:
            entries = list(lnr._mcp_spans.values())
        assert len(entries) == 1, "MCP tool span not opened"
        attrs = entries[0]["span"].attributes
        assert attrs["mcp.operation"] == "tool_call"
        assert attrs.get("mcp.server_name") == "weather-mcp"
        assert attrs.get("mcp.tool_name") == "get_forecast"
        # KNOWN BUG (MCP field drift): real args field is ``tool_args``; handler
        # reads arguments/args/input/params → mcp.arguments never set.
        assert "mcp.arguments" not in attrs
        # The args ARE recoverable on the real event via ``tool_args``.
        assert ev.tool_args == {"city": "Paris"}
        lnr.shutdown()

    def test_tool_execution_completed_happy_path(self) -> None:
        # Prime directly to bypass the pairing bug and exercise the result path.
        lnr = _make_listener()
        span = _make_rich_span()
        with lnr._lock:
            lnr._mcp_spans["tool-1"] = {"span": span, "start_t": time.monotonic()}

        completed = MCPToolExecutionCompletedEvent(
            server_name="weather-mcp",
            tool_name="get_forecast",
            result="Sunny, 22C",
            started_event_id="tool-1",
        )
        lnr.on_mcp_tool_execution_completed(None, completed)

        attrs = span.attributes
        assert "tool-1" not in lnr._mcp_spans
        assert attrs["mcp.status"] == ATTR_STATUS_SUCCESS
        # result IS a real field → captured as mcp.result.
        assert "Sunny, 22C" in str(attrs.get("mcp.result", ""))
        assert "mcp.duration_ms" in attrs
        lnr.shutdown()

    def test_tool_execution_started_completed_pairing_broken_known_bug(self) -> None:
        lnr = _make_listener()
        _prime_agent(lnr, "agent-1")
        started = MCPToolExecutionStartedEvent(
            server_name="srv", tool_name="t", agent_id="agent-1"
        )
        lnr.on_mcp_tool_execution_started(None, started)
        with lnr._lock:
            keys_after_start = set(lnr._mcp_spans.keys())

        completed = MCPToolExecutionCompletedEvent(
            server_name="srv",
            tool_name="t",
            result="ok",
            started_event_id=started.event_id,
        )
        lnr.on_mcp_tool_execution_completed(None, completed)

        # KNOWN BUG (MCP pairing): tool span leaks (started key != completed key).
        with lnr._lock:
            keys_after_complete = set(lnr._mcp_spans.keys())
        assert keys_after_complete == keys_after_start
        leaked = lnr._mcp_spans[next(iter(keys_after_complete))]["span"]
        assert "mcp.status" not in leaked.attributes
        lnr.shutdown()

    def test_tool_execution_failed_error_type(self) -> None:
        lnr = _make_listener()
        span = _make_rich_span()
        with lnr._lock:
            lnr._mcp_spans["tool-f"] = {"span": span, "start_t": time.monotonic()}

        failed = MCPToolExecutionFailedEvent(
            server_name="srv",
            tool_name="t",
            error="tool not found",
            error_type="tool_not_found",
            started_event_id="tool-f",
        )
        lnr.on_mcp_tool_execution_failed(None, failed)

        attrs = span.attributes
        assert "tool-f" not in lnr._mcp_spans
        assert attrs["mcp.status"] == ATTR_STATUS_ERROR
        assert attrs.get("mcp.error_type") == "tool_not_found"
        lnr.shutdown()

    def test_config_fetch_failed_annotates_agent_span_no_config_known_bug(self) -> None:
        lnr = _make_listener()
        agent_span = _prime_agent(lnr, "agent-1")
        ev = MCPConfigFetchFailedEvent(
            slug="weather-mcp",
            error="registry unreachable",
            error_type="NetworkError",
            agent_id="agent-1",
        )
        lnr.on_mcp_config_fetch_failed(None, ev)

        attrs = agent_span.attributes
        assert attrs.get("mcp.config_fetch_failed") is True
        assert "registry unreachable" in str(attrs.get("mcp.config_fetch_error", ""))
        # KNOWN BUG (§2 #19): MCPConfigFetchFailedEvent has slug/error/error_type
        # but NO ``config`` → config snapshot never written, and server name is
        # read from server_name/server (absent), not ``slug`` → also lost.
        assert "mcp.config_fetch_error.config" not in attrs
        assert "mcp.config_fetch_error.server" not in attrs
        # error is a string → error.type == 'str'.
        assert attrs.get("mcp.config_fetch_error.type") == "str"
        # The server identity IS recoverable on the real event via ``slug``.
        assert ev.slug == "weather-mcp"
        lnr.shutdown()


class TestMCPConfigRedaction:
    """``_redact_config`` is the only credential-safety logic in the MCP path.

    Note: no real MCP event in this CrewAI version carries a ``config`` field,
    so redaction is exercised as a pure function (it cannot be driven through a
    real event — see §2 #19).
    """

    def test_redacts_sensitive_dict_keys(self) -> None:
        cfg = {
            "url": "https://mcp.example.com",
            "api_key": "sk-secret-123",
            "Authorization": "Bearer tok-456",
            "password": "hunter2",
            "client_secret": "csx",
            "transport": "sse",
        }
        out = _redact_config(cfg)
        assert out["api_key"] == "<redacted>"
        assert out["Authorization"] == "<redacted>"
        assert out["password"] == "<redacted>"
        assert out["client_secret"] == "<redacted>"
        # Non-sensitive keys pass through unchanged.
        assert out["url"] == "https://mcp.example.com"
        assert out["transport"] == "sse"

    def test_redacts_nested_and_header_values(self) -> None:
        cfg = {
            "headers": {"Authorization": "Bearer xyz", "Accept": "application/json"},
            "env": {"OPENAI_API_KEY": "sk-zzz", "DEBUG": "1"},
        }
        out = _redact_config(cfg)
        assert out["headers"]["Authorization"] == "<redacted>"
        assert out["headers"]["Accept"] == "application/json"
        assert out["env"]["OPENAI_API_KEY"] == "<redacted>"
        assert out["env"]["DEBUG"] == "1"

    def test_redacts_cli_secret_flag_in_args(self) -> None:
        cfg = {"command": "server", "args": ["--api-key", "sk-secret", "--port", "80"]}
        out = _redact_config(cfg)
        # The token following a secret flag is redacted; the port is not.
        assert "<redacted>" in out["args"]
        assert "sk-secret" not in out["args"]
        assert "80" in out["args"]


# ===========================================================================
# Reasoning
# ===========================================================================


class TestReasoningRealEvents:
    def test_reasoning_started_opens_span(self) -> None:
        lnr = _make_listener()
        _prime_agent(lnr, "agent-1")
        ev = AgentReasoningStartedEvent(
            task_id="t1", agent_role="Researcher", attempt=2, agent_id="agent-1"
        )
        lnr.on_agent_reasoning_started(None, ev)

        with lnr._lock:
            entries = list(lnr._reasoning_spans.values())
        assert len(entries) == 1, "reasoning span not opened"
        attrs = entries[0]["span"].attributes
        assert attrs.get(ATTR_AGENT_ROLE) == "Researcher"
        assert attrs.get("task.id") == "t1"
        assert attrs.get("reasoning.attempt") == 2
        # REFUTED-adjacent: reasoning_started reads ``is_ready`` which the started
        # event does not define → never set (no started-event is_ready exists).
        assert "reasoning.is_ready" not in attrs
        lnr.shutdown()

    def test_reasoning_completed_loses_ready_flag_known_bug(self) -> None:
        # Prime directly under a known key so the completed handler runs (the
        # natural started↔completed pairing is broken — see dedicated test).
        lnr = _make_listener()
        span = _make_rich_span()
        with lnr._lock:
            lnr._reasoning_spans["rsn-1"] = {
                "span": span,
                "start_t": time.monotonic(),
                "agent_id": "agent-1",
            }

        completed = AgentReasoningCompletedEvent(
            task_id="t1",
            agent_role="Researcher",
            plan="Final plan: do X then Y",
            ready=True,
            started_event_id="rsn-1",
        )
        lnr.on_agent_reasoning_completed(None, completed)

        attrs = span.attributes
        # Witness: handler ran, closed the span, captured the plan.
        assert "rsn-1" not in lnr._reasoning_spans
        assert attrs["reasoning.status"] == ATTR_STATUS_SUCCESS
        assert attrs.get("reasoning.final_plan") == "Final plan: do X then Y"
        assert "reasoning.duration_ms" in attrs

        # KNOWN BUG (§2 #20): real field is ``ready``; handler reads ``is_ready``
        # → reasoning.is_ready never set.
        assert "reasoning.is_ready" not in attrs
        # The flag IS recoverable on the real event.
        assert completed.ready is True
        lnr.shutdown()

    def test_reasoning_started_completed_pairing_broken_known_bug(self) -> None:
        lnr = _make_listener()
        _prime_agent(lnr, "agent-1")
        started = AgentReasoningStartedEvent(
            task_id="t1", agent_role="R", agent_id="agent-1"
        )
        lnr.on_agent_reasoning_started(None, started)
        with lnr._lock:
            keys_after_start = set(lnr._reasoning_spans.keys())
        assert len(keys_after_start) == 1

        completed = AgentReasoningCompletedEvent(
            task_id="t1",
            agent_role="R",
            plan="done",
            ready=True,
            agent_id="agent-1",
            started_event_id=started.event_id,
        )
        lnr.on_agent_reasoning_completed(None, completed)

        # KNOWN BUG (reasoning pairing): _resolve_reasoning_id omits event_id, so
        # started keys off id(event) and completed off started_event_id → the
        # span never pairs and leaks open.
        with lnr._lock:
            keys_after_complete = set(lnr._reasoning_spans.keys())
        assert keys_after_complete == keys_after_start, "span should leak (unpaired)"
        leaked = lnr._reasoning_spans[next(iter(keys_after_complete))]["span"]
        assert "reasoning.status" not in leaked.attributes
        lnr.shutdown()

    def test_reasoning_failed_error_status(self) -> None:
        lnr = _make_listener()
        span = _make_rich_span()
        with lnr._lock:
            lnr._reasoning_spans["rsn-f"] = {
                "span": span,
                "start_t": time.monotonic(),
                "agent_id": "agent-1",
            }

        failed = AgentReasoningFailedEvent(
            task_id="t1",
            agent_role="R",
            error="planner crashed",
            started_event_id="rsn-f",
        )
        lnr.on_agent_reasoning_failed(None, failed)

        attrs = span.attributes
        assert "rsn-f" not in lnr._reasoning_spans
        assert attrs["reasoning.status"] == ATTR_STATUS_ERROR
        assert "planner crashed" in str(attrs.get("error.message", ""))
        lnr.shutdown()

    def test_step_observation_started_captures_step_number(self) -> None:
        lnr = _make_listener()
        _prime_agent(lnr, "agent-1")
        ev = StepObservationStartedEvent(
            agent_role="Researcher",
            step_number=1,
            step_description="search the web",
            agent_id="agent-1",
        )
        lnr.on_step_observation_started(None, ev)

        with lnr._lock:
            entries = list(lnr._observation_spans.values())
        assert len(entries) == 1, "observation span not opened"
        attrs = entries[0]["span"].attributes
        # step_number is a real field and IS captured.
        assert attrs.get("step.number") == 1
        # KNOWN BUG (observation field drift): real field is ``step_description``;
        # handler reads description/step_name → step.description never set.
        assert "step.description" not in attrs
        # The description IS recoverable on the real event.
        assert ev.step_description == "search the web"
        lnr.shutdown()

    def test_step_observation_completed_happy_path(self) -> None:
        # Prime directly (pairing for observations has the same gap).
        lnr = _make_listener()
        span = _make_rich_span()
        with lnr._lock:
            lnr._observation_spans["obs-1"] = {
                "span": span,
                "start_t": time.monotonic(),
                "agent_id": "agent-1",
            }

        completed = StepObservationCompletedEvent(
            agent_role="Researcher",
            step_number=1,
            suggested_refinements=["narrow the query", "add a date filter"],
            started_event_id="obs-1",
        )
        lnr.on_step_observation_completed(None, completed)

        attrs = span.attributes
        assert "obs-1" not in lnr._observation_spans
        assert attrs["step.status"] == ATTR_STATUS_SUCCESS
        # suggested_refinements is a real field → captured as JSON.
        assert "narrow the query" in str(attrs.get("step.suggested_refinements", ""))
        assert "step.duration_ms" in attrs
        lnr.shutdown()

    def test_goal_achieved_early_loses_step_number_known_bug(self) -> None:
        lnr = _make_listener()
        agent_span = _prime_agent(lnr, "agent-1")
        ev = GoalAchievedEarlyEvent(
            agent_role="Researcher",
            step_number=2,
            steps_remaining=5,
            steps_completed=2,
            agent_id="agent-1",
        )
        # No open reasoning span → annotation falls back to the agent span.
        lnr.on_goal_achieved_early(None, ev)

        attrs = agent_span.attributes
        assert attrs.get("reasoning.goal_achieved_early") is True
        # steps_remaining IS a real field and IS captured.
        assert attrs.get("reasoning.steps_remaining") == 5

        # KNOWN BUG (§2 #22): real field is ``step_number``; handler reads
        # ``step``/``current_step`` → agent.step never set.
        assert ATTR_AGENT_STEP not in attrs
        # ``steps_completed`` is real but the handler never reads it.
        assert ev.step_number == 2 and ev.steps_completed == 2
        lnr.shutdown()

    def test_replan_triggered_loses_completed_and_reason_known_bug(self) -> None:
        lnr = _make_listener()
        agent_span = _prime_agent(lnr, "agent-1")
        ev = PlanReplanTriggeredEvent(
            agent_role="Researcher",
            step_number=1,
            replan_reason="tool kept failing",
            replan_count=2,
            completed_steps_preserved=3,
            agent_id="agent-1",
        )
        lnr.on_plan_replan_triggered(None, ev)

        attrs = agent_span.attributes
        assert attrs.get("reasoning.replan_triggered") is True
        # replan_count IS a real field and IS captured.
        assert attrs.get("reasoning.replan_count") == 2

        # KNOWN BUG (§2 #21): real field is ``completed_steps_preserved`` (int);
        # handler reads ``completed_steps``/``steps_done`` → never set.
        assert "reasoning.completed_steps" not in attrs
        # BONUS BUG: real field is ``replan_reason``; handler reads
        # ``reason``/``message`` → replan_reason dropped too.
        assert "reasoning.replan_reason" not in attrs
        # Both values ARE recoverable on the real event.
        assert ev.completed_steps_preserved == 3
        assert ev.replan_reason == "tool kept failing"
        lnr.shutdown()

    def test_plan_refinement_annotates_open_span_refuted(self) -> None:
        # REFUTED (vicinity of §2): PlanRefinementEvent.refinements (list[str])
        # and refined_step_count are real fields the handler reads correctly.
        lnr = _make_listener()
        agent_span = _prime_agent(lnr, "agent-1")
        ev = PlanRefinementEvent(
            agent_role="Researcher",
            step_number=1,
            refinements=["step a refined", "step b refined"],
            refined_step_count=2,
            agent_id="agent-1",
        )
        lnr.on_plan_refinement(None, ev)

        attrs = agent_span.attributes
        assert "step a refined" in str(attrs.get("reasoning.refined_steps", ""))
        assert attrs.get("reasoning.refined_step_count") == 2
        lnr.shutdown()

    def test_capture_reasoning_false_gates_all_handlers(self) -> None:
        # All 9 reasoning-mixin handlers share the same
        # ``if not self.capture_reasoning: return`` gate — exercise every one.
        lnr = _make_listener(capture_reasoning=False)
        _prime_agent(lnr, "agent-1")

        lnr.on_agent_reasoning_started(
            None,
            AgentReasoningStartedEvent(
                task_id="t1", agent_role="R", agent_id="agent-1"
            ),
        )
        lnr.on_agent_reasoning_completed(
            None,
            AgentReasoningCompletedEvent(
                task_id="t1", agent_role="R", plan="p", ready=True, agent_id="agent-1"
            ),
        )
        lnr.on_agent_reasoning_failed(
            None,
            AgentReasoningFailedEvent(
                task_id="t1", agent_role="R", error="boom", agent_id="agent-1"
            ),
        )
        lnr.on_step_observation_started(
            None,
            StepObservationStartedEvent(
                agent_role="R", step_number=1, agent_id="agent-1"
            ),
        )
        lnr.on_step_observation_completed(
            None,
            StepObservationCompletedEvent(
                agent_role="R", step_number=1, agent_id="agent-1"
            ),
        )
        lnr.on_step_observation_failed(
            None,
            StepObservationFailedEvent(
                agent_role="R", step_number=1, error="boom", agent_id="agent-1"
            ),
        )
        lnr.on_goal_achieved_early(
            None,
            GoalAchievedEarlyEvent(agent_role="R", step_number=1, agent_id="agent-1"),
        )
        lnr.on_plan_replan_triggered(
            None,
            PlanReplanTriggeredEvent(agent_role="R", step_number=1, agent_id="agent-1"),
        )
        lnr.on_plan_refinement(
            None,
            PlanRefinementEvent(agent_role="R", step_number=1, agent_id="agent-1"),
        )

        # No spans opened, no annotations written when capture is disabled.
        with lnr._lock:
            assert lnr._reasoning_spans == {}
            assert lnr._observation_spans == {}
            assert lnr._agent_spans["agent-1"].attributes == {}
        lnr.shutdown()
