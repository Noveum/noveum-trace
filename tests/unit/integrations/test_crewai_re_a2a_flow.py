"""
Real-event contract & regression tests — A2A + FLOW families.

Unlike ``test_crewai_integration.py`` (which feeds handlers ``MagicMock``
events), these construct the REAL ``crewai.events.types.*`` Pydantic events and
assert what actually lands on the span. They catch upstream field drift and
expose handler/event field mismatches that MagicMock tests silently pass.

``# KNOWN BUG`` baselines current (buggy) behavior so the suite stays green and
the defect is documented (see CREWAI_TEST_PLAN.md §2). Flip the assertion when
the handler is fixed. Installed CrewAI version: 1.14.2a2.

Each bug test additionally asserts (a) the target span is actually open (so the
"absent attribute" assertion proves the handler ran and dropped the value rather
than no-op'd), and (b) the real data IS present on the event (recoverable).
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("crewai", reason="requires optional 'crewai' extra")

_src = Path(__file__).parents[3] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from crewai.events.types.a2a_events import (  # noqa: E402
    A2AArtifactReceivedEvent,
    A2AAuthenticationFailedEvent,
    A2AConnectionErrorEvent,
    A2AConversationCompletedEvent,
    A2AConversationStartedEvent,
    A2ADelegationCompletedEvent,
    A2ADelegationStartedEvent,
    A2AMessageSentEvent,
    A2APollingStatusEvent,
    A2AResponseReceivedEvent,
    A2AServerTaskFailedEvent,
    A2AServerTaskStartedEvent,
    A2AStreamingChunkEvent,
)
from crewai.events.types.flow_events import (  # noqa: E402
    FlowFinishedEvent,
    FlowInputReceivedEvent,
    FlowPausedEvent,
    FlowStartedEvent,
    MethodExecutionFailedEvent,
    MethodExecutionFinishedEvent,
    MethodExecutionStartedEvent,
)

from noveum_trace.integrations.crewai._handlers_flow import (  # noqa: E402
    _make_method_key,
)
from noveum_trace.integrations.crewai.crewai_constants import (  # noqa: E402
    ATTR_STATUS_ERROR,
    ATTR_STATUS_SUCCESS,
)

TRACE_ID = "trace-x"


# ---------------------------------------------------------------------------
# Self-contained harness (rich mocks so attribute assertions work)
# (copied verbatim from test_crewai_re_guardrail_task.py)
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


# ---------------------------------------------------------------------------
# A2A-specific helpers
# ---------------------------------------------------------------------------


def _open_delegation(lnr: Any, context_id: str) -> MagicMock:
    """Open a delegation span via the real handler; return the open Span."""
    ev = A2ADelegationStartedEvent(
        agent_id="remote-agent-1",
        endpoint="http://a2a.example/agent",
        task_description="summarize the report",
        context_id=context_id,
    )
    lnr.on_a2a_delegation_started(SimpleNamespace(role="Manager"), ev)
    entry = lnr._a2a_spans[(context_id, "delegation")]
    span = entry["span"]
    assert span is not None, "delegation span did not open (no resolvable trace)"
    return span


def _open_conversation(lnr: Any, context_id: str) -> MagicMock:
    """Open a conversation span via the real handler; return the open Span."""
    ev = A2AConversationStartedEvent(
        agent_id="remote-agent-1",
        endpoint="http://a2a.example/agent",
        context_id=context_id,
    )
    lnr.on_a2a_conversation_started(SimpleNamespace(role="Manager"), ev)
    entry = lnr._a2a_spans[(context_id, "conversation")]
    span = entry["span"]
    assert span is not None, "conversation span did not open (no resolvable trace)"
    return span


# ===========================================================================
# A2A — dead-handler confirmations (§2C)
# ===========================================================================


class TestA2ADeadHandlers:
    """The handler exists but the subscribed event class does NOT in 1.14.2a2.

    These import attempts must raise; failures arrive via the corresponding
    ``*Completed`` event with ``status='failed'`` instead (tested below).
    """

    def test_delegation_failed_event_does_not_exist(self) -> None:
        with pytest.raises(ImportError):
            from crewai.events.types.a2a_events import (  # noqa: F401
                A2ADelegationFailedEvent,
            )

    def test_conversation_failed_event_does_not_exist(self) -> None:
        with pytest.raises(ImportError):
            from crewai.events.types.a2a_events import (  # noqa: F401
                A2AConversationFailedEvent,
            )

    def test_message_received_event_does_not_exist(self) -> None:
        with pytest.raises(ImportError):
            from crewai.events.types.a2a_events import (  # noqa: F401
                A2AMessageReceivedEvent,
            )

    def test_streaming_completed_event_does_not_exist(self) -> None:
        with pytest.raises(ImportError):
            from crewai.events.types.a2a_events import (  # noqa: F401
                A2AStreamingCompletedEvent,
            )


# ===========================================================================
# A2A — delegation lifecycle (happy + live failure path)
# ===========================================================================


class TestA2ADelegationRealEvents:
    def test_delegation_started_captures_identity(self) -> None:
        lnr = _make_listener()
        _prime_crew(lnr)  # one open trace → child span resolves
        span = _open_delegation(lnr, "ctx-d1")
        attrs = span.attributes
        assert attrs["a2a.context_id"] == "ctx-d1"
        assert attrs["a2a.endpoint"] == "http://a2a.example/agent"
        assert attrs["a2a.task_description"] == "summarize the report"
        # receiving agent resolved from event.agent_id (remote)
        assert attrs.get("a2a.remote_agent_id") == "remote-agent-1"
        lnr.shutdown()

    def test_delegation_completed_ok_status_is_raw_string(self) -> None:
        lnr = _make_listener()
        _prime_crew(lnr)
        span = _open_delegation(lnr, "ctx-d2")
        ev = A2ADelegationCompletedEvent(
            status="completed", result="final result text", context_id="ctx-d2"
        )
        lnr.on_a2a_delegation_completed(SimpleNamespace(), ev)

        # span closed/removed
        assert ("ctx-d2", "delegation") not in lnr._a2a_spans
        attrs = span.attributes
        # NOTE: delegation writes the RAW status string ("completed"), not
        # ATTR_STATUS_SUCCESS ("ok"); set_status(OK) is applied separately.
        assert attrs["a2a.status"] == "completed"
        assert attrs["a2a.result"] == "final result text"
        assert "a2a.duration_ms" in attrs
        # span closed OK (set_status(OK) applied separately from a2a.status)
        span.set_status.assert_called()
        lnr.shutdown()

    def test_delegation_completed_failed_closes_error(self) -> None:
        """Live failure path — A2ADelegationFailedEvent is dead (§2C)."""
        lnr = _make_listener()
        _prime_crew(lnr)
        span = _open_delegation(lnr, "ctx-d3")
        ev = A2ADelegationCompletedEvent(
            status="failed", error="remote agent rejected task", context_id="ctx-d3"
        )
        lnr.on_a2a_delegation_completed(SimpleNamespace(), ev)

        assert ("ctx-d3", "delegation") not in lnr._a2a_spans
        attrs = span.attributes
        assert attrs["a2a.status"] == ATTR_STATUS_ERROR
        # error is a plain string on the real event → error.type == "str"
        assert attrs["error.type"] == "str"
        assert "remote agent rejected task" in str(attrs.get("error.message", ""))
        span.set_status.assert_called()
        lnr.shutdown()


# ===========================================================================
# A2A — conversation lifecycle (happy + live failure path) + streaming
# ===========================================================================


class TestA2AConversationRealEvents:
    def test_conversation_started_separate_span_and_buffers(self) -> None:
        lnr = _make_listener()
        _prime_crew(lnr)
        # delegation and conversation can coexist for the same context_id
        _open_delegation(lnr, "ctx-c1")
        span = _open_conversation(lnr, "ctx-c1")
        assert ("ctx-c1", "delegation") in lnr._a2a_spans
        assert ("ctx-c1", "conversation") in lnr._a2a_spans
        # conversation initializes its streaming/message buffers
        assert ("ctx-c1", "conversation") in lnr._a2a_streaming_chunks
        assert ("ctx-c1", "conversation") in lnr._a2a_stream_buffers
        assert span.attributes["a2a.context_id"] == "ctx-c1"
        lnr.shutdown()

    def test_streaming_chunk_accumulate_and_flush_on_final(self) -> None:
        lnr = _make_listener()
        _prime_crew(lnr)
        span = _open_conversation(lnr, "ctx-c2")

        # non-final chunk: buffered, NOT yet flushed to the span
        ch1 = A2AStreamingChunkEvent(
            chunk="Hello ", chunk_index=0, final=False, context_id="ctx-c2"
        )
        lnr.on_a2a_streaming_chunk(SimpleNamespace(), ch1)
        assert "a2a.streaming_content" not in span.attributes

        # final chunk: flush joins all chunks onto the span
        ch2 = A2AStreamingChunkEvent(
            chunk="World", chunk_index=1, final=True, context_id="ctx-c2"
        )
        lnr.on_a2a_streaming_chunk(SimpleNamespace(), ch2)
        assert span.attributes["a2a.streaming_content"] == "Hello World"
        lnr.shutdown()

    def test_conversation_completed_success_uses_status_ok(self) -> None:
        lnr = _make_listener()
        _prime_crew(lnr)
        span = _open_conversation(lnr, "ctx-c3")
        ev = A2AConversationCompletedEvent(
            status="completed",
            total_turns=3,
            final_result="conversation answer",
            context_id="ctx-c3",
        )
        lnr.on_a2a_conversation_completed(SimpleNamespace(), ev)

        assert ("ctx-c3", "conversation") not in lnr._a2a_spans
        attrs = span.attributes
        # NOTE: conversation (unlike delegation) writes ATTR_STATUS_SUCCESS.
        assert attrs["a2a.status"] == ATTR_STATUS_SUCCESS
        assert attrs["a2a.final_result"] == "conversation answer"
        assert attrs["a2a.total_turns"] == 3
        lnr.shutdown()

    def test_conversation_completed_failed_closes_error(self) -> None:
        """Live failure path — A2AConversationFailedEvent is dead (§2C)."""
        lnr = _make_listener()
        _prime_crew(lnr)
        span = _open_conversation(lnr, "ctx-c4")
        ev = A2AConversationCompletedEvent(
            status="failed",
            total_turns=1,
            error="conversation aborted",
            context_id="ctx-c4",
        )
        lnr.on_a2a_conversation_completed(SimpleNamespace(), ev)

        assert ("ctx-c4", "conversation") not in lnr._a2a_spans
        attrs = span.attributes
        assert attrs["a2a.status"] == ATTR_STATUS_ERROR
        assert "conversation aborted" in str(attrs.get("error.message", ""))
        lnr.shutdown()

    def test_message_sent_buffered_into_conversation(self) -> None:
        lnr = _make_listener()
        _prime_crew(lnr)
        _open_conversation(lnr, "ctx-c5")
        ev = A2AMessageSentEvent(
            message="What is the weather?",
            turn_number=1,
            context_id="ctx-c5",
            agent_role="Reporter",
        )
        lnr.on_a2a_message_sent(SimpleNamespace(), ev)
        buf = lnr._a2a_stream_buffers[("ctx-c5", "conversation")]
        assert len(buf) == 1
        assert buf[0]["type"] == "sent"
        assert buf[0]["content"] == "What is the weather?"
        lnr.shutdown()

    def test_response_received_buffered_into_conversation(self) -> None:
        lnr = _make_listener()
        _prime_crew(lnr)
        _open_conversation(lnr, "ctx-c6")
        ev = A2AResponseReceivedEvent(
            response="It is sunny.",
            turn_number=2,
            status="completed",
            context_id="ctx-c6",
            final=True,
        )
        lnr.on_a2a_response_received(SimpleNamespace(), ev)
        buf = lnr._a2a_stream_buffers[("ctx-c6", "conversation")]
        assert len(buf) == 1
        assert buf[0]["type"] == "response_received"
        assert buf[0]["content"] == "It is sunny."
        assert buf[0]["final"] is True
        lnr.shutdown()


# ===========================================================================
# A2A — annotation handlers writing onto the open delegation span
# ===========================================================================


class TestA2AAnnotationRealEvents:
    def test_polling_status_field_mismatch_known_bug(self) -> None:
        lnr = _make_listener()
        _prime_crew(lnr)
        span = _open_delegation(lnr, "ctx-p1")

        ev = A2APollingStatusEvent(
            task_id="task-9",
            context_id="ctx-p1",
            state="pending",
            elapsed_seconds=2.5,
            poll_count=2,
        )
        lnr.on_a2a_polling_status(SimpleNamespace(), ev)

        # Span is still open (handler annotates, does not close).
        assert ("ctx-p1", "delegation") in lnr._a2a_spans

        # KNOWN BUG (§2 #12): real fields are ``state`` / ``poll_count``; the
        # handler reads ``status`` / ``attempt`` → both attrs never set.
        assert "a2a.polling_status" not in span.attributes
        assert "a2a.polling_attempt" not in span.attributes
        # The real data IS on the event (recoverable once the handler is fixed).
        assert ev.state == "pending"
        assert ev.poll_count == 2
        lnr.shutdown()

    def test_auth_failed_field_mismatch_known_bug(self) -> None:
        lnr = _make_listener()
        _prime_crew(lnr)
        span = _open_delegation(lnr, "ctx-a1")

        # A2AAuthenticationFailedEvent has NO context_id field → handler resolves
        # context from source.context_id.
        ev = A2AAuthenticationFailedEvent(
            endpoint="http://a2a.example/agent",
            auth_type="bearer",
            error="token expired",
        )
        lnr.on_a2a_auth_failed(SimpleNamespace(context_id="ctx-a1"), ev)

        attrs = span.attributes
        # Handler DID run: error.type + a2a.auth_error are written.
        assert attrs["error.type"] == "AuthenticationError"
        assert "token expired" in str(attrs.get("a2a.auth_error", ""))
        # KNOWN BUG (§2 #13): real field is ``auth_type``; handler reads
        # ``auth_method`` / ``method`` → a2a.auth_method never set.
        assert "a2a.auth_method" not in attrs
        # The real data IS on the event (recoverable once the handler is fixed).
        assert ev.auth_type == "bearer"
        lnr.shutdown()

    def test_connection_error_captures_type_and_endpoint(self) -> None:
        lnr = _make_listener()
        _prime_crew(lnr)
        span = _open_delegation(lnr, "ctx-ce1")
        ev = A2AConnectionErrorEvent(
            endpoint="http://a2a.example/agent",
            error="connection refused",
            error_type="connection_refused",
            context_id="ctx-ce1",
        )
        lnr.on_a2a_connection_error(SimpleNamespace(), ev)

        attrs = span.attributes
        assert attrs["error.type"] == "ConnectionError"
        assert "connection refused" in str(attrs.get("error.message", ""))
        assert attrs["a2a.endpoint"] == "http://a2a.example/agent"
        lnr.shutdown()

    def test_artifact_received_non_image_metadata(self) -> None:
        lnr = _make_listener()
        _prime_crew(lnr)
        span = _open_delegation(lnr, "ctx-art1")
        ev = A2AArtifactReceivedEvent(
            task_id="task-art",
            artifact_id="artifact-1",
            artifact_name="report.txt",
            mime_type="text/plain",
            size_bytes=2048,
            context_id="ctx-art1",
        )
        lnr.on_a2a_artifact_received(SimpleNamespace(), ev)

        attrs = span.attributes
        assert attrs["a2a.artifact_name"] == "report.txt"
        assert attrs["a2a.artifact_mime_type"] == "text/plain"
        assert attrs["a2a.artifact_size_bytes"] == 2048
        assert attrs["a2a.artifact_id"] == "artifact-1"
        # non-image → no image export / uuid
        assert "a2a.artifact_image_uuid" not in attrs
        lnr.shutdown()

    def test_server_task_started_and_failed_annotate_delegation(self) -> None:
        lnr = _make_listener()
        _prime_crew(lnr)
        span = _open_delegation(lnr, "ctx-st1")

        started = A2AServerTaskStartedEvent(task_id="srv-1", context_id="ctx-st1")
        lnr.on_a2a_server_task_started(SimpleNamespace(), started)
        assert span.attributes["a2a.server_task.phase"] == "started"
        assert span.attributes["a2a.server_task.task_id"] == "srv-1"

        failed = A2AServerTaskFailedEvent(
            task_id="srv-1", context_id="ctx-st1", error="server task crashed"
        )
        lnr.on_a2a_server_task_failed(SimpleNamespace(), failed)
        assert span.attributes["a2a.server_task.phase"] == "failed"
        assert "server task crashed" in str(
            span.attributes.get("a2a.server_task.error", "")
        )
        # annotation only — span still open
        assert ("ctx-st1", "delegation") in lnr._a2a_spans
        lnr.shutdown()


# ===========================================================================
# FLOW — dead/absent event confirmation (§2C)
# ===========================================================================


class TestFlowDeadHandlers:
    def test_flow_failed_event_does_not_exist(self) -> None:
        # No FlowFailedEvent in 1.14.2a2 — flow-level failure surfaces via
        # MethodExecutionFailedEvent (tested below).
        with pytest.raises(ImportError):
            from crewai.events.types.flow_events import (  # noqa: F401
                FlowFailedEvent,
            )


# ===========================================================================
# FLOW — lifecycle (started / method / finished)
# ===========================================================================


class TestFlowLifecycleRealEvents:
    def test_flow_started_inputs_and_state(self) -> None:
        lnr = _make_listener()
        # FlowStartedEvent has no flow_id/state → resolved from source.
        src = SimpleNamespace(flow_id="f1", name="ResearchFlow", state={"step": 0})
        ev = FlowStartedEvent(flow_name="ResearchFlow", inputs={"topic": "AI"})
        lnr.on_flow_started(src, ev)

        entry = lnr._flow_spans.get("f1")
        assert entry is not None, "flow span not opened"
        attrs = entry["span"].attributes
        assert attrs["flow.id"] == "f1"
        assert attrs["flow.name"] == "ResearchFlow"
        assert '"topic": "AI"' in attrs["flow.inputs"]
        assert '"step": 0' in attrs["flow.state"]
        lnr.shutdown()

    def test_method_started_name_and_type_inference(self) -> None:
        lnr = _make_listener()
        # @start method inference: method_name in source._start_methods → "start".
        src = SimpleNamespace(
            flow_id="f2",
            name="ResearchFlow",
            state={},
            _start_methods={"begin"},
            _routers=set(),
        )
        # Open the parent flow span first so the method child span resolves its
        # trace (the method span parents off the open flow span).
        lnr.on_flow_started(src, FlowStartedEvent(flow_name="ResearchFlow", inputs={}))
        ev = MethodExecutionStartedEvent(
            flow_name="ResearchFlow", method_name="begin", state={}
        )
        lnr.on_method_execution_started(src, ev)

        method_key = _make_method_key("f2", str(id(ev)))
        entry = lnr._flow_method_spans.get(method_key)
        assert entry is not None, "flow method span not opened"
        attrs = entry["span"].attributes
        assert attrs["flow.method.name"] == "begin"
        assert attrs["flow.method.type"] == "start"
        lnr.shutdown()

    def test_method_finished_reads_result_not_output(self) -> None:
        lnr = _make_listener()
        src = SimpleNamespace(flow_id="f3", name="ResearchFlow", state={})
        # _resolve_method_id falls back to id(event); pre-seed the method span
        # under that key so finish pairs (started/finished are distinct events).
        ev = MethodExecutionFinishedEvent(
            flow_name="ResearchFlow",
            method_name="begin",
            result="method return value",
            state={},
        )
        method_key = _make_method_key("f3", str(id(ev)))
        span = _make_rich_span(span_id="flow-method-3")
        with lnr._lock:
            lnr._flow_method_spans[method_key] = {
                "span": span,
                "start_t": time.monotonic(),
            }
        lnr.on_method_execution_finished(src, ev)

        assert method_key not in lnr._flow_method_spans  # closed
        attrs = span.attributes
        # real field is ``result`` (not ``output``) → flow.method.output captured
        assert attrs["flow.method.output"] == "method return value"
        assert attrs["flow.method.status"] == ATTR_STATUS_SUCCESS
        assert "flow.method.duration_ms" in attrs
        lnr.shutdown()

    def test_method_failed_marks_error(self) -> None:
        """Flow-side failure path (FlowFailedEvent does not exist — §2C)."""
        lnr = _make_listener()
        src = SimpleNamespace(flow_id="f4", name="ResearchFlow", state={})
        ev = MethodExecutionFailedEvent(
            flow_name="ResearchFlow",
            method_name="begin",
            error=RuntimeError("method blew up"),
        )
        method_key = _make_method_key("f4", str(id(ev)))
        span = _make_rich_span(span_id="flow-method-4")
        with lnr._lock:
            lnr._flow_method_spans[method_key] = {
                "span": span,
                "start_t": time.monotonic(),
            }
        lnr.on_method_execution_failed(src, ev)

        assert method_key not in lnr._flow_method_spans
        attrs = span.attributes
        assert attrs["flow.method.status"] == ATTR_STATUS_ERROR
        assert attrs["error.type"] == "RuntimeError"
        assert "method blew up" in str(attrs.get("error.message", ""))
        span.set_status.assert_called()
        lnr.shutdown()

    def test_flow_finished_closes_span(self) -> None:
        lnr = _make_listener()
        src = SimpleNamespace(flow_id="f5", name="ResearchFlow", state={"step": 0})
        lnr.on_flow_started(src, FlowStartedEvent(flow_name="ResearchFlow", inputs={}))
        span = lnr._flow_spans["f5"]["span"]

        # FlowFinishedEvent has no flow_id → source.flow_id resolves the entry.
        ev = FlowFinishedEvent(
            flow_name="ResearchFlow", result="final flow output", state={"step": 9}
        )
        lnr.on_flow_finished(src, ev)

        assert "f5" not in lnr._flow_spans
        attrs = span.attributes
        assert attrs["flow.result"] == "final flow output"
        assert attrs["flow.status"] == ATTR_STATUS_SUCCESS
        assert "flow.duration_ms" in attrs
        lnr.shutdown()


# ===========================================================================
# FLOW — pause / input annotations (known bugs)
# ===========================================================================


class TestFlowAnnotationRealEvents:
    def test_flow_input_received_loses_value_known_bug(self) -> None:
        lnr = _make_listener()
        src = SimpleNamespace(flow_id="f6", name="ResearchFlow", state={})
        lnr.on_flow_started(src, FlowStartedEvent(flow_name="ResearchFlow", inputs={}))
        span = lnr._flow_spans["f6"]["span"]

        ev = FlowInputReceivedEvent(
            flow_name="ResearchFlow",
            method_name="ask_topic",
            message="What topic?",
            response="quantum computing",
        )
        lnr.on_flow_input_received(src, ev)

        attrs = span.attributes
        # Handler DID run: the marker is set.
        assert attrs.get("flow.input_received") is True
        # KNOWN BUG (§2 #10): real field is ``response``; handler reads
        # ``value`` / ``input`` → flow.input_value never set. The event also
        # carries no ``field`` → flow.input_field never set.
        assert "flow.input_value" not in attrs
        assert "flow.input_field" not in attrs
        # The real data IS on the event (recoverable once the handler is fixed).
        assert ev.response == "quantum computing"
        lnr.shutdown()

    def test_flow_paused_loses_outcomes_known_bug(self) -> None:
        lnr = _make_listener()
        # FlowPausedEvent carries its own flow_id; prime the flow span under it.
        src = SimpleNamespace(flow_id="f7", name="ResearchFlow", state={})
        lnr.on_flow_started(src, FlowStartedEvent(flow_name="ResearchFlow", inputs={}))
        span = lnr._flow_spans["f7"]["span"]

        ev = FlowPausedEvent(
            flow_name="ResearchFlow",
            flow_id="f7",
            method_name="await_review",
            state={},
            message="awaiting human review",
            emit=["approve", "reject"],
        )
        lnr.on_flow_paused(src, ev)

        attrs = span.attributes
        # Handler DID run: pause marker + message are set.
        assert attrs.get("flow.pause") is True
        assert "awaiting human review" in str(attrs.get("flow.pause_message", ""))
        # KNOWN BUG (§2 #11): real field is ``emit``; handler reads
        # ``possible_outcomes`` / ``outcomes`` → pause outcomes never set.
        assert "flow.pause_possible_outcomes" not in attrs
        # The real data IS on the event (recoverable once the handler is fixed).
        assert ev.emit == ["approve", "reject"]
        lnr.shutdown()
