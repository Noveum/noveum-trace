"""
Real-event contract & regression tests — guardrail + task families.

Unlike ``test_crewai_integration.py`` (which feeds handlers ``MagicMock``
events), these construct the REAL ``crewai.events.types.*`` Pydantic events and
assert what actually lands on the span. They catch upstream field drift and
expose handler/event field mismatches that MagicMock tests silently pass.

``# KNOWN BUG`` baselines current (buggy) behavior so the suite stays green and
the defect is documented (see CREWAI_TEST_PLAN.md §2). Flip the assertion when
the handler is fixed.
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

from crewai.events.types.llm_guardrail_events import (  # noqa: E402
    LLMGuardrailCompletedEvent,
    LLMGuardrailStartedEvent,
)
from crewai.events.types.task_events import (  # noqa: E402
    TaskCompletedEvent,
    TaskEvaluationEvent,
    TaskFailedEvent,
    TaskStartedEvent,
)
from crewai.tasks.task_output import TaskOutput  # noqa: E402

from noveum_trace.integrations.crewai._handlers_task import (  # noqa: E402
    _extract_evaluation_attributes,
)
from noveum_trace.integrations.crewai.crewai_constants import (  # noqa: E402
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
# Guardrail
# ===========================================================================


class TestGuardrailRealEvents:
    def test_started_captures_identity_and_retry(self) -> None:
        lnr = _make_listener()
        _prime_agent(lnr, "agent-1")
        src = SimpleNamespace(name="PolitenessGuard", role="Researcher")
        ev = LLMGuardrailStartedEvent(
            guardrail="must be polite", retry_count=0, agent_id="agent-1"
        )
        lnr.on_llm_guardrail_started(src, ev)

        with lnr._lock:
            entry = lnr._guardrail_spans.get(ev.event_id)
        assert entry is not None, "guardrail span not opened"
        attrs = entry["span"].attributes
        assert attrs["guardrail.id"] == ev.event_id
        assert attrs["guardrail.retry_count"] == 0
        assert attrs.get("guardrail.name") == "PolitenessGuard"

        # KNOWN BUG (§2 #5): LLMGuardrailStartedEvent has no call_id/input/output
        # fields, so the handler never records the LLM call being guarded nor the
        # text being validated.
        assert "llm.call_id" not in attrs
        assert "guardrail.input" not in attrs
        lnr.shutdown()

    def test_completed_rejection_loses_signal_known_bug(self) -> None:
        lnr = _make_listener()
        _prime_agent(lnr, "agent-1")
        started = LLMGuardrailStartedEvent(
            guardrail="must be polite", retry_count=0, agent_id="agent-1"
        )
        lnr.on_llm_guardrail_started(SimpleNamespace(name="G"), started)
        span = lnr._guardrail_spans[started.event_id]["span"]

        completed = LLMGuardrailCompletedEvent(
            success=False,
            result={"checks": [{"name": "toxicity", "passed": False}]},
            error="Output contains disallowed content",
            retry_count=2,
            started_event_id=started.event_id,
        )
        lnr.on_llm_guardrail_completed(SimpleNamespace(name="G"), completed)

        # Span paired + closed correctly.
        assert started.event_id not in lnr._guardrail_spans
        attrs = span.attributes
        assert attrs["guardrail.retry_count"] == 2  # this field works
        assert "guardrail.duration_ms" in attrs

        # KNOWN BUG (§2 #2): real field is ``success``; handler reads
        # ``validation_success`` → pass/fail signal silently dropped.
        assert "guardrail.validation_success" not in attrs
        # KNOWN BUG (§2 #3): real field is ``result`` (singular); handler reads
        # ``results``/``checks`` → never recorded.
        assert "guardrail.results" not in attrs
        # KNOWN BUG (§2 #4): a rejection (success=False) is reported as status
        # ``ok`` and the error message is dropped — a failed guardrail looks like
        # a passing one in traces.
        assert attrs["guardrail.status"] == ATTR_STATUS_SUCCESS
        assert "error.message" not in attrs
        lnr.shutdown()


# ===========================================================================
# Task
# ===========================================================================


class TestTaskRealEvents:
    def test_task_started_captures_attributes(self) -> None:
        lnr = _make_listener()
        _prime_crew(lnr, "crew-1")  # one open trace → child resolves
        src = SimpleNamespace(
            id="t1",
            name="research",
            description="Research AI trends",
            expected_output="A report",
            agent=SimpleNamespace(role="Researcher"),
            human_input=False,
            async_execution=False,
            output_file=None,
            context=[],
        )
        ev = TaskStartedEvent(task_id="t1", context="ctx")
        lnr.on_task_started(src, ev)

        with lnr._lock:
            span = lnr._task_spans.get("t1")
        assert span is not None, "task span not opened"
        assert span.attributes.get("task.description") == "Research AI trends"
        lnr.shutdown()

    def test_task_completed_extracts_output_raw(self) -> None:
        lnr = _make_listener()
        _prime_task(lnr, "t1")
        out = TaskOutput(description="d", agent="Researcher", raw="the final answer")
        ev = TaskCompletedEvent(task_id="t1", output=out)
        lnr.on_task_completed(SimpleNamespace(), ev)

        with lnr._lock:
            assert "t1" not in lnr._task_spans  # span closed/removed
        lnr.shutdown()

    def test_task_failed_marks_error(self) -> None:
        lnr = _make_listener()
        span = _prime_task(lnr, "t1")
        ev = TaskFailedEvent(task_id="t1", error="ValueError: boom")
        lnr.on_task_failed(SimpleNamespace(), ev)

        assert "t1" not in lnr._task_spans
        attrs = span.attributes
        assert attrs.get("task.status") == ATTR_STATUS_ERROR
        assert "boom" in str(attrs.get("error.message", ""))
        lnr.shutdown()

    def test_task_evaluation_extracts_nothing_known_bug(self) -> None:
        # KNOWN BUG (§2 #1): TaskEvaluationEvent defines only
        # {type, evaluation_type, task}. The handler reads score/feedback/model/
        # criteria/result.* — none exist — so _extract_evaluation_attributes
        # returns {} and no task.evaluation_* attribute is ever written.
        ev = TaskEvaluationEvent(evaluation_type="score", task_id="t1")
        attrs = _extract_evaluation_attributes(ev)
        assert attrs == {}

    def test_task_evaluation_writes_no_score_to_span_known_bug(self) -> None:
        lnr = _make_listener()
        span = _prime_task(lnr, "t1")
        ev = TaskEvaluationEvent(evaluation_type="score", task_id="t1")
        lnr.on_task_evaluation(SimpleNamespace(id="t1"), ev)
        # KNOWN BUG (§2 #1): evaluation score never reaches the span.
        assert "task.evaluation_score" not in span.attributes
        assert "task.evaluation_feedback" not in span.attributes
        lnr.shutdown()
