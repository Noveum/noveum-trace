"""
Real-event contract & regression tests — crew + agent families.

Unlike ``test_crewai_integration.py`` (which feeds handlers ``MagicMock``
events), these construct the REAL ``crewai.events.types.*`` Pydantic events and
assert what actually lands on the span. They catch upstream field drift and
expose handler/event field mismatches that MagicMock tests silently pass.

``# KNOWN BUG`` baselines current (buggy) behavior so the suite stays green and
the defect is documented (see CREWAI_TEST_PLAN.md §2). Flip the assertion when
the handler is fixed.

Field names were verified empirically against the installed CrewAI version
(``1.14.2a2``) — the installed version is authoritative, not the plan.
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

from crewai.events.types.agent_events import (  # noqa: E402
    AgentEvaluationCompletedEvent,
    AgentEvaluationStartedEvent,
    AgentExecutionCompletedEvent,
    AgentExecutionStartedEvent,
    LiteAgentExecutionStartedEvent,
)
from crewai.events.types.crew_events import (  # noqa: E402
    CrewKickoffCompletedEvent,
    CrewKickoffFailedEvent,
    CrewTestCompletedEvent,
    CrewTestResultEvent,
    CrewTestStartedEvent,
)

from noveum_trace.integrations.crewai.crewai_constants import (  # noqa: E402
    ATTR_STATUS_ERROR,
    ATTR_STATUS_SUCCESS,
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


def _crew_source(crew_id: str = "crew-1", name: str = "C") -> SimpleNamespace:
    """Minimal Crew-like source: crew_id resolves from ``source.id``."""
    return SimpleNamespace(
        id=crew_id,
        name=name,
        agents=[],
        tasks=[],
        process=SimpleNamespace(value="sequential"),
        memory=False,
        verbose=False,
        max_rpm=None,
    )


# ===========================================================================
# Crew (mostly correct — verify happy paths + error parsing)
# ===========================================================================


class TestCrewRealEvents:
    def test_kickoff_failed_string_error_parses_type_and_message(self) -> None:
        """A real ``CrewKickoffFailedEvent`` carries ``error`` as a *string*.

        The crew handler's string-error branch splits ``'Type: msg'`` and
        records the inferred error type — verify it actually parses
        ``ValueError`` out of the message rather than recording ``error.type``
        as ``'str'`` (the Exception branch is never hit for real events).
        """
        lnr = _make_listener()
        span = _prime_crew(lnr, "crew-1")
        src = _crew_source("crew-1")
        ev = CrewKickoffFailedEvent(error="ValueError: invalid config", crew_name="C")
        lnr.on_crew_kickoff_failed(src, ev)

        # Span closed + removed, trace finished.
        with lnr._lock:
            assert "crew-1" not in lnr._crew_spans
        attrs = span.attributes
        assert attrs["crew.status"] == ATTR_STATUS_ERROR
        # String parsing: 'ValueError: invalid config' -> type + trimmed message.
        assert attrs["error.type"] == "ValueError"
        assert attrs["error.message"] == "invalid config"
        span.finish.assert_called_once()
        lnr._test_trace.finish.assert_not_called()  # closed via client.finish_trace
        lnr.shutdown()

    def test_kickoff_completed_aggregates_output_tokens_cost(self) -> None:
        """Happy path: output text, aggregated token/cost totals, status ok."""
        lnr = _make_listener()
        span = _prime_crew(lnr, "crew-1")
        # Pre-seed the per-crew accumulators (exact attr names from
        # _handlers_crew.py: _total_tokens_by_crew / _total_cost_by_crew).
        with lnr._lock:
            lnr._total_tokens_by_crew["crew-1"] = 150
            lnr._total_cost_by_crew["crew-1"] = 0.003

        src = _crew_source("crew-1")
        ev = CrewKickoffCompletedEvent(output="the final answer", crew_name="C")
        lnr.on_crew_kickoff_completed(src, ev)

        with lnr._lock:
            assert "crew-1" not in lnr._crew_spans
        attrs = span.attributes
        assert attrs["crew.status"] == ATTR_STATUS_SUCCESS
        assert attrs["crew.output"] == "the final answer"
        assert attrs["crew.total_tokens"] == 150
        assert attrs["crew.total_cost"] == 0.003
        assert "crew.duration_ms" in attrs
        span.finish.assert_called_once()
        lnr.shutdown()

    def test_full_test_lifecycle_captures_quality_and_model(self) -> None:
        """CrewTestStarted -> CrewTestResult -> CrewTestCompleted attribute capture.

        Result event fires *before* completed, so the span is still open when
        quality_score / test_model / execution_duration are written.
        """
        lnr = _make_listener()
        span = _prime_crew(lnr, "crew-1")
        src = _crew_source("crew-1")

        started = CrewTestStartedEvent(
            n_iterations=3,
            eval_llm="gpt-4o",
            inputs={"topic": "AI"},
            crew_name="C",
        )
        lnr.on_crew_test_started(src, started)
        assert span.attributes["crew.mode"] == "test"
        assert span.attributes["crew.test.n_iterations"] == 3
        assert span.attributes["crew.test.eval_llm"] == "gpt-4o"

        result = CrewTestResultEvent(
            quality=8.5,
            execution_duration=12.0,
            model="gpt-4o",
            crew_name="C",
        )
        lnr.on_crew_test_result(src, result)
        assert span.attributes["crew.quality_score"] == 8.5
        assert span.attributes["crew.test.execution_duration_s"] == 12.0
        assert span.attributes["crew.test_model"] == "gpt-4o"

        completed = CrewTestCompletedEvent(crew_name="C")
        lnr.on_crew_test_completed(src, completed)
        with lnr._lock:
            assert "crew-1" not in lnr._crew_spans
        assert span.attributes["crew.status"] == ATTR_STATUS_SUCCESS
        assert span.attributes["crew.mode"] == "test"
        span.finish.assert_called_once()
        lnr.shutdown()


# ===========================================================================
# Agent (several KNOWN BUGS to baseline — verified against real event fields)
# ===========================================================================


class TestAgentRealEvents:
    def test_execution_started_full_identity(self) -> None:
        """Happy path: role/goal/backstory/llm_model/task_prompt captured.

        ``AgentExecutionStartedEvent`` types ``agent: BaseAgent`` (and requires
        ``task``/``tools``), so we ``model_construct`` to bypass Pydantic
        validation and pass a SimpleNamespace agent. Identity is read from the
        *source* (the Agent object), not the event payload.
        """
        lnr = _make_listener()
        _prime_crew(lnr, "crew-1")  # single open trace -> child span resolves
        agent_obj = SimpleNamespace(
            id="agent-1",
            role="Researcher",
            goal="Find AI trends",
            backstory="A veteran analyst",
            tools=[],
            allow_delegation=False,
            max_iter=5,
            max_rpm=None,
            llm=SimpleNamespace(model="gpt-4"),
        )
        ev = AgentExecutionStartedEvent.model_construct(
            agent=agent_obj,
            task=SimpleNamespace(id="t1"),
            tools=[],
            task_prompt="Research the AI landscape",
        )
        lnr.on_agent_execution_started(agent_obj, ev)

        with lnr._lock:
            span = lnr._agent_spans.get("agent-1")
        assert span is not None, "agent span not opened"
        attrs = span.attributes
        assert attrs["agent.type"] == "full"
        assert attrs["agent.id"] == "agent-1"
        assert attrs["agent.role"] == "Researcher"
        assert attrs["agent.goal"] == "Find AI trends"
        assert attrs["agent.backstory"] == "A veteran analyst"
        assert attrs["agent.llm_model"] == "gpt-4"
        assert attrs["agent.task_prompt"] == "Research the AI landscape"
        lnr.shutdown()

    def test_execution_completed_output_set_no_iterations(self) -> None:
        """Output captured; KNOWN BUG (§2 #14): no iteration count.

        ``AgentExecutionCompletedEvent`` defines only ``agent, task, output``
        (verified on 1.14.2a2). The handler reads ``event.iterations``/``step``
        — neither exists — so ``agent.iterations`` is never written.
        """
        lnr = _make_listener()
        span = _prime_agent(lnr, "agent-1")
        ev = AgentExecutionCompletedEvent.model_construct(
            agent=SimpleNamespace(id="agent-1"),
            task=SimpleNamespace(id="t1"),
            output="the agent's final output",
            agent_id="agent-1",
        )
        lnr.on_agent_execution_completed(SimpleNamespace(id="agent-1"), ev)

        with lnr._lock:
            assert "agent-1" not in lnr._agent_spans  # closed
        attrs = span.attributes
        assert attrs["agent.status"] == ATTR_STATUS_SUCCESS
        assert attrs["agent.output"] == "the agent's final output"
        # KNOWN BUG (§2 #14): real event has no iterations/step field, so the
        # reasoning-iteration count silently never reaches the span.
        assert "agent.iterations" not in attrs
        lnr.shutdown()

    def test_lite_agent_started_ignores_agent_info_dict(self) -> None:
        """KNOWN BUG (§2 #15): lite identity comes from ``agent_info`` (a dict),
        but the handler probes *source*/event scalar attrs — never the dict —
        so the lite agent's role/id/goal are silently lost.

        ``agent_info`` is the only carrier of identity on
        ``LiteAgentExecutionStartedEvent`` (verified on 1.14.2a2); the event has
        no ``role``/``goal``/``id`` fields and ``source`` here carries none
        either.
        """
        lnr = _make_listener()
        _prime_crew(lnr, "crew-1")  # single open trace -> child span resolves
        ev = LiteAgentExecutionStartedEvent(
            agent_info={"id": "lite-1", "role": "Helper", "goal": "assist"},
            tools=None,
            messages="hello",
        )
        # source carries no identity -> agent_id falls back to id(source).
        lnr.on_lite_agent_started(SimpleNamespace(), ev)

        with lnr._lock:
            assert len(lnr._agent_spans) == 1, "lite agent span not opened"
            span = next(iter(lnr._agent_spans.values()))
        attrs = span.attributes
        # The handler DOES tag the agent type correctly.
        assert attrs["agent.type"] == "lite"
        # KNOWN BUG (§2 #15): identity from agent_info dict is never captured.
        assert "agent.role" not in attrs
        assert "Helper" not in str(attrs)
        assert "lite-1" not in str(attrs)
        lnr.shutdown()

    def test_evaluation_completed_reads_score_not_passed_or_model(self) -> None:
        """KNOWN BUG (§2 #16): ``AgentEvaluationCompletedEvent`` carries ``score``
        (+ agent_id/agent_role/iteration/metric_category) but NOT
        ``passed``/``result``/``model`` (verified on 1.14.2a2).

        ``score`` IS read by the handler — assert it lands; ``passed`` and
        ``model`` are absent because the real event has no such fields.
        """
        lnr = _make_listener()
        span = _prime_agent(lnr, "agent-1")
        ev = AgentEvaluationCompletedEvent(
            agent_id="agent-1",
            agent_role="Researcher",
            iteration=1,
            metric_category="quality",
            score=0.95,
        )
        lnr.on_agent_evaluation_completed(SimpleNamespace(id="agent-1"), ev)

        attrs = span.attributes
        # score IS captured (the handler reads event.score).
        assert attrs["agent.evaluation.score"] == 0.95
        # KNOWN BUG (§2 #16): real event has no passed/model fields.
        assert "agent.evaluation.passed" not in attrs
        assert "agent.evaluation.model" not in attrs
        lnr.shutdown()

    def test_evaluation_started_no_criteria(self) -> None:
        """KNOWN BUG (§2 #17): ``AgentEvaluationStartedEvent`` has no ``criteria``
        field (verified on 1.14.2a2: agent_id/agent_role/iteration only), so the
        handler never records ``agent.evaluation.criteria``.

        It also has no ``model``/``evaluator_model`` field, so
        ``agent.evaluation.model`` is likewise absent.
        """
        lnr = _make_listener(capture_inputs=True)
        span = _prime_agent(lnr, "agent-1")
        ev = AgentEvaluationStartedEvent(
            agent_id="agent-1",
            agent_role="Researcher",
            iteration=1,
        )
        lnr.on_agent_evaluation_started(SimpleNamespace(id="agent-1"), ev)

        attrs = span.attributes
        # KNOWN BUG (§2 #17): no criteria field on the real event.
        assert "agent.evaluation.criteria" not in attrs
        # No model field either -> evaluator model never recorded at start.
        assert "agent.evaluation.model" not in attrs
        lnr.shutdown()
