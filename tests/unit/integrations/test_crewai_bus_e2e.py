"""
Bus-routing and end-to-end tests for the CrewAI integration.

Every other crewAI test calls handler methods directly. These go through the
REAL ``crewai_event_bus`` (proving ``setup_listeners`` registration actually
routes events) and run a REAL ``crew.kickoff()`` with a network-free fake LLM
(proving the whole integration produces a correct span tree). See
CREWAI_TEST_PLAN.md §6.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest

# Disable crewAI telemetry / network before importing crewai.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("OPENAI_API_KEY", "sk-fake-test-key")

pytest.importorskip("crewai", reason="requires optional 'crewai' extra")

_src = Path(__file__).parents[3] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from crewai import Agent, Crew, Task  # noqa: E402
from crewai.events import crewai_event_bus  # noqa: E402
from crewai.events.types.crew_events import (  # noqa: E402
    CrewKickoffCompletedEvent,
    CrewKickoffStartedEvent,
)
from crewai.llms.base_llm import BaseLLM  # noqa: E402

from noveum_trace.integrations.crewai.crewai_listener import (  # noqa: E402
    NoveumCrewAIListener,
)


class _FakeLLM(BaseLLM):
    """Deterministic, network-free LLM that returns a fixed final answer."""

    def __init__(self, answer: str = "Final Answer: 42", raises: bool = False) -> None:
        super().__init__(model="fake-model")
        self._answer = answer
        self._raises = raises

    def call(self, messages: Any, *args: Any, **kwargs: Any) -> str:
        if self._raises:
            raise RuntimeError("intentional fake-LLM failure")
        return self._answer

    def supports_function_calling(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Span extraction helpers (mirror tests/integration/langchain/_helpers.py)
# ---------------------------------------------------------------------------


def _exported_spans(client: Any) -> list:
    spans: list = []
    for call in client.transport.export_trace.call_args_list:
        trace = call.args[0] if call.args else call.kwargs.get("trace")
        for s in getattr(trace, "spans", None) or []:
            spans.append(s)
    return spans


def _status(span: Any) -> Optional[str]:
    st = getattr(span, "status", None)
    return getattr(st, "value", st)


def _by_name(spans: list, name: str) -> list:
    return [s for s in spans if getattr(s, "name", None) == name]


@pytest.fixture(autouse=True)
def _reset_patch_state() -> Any:
    import noveum_trace.integrations.crewai.crewai_listener as m

    def _restore() -> None:
        if getattr(m, "_patch_applied", False) and m._original_track_token_usage:
            try:
                from crewai.llms.base_llm import BaseLLM as _B

                _B._track_token_usage_internal = m._original_track_token_usage
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
# Bus routing
# ===========================================================================


class TestBusRouting:
    def test_listener_registers_many_handlers(
        self, client_with_mocked_transport: Any
    ) -> None:
        lnr = NoveumCrewAIListener(client_with_mocked_transport)
        try:
            # Registration happens in __init__ via BaseEventListener → setup_listeners.
            assert len(lnr._handlers) > 60
        finally:
            lnr.shutdown()

    def test_bus_emit_crew_event_creates_span(
        self, client_with_mocked_transport: Any
    ) -> None:
        client = client_with_mocked_transport
        lnr = NoveumCrewAIListener(client)
        try:
            src = SimpleNamespace(
                id="crew-bus-1",
                name="BusCrew",
                agents=[],
                tasks=[],
                process=SimpleNamespace(value="sequential"),
                memory=False,
                verbose=False,
                max_rpm=None,
            )
            crewai_event_bus.emit(
                src, CrewKickoffStartedEvent(crew_name="BusCrew", inputs={"q": "x"})
            )
            crewai_event_bus.flush()
            # The Noveum handler created the crew span via the bus (crewAI's own
            # console listener may log an unrelated error on the stub source).
            assert "crew-bus-1" in lnr._crew_spans

            crewai_event_bus.emit(
                src, CrewKickoffCompletedEvent(crew_name="BusCrew", output="done")
            )
            crewai_event_bus.flush()
            assert "crew-bus-1" not in lnr._crew_spans  # closed
        finally:
            lnr.shutdown()


# ===========================================================================
# End-to-end kickoff
# ===========================================================================


class TestE2EKickoff:
    def test_e2e_minimal_crew_builds_span_tree(
        self, client_with_mocked_transport: Any
    ) -> None:
        import noveum_trace

        client = client_with_mocked_transport
        lnr = NoveumCrewAIListener(client)
        try:
            agent = Agent(
                role="Solver",
                goal="Solve the problem",
                backstory="You solve problems.",
                llm=_FakeLLM(),
                verbose=False,
            )
            task = Task(
                description="What is 6 times 7?",
                expected_output="A number",
                agent=agent,
            )
            crew = Crew(agents=[agent], tasks=[task], verbose=False)
            result = crew.kickoff()
            crewai_event_bus.flush()
            noveum_trace.flush()

            assert "42" in str(result)

            spans = _exported_spans(client)
            crew_spans = _by_name(spans, "crewai.crew")
            task_spans = _by_name(spans, "crewai.task")
            agent_spans = _by_name(spans, "crewai.agent")
            assert (
                crew_spans
            ), f"no crew span; got {[getattr(s,'name',None) for s in spans]}"
            assert task_spans, "no task span"
            assert agent_spans, "no agent span"

            crew_span = crew_spans[0]
            task_span = task_spans[0]
            agent_span = agent_spans[0]

            # Nesting: task is a child of crew, agent a child of task.
            assert task_span.parent_span_id == crew_span.span_id
            assert agent_span.parent_span_id == task_span.span_id

            # All closed successfully.
            assert _status(crew_span) == "ok"
            assert _status(task_span) == "ok"
            assert _status(agent_span) == "ok"
        finally:
            lnr.shutdown()

    def test_e2e_task_error_propagates_to_crew(
        self, client_with_mocked_transport: Any
    ) -> None:
        import noveum_trace

        client = client_with_mocked_transport
        lnr = NoveumCrewAIListener(client)
        try:
            agent = Agent(
                role="Faulty",
                goal="Fail",
                backstory="You always error.",
                llm=_FakeLLM(raises=True),
                max_iter=1,
                verbose=False,
            )
            task = Task(
                description="Do the impossible.",
                expected_output="nothing",
                agent=agent,
            )
            crew = Crew(agents=[agent], tasks=[task], verbose=False)
            with pytest.raises(Exception):  # noqa: B017 - crewAI wraps the cause
                crew.kickoff()
            crewai_event_bus.flush()
            noveum_trace.flush()

            spans = _exported_spans(client)
            crew_spans = _by_name(spans, "crewai.crew")
            # A crew span must be exported and reflect the failure, not "ok".
            assert crew_spans, "expected a crew span export on task failure"
            assert _status(crew_spans[0]) == "error"
        finally:
            lnr.shutdown()
