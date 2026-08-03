"""
Unit tests for the OpenAI Agents SDK trace processor.

These tests fake the OpenAI Agents ``Trace`` / ``Span`` payloads and the Noveum
client, so the suite runs on every supported Python version *without* the
optional ``openai-agents`` dependency installed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

_src = Path(__file__).parents[3] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from noveum_trace.integrations.openai_agents import (  # noqa: E402
    processor as processor_module,
)
from noveum_trace.integrations.openai_agents.processor import (  # noqa: E402
    NoveumTraceProcessor,
    setup_openai_agents_tracing,
)

# ---------------------------------------------------------------------------
# Fakes — Noveum side
# ---------------------------------------------------------------------------


def _make_noveum_span(span_id: str, trace_id: str = "ntrace-1") -> MagicMock:
    span = MagicMock()
    span.span_id = span_id
    span.trace_id = trace_id
    span.attributes = {}
    span.set_attributes = MagicMock(
        side_effect=lambda attrs: span.attributes.update(attrs)
    )
    span.set_status = MagicMock()
    span.finish = MagicMock()
    span.is_finished = MagicMock(return_value=False)
    return span


def _make_noveum_trace(trace_id: str = "ntrace-1") -> MagicMock:
    trace = MagicMock()
    trace.trace_id = trace_id
    counter = {"n": 0}

    def _create_span(name, parent_span_id=None, attributes=None, start_time=None):
        counter["n"] += 1
        span = _make_noveum_span(f"nspan-{counter['n']}", trace_id=trace_id)
        span.name = name
        span.parent_span_id = parent_span_id
        if attributes:
            span.attributes.update(attributes)
        return span

    trace.create_span = MagicMock(side_effect=_create_span)
    trace.finish = MagicMock()
    return trace


def _make_client(trace_id: str = "ntrace-1") -> MagicMock:
    client = MagicMock()
    trace = _make_noveum_trace(trace_id)
    client.start_trace = MagicMock(return_value=trace)
    client.finish_trace = MagicMock()
    client.flush = MagicMock()
    client._trace = trace
    return client


# ---------------------------------------------------------------------------
# Fakes — OpenAI Agents side
# ---------------------------------------------------------------------------


def _oai_trace(
    trace_id: str = "oai-trace-1",
    name: str = "my-workflow",
    group_id=None,
    metadata=None,
) -> SimpleNamespace:
    return SimpleNamespace(
        trace_id=trace_id, name=name, group_id=group_id, metadata=metadata
    )


def _oai_span(
    span_id: str,
    span_data,
    trace_id: str = "oai-trace-1",
    parent_id=None,
    started_at: str = "2026-08-03T12:00:00+00:00",
    ended_at: str = "2026-08-03T12:00:01+00:00",
    error=None,
) -> SimpleNamespace:
    return SimpleNamespace(
        span_id=span_id,
        trace_id=trace_id,
        parent_id=parent_id,
        started_at=started_at,
        ended_at=ended_at,
        error=error,
        span_data=span_data,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_module_imports_without_optional_dependency(self) -> None:
        assert hasattr(processor_module, "OPENAI_AGENTS_AVAILABLE")
        assert callable(NoveumTraceProcessor)

    def test_privacy_safe_defaults(self) -> None:
        proc = NoveumTraceProcessor()
        assert proc.capture_inputs is False
        assert proc.capture_outputs is False
        assert proc.capture_llm_messages is False
        assert proc.capture_tool_schemas is True
        assert proc.capture_trace_metadata is True


# ---------------------------------------------------------------------------
# Trace / span mapping
# ---------------------------------------------------------------------------


class TestTraceMapping:
    def test_on_trace_start_opens_noveum_trace(self) -> None:
        client = _make_client()
        proc = NoveumTraceProcessor(client=client)
        proc.on_trace_start(_oai_trace(name="wf", group_id="g1"))

        client.start_trace.assert_called_once()
        kwargs = client.start_trace.call_args.kwargs
        assert kwargs["name"] == "wf"
        assert kwargs["set_as_current"] is False
        assert kwargs["attributes"]["openai_agents.workflow_name"] == "wf"
        assert kwargs["attributes"]["openai_agents.group_id"] == "g1"

    def test_on_trace_end_finishes_and_exports(self) -> None:
        client = _make_client()
        proc = NoveumTraceProcessor(client=client)
        oai = _oai_trace()
        proc.on_trace_start(oai)
        proc.on_trace_end(oai)
        client.finish_trace.assert_called_once_with(client._trace)

    def test_unnamed_workflow_uses_prefix(self) -> None:
        client = _make_client()
        proc = NoveumTraceProcessor(client=client)
        proc.on_trace_start(_oai_trace(name=""))
        assert client.start_trace.call_args.kwargs["name"] == "openai_agents.workflow"


class TestSpanMapping:
    def test_parent_child_linkage(self) -> None:
        client = _make_client()
        proc = NoveumTraceProcessor(client=client)
        proc.on_trace_start(_oai_trace())

        parent_sd = SimpleNamespace(
            type="agent", name="triage", handoffs=None, tools=None, output_type=None
        )
        proc.on_span_start(_oai_span("oai-parent", parent_sd))

        child_sd = SimpleNamespace(
            type="generation",
            model="gpt-4o",
            usage={"input_tokens": 1, "output_tokens": 2},
        )
        proc.on_span_start(_oai_span("oai-child", child_sd, parent_id="oai-parent"))

        trace = client._trace
        assert trace.create_span.call_count == 2
        parent_call = trace.create_span.call_args_list[0]
        child_call = trace.create_span.call_args_list[1]
        assert parent_call.kwargs["parent_span_id"] is None
        assert child_call.kwargs["parent_span_id"] == "nspan-1"

    def test_generation_attributes_mapped(self) -> None:
        client = _make_client()
        proc = NoveumTraceProcessor(client=client)
        proc.on_trace_start(_oai_trace())

        sd = SimpleNamespace(
            type="generation",
            model="gpt-4o",
            usage={"input_tokens": 10, "output_tokens": 5},
            model_config={"temperature": 0.7, "top_p": 0.9},
        )
        span = _oai_span("s1", sd)
        proc.on_span_start(span)
        nspan = proc._spans["s1"]
        proc.on_span_end(span)

        assert nspan.attributes["llm.model"] == "gpt-4o"
        assert nspan.attributes["llm.provider"] == "openai"
        assert nspan.attributes["llm.input_tokens"] == 10
        assert nspan.attributes["llm.output_tokens"] == 5
        assert nspan.attributes["llm.total_tokens"] == 15
        assert nspan.attributes["llm.temperature"] == 0.7
        assert nspan.attributes["openai_agents.status"] == "ok"
        nspan.finish.assert_called_once()

    def test_handoff_attributes_mapped(self) -> None:
        client = _make_client()
        proc = NoveumTraceProcessor(client=client)
        proc.on_trace_start(_oai_trace())
        sd = SimpleNamespace(type="handoff", from_agent="triage", to_agent="billing")
        span = _oai_span("s1", sd)
        proc.on_span_start(span)
        nspan = proc._spans["s1"]
        proc.on_span_end(span)
        assert nspan.attributes["handoff.from_agent"] == "triage"
        assert nspan.attributes["handoff.to_agent"] == "billing"


# ---------------------------------------------------------------------------
# Capture flags (privacy)
# ---------------------------------------------------------------------------


class TestCaptureFlags:
    @pytest.mark.parametrize("capture", [False, True])
    def test_function_io_gated_by_flags(self, capture: bool) -> None:
        client = _make_client()
        proc = NoveumTraceProcessor(
            client=client, capture_inputs=capture, capture_outputs=capture
        )
        proc.on_trace_start(_oai_trace())
        sd = SimpleNamespace(
            type="function", name="get_weather", input='{"city":"SF"}', output="sunny"
        )
        span = _oai_span("s1", sd)
        proc.on_span_start(span)
        nspan = proc._spans["s1"]
        proc.on_span_end(span)

        assert nspan.attributes["tool.name"] == "get_weather"
        assert ("tool.input" in nspan.attributes) is capture
        assert ("tool.output" in nspan.attributes) is capture

    @pytest.mark.parametrize("capture", [False, True])
    def test_llm_messages_gated_by_flag(self, capture: bool) -> None:
        client = _make_client()
        proc = NoveumTraceProcessor(client=client, capture_llm_messages=capture)
        proc.on_trace_start(_oai_trace())
        sd = SimpleNamespace(
            type="generation",
            model="gpt-4o",
            usage=None,
            input=[{"role": "user", "content": "hi"}],
            output=[{"role": "assistant", "content": "hello"}],
        )
        span = _oai_span("s1", sd)
        proc.on_span_start(span)
        nspan = proc._spans["s1"]
        proc.on_span_end(span)

        assert ("llm.input" in nspan.attributes) is capture
        assert ("llm.output" in nspan.attributes) is capture


# ---------------------------------------------------------------------------
# Errors, flush, shutdown, resilience
# ---------------------------------------------------------------------------


class TestErrorsAndLifecycle:
    def test_span_error_sets_status(self) -> None:
        client = _make_client()
        proc = NoveumTraceProcessor(client=client)
        proc.on_trace_start(_oai_trace())
        sd = SimpleNamespace(type="function", name="boom", input=None, output=None)
        span = _oai_span("s1", sd, error={"message": "kaboom", "data": {"k": "v"}})
        proc.on_span_start(span)
        nspan = proc._spans["s1"]
        proc.on_span_end(span)

        assert nspan.attributes["openai_agents.status"] == "error"
        assert nspan.attributes["error.message"] == "kaboom"
        nspan.set_status.assert_called_once()

    def test_force_flush_and_shutdown(self) -> None:
        client = _make_client()
        proc = NoveumTraceProcessor(client=client)

        proc.force_flush()
        assert client.flush.call_count == 1

        proc.shutdown()
        assert proc._is_shutdown is True
        assert client.flush.call_count == 2

        client.flush.reset_mock()
        proc.force_flush()
        client.flush.assert_not_called()

    def test_handlers_never_raise(self) -> None:
        proc = NoveumTraceProcessor(client=_make_client())
        # Span callbacks for an unknown trace must be silent no-ops.
        proc.on_span_start(
            SimpleNamespace(
                trace_id="unknown",
                span_id="y",
                parent_id=None,
                started_at=None,
                ended_at=None,
                error=None,
                span_data=None,
            )
        )
        proc.on_span_end(
            SimpleNamespace(span_id="y", ended_at=None, span_data=None, error=None)
        )
        proc.on_trace_end(SimpleNamespace(trace_id="unknown"))

    def test_client_failure_is_non_fatal(self) -> None:
        boom = MagicMock()
        boom.start_trace.side_effect = RuntimeError("nope")
        proc = NoveumTraceProcessor(client=boom)
        proc.on_trace_start(_oai_trace())  # must not raise

    def test_uninitialised_sdk_is_noop(self) -> None:
        proc = NoveumTraceProcessor()  # no client; global SDK not initialised
        proc.on_trace_start(_oai_trace())  # must not raise


class TestSetupFactory:
    def test_setup_rejects_api_key(self) -> None:
        if processor_module.OPENAI_AGENTS_AVAILABLE:
            with pytest.raises(TypeError):
                setup_openai_agents_tracing(api_key="x")
        else:
            with pytest.raises(ImportError):
                setup_openai_agents_tracing(api_key="x")
