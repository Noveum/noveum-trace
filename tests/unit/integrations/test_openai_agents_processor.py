"""
Unit tests for the OpenAI Agents SDK trace processor.

These tests fake the OpenAI Agents ``Trace`` / ``Span`` payloads and the Noveum
client, so the suite runs on every supported Python version *without* the
optional ``openai-agents`` dependency installed.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

_src = Path(__file__).parents[3] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from noveum_trace.integrations._common import coerce_datetime  # noqa: E402

# isort: off
from noveum_trace.integrations.openai_agents import (  # noqa: E402
    processor as processor_module,
)

# isort: on
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

    def test_captures_everything_by_default(self) -> None:
        proc = NoveumTraceProcessor()
        assert proc.capture_inputs is True
        assert proc.capture_outputs is True
        assert proc.capture_llm_messages is True
        assert proc.capture_tool_schemas is True
        assert proc.capture_trace_metadata is True
        assert proc.capture_cost is True

    def test_capture_flags_can_be_disabled(self) -> None:
        proc = NoveumTraceProcessor(
            capture_inputs=False,
            capture_outputs=False,
            capture_llm_messages=False,
            capture_tool_schemas=False,
            capture_trace_metadata=False,
            capture_cost=False,
        )
        assert proc.capture_inputs is False
        assert proc.capture_outputs is False
        assert proc.capture_llm_messages is False
        assert proc.capture_tool_schemas is False
        assert proc.capture_trace_metadata is False
        assert proc.capture_cost is False


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

    def test_response_llm_content_gated_by_llm_messages(self) -> None:
        # Response-span input/output are LLM content, gated on
        # capture_llm_messages — NOT on capture_inputs / capture_outputs.
        response = SimpleNamespace(
            model="gpt-4o", id="resp_1", usage=None, output_text="the answer"
        )
        sd = SimpleNamespace(type="response", response=response, input="the prompt")

        # capture_inputs/outputs on, capture_llm_messages off -> no LLM content
        client = _make_client()
        proc = NoveumTraceProcessor(
            client=client,
            capture_inputs=True,
            capture_outputs=True,
            capture_llm_messages=False,
        )
        proc.on_trace_start(_oai_trace())
        proc.on_span_start(_oai_span("s1", sd))
        nspan = proc._spans["s1"]
        proc.on_span_end(_oai_span("s1", sd))
        assert nspan.attributes["llm.model"] == "gpt-4o"
        assert "llm.input" not in nspan.attributes
        assert "llm.output" not in nspan.attributes

        # capture_llm_messages on -> LLM content captured
        client2 = _make_client()
        proc2 = NoveumTraceProcessor(client=client2, capture_llm_messages=True)
        proc2.on_trace_start(_oai_trace())
        proc2.on_span_start(_oai_span("s2", sd))
        nspan2 = proc2._spans["s2"]
        proc2.on_span_end(_oai_span("s2", sd))
        assert nspan2.attributes["llm.input"] == "the prompt"
        assert nspan2.attributes["llm.output"] == "the answer"


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


# ---------------------------------------------------------------------------
# Full-payload capture (no truncation, richer LLM/agent/tool attributes)
# ---------------------------------------------------------------------------


class TestFullPayloadCapture:
    def test_long_payloads_are_not_truncated(self) -> None:
        huge = "x" * 50_000
        sd = SimpleNamespace(type="function", name="tool", input=huge, output=huge)
        client = _make_client()
        proc = NoveumTraceProcessor(client=client)
        proc.on_trace_start(_oai_trace())
        proc.on_span_start(_oai_span("s1", sd))
        nspan = proc._spans["s1"]
        proc.on_span_end(_oai_span("s1", sd))

        assert nspan.attributes["tool.input"] == huge
        assert nspan.attributes["tool.output"] == huge
        assert "…" not in nspan.attributes["tool.output"]

    def test_long_system_prompt_survives_intact(self) -> None:
        prompt = "system instructions " * 2_000
        sd = SimpleNamespace(
            type="generation",
            model="gpt-4o",
            model_config=None,
            usage=None,
            input=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "hi"},
            ],
            output=[{"role": "assistant", "content": "hello"}],
        )
        client = _make_client()
        proc = NoveumTraceProcessor(client=client)
        proc.on_trace_start(_oai_trace())
        proc.on_span_start(_oai_span("s1", sd))
        nspan = proc._spans["s1"]
        proc.on_span_end(_oai_span("s1", sd))

        assert nspan.attributes["llm.system_prompt"] == prompt
        assert nspan.attributes["llm.input_text"] == "user: hi"
        assert nspan.attributes["llm.output_text"] == "assistant: hello"

    def test_generation_captures_tool_calls(self) -> None:
        sd = SimpleNamespace(
            type="generation",
            model="gpt-4o",
            model_config={"temperature": 0.2, "reasoning": {"effort": "high"}},
            usage=None,
            input=[{"role": "user", "content": "weather?"}],
            output=[
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"Paris"}',
                            },
                        }
                    ],
                }
            ],
        )
        client = _make_client()
        proc = NoveumTraceProcessor(client=client)
        proc.on_trace_start(_oai_trace())
        proc.on_span_start(_oai_span("s1", sd))
        nspan = proc._spans["s1"]
        proc.on_span_end(_oai_span("s1", sd))

        assert nspan.attributes["llm.tool_call_count"] == 1
        assert nspan.attributes["llm.tool_calls"][0]["name"] == "get_weather"
        assert nspan.attributes["llm.tool_calls"][0]["arguments"] == '{"city":"Paris"}'
        assert nspan.attributes["llm.temperature"] == 0.2
        assert nspan.attributes["llm.reasoning_effort"] == "high"

    def test_generation_captures_cache_and_reasoning_tokens(self) -> None:
        sd = SimpleNamespace(
            type="generation",
            model="gpt-4o",
            model_config=None,
            input=None,
            output=None,
            usage={
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "input_tokens_details": {
                    "cached_tokens": 64,
                    "cache_write_tokens": 8,
                },
                "output_tokens_details": {"reasoning_tokens": 12},
            },
        )
        client = _make_client()
        proc = NoveumTraceProcessor(client=client)
        proc.on_trace_start(_oai_trace())
        proc.on_span_start(_oai_span("s1", sd))
        nspan = proc._spans["s1"]
        proc.on_span_end(_oai_span("s1", sd))

        assert nspan.attributes["llm.cached_input_tokens"] == 64
        assert nspan.attributes["llm.cache_write_input_tokens"] == 8
        assert nspan.attributes["llm.reasoning_tokens"] == 12
        assert nspan.attributes["llm.cache_hit"] is True

    def test_response_span_flat_cache_usage(self) -> None:
        response = SimpleNamespace(
            model="gpt-4o", id="resp_1", usage=None, output_text="ok", output=[]
        )
        sd = SimpleNamespace(
            type="response",
            response=response,
            input=None,
            usage={
                "input_tokens": 10,
                "output_tokens": 2,
                "total_tokens": 12,
                "cached_input_tokens": 0,
                "cache_write_input_tokens": 0,
            },
        )
        client = _make_client()
        proc = NoveumTraceProcessor(client=client)
        proc.on_trace_start(_oai_trace())
        proc.on_span_start(_oai_span("s1", sd))
        nspan = proc._spans["s1"]
        proc.on_span_end(_oai_span("s1", sd))

        assert nspan.attributes["llm.cached_input_tokens"] == 0
        assert nspan.attributes["llm.cache_hit"] is False

    def test_response_span_captures_instructions_tools_and_reasoning(self) -> None:
        response = SimpleNamespace(
            model="gpt-4o",
            id="resp_1",
            usage=None,
            status="completed",
            temperature=0.5,
            top_p=1.0,
            max_output_tokens=512,
            instructions="You are a helpful weather bot.",
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Look up weather",
                    "parameters": {"type": "object"},
                }
            ],
            output=[
                {"type": "reasoning", "summary": [{"text": "The user wants weather."}]},
                {
                    "type": "function_call",
                    "name": "get_weather",
                    "call_id": "call_1",
                    "arguments": '{"city":"Paris"}',
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "It is sunny."}],
                },
            ],
            output_text="It is sunny.",
        )
        sd = SimpleNamespace(type="response", response=response, input="weather?")
        client = _make_client()
        proc = NoveumTraceProcessor(client=client)
        proc.on_trace_start(_oai_trace())
        proc.on_span_start(_oai_span("s1", sd))
        nspan = proc._spans["s1"]
        proc.on_span_end(_oai_span("s1", sd))

        attrs = nspan.attributes
        assert attrs["llm.system_prompt"] == "You are a helpful weather bot."
        assert attrs["llm.available_tool_count"] == 1
        assert attrs["llm.available_tools"][0]["name"] == "get_weather"
        assert attrs["llm.tool_calls"][0]["call_id"] == "call_1"
        assert attrs["llm.reasoning"] == "The user wants weather."
        assert attrs["llm.output_text"] == "It is sunny."
        assert attrs["llm.response_status"] == "completed"
        assert attrs["llm.temperature"] == 0.5
        assert attrs["llm.max_tokens"] == 512

    def test_agent_span_captures_metadata_and_tool_count(self) -> None:
        sd = SimpleNamespace(
            type="agent",
            name="Weather agent",
            handoffs=["billing"],
            tools=["get_weather", "get_forecast"],
            output_type="str",
            metadata={"team": "growth"},
        )
        client = _make_client()
        proc = NoveumTraceProcessor(client=client)
        proc.on_trace_start(_oai_trace())
        proc.on_span_start(_oai_span("s1", sd))
        nspan = proc._spans["s1"]
        proc.on_span_end(_oai_span("s1", sd))

        assert nspan.attributes["agent.tool_count"] == 2
        assert nspan.attributes["agent.metadata"] == {"team": "growth"}

    def test_function_span_flags_mcp_tool_calls(self) -> None:
        sd = SimpleNamespace(
            type="function",
            name="search",
            input="{}",
            output="result",
            mcp_data={"server": "docs"},
        )
        client = _make_client()
        proc = NoveumTraceProcessor(client=client)
        proc.on_trace_start(_oai_trace())
        proc.on_span_start(_oai_span("s1", sd))
        nspan = proc._spans["s1"]
        proc.on_span_end(_oai_span("s1", sd))

        assert nspan.attributes["tool.is_mcp"] is True
        assert nspan.attributes["tool.mcp_data"] == {"server": "docs"}


# ---------------------------------------------------------------------------
# Parent resolution
# ---------------------------------------------------------------------------


class TestParentResolution:
    def test_child_uses_openai_parent_id(self) -> None:
        client = _make_client()
        proc = NoveumTraceProcessor(client=client)
        proc.on_trace_start(_oai_trace())

        parent_sd = SimpleNamespace(type="agent", name="A")
        child_sd = SimpleNamespace(type="function", name="t", input=None, output=None)
        proc.on_span_start(_oai_span("p1", parent_sd))
        parent_noveum_id = proc._spans["p1"].span_id
        proc.on_span_start(_oai_span("c1", child_sd, parent_id="p1"))

        assert proc._spans["c1"].parent_span_id == parent_noveum_id

    def test_child_starting_after_parent_ends_keeps_parentage(self) -> None:
        # Parent lookup must survive the parent span closing: the openai-id to
        # noveum-id map lives for the whole trace, not just while the parent
        # span object is open.
        client = _make_client()
        proc = NoveumTraceProcessor(client=client)
        proc.on_trace_start(_oai_trace())

        parent_sd = SimpleNamespace(type="agent", name="A")
        child_sd = SimpleNamespace(type="function", name="t", input=None, output=None)
        proc.on_span_start(_oai_span("p1", parent_sd))
        parent_noveum_id = proc._spans["p1"].span_id
        proc.on_span_end(_oai_span("p1", parent_sd))

        proc.on_span_start(_oai_span("c1", child_sd, parent_id="p1"))
        assert proc._spans["c1"].parent_span_id == parent_noveum_id

    def test_unknown_parent_falls_back_to_trace_root(self) -> None:
        client = _make_client()
        proc = NoveumTraceProcessor(client=client)
        proc.on_trace_start(_oai_trace())
        sd = SimpleNamespace(type="function", name="t", input=None, output=None)
        proc.on_span_start(_oai_span("c1", sd, parent_id="never-seen"))
        assert proc._spans["c1"].parent_span_id is None

    def test_trace_end_clears_span_id_map(self) -> None:
        client = _make_client()
        proc = NoveumTraceProcessor(client=client)
        trace = _oai_trace()
        proc.on_trace_start(trace)
        sd = SimpleNamespace(type="agent", name="A")
        proc.on_span_start(_oai_span("p1", sd))
        proc.on_trace_end(trace)

        assert proc._noveum_span_ids == {}
        assert proc._trace_span_ids == {}
        assert proc._spans == {}


# ---------------------------------------------------------------------------
# Timestamp coercion
# ---------------------------------------------------------------------------


class TestTimestampCoercion:
    @pytest.mark.parametrize(
        "value",
        ["2026-08-03T12:00:00Z", "2026-08-03T12:00:00z", "2026-08-03T12:00:00+00:00"],
    )
    def test_utc_designator_forms_are_equivalent(self, value: str) -> None:
        # ``datetime.fromisoformat`` only accepts a trailing ``Z`` on 3.11+, so
        # without normalisation this silently returns None on 3.9/3.10.
        parsed = coerce_datetime(value)
        assert parsed == datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc)

    def test_unparseable_string_returns_none(self) -> None:
        assert coerce_datetime("not-a-timestamp") is None
        assert coerce_datetime("Z") is None

    def test_z_timestamps_preserve_span_timing(self) -> None:
        client = _make_client()
        proc = NoveumTraceProcessor(client=client)
        proc.on_trace_start(_oai_trace())

        sd = SimpleNamespace(type="agent", name="A")
        span = _oai_span(
            "s1",
            sd,
            started_at="2026-08-03T12:00:00Z",
            ended_at="2026-08-03T12:00:01Z",
        )
        proc.on_span_start(span)
        noveum_span = proc._spans["s1"]
        proc.on_span_end(span)

        start_time = client._trace.create_span.call_args.kwargs["start_time"]
        assert start_time == datetime(2026, 8, 3, 12, 0, 0, tzinfo=timezone.utc)
        noveum_span.finish.assert_called_once_with(
            datetime(2026, 8, 3, 12, 0, 1, tzinfo=timezone.utc)
        )
