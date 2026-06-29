"""Tests for additive OpenTelemetry fields in the exported trace payload."""

from unittest.mock import patch

from noveum_trace.core.config import Config
from noveum_trace.core.span import SpanStatus
from noveum_trace.core.trace import Trace
from noveum_trace.transport.http_transport import HttpTransport
from noveum_trace.utils import otel_compat


def _build_trace():
    """Real Trace with one finished LLM span (Mocks would be stripped by transport)."""
    trace = Trace(name="test-trace")
    span = trace.create_span(
        name="llm.chat",
        attributes={
            "llm.model": "claude-sonnet-4",
            "llm.provider": "anthropic",
            "llm.input_tokens": 10,
            "llm.output_tokens": 5,
        },
    )
    span.set_status(SpanStatus.OK)
    span.finish()
    return trace, span


def _make_transport(config):
    with patch("noveum_trace.transport.http_transport.BatchProcessor"):
        return HttpTransport(config)


class TestOtelExportAdditive:
    def test_legacy_fields_preserved(self):
        transport = _make_transport(Config.create(project="p", environment="prod"))
        trace, span = _build_trace()
        result = transport._format_trace_for_export(trace)

        out_span = result["spans"][0]
        assert out_span["span_id"] == span.span_id  # unchanged UUID
        assert out_span["status"] == "ok"
        assert out_span["attributes"]["llm.model"] == "claude-sonnet-4"
        assert out_span["attributes"]["llm.input_tokens"] == 10

    def test_schema_version_and_resource_added(self):
        transport = _make_transport(Config.create(project="p", environment="prod"))
        trace, _ = _build_trace()
        result = transport._format_trace_for_export(trace)

        assert result["schema_version"] == otel_compat.SCHEMA_VERSION
        assert result["otel"]["trace_id"] == trace.trace_id.replace("-", "")
        assert result["otel"]["resource"]["service.name"] == "p"
        assert result["otel"]["resource"]["deployment.environment"] == "prod"

    def test_gen_ai_mirrored_beside_llm(self):
        transport = _make_transport(Config.create())
        trace, _ = _build_trace()
        result = transport._format_trace_for_export(trace)

        attrs = result["spans"][0]["attributes"]
        assert attrs["gen_ai.request.model"] == "claude-sonnet-4"
        assert attrs["gen_ai.provider.name"] == "anthropic"
        assert attrs["gen_ai.usage.input_tokens"] == 10
        assert attrs["llm.model"] == "claude-sonnet-4"  # legacy still present

    def test_per_span_otel_block(self):
        transport = _make_transport(Config.create())
        trace, span = _build_trace()
        result = transport._format_trace_for_export(trace)

        span_otel = result["spans"][0]["otel"]
        assert span_otel["span_id"] == span.span_id.replace("-", "")[:16]
        assert len(span_otel["span_id"]) == 16
        assert span_otel["parent_span_id"] is None
        assert span_otel["kind"] == "CLIENT"
        assert span_otel["status"] == {"code": "OK"}
        assert span_otel["name"] == "chat claude-sonnet-4"
        assert span_otel["scope"]["name"] == otel_compat.DEFAULT_SCOPE_NAME

    def test_disabled_emits_legacy_payload_only(self):
        transport = _make_transport(Config.create(otel_compat=False))
        trace, _ = _build_trace()
        result = transport._format_trace_for_export(trace)

        assert "schema_version" not in result
        assert "otel" not in result
        assert "otel" not in result["spans"][0]
        assert "gen_ai.request.model" not in result["spans"][0]["attributes"]
