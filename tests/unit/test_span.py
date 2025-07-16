"""
Unit tests for Span class.
"""

from noveum_trace.core.span import Span
from noveum_trace.types import (
    AISystem,
    LLMRequest,
    LLMResponse,
    OperationType,
    SpanKind,
    SpanStatus,
)


class TestSpan:
    """Test cases for Span class."""

    def test_span_creation(self):
        """Test basic span creation."""
        span = Span("test_span")

        assert span.name == "test_span"
        assert span.kind == SpanKind.INTERNAL
        assert span.status == SpanStatus.UNSET
        assert span.is_recording is True
        assert span.is_ended is False
        assert span.span_id is not None
        assert span.trace_id is not None

    def test_span_with_attributes(self):
        """Test span creation with attributes."""
        attributes = {"user.id": "123", "operation": "test"}
        span = Span("test_span", attributes=attributes)

        assert span.get_attribute("user.id") == "123"
        assert span.get_attribute("operation") == "test"

    def test_set_attribute(self):
        """Test setting span attributes."""
        span = Span("test_span")

        span.set_attribute("key1", "value1")
        span.set_attribute("key2", 42)
        span.set_attribute("key3", True)

        assert span.get_attribute("key1") == "value1"
        assert span.get_attribute("key2") == 42
        assert span.get_attribute("key3") is True

    def test_set_status(self):
        """Test setting span status."""
        span = Span("test_span")

        span.set_status(SpanStatus.OK)
        assert span.status == SpanStatus.OK

        span.set_status("error", "Something went wrong")
        assert span.status == SpanStatus.ERROR
        assert span.get_attribute("status.description") == "Something went wrong"

    def test_add_event(self):
        """Test adding events to span."""
        span = Span("test_span")

        span.add_event("test_event", {"key": "value"})

        span_data = span.get_span_data()
        assert len(span_data.events) == 1
        assert span_data.events[0]["name"] == "test_event"
        assert span_data.events[0]["attributes"]["key"] == "value"

    def test_add_link(self):
        """Test adding links to span."""
        span = Span("test_span")

        span.add_link("other_trace_id", "other_span_id", {"link_attr": "value"})

        span_data = span.get_span_data()
        assert len(span_data.links) == 1
        assert span_data.links[0]["span_id"] == "other_span_id"
        assert span_data.links[0]["trace_id"] == "other_trace_id"

    def test_record_exception(self):
        """Test recording exceptions."""
        span = Span("test_span")

        try:
            raise ValueError("Test error")
        except ValueError as e:
            span.record_exception(e)

        assert span.status == SpanStatus.ERROR
        span_data = span.get_span_data()
        assert len(span_data.events) == 1
        assert span_data.events[0]["name"] == "exception"
        assert "ValueError" in span_data.events[0]["attributes"]["exception.type"]

    def test_llm_request_response(self):
        """Test LLM request and response handling."""
        span = Span("llm_span")

        # Set LLM request
        request = LLMRequest(
            operation_type=OperationType.CHAT,
            ai_system=AISystem.OPENAI,
            model="gpt-4",
            prompt="Hello, world!",
        )
        span.set_llm_request(request)

        # Set LLM response
        response = LLMResponse(
            id="chatcmpl-test",
            model="gpt-4",
            choices=[
                {
                    "message": {
                        "content": "Hello! How can I help you?",
                        "role": "assistant",
                    },
                    "finish_reason": "stop",
                }
            ],
        )
        span.set_llm_response(response)

        # Check attributes
        assert span.get_attribute("gen_ai.operation.name") == "chat"
        assert span.get_attribute("gen_ai.system") == "openai"
        assert span.get_attribute("gen_ai.request.model") == "gpt-4"

    def test_span_end(self):
        """Test span ending."""
        span = Span("test_span")

        assert span.is_ended is False
        span.end()
        assert span.is_ended is True
        assert span.is_recording is False
        assert span.end_time is not None

    def test_span_context_manager(self):
        """Test span as context manager."""
        with Span("test_span") as span:
            assert span.is_recording is True
            span.set_attribute("test", "value")

        assert span.is_ended is True

    def test_span_duration(self):
        """Test span duration calculation."""
        span = Span("test_span")
        span.end()

        duration = span.duration_ms
        assert duration is not None
        assert duration >= 0
