"""
Basic functionality tests for Noveum Trace SDK.

These tests verify that the core components work correctly
and can be used for basic tracing operations.
"""

import pytest

from noveum_trace.core.config import Config
from noveum_trace.core.span import Span, SpanStatus
from noveum_trace.core.trace import Trace
from noveum_trace.utils.exceptions import ConfigurationError, NoveumTraceError


class TestSpan:
    """Test the Span class."""

    def test_span_creation(self):
        """Test basic span creation."""
        span = Span(name="test_span", trace_id="trace_123", span_id="span_456")

        assert span.name == "test_span"
        assert span.trace_id == "trace_123"
        assert span.span_id == "span_456"
        assert span.status == SpanStatus.UNSET
        assert not span.is_finished()

    def test_span_attributes(self):
        """Test span attribute management."""
        span = Span("test", "trace_123")

        # Test setting single attribute
        span.set_attribute("key1", "value1")
        assert span.attributes["key1"] == "value1"

        # Test setting multiple attributes
        span.set_attributes({"key2": "value2", "key3": 123})
        assert span.attributes["key2"] == "value2"
        assert span.attributes["key3"] == 123

    def test_span_events(self):
        """Test span event management."""
        span = Span("test", "trace_123")

        # Add event
        span.add_event("test_event", {"detail": "test"})

        assert len(span.events) == 1
        assert span.events[0].name == "test_event"
        assert span.events[0].attributes["detail"] == "test"

    def test_span_status(self):
        """Test span status management."""
        span = Span("test", "trace_123")

        # Set status
        span.set_status(SpanStatus.OK, "Success")

        assert span.status == SpanStatus.OK
        assert span.status_message == "Success"

    def test_span_exception_handling(self):
        """Test span exception recording."""
        span = Span("test", "trace_123")

        # Record exception
        try:
            raise ValueError("Test error")
        except ValueError as e:
            span.record_exception(e)

        assert span.exception is not None
        assert span.attributes["exception.type"] == "ValueError"
        assert span.attributes["exception.message"] == "Test error"
        assert len(span.events) == 1  # Exception event

    def test_span_context_manager(self):
        """Test span as context manager."""
        span = Span("test", "trace_123")

        with span:
            span.set_attribute("inside", True)

        assert span.is_finished()
        assert span.status == SpanStatus.OK
        assert span.attributes["inside"] is True

    def test_span_context_manager_with_exception(self):
        """Test span context manager with exception."""
        span = Span("test", "trace_123")

        with pytest.raises(ValueError):
            with span:
                raise ValueError("Test error")

        assert span.is_finished()
        assert span.status == SpanStatus.ERROR
        assert span.exception is not None

    def test_span_serialization(self):
        """Test span to_dict and from_dict."""
        original_span = Span("test", "trace_123", "span_456")
        original_span.set_attribute("key", "value")
        original_span.add_event("event", {"data": "test"})
        original_span.finish()

        # Convert to dict
        span_dict = original_span.to_dict()

        # Verify dict structure
        assert span_dict["name"] == "test"
        assert span_dict["trace_id"] == "trace_123"
        assert span_dict["span_id"] == "span_456"
        assert span_dict["attributes"]["key"] == "value"
        assert len(span_dict["events"]) == 1

        # Convert back to span
        restored_span = Span.from_dict(span_dict)

        assert restored_span.name == original_span.name
        assert restored_span.trace_id == original_span.trace_id
        assert restored_span.span_id == original_span.span_id
        assert restored_span.attributes == original_span.attributes


class TestTrace:
    """Test the Trace class."""

    def test_trace_creation(self):
        """Test basic trace creation."""
        trace = Trace("test_trace", "trace_123")

        assert trace.name == "test_trace"
        assert trace.trace_id == "trace_123"
        assert trace.status == SpanStatus.UNSET
        assert not trace._finished
        assert len(trace.spans) == 0

    def test_trace_span_creation(self):
        """Test creating spans within a trace."""
        trace = Trace("test_trace")

        # Create a span
        span = trace.create_span("test_span")

        assert span.name == "test_span"
        assert span.trace_id == trace.trace_id
        assert span in trace.spans
        assert trace.span_count == 1

    def test_trace_context_manager(self):
        """Test trace as context manager."""
        trace = Trace("test_trace")

        with trace:
            span = trace.create_span("test_span")
            span.set_attribute("test", True)

        assert trace._finished
        assert trace.status == SpanStatus.OK

    def test_trace_context_manager_with_exception(self):
        """Test trace context manager with exception."""
        trace = Trace("test_trace")

        with pytest.raises(ValueError):
            with trace:
                raise ValueError("Test error")

        assert trace._finished
        assert trace.status == SpanStatus.ERROR


class TestConfig:
    """Test the Config class."""

    def test_config_creation(self):
        """Test basic config creation."""
        config = Config()

        assert config.environment == "development"
        assert config.tracing.enabled is True
        assert config.tracing.sample_rate == 1.0
        assert config.transport.endpoint == "https://api.noveum.ai"

    def test_config_with_endpoint_parameter(self):
        """Test config creation with endpoint parameter."""
        custom_endpoint = "http://localhost:8082/api/v1"
        config = Config.create(
            api_key="test-config-key",
            project="config-test-project",
            endpoint=custom_endpoint,
        )

        assert config.api_key == "test-config-key"
        assert config.project == "config-test-project"
        assert config.transport.endpoint == custom_endpoint

    def test_config_validation(self):
        """Test config validation."""
        # Test invalid sample rate
        with pytest.raises(ConfigurationError):
            config = Config()
            config.tracing.sample_rate = 1.5  # Invalid
            config._validate()

    def test_config_from_dict_with_top_level_endpoint(self):
        """Test creating config from dictionary with top-level endpoint."""
        config_data = {
            "project": "test_project",
            "api_key": "test_key",
            "environment": "production",
            "endpoint": "http://localhost:8082/api/v1",
            "tracing": {
                "sample_rate": 0.5,
                "capture_errors": False,
            },
        }

        config = Config.from_dict(config_data)

        assert config.project == "test_project"
        assert config.api_key == "test_key"
        assert config.environment == "production"
        assert config.transport.endpoint == "http://localhost:8082/api/v1"
        assert config.tracing.sample_rate == 0.5
        assert config.tracing.capture_errors is False

    def test_config_from_dict_with_transport_endpoint(self):
        """Test creating config from dictionary with transport.endpoint."""
        config_data = {
            "project": "test_project",
            "api_key": "test_key",
            "environment": "production",
            "transport": {
                "endpoint": "http://localhost:8082/api/v1",
                "timeout": 60,
            },
        }

        config = Config.from_dict(config_data)

        assert config.project == "test_project"
        assert config.api_key == "test_key"
        assert config.environment == "production"
        assert config.transport.endpoint == "http://localhost:8082/api/v1"
        assert config.transport.timeout == 60

    def test_config_from_dict_endpoint_precedence(self):
        """Test that top-level endpoint takes precedence over transport.endpoint."""
        config_data = {
            "project": "test_project",
            "api_key": "test_key",
            "endpoint": "http://localhost:8082/api/v1",  # Top-level - should take precedence
            "transport": {
                "endpoint": "http://localhost:9000/api/v1",  # Should be overridden
                "timeout": 60,
            },
        }

        config = Config.from_dict(config_data)

        assert (
            config.transport.endpoint == "http://localhost:8082/api/v1"
        )  # Top-level endpoint wins

    def test_config_from_dict_only_top_level_endpoint(self):
        """Test config when only top-level endpoint is provided (no transport section)."""
        config_data = {
            "project": "test_project",
            "api_key": "test_key",
            "endpoint": "http://localhost:8082/api/v1",
        }

        config = Config.from_dict(config_data)

        assert config.project == "test_project"
        assert config.api_key == "test_key"
        assert config.transport.endpoint == "http://localhost:8082/api/v1"

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_data = {
            "project": "test_project",
            "api_key": "test_key",
            "environment": "production",
            "tracing": {
                "sample_rate": 0.5,
                "capture_errors": False,
            },
            "transport": {
                "endpoint": "https://custom.endpoint.com",
                "timeout": 60,
            },
        }

        config = Config.from_dict(config_data)

        assert config.project == "test_project"
        assert config.api_key == "test_key"
        assert config.environment == "production"
        assert config.tracing.sample_rate == 0.5
        assert config.tracing.capture_errors is False
        assert config.transport.endpoint == "https://custom.endpoint.com"
        assert config.transport.timeout == 60

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = Config()
        config.project = "test_project"
        config.api_key = "test_key"

        config_dict = config.to_dict()

        assert config_dict["project"] == "test_project"
        assert config_dict["api_key"] == "test_key"
        assert "tracing" in config_dict
        assert "transport" in config_dict


class TestBasicImports:
    """Test that basic imports work correctly."""

    def test_main_package_import(self):
        """Test importing the main package."""
        import noveum_trace

        # Should not raise any exceptions
        assert noveum_trace is not None

    def test_decorator_imports(self):
        """Test importing decorators."""
        from noveum_trace import trace, trace_agent, trace_llm, trace_tool

        # Should not raise any exceptions
        assert trace is not None
        assert trace_llm is not None
        assert trace_agent is not None
        assert trace_tool is not None

    def test_core_imports(self):
        """Test importing core classes."""
        from noveum_trace.core.config import Config
        from noveum_trace.core.span import Span
        from noveum_trace.core.trace import Trace

        # Should not raise any exceptions
        assert Span is not None
        assert Trace is not None
        assert Config is not None

    def test_utils_imports(self):
        """Test importing utility classes."""
        from noveum_trace.utils.llm_utils import detect_llm_provider

        # Should not raise any exceptions
        assert NoveumTraceError is not None
        assert detect_llm_provider is not None


class TestUtilities:
    """Test utility functions."""

    def test_llm_provider_detection(self):
        """Test LLM provider detection."""
        from noveum_trace.utils.llm_utils import detect_llm_provider

        # Test OpenAI detection
        openai_args = {"model": "gpt-4", "messages": []}
        assert detect_llm_provider(openai_args) == "openai"

        # Test Anthropic detection
        anthropic_args = {"model": "claude-3-opus-20240229"}
        assert detect_llm_provider(anthropic_args) == "anthropic"

        # Test unknown provider
        unknown_args = {"data": "test"}
        assert detect_llm_provider(unknown_args) is None

    def test_token_estimation(self):
        """Test token count estimation."""
        from noveum_trace.utils.llm_utils import estimate_token_count

        # Test simple text
        text = "Hello, world!"
        tokens = estimate_token_count(text)
        assert tokens > 0
        assert isinstance(tokens, int)

        # Test messages format
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        tokens = estimate_token_count(messages)
        assert tokens > 0

    def test_pii_redaction(self):
        """Test PII redaction functionality."""
        from noveum_trace.utils.pii_redaction import detect_pii_types, redact_pii

        # Test email redaction
        text_with_email = "Contact me at john@example.com"
        redacted = redact_pii(text_with_email)
        assert "john@example.com" not in redacted
        assert "[EMAIL_REDACTED]" in redacted

        # Test PII detection
        pii_types = detect_pii_types(text_with_email)
        assert "email" in pii_types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
