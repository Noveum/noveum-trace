"""
Simplified endpoint validation tests that work with existing mocking infrastructure.

This module validates that trace data structure and content is correct by
inspecting the data passed to the mocked transport layer.
"""

import json
import time
from typing import Any

import pytest

from noveum_trace.core.client import NoveumClient
from noveum_trace.core.config import Config
from noveum_trace.core.span import SpanStatus


class TransportCapture:
    """Captures data sent to transport for validation"""

    def __init__(self):
        self.exported_traces: list[Any] = []
        self.export_calls: list[dict[str, Any]] = []

    def capture_export(self, original_export):
        """Wrap the original export method to capture data"""

        def wrapped_export(trace_data):
            # Store the trace data
            self.exported_traces.append(trace_data)

            # Store call details
            call_info = {"trace_data": trace_data, "timestamp": time.time()}
            self.export_calls.append(call_info)

            # Call original or return mock response
            try:
                return original_export(trace_data)
            except Exception:
                return {
                    "success": True,
                    "trace_id": f"captured-{len(self.exported_traces)}",
                }

        return wrapped_export

    def clear(self):
        """Clear captured data"""
        self.exported_traces.clear()
        self.export_calls.clear()

    def get_latest_trace(self):
        """Get the latest captured trace"""
        return self.exported_traces[-1] if self.exported_traces else None


@pytest.fixture
def transport_capture():
    """Provide a transport capture for validating exported data"""
    return TransportCapture()


@pytest.fixture
def client_with_capture(transport_capture):
    """Provide a client that captures transport exports"""
    config = Config.create(
        api_key="test-api-key", project="test-project", endpoint="https://api.noveum.ai"
    )

    client = NoveumClient(config=config)

    # Wrap the transport export method to capture data
    original_export = client.transport.export_trace
    client.transport.export_trace = transport_capture.capture_export(original_export)

    yield client

    # Clean up
    try:
        client.shutdown()
    except Exception:
        pass


class TestTransportValidation:
    """Test suite for validating data sent to transport layer"""

    def test_basic_trace_structure(self, client_with_capture, transport_capture):
        """Test that basic trace structure is valid"""
        transport_capture.clear()

        # Create a trace
        trace = client_with_capture.start_trace("test-operation")
        span = trace.create_span("test-span")
        span.set_attribute("test.key", "test-value")
        span.finish()
        client_with_capture.finish_trace(trace)

        # The finish should trigger export
        time.sleep(0.1)

        # Validate data was captured
        assert len(transport_capture.exported_traces) > 0, "No traces were exported"

        # Get the trace data
        trace_data = transport_capture.get_latest_trace()
        assert trace_data is not None, "No trace data found"

        # Validate it's a Trace object with expected attributes
        assert hasattr(trace_data, "name"), "Trace missing name attribute"
        assert hasattr(trace_data, "trace_id"), "Trace missing trace_id attribute"
        assert hasattr(trace_data, "spans"), "Trace missing spans attribute"

        # Validate content
        assert trace_data.name == "test-operation"
        assert len(trace_data.spans) > 0, "No spans in trace"

        # Validate span structure
        test_span = trace_data.spans[0]
        assert hasattr(test_span, "name"), "Span missing name attribute"
        assert hasattr(test_span, "span_id"), "Span missing span_id attribute"
        assert hasattr(test_span, "attributes"), "Span missing attributes"

        assert test_span.name == "test-span"
        assert test_span.attributes.get("test.key") == "test-value"

    def test_llm_trace_attributes(self, client_with_capture, transport_capture):
        """Test that LLM-specific attributes are captured correctly"""
        transport_capture.clear()

        # Create LLM trace
        trace = client_with_capture.start_trace("llm-operation")
        span = trace.create_span("llm-call")

        # Add LLM attributes
        llm_attributes = {
            "llm.model": "gpt-4",
            "llm.prompt": "What is 2+2?",
            "llm.completion": "2+2 equals 4",
            "llm.tokens.prompt": 5,
            "llm.tokens.completion": 4,
            "llm.tokens.total": 9,
            "llm.cost": 0.0027,
        }

        for key, value in llm_attributes.items():
            span.set_attribute(key, value)

        span.finish()
        client_with_capture.finish_trace(trace)

        time.sleep(0.1)

        # Validate LLM data
        trace_data = transport_capture.get_latest_trace()
        assert trace_data is not None, "No LLM trace data found"

        assert len(trace_data.spans) > 0, "No spans in LLM trace"
        llm_span = trace_data.spans[0]

        # Validate LLM attributes
        attrs = llm_span.attributes
        assert attrs.get("llm.model") == "gpt-4"
        assert attrs.get("llm.prompt") == "What is 2+2?"
        assert attrs.get("llm.tokens.total") == 9
        assert attrs.get("llm.cost") == 0.0027

    def test_agent_trace_attributes(self, client_with_capture, transport_capture):
        """Test that agent-specific attributes are captured"""
        transport_capture.clear()

        # Create agent trace
        trace = client_with_capture.start_trace("agent-operation")
        span = trace.create_span("agent-action")

        # Add agent attributes
        agent_attributes = {
            "agent.id": "agent-001",
            "agent.name": "TestAgent",
            "agent.role": "assistant",
            "agent.capability": "text-processing",
            "agent.action": "analyze_input",
            "agent.decision": "process_further",
        }

        for key, value in agent_attributes.items():
            span.set_attribute(key, value)

        span.finish()
        client_with_capture.finish_trace(trace)

        time.sleep(0.1)

        # Validate agent data
        trace_data = transport_capture.get_latest_trace()
        assert trace_data is not None, "No agent trace data found"

        agent_span = trace_data.spans[0]
        attrs = agent_span.attributes

        assert attrs.get("agent.id") == "agent-001"
        assert attrs.get("agent.name") == "TestAgent"
        assert attrs.get("agent.role") == "assistant"
        assert attrs.get("agent.action") == "analyze_input"

    def test_tool_call_attributes(self, client_with_capture, transport_capture):
        """Test that tool call attributes are captured"""
        transport_capture.clear()

        # Create tool trace
        trace = client_with_capture.start_trace("tool-operation")
        span = trace.create_span("tool-call")

        # Add tool attributes
        tool_attributes = {
            "tool.name": "web_search",
            "tool.input": '{"query": "Python testing best practices"}',
            "tool.output": '{"results": ["Best practices guide", "Testing frameworks"]}',
            "tool.execution_time": 1.45,
            "tool.success": True,
            "tool.error": None,
        }

        for key, value in tool_attributes.items():
            if value is not None:
                span.set_attribute(key, value)

        span.finish()
        client_with_capture.finish_trace(trace)

        time.sleep(0.1)

        # Validate tool data
        trace_data = transport_capture.get_latest_trace()
        assert trace_data is not None, "No tool trace data found"

        tool_span = trace_data.spans[0]
        attrs = tool_span.attributes

        assert attrs.get("tool.name") == "web_search"
        assert attrs.get("tool.success") is True
        assert attrs.get("tool.execution_time") == 1.45

        # Validate JSON attributes can be parsed
        tool_input = attrs.get("tool.input")
        tool_output = attrs.get("tool.output")

        assert tool_input is not None
        assert tool_output is not None

        # Should be valid JSON strings

        input_data = json.loads(tool_input)
        output_data = json.loads(tool_output)

        assert input_data["query"] == "Python testing best practices"
        assert len(output_data["results"]) == 2

    def test_nested_spans_structure(self, client_with_capture, transport_capture):
        """Test that nested span hierarchies are preserved"""
        transport_capture.clear()

        # Create nested spans
        trace = client_with_capture.start_trace("nested-operation")

        parent_span = trace.create_span("parent-operation")
        parent_span.set_attribute("level", "parent")

        child_span = trace.create_span(
            "child-operation", parent_span_id=parent_span.span_id
        )
        child_span.set_attribute("level", "child")

        grandchild_span = trace.create_span(
            "grandchild-operation", parent_span_id=child_span.span_id
        )
        grandchild_span.set_attribute("level", "grandchild")

        # Finish in reverse order
        grandchild_span.finish()
        child_span.finish()
        parent_span.finish()
        client_with_capture.finish_trace(trace)

        time.sleep(0.1)

        # Validate hierarchy
        trace_data = transport_capture.get_latest_trace()
        assert trace_data is not None, "No nested trace data found"

        assert (
            len(trace_data.spans) == 3
        ), f"Expected 3 spans, got {len(trace_data.spans)}"

        # Find spans by name
        spans_by_name = {span.name: span for span in trace_data.spans}

        parent = spans_by_name["parent-operation"]
        child = spans_by_name["child-operation"]
        grandchild = spans_by_name["grandchild-operation"]

        # Validate hierarchy
        assert parent.parent_span_id is None, "Parent should have no parent"
        assert (
            child.parent_span_id == parent.span_id
        ), "Child should have parent as parent"
        assert (
            grandchild.parent_span_id == child.span_id
        ), "Grandchild should have child as parent"

        # Validate attributes
        assert parent.attributes["level"] == "parent"
        assert child.attributes["level"] == "child"
        assert grandchild.attributes["level"] == "grandchild"

    def test_error_span_handling(self, client_with_capture, transport_capture):
        """Test that error spans are handled correctly"""
        transport_capture.clear()

        # Create trace with error
        trace = client_with_capture.start_trace("error-operation")
        span = trace.create_span("error-span")

        try:
            raise ValueError("Test error for validation")
        except Exception as e:
            span.record_exception(e)
            span.set_status(SpanStatus.ERROR, str(e))

        span.finish()
        client_with_capture.finish_trace(trace)

        time.sleep(0.1)

        # Validate error data
        trace_data = transport_capture.get_latest_trace()
        assert trace_data is not None, "No error trace data found"

        error_span = trace_data.spans[0]

        # Check error status
        assert (
            error_span.status.value == "error"
        ), f"Expected error status, got {error_span.status}"

        # Check exception attributes
        attrs = error_span.attributes
        assert "exception.type" in attrs
        assert "exception.message" in attrs
        assert attrs["exception.type"] == "ValueError"
        assert "Test error for validation" in attrs["exception.message"]

    def test_metrics_and_performance_data(self, client_with_capture, transport_capture):
        """Test that performance metrics are captured"""
        transport_capture.clear()

        # Create trace with metrics
        trace = client_with_capture.start_trace("metrics-operation")
        span = trace.create_span("metrics-span")

        # Add performance metrics
        metrics = {
            "performance.duration_ms": 123.45,
            "performance.memory_mb": 64.2,
            "performance.cpu_percent": 15.8,
            "metrics.requests_count": 5,
            "metrics.success_rate": 0.95,
            "cost.tokens": 1500,
            "cost.estimate_usd": 0.0045,
        }

        for key, value in metrics.items():
            span.set_attribute(key, value)

        span.finish()
        client_with_capture.finish_trace(trace)

        time.sleep(0.1)

        # Validate metrics
        trace_data = transport_capture.get_latest_trace()
        assert trace_data is not None, "No metrics trace data found"

        metrics_span = trace_data.spans[0]
        attrs = metrics_span.attributes

        # Validate numeric metrics
        assert attrs.get("performance.duration_ms") == 123.45
        assert attrs.get("performance.memory_mb") == 64.2
        assert attrs.get("metrics.requests_count") == 5
        assert attrs.get("cost.estimate_usd") == 0.0045

    def test_trace_serialization_compatibility(
        self, client_with_capture, transport_capture
    ):
        """Test that trace data can be serialized properly"""
        transport_capture.clear()

        # Create a complex trace
        trace = client_with_capture.start_trace("serialization-test")
        span = trace.create_span("test-span")

        # Add various data types
        span.set_attribute("string_attr", "test string")
        span.set_attribute("int_attr", 42)
        span.set_attribute("float_attr", 3.14159)
        span.set_attribute("bool_attr", True)
        span.set_attribute("none_attr", None)
        span.set_attribute("list_attr", [1, 2, 3])
        span.set_attribute("dict_attr", {"key": "value", "nested": {"data": 123}})

        span.finish()
        client_with_capture.finish_trace(trace)

        time.sleep(0.1)

        # Validate serializable data
        trace_data = transport_capture.get_latest_trace()
        assert trace_data is not None, "No serialization trace data found"

        test_span = trace_data.spans[0]
        attrs = test_span.attributes

        # Test different data types
        assert attrs["string_attr"] == "test string"
        assert attrs["int_attr"] == 42
        assert attrs["float_attr"] == 3.14159
        assert attrs["bool_attr"] is True

        # Test that the data would be JSON serializable

        try:
            # Convert trace to dict-like structure for JSON test
            serializable_attrs = {
                k: v
                for k, v in attrs.items()
                if v is not None  # Skip None values as they're often excluded
            }
            json_str = json.dumps(serializable_attrs)
            parsed = json.loads(json_str)
            assert parsed["string_attr"] == "test string"
            assert parsed["int_attr"] == 42
        except (TypeError, ValueError) as e:
            pytest.fail(f"Trace data is not JSON serializable: {e}")


class TestConfigurationValidation:
    """Test that configuration is properly applied to traces"""

    def test_project_and_environment_in_traces(self, transport_capture):
        """Test that project and environment info is included in traces"""
        transport_capture.clear()

        # Create client with specific config
        config = Config.create(
            api_key="test-key", project="validation-project", environment="test-env"
        )

        client = NoveumClient(config=config)

        # Wrap transport
        original_export = client.transport.export_trace
        client.transport.export_trace = transport_capture.capture_export(
            original_export
        )

        # Create trace
        trace = client.start_trace("config-test")
        span = trace.create_span("test-span")
        span.finish()
        client.finish_trace(trace)

        time.sleep(0.1)

        # Validate config data in trace
        trace_data = transport_capture.get_latest_trace()
        assert trace_data is not None, "No config trace data found"

        # Check trace-level attributes for config info
        attrs = trace_data.attributes
        assert attrs.get("noveum.project") == "validation-project"
        assert attrs.get("noveum.environment") == "test-env"
        assert "noveum.sdk.version" in attrs

        client.shutdown()

    def test_custom_endpoint_configuration(self, transport_capture):
        """Test that custom endpoints are properly configured"""
        transport_capture.clear()

        custom_endpoint = "https://custom.noveum.ai"

        # Create client with custom endpoint
        config = Config.create(
            api_key="test-key", project="endpoint-test", endpoint=custom_endpoint
        )

        client = NoveumClient(config=config)

        # Validate the transport uses custom endpoint
        assert (
            client.config.endpoint == custom_endpoint
        ), f"Expected {custom_endpoint}, got {client.config.endpoint}"

        client.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])
