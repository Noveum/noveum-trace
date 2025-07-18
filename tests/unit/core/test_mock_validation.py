"""
Mock validation tests that work with the existing mocking infrastructure.

These tests verify that:
1. The mocking infrastructure prevents real API calls
2. Trace and span structures are valid when created
3. Different types of operations generate expected data structures
4. The SDK behaves correctly under mocked conditions
"""

import time

import pytest

import noveum_trace
from noveum_trace.core.span import SpanStatus


class TestMockValidation:
    """Test suite for validating the mocking infrastructure and trace data"""

    def test_client_prevents_real_api_calls(self, client_with_mocked_transport):
        """Test that no real API calls are made"""
        client = client_with_mocked_transport

        # Create a trace and span
        trace = client.start_trace("api-test")
        span = trace.create_span("test-span")
        span.set_attribute("test", "value")
        span.finish()
        client.finish_trace(trace)

        # Flush to trigger any transport calls
        client.flush()

        # Verify transport methods were called (mocked)
        assert (
            client.transport.export_trace.called
        ), "export_trace should have been called"
        assert client.transport.flush.called, "flush should have been called"

        # Verify the transport is mocked and returns expected values
        export_result = client.transport.export_trace.return_value
        assert export_result == {"success": True}, "Mock should return success"

    def test_trace_structure_validation(self, client_with_mocked_transport):
        """Test that traces have correct structure when created"""
        client = client_with_mocked_transport

        # Create a trace
        trace = client.start_trace("structure-test")

        # Validate trace structure
        assert hasattr(trace, "name"), "Trace should have name"
        assert hasattr(trace, "trace_id"), "Trace should have trace_id"
        assert hasattr(trace, "spans"), "Trace should have spans list"
        assert hasattr(trace, "attributes"), "Trace should have attributes dict"
        assert hasattr(trace, "start_time"), "Trace should have start_time"

        # Validate content
        assert trace.name == "structure-test"
        assert len(trace.trace_id) > 0, "Trace ID should not be empty"
        assert isinstance(trace.spans, list), "Spans should be a list"
        assert isinstance(trace.attributes, dict), "Attributes should be a dict"

        trace.finish()

    def test_span_structure_validation(self, client_with_mocked_transport):
        """Test that spans have correct structure when created"""
        client = client_with_mocked_transport

        # Create trace and span
        trace = client.start_trace("span-test")
        span = trace.create_span("test-span")

        # Validate span structure
        assert hasattr(span, "name"), "Span should have name"
        assert hasattr(span, "span_id"), "Span should have span_id"
        assert hasattr(span, "trace_id"), "Span should have trace_id"
        assert hasattr(span, "attributes"), "Span should have attributes dict"
        assert hasattr(span, "start_time"), "Span should have start_time"
        assert hasattr(span, "status"), "Span should have status"

        # Validate content
        assert span.name == "test-span"
        assert len(span.span_id) > 0, "Span ID should not be empty"
        assert span.trace_id == trace.trace_id, "Span should belong to trace"
        assert isinstance(span.attributes, dict), "Attributes should be a dict"

        span.finish()
        trace.finish()

    def test_llm_attributes_validation(self, client_with_mocked_transport):
        """Test that LLM attributes are properly stored"""
        client = client_with_mocked_transport

        # Create LLM trace
        trace = client.start_trace("llm-test")
        span = trace.create_span("llm-call")

        # Add LLM attributes
        llm_attrs = {
            "llm.model": "gpt-4",
            "llm.prompt": "Test prompt",
            "llm.completion": "Test completion",
            "llm.tokens.prompt": 10,
            "llm.tokens.completion": 15,
            "llm.tokens.total": 25,
            "llm.cost": 0.005,
        }

        for key, value in llm_attrs.items():
            span.set_attribute(key, value)

        # Validate attributes were set
        for key, expected_value in llm_attrs.items():
            actual_value = span.attributes.get(key)
            assert (
                actual_value == expected_value
            ), f"Attribute {key}: expected {expected_value}, got {actual_value}"

        span.finish()
        trace.finish()

    def test_agent_attributes_validation(self, client_with_mocked_transport):
        """Test that agent attributes are properly stored"""
        client = client_with_mocked_transport

        # Create agent trace
        trace = client.start_trace("agent-test")
        span = trace.create_span("agent-action")

        # Add agent attributes
        agent_attrs = {
            "agent.id": "agent-001",
            "agent.name": "TestAgent",
            "agent.role": "assistant",
            "agent.capability": "reasoning",
            "agent.action": "analyze",
            "agent.decision": "proceed",
        }

        for key, value in agent_attrs.items():
            span.set_attribute(key, value)

        # Validate attributes
        for key, expected_value in agent_attrs.items():
            actual_value = span.attributes.get(key)
            assert (
                actual_value == expected_value
            ), f"Agent attribute {key}: expected {expected_value}, got {actual_value}"

        span.finish()
        trace.finish()

    def test_tool_attributes_validation(self, client_with_mocked_transport):
        """Test that tool attributes are properly stored"""
        client = client_with_mocked_transport

        # Create tool trace
        trace = client.start_trace("tool-test")
        span = trace.create_span("tool-call")

        # Add tool attributes
        tool_attrs = {
            "tool.name": "web_search",
            "tool.input": '{"query": "test"}',
            "tool.output": '{"results": ["result1"]}',
            "tool.execution_time": 1.5,
            "tool.success": True,
        }

        for key, value in tool_attrs.items():
            span.set_attribute(key, value)

        # Validate attributes
        for key, expected_value in tool_attrs.items():
            actual_value = span.attributes.get(key)
            assert (
                actual_value == expected_value
            ), f"Tool attribute {key}: expected {expected_value}, got {actual_value}"

        # Validate JSON strings are valid
        import json

        tool_input = json.loads(span.attributes["tool.input"])
        tool_output = json.loads(span.attributes["tool.output"])

        assert tool_input["query"] == "test"
        assert tool_output["results"] == ["result1"]

        span.finish()
        trace.finish()

    def test_nested_spans_hierarchy(self, client_with_mocked_transport):
        """Test that nested spans maintain proper hierarchy"""
        client = client_with_mocked_transport

        # Create nested spans
        trace = client.start_trace("nested-test")

        parent_span = trace.create_span("parent")
        parent_span.set_attribute("level", "parent")

        child_span = trace.create_span("child", parent_span_id=parent_span.span_id)
        child_span.set_attribute("level", "child")

        grandchild_span = trace.create_span(
            "grandchild", parent_span_id=child_span.span_id
        )
        grandchild_span.set_attribute("level", "grandchild")

        # Validate hierarchy
        assert parent_span.parent_span_id is None, "Parent should have no parent"
        assert (
            child_span.parent_span_id == parent_span.span_id
        ), "Child should have parent ID"
        assert (
            grandchild_span.parent_span_id == child_span.span_id
        ), "Grandchild should have child ID"

        # Validate all spans are in trace
        assert len(trace.spans) == 3, f"Expected 3 spans, got {len(trace.spans)}"

        # Finish spans
        grandchild_span.finish()
        child_span.finish()
        parent_span.finish()
        trace.finish()

    def test_error_handling_validation(self, client_with_mocked_transport):
        """Test that error spans are handled correctly"""
        client = client_with_mocked_transport

        # Create error trace
        trace = client.start_trace("error-test")
        span = trace.create_span("error-span")

        # Simulate error
        try:
            raise ValueError("Test error for validation")
        except Exception as e:
            span.record_exception(e)
            span.set_status(SpanStatus.ERROR, str(e))

        # Validate error was recorded
        assert (
            span.status == SpanStatus.ERROR
        ), f"Expected ERROR status, got {span.status}"
        assert span.exception is not None, "Exception should be recorded"
        assert isinstance(
            span.exception, ValueError
        ), "Exception type should be preserved"

        # Validate exception attributes
        attrs = span.attributes
        assert "exception.type" in attrs, "Exception type should be in attributes"
        assert "exception.message" in attrs, "Exception message should be in attributes"
        assert attrs["exception.type"] == "ValueError"
        assert "Test error for validation" in attrs["exception.message"]

        span.finish()
        trace.finish()

    def test_performance_metrics_validation(self, client_with_mocked_transport):
        """Test that performance metrics are stored correctly"""
        client = client_with_mocked_transport

        # Create performance trace
        trace = client.start_trace("performance-test")
        span = trace.create_span("performance-span")

        # Add performance metrics
        metrics = {
            "performance.duration_ms": 123.45,
            "performance.memory_mb": 64.0,
            "performance.cpu_percent": 25.5,
            "metrics.throughput": 1000,
            "metrics.latency_p95": 95.5,
        }

        for key, value in metrics.items():
            span.set_attribute(key, value)

        # Validate metrics
        for key, expected_value in metrics.items():
            actual_value = span.attributes.get(key)
            assert (
                actual_value == expected_value
            ), f"Metric {key}: expected {expected_value}, got {actual_value}"

            # Validate numeric types
            if isinstance(expected_value, (int, float)):
                assert isinstance(
                    actual_value, (int, float)
                ), f"Metric {key} should be numeric"

        span.finish()
        trace.finish()

    def test_configuration_injection(self, client_with_mocked_transport):
        """Test that configuration is properly injected into traces"""
        client = client_with_mocked_transport

        # Create trace
        trace = client.start_trace("config-test")

        # Check that configuration attributes are injected
        attrs = trace.attributes

        # These should be automatically added by the client
        assert "noveum.project" in attrs, "Project should be in trace attributes"
        assert (
            "noveum.environment" in attrs
        ), "Environment should be in trace attributes"
        assert (
            "noveum.sdk.version" in attrs
        ), "SDK version should be in trace attributes"

        # Validate values
        assert attrs["noveum.project"] == client.config.project
        assert attrs["noveum.environment"] == client.config.environment

        trace.finish()

    def test_transport_mocking_effectiveness(self, client_with_mocked_transport):
        """Test that transport layer is properly mocked"""
        client = client_with_mocked_transport

        # Verify transport is mocked
        assert hasattr(
            client.transport, "export_trace"
        ), "Transport should have export_trace"
        assert hasattr(client.transport, "flush"), "Transport should have flush"
        assert hasattr(client.transport, "shutdown"), "Transport should have shutdown"

        # Create and finish trace to trigger export
        trace = client.start_trace("transport-test")
        span = trace.create_span("test-span")
        span.finish()
        client.finish_trace(trace)

        # Verify mock calls
        assert (
            client.transport.export_trace.called
        ), "export_trace should have been called"

        # Verify no real HTTP calls by checking the mock return value
        export_call_args = client.transport.export_trace.call_args
        assert (
            export_call_args is not None
        ), "export_trace should have been called with arguments"

        # The first argument should be the trace object
        exported_trace = export_call_args[0][0]
        assert hasattr(exported_trace, "name"), "Exported object should be a trace"
        assert exported_trace.name == "transport-test"

    def test_data_type_compatibility(self, client_with_mocked_transport):
        """Test that various data types can be stored as attributes"""
        client = client_with_mocked_transport

        trace = client.start_trace("datatype-test")
        span = trace.create_span("test-span")

        # Test various data types
        test_data = {
            "string_attr": "test string",
            "int_attr": 42,
            "float_attr": 3.14159,
            "bool_attr": True,
            "none_attr": None,
            "list_attr": [1, 2, 3],
            "dict_attr": {"key": "value", "nested": {"data": 123}},
        }

        for key, value in test_data.items():
            span.set_attribute(key, value)

        # Validate all data types are preserved
        for key, expected_value in test_data.items():
            actual_value = span.attributes.get(key)
            assert (
                actual_value == expected_value
            ), f"Data type test failed for {key}: expected {expected_value}, got {actual_value}"

        # Test JSON serialization compatibility
        import json

        try:
            # Filter out None values for JSON test
            serializable_attrs = {
                k: v for k, v in span.attributes.items() if v is not None
            }
            json_str = json.dumps(serializable_attrs)
            parsed = json.loads(json_str)

            # Verify some key values survived JSON roundtrip
            assert parsed["string_attr"] == "test string"
            assert parsed["int_attr"] == 42
            assert parsed["bool_attr"] is True

        except (TypeError, ValueError) as e:
            pytest.fail(f"Attribute data is not JSON serializable: {e}")

        span.finish()
        trace.finish()


class TestDecoratorIntegration:
    """Test that decorators work correctly with mocked transport"""

    def test_trace_decorator_basic(self, client_with_mocked_transport):
        """Test that @trace decorator works with mocked transport"""

        @noveum_trace.trace
        def test_function():
            return "test result"

        # Execute decorated function
        result = test_function()

        # Validate function still works
        assert result == "test result"

        # Give time for any async operations
        time.sleep(0.1)

    def test_trace_decorator_with_parameters(self, client_with_mocked_transport):
        """Test that @trace decorator handles function parameters"""

        @noveum_trace.trace
        def parameterized_function(x, y, z="default"):
            return f"{x}-{y}-{z}"

        # Execute with parameters
        result = parameterized_function("a", "b", z="custom")

        # Validate function works correctly
        assert result == "a-b-custom"

        time.sleep(0.1)

    def test_trace_decorator_with_exception(self, client_with_mocked_transport):
        """Test that @trace decorator handles exceptions"""

        @noveum_trace.trace
        def failing_function():
            raise ValueError("Test exception from decorator")

        # Execute function that raises exception
        with pytest.raises(ValueError, match="Test exception from decorator"):
            failing_function()

        time.sleep(0.1)


if __name__ == "__main__":
    pytest.main([__file__])
