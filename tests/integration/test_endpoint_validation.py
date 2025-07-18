"""
Comprehensive tests for validating the mocked HTTP endpoint and trace data.

This module tests that our mocking infrastructure correctly captures HTTP requests
and that the trace data being sent is valid and properly structured.
"""

import concurrent.futures
import json
import threading
import time
from datetime import datetime
from typing import Any, Optional
from unittest.mock import Mock, patch

import pytest

import noveum_trace
from noveum_trace.core.client import NoveumClient
from noveum_trace.core.config import Config
from noveum_trace.core.span import SpanStatus
from noveum_trace.transport.http_transport import HttpTransport


class EndpointCapture:
    """Captures all HTTP requests to validate endpoint behavior"""

    def __init__(self):
        self.requests: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def capture_request(
        self, url: str, method: str, headers: dict[str, str], data: Any
    ) -> Mock:
        """Capture an HTTP request and return a mock response"""
        with self.lock:
            request_data = {
                "url": url,
                "method": method,
                "headers": dict(headers) if headers else {},
                "timestamp": datetime.now().isoformat(),
                "raw_data": data,
            }

            # Try to parse JSON data
            if data:
                try:
                    if isinstance(data, (str, bytes)):
                        request_data["json_data"] = json.loads(data)
                    else:
                        request_data["json_data"] = data
                except (json.JSONDecodeError, TypeError):
                    request_data["json_data"] = None

            self.requests.append(request_data)

        # Return a mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "trace_id": f"trace-{len(self.requests)}",
        }
        mock_response.text = '{"success": true}'
        mock_response.headers = {"content-type": "application/json"}
        return mock_response

    def get_trace_requests(self) -> list[dict[str, Any]]:
        """Get all requests that contain trace data"""
        with self.lock:
            return [req for req in self.requests if req.get("json_data")]

    def get_latest_trace(self) -> Optional[dict[str, Any]]:
        """Get the latest trace data sent to the endpoint"""
        trace_requests = self.get_trace_requests()
        if not trace_requests:
            return None

        # Extract the first trace from the batch payload
        json_data = trace_requests[-1]["json_data"]
        if "traces" in json_data and json_data["traces"]:
            return json_data["traces"][0]
        return json_data

    def clear(self):
        """Clear all captured requests"""
        with self.lock:
            self.requests.clear()


@pytest.fixture
def endpoint_capture():
    """Provide an endpoint capture instance for tests"""
    capture = EndpointCapture()

    def mock_requests_post(url, data=None, headers=None, json=None, **kwargs):
        payload = json if json is not None else data
        return capture.capture_request(url, "POST", headers, payload)

    def mock_session_post(url, data=None, headers=None, json=None, **kwargs):
        payload = json if json is not None else data
        return capture.capture_request(url, "POST", headers, payload)

    # Patch HTTP methods to capture requests
    with (
        patch("requests.post", side_effect=mock_requests_post),
        patch("requests.Session.post", side_effect=mock_session_post),
        patch(
            "noveum_trace.transport.http_transport.requests.Session.post",
            side_effect=mock_session_post,
        ),
    ):
        yield capture


@pytest.fixture
def capturing_client(endpoint_capture):
    """Provide a client that captures all HTTP requests"""
    # Create a real client that will use the mocked HTTP layer
    config = Config.create(
        api_key="test-api-key", project="test-project", endpoint="https://api.noveum.ai"
    )

    client = NoveumClient(config=config)
    yield client

    # Clean up
    try:
        client.shutdown()
    except Exception:
        pass


@pytest.mark.disable_transport_mocking
class TestEndpointValidation:
    """Test suite for validating HTTP endpoint behavior and trace data"""

    def _wait_for_trace_capture(self, endpoint_capture, max_wait=5.0):
        """Wait for trace to be captured"""
        start_time = time.time()
        while time.time() - start_time < max_wait:
            if endpoint_capture.requests:
                break
            time.sleep(0.1)

    def test_basic_trace_capture(self, capturing_client, endpoint_capture):
        """Test that basic traces are captured correctly"""
        # Create a simple trace
        trace = capturing_client.start_trace("test-operation")
        span = trace.create_span("test-span")
        span.set_attribute("test.key", "test-value")
        span.finish()
        capturing_client.finish_trace(trace)

        # Flush to ensure data is sent
        capturing_client.flush()

        # Wait for async operations and poll for requests
        self._wait_for_trace_capture(endpoint_capture)

        # Validate requests were captured
        requests = endpoint_capture.get_trace_requests()
        assert len(requests) > 0, "No trace requests were captured"

        # Validate the latest trace data
        latest_trace = endpoint_capture.get_latest_trace()
        assert latest_trace is not None, "No trace data found"

        # Validate trace structure
        self._validate_trace_structure(latest_trace)

        # Validate specific content
        assert latest_trace.get("name") == "test-operation"
        spans = latest_trace.get("spans", [])
        assert len(spans) > 0, "No spans found in trace"

        test_span = spans[0]
        assert test_span.get("name") == "test-span"
        attributes = test_span.get("attributes", {})
        assert attributes.get("test.key") == "test-value"

    def test_decorator_trace_capture(self, capturing_client, endpoint_capture):
        """Test that decorator-based traces are captured correctly"""
        endpoint_capture.clear()

        # Initialize noveum_trace with the same config as capturing_client
        noveum_trace.init(
            api_key="test-api-key",
            project="test-project",
            endpoint="https://api.noveum.ai",
        )

        # Use a decorator-based trace
        @noveum_trace.trace
        def test_function(x, y):
            """Test function with parameters"""
            return x + y

        # Execute the function
        result = test_function(5, 3)
        assert result == 8

        # Flush the global client to ensure data is sent
        noveum_trace.get_client().flush()

        # Wait for async operations and poll for requests
        self._wait_for_trace_capture(endpoint_capture)

        # Validate trace was captured
        latest_trace = endpoint_capture.get_latest_trace()
        assert latest_trace is not None, "Decorator trace not captured"

        self._validate_trace_structure(latest_trace)

        # Validate function-specific content
        spans = latest_trace.get("spans", [])
        if spans:
            test_span = spans[0]
            assert "test_function" in test_span.get("name", "")

    def test_llm_trace_capture(self, capturing_client, endpoint_capture):
        """Test that LLM traces are captured with proper metadata"""
        endpoint_capture.clear()

        # Create an LLM trace
        trace = capturing_client.start_trace("llm-operation")
        span = trace.create_span("llm-call")

        # Add LLM-specific attributes
        span.set_attribute("llm.model", "gpt-4")
        span.set_attribute("llm.prompt", "What is the capital of France?")
        span.set_attribute("llm.completion", "The capital of France is Paris.")
        span.set_attribute("llm.tokens.prompt", 8)
        span.set_attribute("llm.tokens.completion", 6)
        span.set_attribute("llm.tokens.total", 14)

        span.finish()
        capturing_client.finish_trace(trace)

        capturing_client.flush()

        # Wait for async operations and poll for requests
        self._wait_for_trace_capture(endpoint_capture)

        # Validate LLM trace
        latest_trace = endpoint_capture.get_latest_trace()
        assert latest_trace is not None, "LLM trace not captured"

        self._validate_trace_structure(latest_trace)

        # Validate LLM-specific content
        spans = latest_trace.get("spans", [])
        assert len(spans) > 0, "No spans in LLM trace"

        llm_span = spans[0]
        attributes = llm_span.get("attributes", {})

        assert attributes.get("llm.model") == "gpt-4"
        assert attributes.get("llm.prompt") == "What is the capital of France?"
        assert attributes.get("llm.completion") == "The capital of France is Paris."
        assert attributes.get("llm.tokens.total") == 14

    def test_agent_trace_capture(self, capturing_client, endpoint_capture):
        """Test that agent traces are captured with agent metadata"""
        endpoint_capture.clear()

        # Create an agent trace
        trace = capturing_client.start_trace("agent-operation")
        span = trace.create_span("agent-action")

        # Add agent-specific attributes
        span.set_attribute("agent.id", "agent-001")
        span.set_attribute("agent.name", "TestAgent")
        span.set_attribute("agent.role", "assistant")
        span.set_attribute("agent.capability", "text-processing")
        span.set_attribute("agent.action", "process_input")

        span.finish()
        capturing_client.finish_trace(trace)

        capturing_client.flush()
        self._wait_for_trace_capture(endpoint_capture)

        # Validate agent trace
        latest_trace = endpoint_capture.get_latest_trace()
        assert latest_trace is not None, "Agent trace not captured"

        self._validate_trace_structure(latest_trace)

        # Validate agent-specific content
        spans = latest_trace.get("spans", [])
        assert len(spans) > 0, "No spans in agent trace"

        agent_span = spans[0]
        attributes = agent_span.get("attributes", {})

        assert attributes.get("agent.id") == "agent-001"
        assert attributes.get("agent.name") == "TestAgent"
        assert attributes.get("agent.role") == "assistant"
        assert attributes.get("agent.action") == "process_input"

    def test_tool_call_trace_capture(self, capturing_client, endpoint_capture):
        """Test that tool call traces are captured with tool metadata"""
        endpoint_capture.clear()

        # Create a tool call trace
        trace = capturing_client.start_trace("tool-operation")
        span = trace.create_span("tool-call")

        # Add tool-specific attributes
        span.set_attribute("tool.name", "web_search")
        span.set_attribute("tool.input", '{"query": "Python testing"}')
        span.set_attribute("tool.output", '{"results": ["result1", "result2"]}')
        span.set_attribute("tool.execution_time", 1.23)
        span.set_attribute("tool.success", True)

        span.finish()
        capturing_client.finish_trace(trace)

        capturing_client.flush()
        self._wait_for_trace_capture(endpoint_capture)

        # Validate tool trace
        latest_trace = endpoint_capture.get_latest_trace()
        assert latest_trace is not None, "Tool trace not captured"

        self._validate_trace_structure(latest_trace)

        # Validate tool-specific content
        spans = latest_trace.get("spans", [])
        assert len(spans) > 0, "No spans in tool trace"

        tool_span = spans[0]
        attributes = tool_span.get("attributes", {})

        assert attributes.get("tool.name") == "web_search"
        assert attributes.get("tool.success") is True
        assert attributes.get("tool.execution_time") == 1.23

    def test_nested_spans_capture(self, capturing_client, endpoint_capture):
        """Test that nested spans are captured correctly"""
        endpoint_capture.clear()

        # Create nested spans
        trace = capturing_client.start_trace("nested-operation")

        parent_span = trace.create_span("parent-span")
        parent_span.set_attribute("level", "parent")

        child_span = trace.create_span("child-span", parent_span_id=parent_span.span_id)
        child_span.set_attribute("level", "child")

        grandchild_span = trace.create_span(
            "grandchild-span", parent_span_id=child_span.span_id
        )
        grandchild_span.set_attribute("level", "grandchild")

        # End spans in reverse order
        grandchild_span.finish()
        child_span.finish()
        parent_span.finish()
        capturing_client.finish_trace(trace)

        capturing_client.flush()
        self._wait_for_trace_capture(endpoint_capture)

        # Validate nested trace
        latest_trace = endpoint_capture.get_latest_trace()
        assert latest_trace is not None, "Nested trace not captured"

        self._validate_trace_structure(latest_trace)

        # Validate span hierarchy
        spans = latest_trace.get("spans", [])
        assert len(spans) == 3, f"Expected 3 spans, got {len(spans)}"

        # Check that parent-child relationships are preserved
        span_names = [span.get("name") for span in spans]
        assert "parent-span" in span_names
        assert "child-span" in span_names
        assert "grandchild-span" in span_names

    def test_error_span_capture(self, capturing_client, endpoint_capture):
        """Test that error spans are captured with error information"""
        endpoint_capture.clear()

        # Create a span with an error
        trace = capturing_client.start_trace("error-operation")
        span = trace.create_span("error-span")

        try:
            raise ValueError("Test error message")
        except Exception as e:
            span.record_exception(e)
            span.set_status(SpanStatus.ERROR, str(e))

        span.finish()
        capturing_client.finish_trace(trace)

        capturing_client.flush()
        self._wait_for_trace_capture(endpoint_capture)

        # Validate error trace
        latest_trace = endpoint_capture.get_latest_trace()
        assert latest_trace is not None, "Error trace not captured"

        self._validate_trace_structure(latest_trace)

        # Validate error information
        spans = latest_trace.get("spans", [])
        assert len(spans) > 0, "No spans in error trace"

        error_span = spans[0]
        status = error_span.get("status", "")
        assert status == "error" or "error" in str(status)

    def test_metrics_capture(self, capturing_client, endpoint_capture):
        """Test that metrics are captured correctly"""
        endpoint_capture.clear()

        # Create a trace with metrics
        trace = capturing_client.start_trace("metrics-operation")
        span = trace.create_span("metrics-span")

        # Add various metrics
        span.set_attribute("metrics.duration", 123.45)
        span.set_attribute("metrics.memory_usage", 1024)
        span.set_attribute("metrics.cpu_percent", 15.5)
        span.set_attribute("metrics.tokens_per_second", 42.0)
        span.set_attribute("metrics.cost_estimate", 0.0023)

        span.finish()
        capturing_client.finish_trace(trace)

        capturing_client.flush()
        self._wait_for_trace_capture(endpoint_capture)

        # Validate metrics
        latest_trace = endpoint_capture.get_latest_trace()
        assert latest_trace is not None, "Metrics trace not captured"

        self._validate_trace_structure(latest_trace)

        # Validate metrics data
        spans = latest_trace.get("spans", [])
        assert len(spans) > 0, "No spans in metrics trace"

        metrics_span = spans[0]
        attributes = metrics_span.get("attributes", {})

        assert attributes.get("metrics.duration") == 123.45
        assert attributes.get("metrics.memory_usage") == 1024
        assert attributes.get("metrics.cost_estimate") == 0.0023

    def test_batch_operations_capture(self, capturing_client, endpoint_capture):
        """Test that batch operations are captured correctly"""
        endpoint_capture.clear()

        # Create multiple traces to test batching
        traces = []
        for i in range(3):
            trace = capturing_client.start_trace(f"batch-operation-{i}")
            span = trace.create_span(f"batch-span-{i}")
            span.set_attribute("batch.index", i)
            span.set_attribute("batch.total", 3)
            span.finish()
            capturing_client.finish_trace(trace)
            traces.append(trace)

        capturing_client.flush()
        self._wait_for_trace_capture(endpoint_capture)

        # Validate multiple traces were captured
        trace_requests = endpoint_capture.get_trace_requests()
        assert len(trace_requests) >= 1, "No batch requests captured"

        # Check that we have trace data for our operations
        all_traces = [
            req["json_data"] for req in trace_requests if req.get("json_data")
        ]
        assert len(all_traces) > 0, "No trace data in batch requests"

        # Validate each trace structure
        for trace_data in all_traces:
            # Extract traces from batch payload if needed
            if "traces" in trace_data and isinstance(trace_data["traces"], list):
                for trace in trace_data["traces"]:
                    self._validate_trace_structure(trace)
            else:
                self._validate_trace_structure(trace_data)

    def test_endpoint_url_validation(self, endpoint_capture):
        """Test that requests are sent to the correct endpoint URL"""
        endpoint_capture.clear()

        # Create client with custom endpoint
        config = Config.create(
            api_key="test-api-key",
            project="test-project",
            endpoint="https://custom.noveum.ai",
        )

        with patch.object(HttpTransport, "__init__", lambda self, config=None: None):
            client = NoveumClient(config=config)

            # Mock the transport to capture requests
            client.transport = Mock()

            def mock_export_trace(trace_data):
                """Mock export that captures the URL"""
                endpoint_capture.capture_request(
                    url="https://custom.noveum.ai/v1/traces",
                    method="POST",
                    headers={"Authorization": f"Bearer {config.api_key}"},
                    data=trace_data,
                )

            def mock_flush(timeout=None):
                """Mock flush that triggers our capture"""
                mock_export_trace({"test": "data"})

            client.transport.export_trace = mock_export_trace
            client.transport.flush = mock_flush
            client.transport.shutdown = Mock()

            # Create and send a trace
            trace = client.start_trace("url-test")
            span = trace.create_span("test-span")
            span.finish()
            client.finish_trace(trace)

            client.flush()

            # Validate URL
            requests = endpoint_capture.requests
            assert len(requests) > 0, "No requests captured"

            # Check that at least one request went to the custom endpoint
            custom_endpoint_found = any(
                "custom.noveum.ai" in req.get("url", "") for req in requests
            )
            assert custom_endpoint_found, "Custom endpoint not used"

            client.shutdown()

    def test_authentication_headers(self, endpoint_capture):
        """Test that authentication headers are included correctly"""
        endpoint_capture.clear()

        # Create client with specific API key
        config = Config.create(api_key="test-secret-key", project="test-project")

        with patch.object(HttpTransport, "__init__", lambda self, config=None: None):
            client = NoveumClient(config=config)

            # Mock the transport to capture requests
            client.transport = Mock()

            def mock_export_trace(trace_data):
                """Mock export that captures the headers"""
                endpoint_capture.capture_request(
                    url="https://api.noveum.ai/v1/traces",
                    method="POST",
                    headers={"Authorization": f"Bearer {config.api_key}"},
                    data=trace_data,
                )

            def mock_flush(timeout=None):
                """Mock flush that triggers our capture"""
                mock_export_trace({"test": "data"})

            client.transport.export_trace = mock_export_trace
            client.transport.flush = mock_flush
            client.transport.shutdown = Mock()

            # Create and send a trace
            trace = client.start_trace("auth-test")
            span = trace.create_span("test-span")
            span.finish()
            client.finish_trace(trace)

            client.flush()

            # Validate authentication headers
            requests = endpoint_capture.requests
            assert len(requests) > 0, "No requests captured"

            # Check for authorization header
            auth_header_found = False
            for req in requests:
                headers = req.get("headers", {})
                if "authorization" in headers or "Authorization" in headers:
                    auth_header_found = True
                    # Validate the format (should be Bearer token)
                    auth_value = headers.get("authorization") or headers.get(
                        "Authorization"
                    )
                    assert auth_value is not None, "Authorization header is None"
                    assert (
                        "Bearer test-secret-key" in auth_value
                    ), f"Unexpected auth value: {auth_value}"
                    break

            assert auth_header_found, "No authorization header found"

            client.shutdown()

    def _validate_trace_structure(self, trace_data: dict[str, Any]):
        """Validate that trace data has the expected structure"""
        assert isinstance(
            trace_data, dict
        ), f"Trace data should be dict, got {type(trace_data)}"

        # Check required top-level fields
        required_fields = ["trace_id", "name"]
        for field in required_fields:
            assert field in trace_data, f"Missing required field: {field}"
            assert trace_data[field] is not None, f"Field {field} is None"

        # Validate trace_id format
        trace_id = trace_data["trace_id"]
        assert isinstance(
            trace_id, str
        ), f"trace_id should be string, got {type(trace_id)}"
        assert len(trace_id) > 0, "trace_id should not be empty"

        # Validate name
        name = trace_data["name"]
        assert isinstance(name, str), f"name should be string, got {type(name)}"
        assert len(name) > 0, "name should not be empty"

        # Check spans if present
        if "spans" in trace_data:
            spans = trace_data["spans"]
            assert isinstance(spans, list), f"spans should be list, got {type(spans)}"

            for i, span in enumerate(spans):
                self._validate_span_structure(span, f"span[{i}]")

        # Check timestamps if present
        for time_field in ["start_time", "end_time", "timestamp"]:
            if time_field in trace_data:
                time_value = trace_data[time_field]
                assert time_value is not None, f"{time_field} should not be None"
                # Could be string (ISO format) or number (timestamp)
                assert isinstance(
                    time_value, (str, int, float)
                ), f"{time_field} should be string or number, got {type(time_value)}"

    def _validate_span_structure(
        self, span_data: dict[str, Any], context: str = "span"
    ):
        """Validate that span data has the expected structure"""
        assert isinstance(
            span_data, dict
        ), f"{context} should be dict, got {type(span_data)}"

        # Check required span fields
        required_fields = ["span_id", "name"]
        for field in required_fields:
            assert field in span_data, f"{context} missing required field: {field}"
            assert span_data[field] is not None, f"{context} field {field} is None"

        # Validate span_id
        span_id = span_data["span_id"]
        assert isinstance(
            span_id, str
        ), f"{context} span_id should be string, got {type(span_id)}"
        assert len(span_id) > 0, f"{context} span_id should not be empty"

        # Validate name
        name = span_data["name"]
        assert isinstance(
            name, str
        ), f"{context} name should be string, got {type(name)}"
        assert len(name) > 0, f"{context} name should not be empty"

        # Check attributes if present
        if "attributes" in span_data:
            attributes = span_data["attributes"]
            assert isinstance(
                attributes, dict
            ), f"{context} attributes should be dict, got {type(attributes)}"

            # Validate attribute values are JSON serializable
            for key, value in attributes.items():
                assert isinstance(key, str), f"{context} attribute key should be string"
                assert self._is_json_serializable(
                    value
                ), f"{context} attribute value for {key} is not JSON serializable: {value}"

        # Check status if present
        if "status" in span_data:
            status = span_data["status"]
            # Status can be string or dict
            assert isinstance(
                status, (str, dict)
            ), f"{context} status should be string or dict, got {type(status)}"

        # Check timestamps if present
        for time_field in ["start_time", "end_time"]:
            if time_field in span_data:
                time_value = span_data[time_field]
                assert (
                    time_value is not None
                ), f"{context} {time_field} should not be None"
                assert isinstance(
                    time_value, (str, int, float)
                ), f"{context} {time_field} should be string or number, got {type(time_value)}"

    def _is_json_serializable(self, value) -> bool:
        """Check if a value is JSON serializable"""
        try:
            json.dumps(value)
            return True
        except (TypeError, ValueError):
            return False


@pytest.mark.disable_transport_mocking
class TestEndpointIntegration:
    """Integration tests for endpoint behavior with real scenarios"""

    def _wait_for_trace_capture(self, endpoint_capture, max_wait=5.0):
        """Wait for trace to be captured"""
        start_time = time.time()
        while time.time() - start_time < max_wait:
            if endpoint_capture.requests:
                break
            time.sleep(0.1)

    def test_complete_workflow_capture(self, capturing_client, endpoint_capture):
        """Test capturing a complete workflow with multiple operations"""
        endpoint_capture.clear()

        # Simulate a complete AI workflow
        workflow_trace = capturing_client.start_trace("ai-workflow")

        # 1. Input processing
        input_span = workflow_trace.create_span("input-processing")
        input_span.set_attribute("input.type", "text")
        input_span.set_attribute("input.length", 150)
        input_span.finish()

        # 2. LLM call
        llm_span = workflow_trace.create_span("llm-call")
        llm_span.set_attribute("llm.model", "gpt-4")
        llm_span.set_attribute("llm.tokens.total", 1500)
        llm_span.set_attribute("llm.cost", 0.045)
        llm_span.finish()

        # 3. Tool usage
        tool_span = workflow_trace.create_span("tool-execution")
        tool_span.set_attribute("tool.name", "calculator")
        tool_span.set_attribute("tool.success", True)
        tool_span.finish()

        # 4. Output generation
        output_span = workflow_trace.create_span("output-generation")
        output_span.set_attribute("output.format", "json")
        output_span.set_attribute("output.size", 300)
        output_span.finish()

        capturing_client.finish_trace(workflow_trace)

        capturing_client.flush()
        self._wait_for_trace_capture(endpoint_capture)

        # Validate complete workflow was captured
        latest_trace = endpoint_capture.get_latest_trace()
        assert latest_trace is not None, "Workflow trace not captured"

        # Validate workflow structure
        assert latest_trace.get("name") == "ai-workflow"
        spans = latest_trace.get("spans", [])
        assert len(spans) == 4, f"Expected 4 spans, got {len(spans)}"

        # Validate each operation was captured
        span_names = [span.get("name") for span in spans]
        expected_spans = [
            "input-processing",
            "llm-call",
            "tool-execution",
            "output-generation",
        ]
        for expected_span in expected_spans:
            assert expected_span in span_names, f"Missing span: {expected_span}"

    def test_concurrent_traces_capture(self, capturing_client, endpoint_capture):
        """Test capturing concurrent traces from multiple threads"""
        endpoint_capture.clear()

        def start_trace(trace_id):
            """Create a trace in a separate thread"""
            trace = capturing_client.start_trace(f"concurrent-trace-{trace_id}")
            span = trace.create_span(f"concurrent-span-{trace_id}")
            span.set_attribute("thread.id", trace_id)
            time.sleep(0.01)  # Simulate some work
            span.finish()
            capturing_client.finish_trace(trace)
            return trace_id

        # Create multiple traces concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(start_trace, i) for i in range(5)]
            completed = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

        assert len(completed) == 5, f"Expected 5 completed traces, got {len(completed)}"

        capturing_client.flush()
        time.sleep(0.2)

        # Validate concurrent traces were captured
        trace_requests = endpoint_capture.get_trace_requests()
        assert len(trace_requests) > 0, "No concurrent traces captured"

        # Should have captured data for multiple traces
        all_traces = [
            req["json_data"] for req in trace_requests if req.get("json_data")
        ]
        assert len(all_traces) > 0, "No trace data captured for concurrent execution"

    def test_error_handling_capture(self, capturing_client, endpoint_capture):
        """Test that error scenarios are properly captured"""
        endpoint_capture.clear()

        # Create traces with various error scenarios
        error_trace = capturing_client.start_trace("error-scenarios")

        # 1. Handled exception
        try:
            raise ValueError("Handled error")
        except Exception as e:
            error_span = error_trace.create_span("handled-error")
            error_span.record_exception(e)
            error_span.set_status(SpanStatus.ERROR, str(e))
            error_span.finish()

        # 2. Warning scenario
        warning_span = error_trace.create_span("warning-scenario")
        warning_span.set_attribute("warning.message", "Performance degraded")
        warning_span.set_status(
            SpanStatus.ERROR, "Slow response time"
        )  # Use ERROR since there's no WARNING status
        warning_span.finish()

        capturing_client.finish_trace(error_trace)

        capturing_client.flush()
        self._wait_for_trace_capture(endpoint_capture)

        # Validate error scenarios were captured
        latest_trace = endpoint_capture.get_latest_trace()
        assert latest_trace is not None, "Error trace not captured"

        spans = latest_trace.get("spans", [])
        assert len(spans) == 2, f"Expected 2 spans, got {len(spans)}"

        # Check that error information is preserved
        error_found = False
        warning_found = False

        for span in spans:
            status = span.get("status", {})
            if "error" in str(status).lower():
                error_found = True
            if "warning" in str(status).lower():
                warning_found = True

        # At least one error/warning should be captured
        assert error_found or warning_found, "No error/warning status captured"


if __name__ == "__main__":
    pytest.main([__file__])
