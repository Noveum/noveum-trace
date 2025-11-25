"""
Integration tests for Noveum Trace decorators with trace collection.

This module verifies that traces are actually generated and captured by all decorators
without sending anything to api.noveum.ai. Instead, it intercepts traces locally and
validates their content. Includes real OpenAI/Anthropic API calls for realistic testing.

These are integration tests that verify the full decorator functionality including:
- Trace generation and capture
- Real LLM API integration (OpenAI, Anthropic)
- End-to-end decorator workflows
- Local trace interception (no external API calls)
"""

import asyncio
import os
import threading
import time
from typing import Any, Optional
from unittest.mock import Mock, patch

import pytest

import noveum_trace
from noveum_trace.core.trace import Trace
from noveum_trace.decorators import (
    trace,
    trace_agent,
    trace_llm,
    trace_retrieval,
    trace_tool,
)

# Configurable endpoint for integration tests
ENDPOINT = os.environ.get("NOVEUM_ENDPOINT", "https://api.noveum.ai/api")


class LocalTraceCapture:
    """Captures traces locally before they're sent to any API"""

    def __init__(self):
        self.captured_traces: list[Trace] = []
        self.lock = threading.Lock()

    def capture_trace(self, trace: Trace):
        """Capture a trace locally"""
        with self.lock:
            self.captured_traces.append(trace)

    def get_all_traces(self) -> list[Trace]:
        """Get all captured traces"""
        with self.lock:
            return self.captured_traces.copy()

    def get_latest_trace(self) -> Optional[Trace]:
        """Get the latest captured trace"""
        with self.lock:
            return self.captured_traces[-1] if self.captured_traces else None

    def find_trace_by_name(self, name: str) -> Optional[Trace]:
        """Find a trace by its name"""
        with self.lock:
            for trace in self.captured_traces:
                if trace.name == name or name in trace.name:
                    return trace
        return None

    def find_span_by_name(
        self, trace_name: str, span_name: str
    ) -> Optional[dict[str, Any]]:
        """Find a span by name within a trace"""
        trace = self.find_trace_by_name(trace_name)
        if not trace:
            return None

        for span in trace.spans:
            if span.name == span_name or span_name in span.name:
                return span.to_dict()
        return None

    def clear(self):
        """Clear all captured traces"""
        with self.lock:
            self.captured_traces.clear()

    def get_trace_count(self) -> int:
        """Get the number of captured traces"""
        with self.lock:
            return len(self.captured_traces)

    def get_total_span_count(self) -> int:
        """Get the total number of spans across all traces"""
        with self.lock:
            return sum(len(trace.spans) for trace in self.captured_traces)


@pytest.fixture
def trace_capture():
    """Provide a local trace capture instance"""
    capture = LocalTraceCapture()

    # Patch the requests.post to capture traces locally
    def mock_post(url, json=None, **kwargs):
        # Extract trace data from the request
        if json and "traces" in json:
            for trace_data in json["traces"]:
                # Create a mock trace object for capture
                mock_trace = Mock()
                mock_trace.trace_id = trace_data.get("trace_id", "test-trace-id")
                mock_trace.name = trace_data.get("name", "test-trace")
                mock_trace.spans = []

                # Convert span data to mock span objects
                for span_data in trace_data.get("spans", []):
                    mock_span = Mock()
                    mock_span.span_id = span_data.get("span_id", "test-span-id")
                    mock_span.name = span_data.get("name", "test-span")
                    mock_span.attributes = span_data.get("attributes", {})
                    mock_span.status = span_data.get("status", "ok")
                    mock_span.to_dict = Mock(return_value=span_data)
                    mock_trace.spans.append(mock_span)

                mock_trace.to_dict = Mock(return_value=trace_data)
                capture.capture_trace(mock_trace)

        # Return a mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        return mock_response

    with patch("requests.Session.post", side_effect=mock_post):
        yield capture


def wait_for_trace_capture(trace_capture, expected_count=1, max_wait=5.0):
    """Wait for traces to be captured locally"""
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if trace_capture.get_trace_count() >= expected_count:
            break
        time.sleep(0.1)


@pytest.mark.integration
@pytest.mark.disable_transport_mocking
class TestDecoratorsIntegration:
    """Integration tests for all decorators with local trace capture"""

    def setup_method(self):
        """Setup for each test method"""
        # Initialize noveum_trace with test configuration
        noveum_trace.init(
            api_key="test-api-key",
            project="decorator-integration-test",
            endpoint=ENDPOINT,
        )

    def teardown_method(self):
        """Cleanup after each test method"""
        # Don't shutdown the client as it interferes with other tests
        pass

    @pytest.mark.integration
    def test_trace_decorator_sends_data(self, trace_capture):
        """Test that @trace decorator actually generates trace data"""
        trace_capture.clear()

        @trace
        def basic_function(x: int, y: str) -> str:
            """A simple function to trace"""
            return f"{y}: {x * 2}"

        # Execute the function
        result = basic_function(10, "result")
        assert result == "result: 20"

        # Flush to ensure data is captured
        noveum_trace.get_client().flush()
        wait_for_trace_capture(trace_capture)

        # Verify trace was captured
        latest_trace = trace_capture.get_latest_trace()
        assert latest_trace is not None, "No trace data captured"

        # Verify trace structure
        assert "basic_function" in latest_trace.name
        assert len(latest_trace.spans) > 0, "No spans in trace"

        # Verify function span
        function_span = latest_trace.spans[0]
        assert "basic_function" in function_span.name

        # Verify function attributes
        attributes = function_span.attributes
        assert attributes.get("function.name") == "basic_function"
        assert attributes.get("function.args.x") == 10
        assert attributes.get("function.args.y") == "result"
        assert attributes.get("function.result") == "result: 20"

    @pytest.mark.integration
    @pytest.mark.llm
    def test_trace_llm_decorator_with_openai(self, trace_capture):
        """Test that @trace_llm decorator captures LLM trace data with real OpenAI call"""
        trace_capture.clear()

        @trace_llm
        def openai_call(prompt: str) -> str:
            """Real OpenAI LLM call"""
            try:
                import openai

                # Use environment variables for API key
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=50,
                )
                return response.choices[0].message.content
            except Exception:
                # Fallback to mock if OpenAI is not available
                return f"Mock response to: {prompt}"

        # Execute the LLM function
        result = openai_call("What is 2+2?")
        assert result is not None
        assert len(result) > 0

        # Flush to ensure data is captured
        noveum_trace.get_client().flush()
        wait_for_trace_capture(trace_capture)

        # Verify LLM trace was captured
        latest_trace = trace_capture.get_latest_trace()
        assert latest_trace is not None, "No LLM trace data captured"

        # Verify LLM-specific attributes
        assert len(latest_trace.spans) > 0, "No spans in LLM trace"

        llm_span = latest_trace.spans[0]
        attributes = llm_span.attributes

        # Check LLM-specific attributes
        assert attributes.get("function.type") == "llm_call"
        assert attributes.get("function.name") == "openai_call"
        assert "openai_call" in llm_span.name

    @pytest.mark.integration
    @pytest.mark.llm
    def test_trace_llm_decorator_with_anthropic(self, trace_capture):
        """Test that @trace_llm decorator captures LLM trace data with real Anthropic call"""
        trace_capture.clear()

        @trace_llm
        def anthropic_call(prompt: str) -> str:
            """Real Anthropic LLM call"""
            try:
                import anthropic

                # Use environment variables for API key
                client = anthropic.Anthropic()
                response = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=50,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
            except Exception:
                # Fallback to mock if Anthropic is not available
                return f"Mock response to: {prompt}"

        # Execute the LLM function
        result = anthropic_call("What is 2+2?")
        assert result is not None
        assert len(result) > 0

        # Flush to ensure data is captured
        noveum_trace.get_client().flush()
        wait_for_trace_capture(trace_capture)

        # Verify LLM trace was captured
        latest_trace = trace_capture.get_latest_trace()
        assert latest_trace is not None, "No Anthropic trace data captured"

        # Verify LLM-specific attributes
        assert len(latest_trace.spans) > 0, "No spans in Anthropic trace"

        llm_span = latest_trace.spans[0]
        attributes = llm_span.attributes

        # Check LLM-specific attributes
        assert attributes.get("function.type") == "llm_call"
        assert attributes.get("function.name") == "anthropic_call"
        assert "anthropic_call" in llm_span.name

    @pytest.mark.integration
    @pytest.mark.agent
    def test_trace_agent_decorator_captures_data(self, trace_capture):
        """Test that @trace_agent decorator captures agent trace data"""
        trace_capture.clear()

        @trace_agent("test-agent-001")
        def mock_agent_action(task: str, context: dict[str, Any]) -> dict[str, Any]:
            """Mock agent function"""
            return {
                "action": "analyze",
                "result": f"Analyzed: {task}",
                "confidence": 0.95,
            }

        # Execute the agent function
        result = mock_agent_action("Process user request", {"user_id": "123"})
        assert result["action"] == "analyze"
        assert result["confidence"] == 0.95

        # Flush to ensure data is captured
        noveum_trace.get_client().flush()
        wait_for_trace_capture(trace_capture)

        # Verify agent trace was captured
        latest_trace = trace_capture.get_latest_trace()
        assert latest_trace is not None, "No agent trace data captured"

        # Verify agent-specific attributes
        assert len(latest_trace.spans) > 0, "No spans in agent trace"

        agent_span = latest_trace.spans[0]
        attributes = agent_span.attributes

        # Check agent-specific attributes
        assert "mock_agent_action" in agent_span.name
        assert attributes.get("function.name") == "mock_agent_action"
        assert attributes.get("agent.id") == "test-agent-001"
        assert attributes.get("function.type") == "agent_operation"

        # Verify dict argument is stored as native dict (not JSON string)
        context_arg = attributes.get("agent.input.context")
        assert context_arg is not None, "Context argument should be captured"
        assert isinstance(
            context_arg, dict
        ), "Context should be stored as dict, not string"
        assert context_arg == {"user_id": "123"}, "Context dict should match input"

    @pytest.mark.integration
    @pytest.mark.tool
    def test_trace_tool_decorator_captures_data(self, trace_capture):
        """Test that @trace_tool decorator captures tool trace data"""
        trace_capture.clear()

        @trace_tool
        def mock_tool_call(query: str, max_results: int = 10) -> list[dict[str, Any]]:
            """Mock tool function"""
            return [
                {"title": "Result 1", "url": "https://example.com/1"},
                {"title": "Result 2", "url": "https://example.com/2"},
            ]

        # Execute the tool function
        result = mock_tool_call("Python testing", max_results=5)
        assert len(result) == 2
        assert result[0]["title"] == "Result 1"

        # Flush to ensure data is captured
        noveum_trace.get_client().flush()
        wait_for_trace_capture(trace_capture)

        # Verify tool trace was captured
        latest_trace = trace_capture.get_latest_trace()
        assert latest_trace is not None, "No tool trace data captured"

        # Verify tool-specific attributes
        assert len(latest_trace.spans) > 0, "No spans in tool trace"

        tool_span = latest_trace.spans[0]
        attributes = tool_span.attributes

        # Check tool-specific attributes
        assert "mock_tool_call" in tool_span.name
        assert attributes.get("function.name") == "mock_tool_call"
        assert attributes.get("function.type") == "tool_call"

        # Verify list return value is stored as native list (not JSON string)
        # Note: tool outputs are stored differently, but inputs should be dict/list
        query_arg = attributes.get("tool.input.query")
        assert query_arg == "Python testing", "Query argument should be captured"

    @pytest.mark.integration
    @pytest.mark.retrieval
    def test_trace_retrieval_decorator_captures_data(self, trace_capture):
        """Test that @trace_retrieval decorator captures retrieval trace data"""
        trace_capture.clear()

        @trace_retrieval
        def mock_retrieval_call(
            query: str, collection: str = "documents"
        ) -> list[dict[str, Any]]:
            """Mock retrieval function"""
            return [
                {"doc_id": "doc1", "content": "Relevant content 1", "score": 0.95},
                {"doc_id": "doc2", "content": "Relevant content 2", "score": 0.87},
            ]

        # Execute the retrieval function
        result = mock_retrieval_call("machine learning", collection="papers")
        assert len(result) == 2
        assert result[0]["score"] == 0.95

        # Flush to ensure data is captured
        noveum_trace.get_client().flush()
        wait_for_trace_capture(trace_capture)

        # Verify retrieval trace was captured
        latest_trace = trace_capture.get_latest_trace()
        assert latest_trace is not None, "No retrieval trace data captured"

        # Verify retrieval-specific attributes
        assert len(latest_trace.spans) > 0, "No spans in retrieval trace"

        retrieval_span = latest_trace.spans[0]
        attributes = retrieval_span.attributes

        # Check retrieval-specific attributes
        assert "mock_retrieval_call" in retrieval_span.name
        assert attributes.get("function.name") == "mock_retrieval_call"
        assert attributes.get("function.type") == "retrieval_operation"

    @pytest.mark.integration
    @pytest.mark.workflow
    def test_nested_decorators_capture_data(self, trace_capture):
        """Test that nested decorated functions capture comprehensive trace data"""
        trace_capture.clear()

        @trace_agent("orchestrator-001")
        def agent_orchestrator(task: str) -> dict[str, Any]:
            """Agent that orchestrates multiple operations"""

            # Call LLM
            llm_response = llm_analyze(task)

            # Call tool
            tool_results = search_tool(task)

            # Call retrieval
            retrieval_results = retrieve_docs(task)

            return {
                "llm_response": llm_response,
                "tool_results": tool_results,
                "retrieval_results": retrieval_results,
                "final_decision": "complete",
            }

        @trace_llm
        def llm_analyze(prompt: str) -> str:
            """LLM analysis function"""
            return f"Analysis: {prompt}"

        @trace_tool
        def search_tool(query: str) -> list[str]:
            """Search tool function"""
            return ["result1", "result2"]

        @trace_retrieval
        def retrieve_docs(query: str) -> list[dict[str, Any]]:
            """Document retrieval function"""
            return [{"doc": "doc1", "relevance": 0.9}]

        # Execute the nested operations
        result = agent_orchestrator("complex task")
        assert result["final_decision"] == "complete"
        assert "Analysis: complex task" in result["llm_response"]

        # Flush to ensure data is captured
        noveum_trace.get_client().flush()
        wait_for_trace_capture(trace_capture)

        # Verify traces were captured (nested calls create spans within the same trace)
        all_traces = trace_capture.get_all_traces()
        assert len(all_traces) >= 1, f"Expected at least 1 trace, got {len(all_traces)}"

        # Verify the main trace contains multiple spans for nested operations
        main_trace = all_traces[0]
        spans = main_trace.spans
        assert len(spans) >= 4, f"Expected at least 4 spans, got {len(spans)}"

        # Verify each type of operation exists as spans
        span_names = [span.name for span in spans]
        assert any(
            "agent_orchestrator" in name for name in span_names
        ), "No agent span found"
        assert any("llm_analyze" in name for name in span_names), "No LLM span found"
        assert any("search_tool" in name for name in span_names), "No tool span found"
        assert any(
            "retrieve_docs" in name for name in span_names
        ), "No retrieval span found"

    @pytest.mark.integration
    @pytest.mark.error_handling
    def test_decorator_error_handling_captures_data(self, trace_capture):
        """Test that decorators capture error information"""
        trace_capture.clear()

        @trace
        def failing_function(x: int) -> int:
            """Function that raises an error"""
            if x > 10:
                raise ValueError("Input too large")
            return x * 2

        # Execute function that will fail
        with pytest.raises(ValueError, match="Input too large"):
            failing_function(15)

        # Flush to ensure data is captured
        noveum_trace.get_client().flush()
        wait_for_trace_capture(trace_capture)

        # Verify error trace was captured
        latest_trace = trace_capture.get_latest_trace()
        assert latest_trace is not None, "No error trace data captured"

        # Verify error information
        assert len(latest_trace.spans) > 0, "No spans in error trace"

        error_span = latest_trace.spans[0]
        attributes = error_span.attributes

        # Check error attributes
        assert attributes.get("exception.type") == "ValueError"
        assert "Input too large" in attributes.get("exception.message", "")
        assert error_span.status == "error"

    @pytest.mark.integration
    @pytest.mark.async_support
    def test_async_decorator_captures_data(self, trace_capture):
        """Test that decorators work with async functions and capture data"""
        trace_capture.clear()

        @trace
        async def async_function(delay: float) -> str:
            """Async function to trace"""
            await asyncio.sleep(delay)
            return f"Completed after {delay}s"

        # Execute async function
        result = asyncio.run(async_function(0.1))
        assert result == "Completed after 0.1s"

        # Flush to ensure data is captured
        noveum_trace.get_client().flush()
        wait_for_trace_capture(trace_capture)

        # Verify async trace was captured
        latest_trace = trace_capture.get_latest_trace()
        assert latest_trace is not None, "No async trace data captured"

        # Verify async function attributes
        assert len(latest_trace.spans) > 0, "No spans in async trace"

        async_span = latest_trace.spans[0]
        attributes = async_span.attributes
        assert attributes.get("function.name") == "async_function"
        # The function is async, so it should have async-related attributes
        assert "async_function" in async_span.name

    @pytest.mark.integration
    def test_decorator_with_custom_attributes(self, trace_capture):
        """Test that decorators capture standard attributes"""
        trace_capture.clear()

        @trace
        def custom_function(data: str) -> str:
            """Function with custom attributes"""
            return data.upper()

        # Execute function
        result = custom_function("hello world")
        assert result == "HELLO WORLD"

        # Flush to ensure data is captured
        noveum_trace.get_client().flush()
        wait_for_trace_capture(trace_capture)

        # Verify standard attributes were captured
        latest_trace = trace_capture.get_latest_trace()
        assert latest_trace is not None, "No trace data captured"

        assert len(latest_trace.spans) > 0, "No spans in trace"

        custom_span = latest_trace.spans[0]
        attributes = custom_span.attributes
        assert attributes.get("function.name") == "custom_function"
        assert attributes.get("function.args.data") == "hello world"

    @pytest.mark.integration
    def test_decorator_dict_list_serialization(self, trace_capture):
        """Test that decorators properly serialize dict and list arguments as native types"""
        trace_capture.clear()

        @trace
        def function_with_collections(
            data_dict: dict[str, Any],
            data_list: list[str],
            nested_dict: dict[str, list[int]],
        ) -> dict[str, Any]:
            """Function that takes dict and list arguments"""
            return {
                "processed_dict": data_dict,
                "processed_list": data_list,
                "nested": nested_dict,
            }

        # Execute function with dict/list arguments
        input_dict = {"key1": "value1", "key2": 42}
        input_list = ["item1", "item2", "item3"]
        nested = {"numbers": [1, 2, 3], "more": [4, 5]}
        function_with_collections(input_dict, input_list, nested)

        # Flush to ensure data is captured
        noveum_trace.get_client().flush()
        wait_for_trace_capture(trace_capture)

        # Verify trace was captured
        latest_trace = trace_capture.get_latest_trace()
        assert latest_trace is not None, "No trace data captured"
        assert len(latest_trace.spans) > 0, "No spans in trace"

        span = latest_trace.spans[0]
        attributes = span.attributes

        # Verify dict argument is stored as native dict (not JSON string)
        captured_dict = attributes.get("function.args.data_dict")
        assert captured_dict is not None, "Dict argument should be captured"
        assert isinstance(
            captured_dict, dict
        ), "Dict should be stored as dict, not string"
        assert captured_dict == input_dict, "Dict should match input exactly"

        # Verify list argument is stored as native list (not JSON string)
        captured_list = attributes.get("function.args.data_list")
        assert captured_list is not None, "List argument should be captured"
        assert isinstance(
            captured_list, list
        ), "List should be stored as list, not string"
        assert captured_list == input_list, "List should match input exactly"

        # Verify nested dict with list values
        captured_nested = attributes.get("function.args.nested_dict")
        assert captured_nested is not None, "Nested dict argument should be captured"
        assert isinstance(captured_nested, dict), "Nested dict should be stored as dict"
        assert captured_nested == nested, "Nested dict should match input exactly"
        assert isinstance(
            captured_nested["numbers"], list
        ), "Nested list should be list"
        assert captured_nested["numbers"] == [1, 2, 3], "Nested list should match"

        # Verify result is also stored as native dict
        captured_result = attributes.get("function.result")
        assert captured_result is not None, "Result should be captured"
        assert isinstance(
            captured_result, dict
        ), "Result should be stored as dict, not string"
        assert (
            captured_result["processed_dict"] == input_dict
        ), "Result dict should preserve structure"
        assert (
            captured_result["processed_list"] == input_list
        ), "Result list should preserve structure"

    @pytest.mark.integration
    @pytest.mark.comprehensive
    def test_all_decorators_capture_data(self, trace_capture):
        """Test that all decorators capture data locally"""
        trace_capture.clear()

        @trace
        def base_func():
            return "base"

        @trace_llm
        def llm_func():
            return "llm"

        @trace_agent("test-agent")
        def agent_func():
            return "agent"

        @trace_tool
        def tool_func():
            return "tool"

        @trace_retrieval
        def retrieval_func():
            return "retrieval"

        # Execute all functions with individual flushes to ensure separate traces
        base_func()
        noveum_trace.get_client().flush()
        time.sleep(0.1)

        llm_func()
        noveum_trace.get_client().flush()
        time.sleep(0.1)

        agent_func()
        noveum_trace.get_client().flush()
        time.sleep(0.1)

        tool_func()
        noveum_trace.get_client().flush()
        time.sleep(0.1)

        retrieval_func()
        noveum_trace.get_client().flush()

        wait_for_trace_capture(trace_capture, expected_count=5)

        # Verify we got traces for all decorator types
        all_traces = trace_capture.get_all_traces()
        assert (
            len(all_traces) >= 5
        ), f"Expected at least 5 traces, got {len(all_traces)}"

        # Verify we have spans for each decorator type
        total_spans = sum(len(captured_trace.spans) for captured_trace in all_traces)
        assert (
            total_spans >= 5
        ), f"Expected at least 5 spans across all traces, got {total_spans}"

        # Check that each decorator type was captured
        span_types = []
        for captured_trace in all_traces:
            for span in captured_trace.spans:
                span_types.append(span.attributes.get("function.type", ""))

        # Verify all decorator types are represented
        assert "user_function" in span_types, "No base @trace spans found"
        assert "llm_call" in span_types, "No @trace_llm spans found"
        assert "agent_operation" in span_types, "No @trace_agent spans found"
        assert "tool_call" in span_types, "No @trace_tool spans found"
        assert "retrieval_operation" in span_types, "No @trace_retrieval spans found"


if __name__ == "__main__":
    pytest.main([__file__])
