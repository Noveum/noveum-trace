#!/usr/bin/env python3
"""
Test script to verify core tracing functionality is working correctly.
This tests the end-to-end tracing flow to ensure traces are created and exported properly.
"""

import os
import sys
from unittest.mock import Mock, patch

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import noveum_trace
from noveum_trace.core.client import NoveumClient
from noveum_trace.core.config import Config
from noveum_trace.core.trace import Trace
from noveum_trace.transport.http_transport import HttpTransport


def test_basic_trace_creation():
    """Test that we can create traces and spans properly."""
    print("🧪 Testing basic trace creation...")

    # Test trace creation
    trace = Trace("test-trace")
    assert trace.name == "test-trace"
    assert trace.trace_id is not None
    assert len(trace.spans) == 0
    print("✅ Trace creation works")

    # Test span creation
    span = trace.create_span("test-span")
    assert span.name == "test-span"
    assert span.trace_id == trace.trace_id
    assert span.span_id is not None
    print("✅ Span creation works")

    # Test span finishing
    span.finish()
    assert span.is_finished()
    assert len(trace.spans) == 1
    print("✅ Span finishing works")

    # Test trace attributes
    trace.set_attribute("test_key", "test_value")
    assert trace.attributes["test_key"] == "test_value"
    print("✅ Trace attributes work")


def test_client_functionality():
    """Test client creation and basic operations."""
    print("\n🧪 Testing client functionality...")

    config = Config.create(
        api_key="test-key", project="test-project", endpoint="https://api.test.com"
    )

    # Mock the HTTP transport to avoid real network calls
    with patch("noveum_trace.transport.http_transport.BatchProcessor"):
        client = NoveumClient(config=config)
        assert client.config.api_key == "test-key"
        assert client.config.project == "test-project"
        print("✅ Client creation works")

        # Test starting a trace
        trace = client.start_trace("client-trace")
        assert trace.name == "client-trace"
        assert trace.trace_id in client._active_traces
        print("✅ Client trace creation works")

        # Test starting a span
        span = client.start_span("client-span")
        assert span.name == "client-span"
        print("✅ Client span creation works")

        # Test finishing span
        client.finish_span(span)
        assert span.is_finished()
        print("✅ Client span finishing works")

        # Test finishing trace
        client.finish_trace(trace)
        assert trace.trace_id not in client._active_traces
        print("✅ Client trace finishing works")


def test_decorator_functionality():
    """Test that decorators work correctly."""
    print("\n🧪 Testing decorator functionality...")

    # Initialize the SDK with test config
    noveum_trace.init(
        project="test-project", api_key="test-key", endpoint="https://api.test.com"
    )

    # Test basic trace decorator
    @noveum_trace.trace
    def test_function(x, y):
        return x + y

    # Mock the transport to capture calls
    client = noveum_trace.get_client()
    original_start_trace = client.start_trace
    original_finish_trace = client.finish_trace
    original_start_span = client.start_span
    original_finish_span = client.finish_span

    trace_calls = []
    span_calls = []

    def mock_start_trace(*args, **kwargs):
        result = original_start_trace(*args, **kwargs)
        trace_calls.append(("start_trace", args, kwargs))
        return result

    def mock_finish_trace(*args, **kwargs):
        result = original_finish_trace(*args, **kwargs)
        trace_calls.append(("finish_trace", args, kwargs))
        return result

    def mock_start_span(*args, **kwargs):
        result = original_start_span(*args, **kwargs)
        span_calls.append(("start_span", args, kwargs))
        return result

    def mock_finish_span(*args, **kwargs):
        result = original_finish_span(*args, **kwargs)
        span_calls.append(("finish_span", args, kwargs))
        return result

    client.start_trace = mock_start_trace
    client.finish_trace = mock_finish_trace
    client.start_span = mock_start_span
    client.finish_span = mock_finish_span

    # Call the decorated function
    result = test_function(3, 4)
    assert result == 7
    print("✅ Decorated function executes correctly")

    # Check that tracing calls were made
    assert len(trace_calls) >= 2  # start and finish
    assert len(span_calls) >= 2  # start and finish
    print("✅ Decorator creates traces and spans")

    # Test LLM decorator
    @noveum_trace.trace_llm(provider="openai")
    def llm_function(prompt):
        return f"Response to: {prompt}"

    span_calls.clear()
    result = llm_function("Hello")
    assert result == "Response to: Hello"
    assert len(span_calls) >= 2  # start and finish span
    print("✅ LLM decorator works")


def test_transport_functionality():
    """Test that transport can format and handle traces."""
    print("\n🧪 Testing transport functionality...")

    config = Config.create(
        api_key="test-key", project="test-project", environment="test-env"
    )

    # Create transport with mocked batch processor
    captured_traces = []

    def mock_add_trace(trace_data):
        captured_traces.append(trace_data)

    with patch(
        "noveum_trace.transport.http_transport.BatchProcessor"
    ) as mock_batch_class:
        mock_batch_instance = Mock()
        mock_batch_instance.add_trace.side_effect = mock_add_trace
        mock_batch_class.return_value = mock_batch_instance

        transport = HttpTransport(config)
        transport.batch_processor = mock_batch_instance

        # Create a test trace
        trace = Trace("transport-test")
        trace.set_attribute("test", "value")
        span = trace.create_span("test-span")
        span.set_attribute("span_attr", "span_value")
        span.finish()

        # Export the trace
        transport.export_trace(trace)

        # Verify the trace was captured
        assert len(captured_traces) == 1
        trace_data = captured_traces[0]

        # Verify trace formatting
        assert trace_data["trace_id"] == trace.trace_id
        assert trace_data["name"] == "transport-test"
        assert trace_data["project"] == "test-project"
        assert trace_data["environment"] == "test-env"
        assert trace_data["sdk"]["name"] == "noveum-trace-python"
        assert "attributes" in trace_data
        assert "spans" in trace_data
        print("✅ Transport formats traces correctly")

        # Test that no-op traces are skipped
        noop_trace = Trace("noop-test")
        noop_trace._noop = True

        captured_traces.clear()
        transport.export_trace(noop_trace)
        assert len(captured_traces) == 0
        print("✅ No-op traces are properly skipped")


def test_context_management():
    """Test context management functionality."""
    print("\n🧪 Testing context management...")

    from noveum_trace.core.context import (
        clear_context,
        get_current_span,
        get_current_trace,
        set_current_span,
        set_current_trace,
    )

    # Clear any existing context
    clear_context()

    # Test that context starts empty
    assert get_current_trace() is None
    assert get_current_span() is None
    print("✅ Initial context is empty")

    # Test setting trace context
    trace = Trace("context-test")
    set_current_trace(trace)
    assert get_current_trace() == trace
    print("✅ Trace context setting works")

    # Test setting span context
    span = trace.create_span("context-span")
    set_current_span(span)
    assert get_current_span() == span
    print("✅ Span context setting works")

    # Test clearing context
    clear_context()
    assert get_current_trace() is None
    assert get_current_span() is None
    print("✅ Context clearing works")


def test_end_to_end_flow():
    """Test complete end-to-end tracing flow."""
    print("\n🧪 Testing end-to-end tracing flow...")

    # Capture all transport calls
    captured_traces = []

    def mock_add_trace(trace_data):
        captured_traces.append(trace_data)

    # Initialize SDK
    noveum_trace.shutdown()  # Clear any existing state

    with patch(
        "noveum_trace.transport.http_transport.BatchProcessor"
    ) as mock_batch_class:
        mock_batch_instance = Mock()
        mock_batch_instance.add_trace.side_effect = mock_add_trace
        mock_batch_class.return_value = mock_batch_instance

        # Initialize with test config
        noveum_trace.init(project="e2e-test", api_key="test-key", environment="test")

        # Override the transport's batch processor
        client = noveum_trace.get_client()
        client.transport.batch_processor = mock_batch_instance

        # Test manual trace creation
        trace = noveum_trace.start_trace("manual-trace")
        trace.set_attribute("manual", True)

        span = client.start_span("manual-span")
        span.set_attribute("span_type", "manual")

        # Simulate some work
        span.add_event("work_started")
        span.add_event("work_completed")

        # Finish span and trace
        client.finish_span(span)
        client.finish_trace(trace)

        # Test decorator-based tracing
        @noveum_trace.trace(name="e2e_function")
        def e2e_test_function(data):
            with noveum_trace.trace_operation("processing") as op_span:
                op_span.set_attribute("data_size", len(data))
                return data.upper()

        result = e2e_test_function("hello world")
        assert result == "HELLO WORLD"

        # Verify traces were captured
        assert len(captured_traces) >= 2  # manual trace + auto traces from decorator

        # Check manual trace
        manual_traces = [t for t in captured_traces if t.get("name") == "manual-trace"]
        assert len(manual_traces) == 1
        manual_trace = manual_traces[0]

        assert manual_trace["project"] == "e2e-test"
        assert manual_trace["environment"] == "test"
        assert manual_trace["attributes"]["manual"] is True
        assert len(manual_trace["spans"]) == 1
        assert manual_trace["spans"][0]["name"] == "manual-span"
        print("✅ Manual tracing works end-to-end")

        # Check decorator traces
        auto_traces = [t for t in captured_traces if "auto_trace_" in t.get("name", "")]
        assert len(auto_traces) >= 1
        print("✅ Decorator tracing works end-to-end")

        print("✅ End-to-end flow works correctly")


def main():
    """Run all functionality tests."""
    print("🚀 Starting comprehensive tracing functionality tests...\n")

    try:
        test_basic_trace_creation()
        test_client_functionality()
        test_decorator_functionality()
        test_transport_functionality()
        test_context_management()
        test_end_to_end_flow()

        print("\n🎉 All functionality tests passed!")
        print("✅ Core tracing functionality is working correctly")
        print("✅ Traces are being created, formatted, and exported properly")
        print("✅ Decorators are functioning as expected")
        print("✅ Transport layer is working correctly")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        # Clean up
        try:
            noveum_trace.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
