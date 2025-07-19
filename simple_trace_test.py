#!/usr/bin/env python3
"""
Simple test to verify core tracing functionality works.
"""

import os
import sys

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_basic_functionality():
    print("🧪 Testing basic trace and span creation...")

    from noveum_trace.core.trace import Trace

    # Test trace creation
    trace = Trace("test-trace")
    assert trace.name == "test-trace"
    assert trace.trace_id is not None
    print("✅ Trace creation works")

    # Test span creation
    span = trace.create_span("test-span")
    assert span.name == "test-span"
    assert span.trace_id == trace.trace_id
    print("✅ Span creation works")

    # Test attributes
    trace.set_attribute("key", "value")
    assert trace.attributes["key"] == "value"
    print("✅ Attributes work")

    # Test span finishing
    span.finish()
    assert span.is_finished()
    print("✅ Span finishing works")


def test_client_basic():
    print("\n🧪 Testing client basic functionality...")

    from unittest.mock import patch

    from noveum_trace.core.client import NoveumClient
    from noveum_trace.core.config import Config

    config = Config.create(api_key="test", project="test")

    with patch("noveum_trace.transport.http_transport.BatchProcessor"):
        client = NoveumClient(config=config)
        assert client.config.api_key == "test"
        print("✅ Client creation works")

        # Test trace creation via client
        trace = client.start_trace("client-trace")
        assert trace.name == "client-trace"
        print("✅ Client trace creation works")

        # Test span creation via client
        span = client.start_span("client-span")
        assert span.name == "client-span"
        print("✅ Client span creation works")


def test_transport_basic():
    print("\n🧪 Testing transport basic functionality...")

    from unittest.mock import Mock, patch

    from noveum_trace.core.config import Config
    from noveum_trace.core.trace import Trace
    from noveum_trace.transport.http_transport import HttpTransport

    config = Config.create(api_key="test", project="test")

    with patch(
        "noveum_trace.transport.http_transport.BatchProcessor"
    ) as mock_batch_class:
        mock_batch_instance = Mock()
        mock_batch_class.return_value = mock_batch_instance

        transport = HttpTransport(config)
        transport.batch_processor = mock_batch_instance

        # Test trace export
        trace = Trace("transport-test")
        transport.export_trace(trace)

        # Verify add_trace was called
        mock_batch_instance.add_trace.assert_called_once()
        print("✅ Transport export works")


def test_sdk_init():
    print("\n🧪 Testing SDK initialization...")

    from unittest.mock import patch

    import noveum_trace

    # Clear any existing state
    try:
        noveum_trace.shutdown()
    except Exception:
        pass

    with patch("noveum_trace.transport.http_transport.BatchProcessor"):
        # Test SDK initialization
        noveum_trace.init(project="test", api_key="test")
        assert noveum_trace.is_initialized()
        print("✅ SDK initialization works")

        # Test getting client
        client = noveum_trace.get_client()
        assert client is not None
        print("✅ Client retrieval works")


def main():
    print("🚀 Running simple trace functionality tests...\n")

    try:
        test_basic_functionality()
        test_client_basic()
        test_transport_basic()
        test_sdk_init()

        print("\n🎉 All basic tests passed!")
        print("✅ Core tracing functionality is working")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
