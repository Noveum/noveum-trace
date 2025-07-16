"""
Pytest configuration and shared fixtures for Noveum Trace SDK tests.
"""

import pytest

from noveum_trace import NoveumTracer, TracerConfig
from noveum_trace.sinks.base import BaseSink, SinkConfig


class MockSink(BaseSink):
    """Mock sink for testing."""

    def __init__(self, name: str = "mock-sink"):
        config = SinkConfig(name=name)
        self.sent_spans = []
        super().__init__(config)

    def _initialize(self) -> None:
        """Initialize mock sink."""
        pass

    def _send_batch(self, spans) -> None:
        """Mock send batch implementation."""
        self.sent_spans.extend(spans)

    def _health_check(self) -> bool:
        """Mock health check."""
        return True

    def _shutdown(self) -> None:
        """Mock shutdown implementation."""
        pass

    def clear_spans(self):
        """Clear sent spans for testing."""
        self.sent_spans.clear()


@pytest.fixture
def mock_sink():
    """Provide a mock sink for testing."""
    return MockSink()


@pytest.fixture
def tracer_config(mock_sink):
    """Provide a test tracer configuration."""
    return TracerConfig(
        project_id="test-project",
        project_name="test-project",
        environment="test",
        sinks=[mock_sink],
        batch_size=1,  # Small batch for immediate processing
        batch_timeout_ms=100,
        max_queue_size=100,
        sampling_rate=1.0,
    )


@pytest.fixture
def tracer(tracer_config):
    """Provide a test tracer instance."""
    tracer = NoveumTracer(tracer_config)
    yield tracer
    tracer.shutdown()
