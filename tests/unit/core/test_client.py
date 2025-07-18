"""
Comprehensive unit tests for NoveumClient.

This module tests all aspects of the NoveumClient including trace/span lifecycle,
context management, error handling, shutdown behavior, and thread safety.
Includes parameterized tests for different configurations and models.
"""

from unittest.mock import patch

import pytest

from noveum_trace.core.client import NoveumClient, SamplingDecision, should_sample
from noveum_trace.core.config import Config
from noveum_trace.utils.exceptions import NoveumTraceError


class TestSamplingDecision:
    """Test sampling decision logic."""

    @pytest.mark.parametrize(
        "sample_rate,expected",
        [
            (1.0, SamplingDecision.RECORD),
            (0.0, SamplingDecision.DROP),
            (1.1, SamplingDecision.RECORD),  # Above 1.0 should record
            (-0.1, SamplingDecision.DROP),  # Below 0.0 should drop
        ],
    )
    def test_should_sample_deterministic_rates(self, sample_rate, expected):
        """Test sampling with deterministic rates."""
        decision = should_sample("test", sample_rate=sample_rate)
        assert decision == expected

    def test_should_sample_partial_rate_distribution(self):
        """Test sampling with partial rate shows expected distribution."""
        # Test multiple times to verify randomness
        record_count = 0
        drop_count = 0

        for _ in range(100):
            decision = should_sample("test", sample_rate=0.5)
            if decision == SamplingDecision.RECORD:
                record_count += 1
            else:
                drop_count += 1

        # With 100 samples at 50% rate, we should get roughly half
        # Allow for some variance due to randomness
        assert 30 <= record_count <= 70
        assert 30 <= drop_count <= 70
        assert record_count + drop_count == 100

    @pytest.mark.parametrize(
        "name,attributes",
        [
            ("simple_trace", None),
            ("trace_with_attrs", {"key": "value"}),
            ("", {}),  # Edge case: empty name
        ],
    )
    def test_should_sample_with_different_inputs(self, name, attributes):
        """Test sampling with different input parameters."""
        decision = should_sample(name, attributes=attributes, sample_rate=1.0)
        assert decision == SamplingDecision.RECORD


class TestNoveumClientBasic:
    """Test basic NoveumClient functionality."""

    def test_client_initialization(self):
        """Test client initializes correctly."""
        client = NoveumClient()
        assert client is not None
        assert hasattr(client, "config")
        assert hasattr(client, "transport")
        assert not client._shutdown
        assert len(client._active_traces) == 0

    def test_client_with_custom_config(self):
        """Test client initialization with custom config."""
        config = Config.create(
            api_key="custom-key",
            project="custom-project",
            endpoint="http://localhost:8080",
        )

        client = NoveumClient(config=config)
        assert client.config.api_key == config.api_key
        assert client.config.project == config.project
        assert client.config.endpoint == config.endpoint

    def test_client_repr(self):
        """Test client string representation."""
        client = NoveumClient()
        repr_str = repr(client)
        assert "NoveumClient" in repr_str
        assert "active_traces=0" in repr_str


class TestNoveumClientTraceManagement:
    """Test trace creation and management."""

    def test_start_trace_basic(self):
        """Test basic trace creation."""
        client = NoveumClient()
        trace = client.start_trace("test_trace")

        assert trace is not None
        assert trace.name == "test_trace"
        assert trace.trace_id in client._active_traces
        assert len(client._active_traces) == 1

    def test_start_trace_with_attributes(self):
        """Test trace creation with attributes."""
        client = NoveumClient()
        attributes = {"user_id": "123", "session": "abc"}

        trace = client.start_trace("test_trace", attributes=attributes)

        assert "user_id" in trace.attributes
        assert trace.attributes["user_id"] == "123"

    def test_start_trace_when_shutdown(self):
        """Test trace creation fails when client is shutdown."""
        client = NoveumClient()
        client._shutdown = True

        with pytest.raises(NoveumTraceError, match="Client has been shutdown"):
            client.start_trace("test_trace")

    def test_get_active_traces(self):
        """Test getting active traces."""
        client = NoveumClient()

        trace1 = client.start_trace("trace1")
        trace2 = client.start_trace("trace2")

        active_traces = client.get_active_traces()
        assert len(active_traces) == 2
        assert trace1 in active_traces
        assert trace2 in active_traces

    def test_get_trace_by_id(self):
        """Test getting trace by ID."""
        client = NoveumClient()
        trace = client.start_trace("test_trace")

        found_trace = client.get_trace(trace.trace_id)
        assert found_trace == trace

        # Test non-existent trace
        assert client.get_trace("non-existent") is None

    def test_finish_trace(self):
        """Test finishing a trace."""
        client = NoveumClient()
        trace = client.start_trace("test_trace")

        # Mock _export_trace to avoid transport calls
        with patch.object(client, "_export_trace"):
            client.finish_trace(trace)

        assert trace.is_finished()
        assert trace.trace_id not in client._active_traces


class TestNoveumClientSpanManagement:
    """Test span creation and management."""

    def test_start_span_requires_active_trace(self):
        """Test span creation fails when no trace is active."""
        client = NoveumClient()

        # Mock context to return None (no active trace)
        with patch("noveum_trace.core.client.get_current_trace", return_value=None):
            with pytest.raises(NoveumTraceError, match="No active trace found"):
                client.start_span("test_span")

    def test_start_span_when_shutdown(self):
        """Test span creation fails when client is shutdown."""
        client = NoveumClient()
        client._shutdown = True

        with pytest.raises(NoveumTraceError, match="Client has been shutdown"):
            client.start_span("test_span")

    def test_start_span_with_active_trace(self):
        """Test span creation with active trace."""
        client = NoveumClient()
        trace = client.start_trace("test_trace")

        with patch("noveum_trace.core.client.get_current_trace", return_value=trace):
            span = client.start_span("test_span")

            assert span is not None
            assert span.name == "test_span"
            assert span.trace_id == trace.trace_id

    def test_start_span_with_parent_context(self):
        """Test span creation with parent span from context."""
        client = NoveumClient()

        # Create trace and parent span
        trace = client.start_trace("test_trace")
        parent_span = trace.create_span("parent_span")

        with (
            patch("noveum_trace.core.client.get_current_trace", return_value=trace),
            patch(
                "noveum_trace.core.client.get_current_span", return_value=parent_span
            ),
        ):

            span = client.start_span("child_span")

            assert span.parent_span_id == parent_span.span_id

    def test_finish_span(self):
        """Test finishing a span."""
        client = NoveumClient()
        trace = client.start_trace("test_trace")
        span = trace.create_span("test_span")

        client.finish_span(span)

        assert span.is_finished()

    def test_finish_span_clears_context(self):
        """Test finishing span clears it from context."""
        client = NoveumClient()
        trace = client.start_trace("test_trace")
        span = trace.create_span("test_span")

        with (
            patch("noveum_trace.core.client.get_current_span", return_value=span),
            patch("noveum_trace.core.client.set_current_span") as mock_set_span,
        ):

            client.finish_span(span)

            # Should call set_current_span to clear the span
            mock_set_span.assert_called()


class TestNoveumClientFlushAndShutdown:
    """Test flush and shutdown functionality."""

    def test_flush_active_traces(self):
        """Test flushing with active traces."""
        with patch("noveum_trace.core.client.HttpTransport"):
            client = NoveumClient()

            # Create some active traces
            trace1 = client.start_trace("trace1")
            trace2 = client.start_trace("trace2")

            # Mock the export to avoid actual HTTP calls
            with patch.object(client, "_export_trace"):
                client.flush()

                # Should finish all active traces
                assert trace1.is_finished()
                assert trace2.is_finished()
                assert len(client._active_traces) == 0

                # Transport flush should be called
                client.transport.flush.assert_called_once_with(None)

    def test_flush_when_shutdown(self):
        """Test flush when client is already shutdown."""
        with patch("noveum_trace.core.client.HttpTransport"):
            client = NoveumClient()
            client._shutdown = True

            # Should return early without doing anything
            client.flush()

            # No transport calls should be made
            client.transport.flush.assert_not_called()

    def test_shutdown(self):
        """Test complete shutdown process."""
        with patch("noveum_trace.core.client.HttpTransport"):
            client = NoveumClient()

            # Create some active traces
            trace1 = client.start_trace("trace1")

            with patch.object(client, "_export_trace"):
                client.shutdown()

                assert client._shutdown is True
                assert trace1.is_finished()
                assert trace1.trace_id not in client._active_traces

    def test_shutdown_idempotent(self):
        """Test shutdown is idempotent."""
        with patch("noveum_trace.core.client.HttpTransport"):
            client = NoveumClient()

            with patch.object(client, "_export_trace"):
                client.shutdown()
                client.shutdown()  # Second call should be safe

                assert client._shutdown is True

    def test_is_shutdown(self):
        """Test is_shutdown status check."""
        with patch("noveum_trace.core.client.HttpTransport"):
            client = NoveumClient()

            assert not client.is_shutdown()

            with patch.object(client, "_export_trace"):
                client.shutdown()

            assert client.is_shutdown()


class TestNoveumClientContextual:
    """Test contextual trace and span creation."""

    def test_create_contextual_trace(self):
        """Test creating contextual trace."""
        client = NoveumClient()

        contextual_trace = client.create_contextual_trace("test_trace")

        assert contextual_trace is not None
        assert hasattr(contextual_trace, "trace")
        assert contextual_trace.trace.name == "test_trace"

    def test_create_contextual_span(self):
        """Test creating contextual span."""
        client = NoveumClient()
        trace = client.start_trace("test_trace")

        with patch("noveum_trace.core.client.get_current_trace", return_value=trace):
            contextual_span = client.create_contextual_span("test_span")

            assert contextual_span is not None
            assert hasattr(contextual_span, "span")
            assert contextual_span.span.name == "test_span"


class TestNoveumClientSampling:
    """Test sampling functionality."""

    @patch("noveum_trace.core.client.should_sample")
    def test_trace_sampling_drop(self, mock_should_sample):
        """Test trace is dropped when sampling says drop."""
        from noveum_trace.core.client import SamplingDecision

        mock_should_sample.return_value = SamplingDecision.DROP

        client = NoveumClient()
        trace = client.start_trace("test_trace")

        # Should return a no-op trace
        assert hasattr(trace, "_noop")
        assert trace._noop is True

    @patch("noveum_trace.core.client.should_sample")
    def test_trace_sampling_record(self, mock_should_sample):
        """Test trace is recorded when sampling says record."""
        from noveum_trace.core.client import SamplingDecision

        mock_should_sample.return_value = SamplingDecision.RECORD

        client = NoveumClient()
        trace = client.start_trace("test_trace")

        # Should be a normal trace
        assert not hasattr(trace, "_noop") or not trace._noop
        assert trace.trace_id in client._active_traces


class TestNoveumClientIntegration:
    """Test client integration scenarios."""

    def test_complex_span_hierarchy(self):
        """Test complex span parent-child hierarchies."""
        client = NoveumClient()
        trace = client.start_trace("hierarchy_test")

        with patch("noveum_trace.core.client.get_current_trace", return_value=trace):
            # Create root span
            root_span = client.start_span("root")

            # Create child spans with explicit parent context
            with patch(
                "noveum_trace.core.client.get_current_span", return_value=root_span
            ):
                child1 = client.start_span("child1")

            with patch(
                "noveum_trace.core.client.get_current_span", return_value=root_span
            ):
                child2 = client.start_span("child2")

                # Create grandchild spans
                with patch(
                    "noveum_trace.core.client.get_current_span", return_value=child1
                ):
                    grandchild1 = client.start_span("grandchild1")

                with patch(
                    "noveum_trace.core.client.get_current_span", return_value=child1
                ):
                    grandchild2 = client.start_span("grandchild2")

            # Verify hierarchy
            assert root_span.parent_span_id is None
            assert child1.parent_span_id == root_span.span_id
            assert child2.parent_span_id == root_span.span_id
            assert grandchild1.parent_span_id == child1.span_id
            assert grandchild2.parent_span_id == child1.span_id

    def test_trace_lifecycle_complete(self):
        """Test complete trace lifecycle."""
        client = NoveumClient()

        # Start trace
        trace = client.start_trace("lifecycle_test")

        with patch("noveum_trace.core.client.get_current_trace", return_value=trace):
            # Add spans
            span1 = client.start_span("span1")
            span2 = client.start_span("span2")

            # Add attributes
            trace.set_attributes({"test": "value"})
            span1.set_attributes({"span_attr": "value"})

            # Finish spans
            client.finish_span(span1)
            client.finish_span(span2)

        # Finish trace
        with patch.object(client, "_export_trace"):
            client.finish_trace(trace)

        assert trace.is_finished()
        assert span1.is_finished()
        assert span2.is_finished()
        assert trace.trace_id not in client._active_traces

    def test_error_handling_in_export(self):
        """Test error handling during trace export."""
        with patch("noveum_trace.core.client.HttpTransport"):
            client = NoveumClient()
            trace = client.start_trace("test_trace")

            # Mock transport to raise exception
            client.transport.export_trace.side_effect = Exception("Export failed")

            # Should not raise exception, should log error
            with patch("noveum_trace.core.client.logger") as mock_logger:
                client._export_trace(trace)
                # Error should be logged by the implementation
                mock_logger.error.assert_called_once()

    def test_multiple_client_instances(self):
        """Test multiple client instances work independently."""
        client1 = NoveumClient()
        client2 = NoveumClient()

        trace1 = client1.start_trace("trace1")
        trace2 = client2.start_trace("trace2")

        # Each client should have its own active traces
        assert len(client1._active_traces) == 1
        assert len(client2._active_traces) == 1
        assert trace1.trace_id in client1._active_traces
        assert trace2.trace_id in client2._active_traces
        assert trace1.trace_id not in client2._active_traces
        assert trace2.trace_id not in client1._active_traces
