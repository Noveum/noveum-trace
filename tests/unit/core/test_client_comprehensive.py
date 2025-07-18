"""
Comprehensive tests for NoveumClient implementation.

This module provides extensive test coverage for the NoveumClient class,
including initialization, configuration, trace management, and all edge cases.
"""

import threading
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from noveum_trace.core.client import NoveumClient, SamplingDecision, should_sample
from noveum_trace.core.config import Config
from noveum_trace.core.span import Span
from noveum_trace.core.trace import Trace
from noveum_trace.utils.exceptions import NoveumTraceError


class TestSamplingDecision:
    """Test sampling decision functionality."""

    def test_should_sample_always_record(self):
        """Test sampling always records when rate is 1.0."""
        decision = should_sample("test", sample_rate=1.0)
        assert decision == SamplingDecision.RECORD

    def test_should_sample_never_record(self):
        """Test sampling never records when rate is 0.0."""
        decision = should_sample("test", sample_rate=0.0)
        assert decision == SamplingDecision.DROP

    def test_should_sample_above_one(self):
        """Test sampling records when rate is above 1.0."""
        decision = should_sample("test", sample_rate=1.5)
        assert decision == SamplingDecision.RECORD

    def test_should_sample_below_zero(self):
        """Test sampling drops when rate is below 0.0."""
        decision = should_sample("test", sample_rate=-0.5)
        assert decision == SamplingDecision.DROP

    def test_should_sample_random(self):
        """Test random sampling behavior."""
        # Test with 50% sample rate - should have both outcomes
        results = []
        for i in range(100):
            decision = should_sample(f"test-{i}", sample_rate=0.5)
            results.append(decision)

        # Should have both RECORD and DROP decisions
        assert SamplingDecision.RECORD in results
        assert SamplingDecision.DROP in results

    def test_should_sample_with_attributes(self):
        """Test sampling with attributes (currently unused)."""
        decision = should_sample("test", attributes={"key": "value"}, sample_rate=1.0)
        assert decision == SamplingDecision.RECORD


class TestNoveumClientInitialization:
    """Test NoveumClient initialization and configuration."""

    def test_init_with_config(self):
        """Test initialization with provided config."""
        config = Config.create(api_key="test-key", project="test-project")

        with patch("noveum_trace.core.client.HttpTransport") as mock_transport:
            client = NoveumClient(config=config)

            assert client.config == config
            assert client._shutdown is False
            assert isinstance(client._active_traces, dict)
            assert isinstance(client._lock, threading.RLock)
            mock_transport.assert_called_once_with(config)

    def test_init_with_parameters(self):
        """Test initialization with individual parameters."""
        with patch("noveum_trace.core.client.HttpTransport"):
            with patch("noveum_trace.core.client.configure") as mock_configure:
                with patch("noveum_trace.core.client.get_config") as mock_get_config:
                    mock_config = Mock()
                    mock_get_config.return_value = mock_config

                    client = NoveumClient(
                        api_key="test-key",
                        project="test-project",
                        custom_param="custom-value",
                    )

                    # Verify configuration was called
                    mock_configure.assert_called_once_with(
                        {
                            "api_key": "test-key",
                            "project": "test-project",
                            "custom_param": "custom-value",
                        }
                    )

                    assert client.config == mock_config

    def test_init_with_only_api_key(self):
        """Test initialization with only API key."""
        with patch("noveum_trace.core.client.HttpTransport"):
            with patch("noveum_trace.core.client.configure") as mock_configure:
                with patch("noveum_trace.core.client.get_config") as mock_get_config:
                    mock_config = Mock()
                    mock_get_config.return_value = mock_config

                    client = NoveumClient(api_key="test-key")

                    mock_configure.assert_called_once_with({"api_key": "test-key"})
                    assert client.config == mock_config

    def test_init_with_only_project(self):
        """Test initialization with only project."""
        with patch("noveum_trace.core.client.HttpTransport"):
            with patch("noveum_trace.core.client.configure") as mock_configure:
                with patch("noveum_trace.core.client.get_config") as mock_get_config:
                    mock_config = Mock()
                    mock_get_config.return_value = mock_config

                    client = NoveumClient(project="test-project")

                    mock_configure.assert_called_once_with({"project": "test-project"})
                    assert client.config == mock_config

    def test_init_with_only_kwargs(self):
        """Test initialization with only kwargs."""
        with patch("noveum_trace.core.client.HttpTransport"):
            with patch("noveum_trace.core.client.configure") as mock_configure:
                with patch("noveum_trace.core.client.get_config") as mock_get_config:
                    mock_config = Mock()
                    mock_get_config.return_value = mock_config

                    client = NoveumClient(endpoint="https://custom.api.com")

                    mock_configure.assert_called_once_with(
                        {"endpoint": "https://custom.api.com"}
                    )
                    assert client.config == mock_config

    def test_init_without_parameters(self):
        """Test initialization without any parameters."""
        with patch("noveum_trace.core.client.HttpTransport"):
            with patch("noveum_trace.core.client.get_config") as mock_get_config:
                mock_config = Mock()
                mock_get_config.return_value = mock_config

                client = NoveumClient()

                assert client.config == mock_config

    def test_init_registers_shutdown_handler(self):
        """Test that initialization registers shutdown handler."""
        with patch("noveum_trace.core.client.HttpTransport"):
            with patch("noveum_trace.core.client.atexit.register") as mock_register:
                client = NoveumClient()

                mock_register.assert_called_once_with(client.shutdown)

    def test_init_logs_message(self, caplog):
        """Test that initialization logs message."""
        with patch("noveum_trace.core.client.HttpTransport"):
            NoveumClient()

            assert "Noveum Trace client initialized" in caplog.text

    def test_get_sdk_version(self):
        """Test SDK version retrieval."""
        with patch("noveum_trace.core.client.HttpTransport"):
            client = NoveumClient()

            version = client._get_sdk_version()
            assert isinstance(version, str)
            assert len(version) > 0


class TestNoveumClientTraceOperations:
    """Test trace creation and management operations."""

    def test_start_trace_basic(self):
        """Test basic trace creation."""
        config = Config.create()

        with patch("noveum_trace.core.client.HttpTransport"):
            client = NoveumClient(config=config)

            trace = client.start_trace("test-trace")

            assert isinstance(trace, Trace)
            assert trace.name == "test-trace"
            assert trace.trace_id in client._active_traces
            assert client._active_traces[trace.trace_id] == trace

    def test_start_trace_with_attributes(self):
        """Test trace creation with attributes."""
        config = Config.create()

        with patch("noveum_trace.core.client.HttpTransport"):
            client = NoveumClient(config=config)

            attributes = {"key": "value", "number": 42}
            trace = client.start_trace("test-trace", attributes=attributes)

            assert trace.attributes.get("key") == "value"
            assert trace.attributes.get("number") == 42

    def test_start_trace_with_start_time(self):
        """Test trace creation with custom start time."""
        config = Config.create()

        with patch("noveum_trace.core.client.HttpTransport"):
            client = NoveumClient(config=config)

            start_time = datetime.now()
            trace = client.start_trace("test-trace", start_time=start_time)

            assert trace.start_time == start_time

    def test_start_trace_not_set_as_current(self):
        """Test trace creation without setting as current."""
        config = Config.create()

        with patch("noveum_trace.core.client.HttpTransport"):
            with patch(
                "noveum_trace.core.client.set_current_trace"
            ) as mock_set_current:
                client = NoveumClient(config=config)

                client.start_trace("test-trace", set_as_current=False)

                mock_set_current.assert_not_called()

    def test_start_trace_set_as_current(self):
        """Test trace creation with setting as current."""
        config = Config.create()

        with patch("noveum_trace.core.client.HttpTransport"):
            with patch(
                "noveum_trace.core.client.set_current_trace"
            ) as mock_set_current:
                client = NoveumClient(config=config)

                trace = client.start_trace("test-trace", set_as_current=True)

                mock_set_current.assert_called_once_with(trace)

    def test_start_trace_adds_config_attributes(self):
        """Test trace creation adds configuration attributes."""
        config = Config.create(project="test-project", environment="test-env")

        with patch("noveum_trace.core.client.HttpTransport"):
            client = NoveumClient(config=config)

            trace = client.start_trace("test-trace")

            assert trace.attributes.get("noveum.project") == "test-project"
            assert trace.attributes.get("noveum.environment") == "test-env"
            assert (
                trace.attributes.get("noveum.sdk.version") == client._get_sdk_version()
            )
            assert (
                trace.attributes.get("noveum.sampling.decision")
                == SamplingDecision.RECORD.value
            )

    def test_start_trace_when_shutdown(self):
        """Test trace creation when client is shutdown."""
        config = Config.create()

        with patch("noveum_trace.core.client.HttpTransport"):
            client = NoveumClient(config=config)
            client._shutdown = True

            with pytest.raises(NoveumTraceError, match="Client has been shutdown"):
                client.start_trace("test-trace")

    def test_start_trace_when_tracing_disabled(self):
        """Test trace creation when tracing is disabled."""
        config = Config.create()
        config.tracing.enabled = False

        with patch("noveum_trace.core.client.HttpTransport"):
            client = NoveumClient(config=config)

            trace = client.start_trace("test-trace")

            assert hasattr(trace, "_noop")
            assert trace._noop is True

    def test_start_trace_when_sampled_out(self):
        """Test trace creation when sampled out."""
        config = Config.create()
        config.tracing.sample_rate = 0.0

        with patch("noveum_trace.core.client.HttpTransport"):
            client = NoveumClient(config=config)

            trace = client.start_trace("test-trace")

            assert hasattr(trace, "_noop")
            assert trace._noop is True

    def test_start_trace_logs_debug(self, caplog):
        """Test trace creation logs debug message."""
        config = Config.create()

        with patch("noveum_trace.core.client.HttpTransport"):
            client = NoveumClient(config=config)

            with caplog.at_level("DEBUG"):
                trace = client.start_trace("test-trace")

            assert f"Started trace: {trace.trace_id}" in caplog.text

    def test_finish_trace_basic(self):
        """Test basic trace finishing."""
        config = Config.create()

        with patch("noveum_trace.core.client.HttpTransport"):
            client = NoveumClient(config=config)

            trace = client.start_trace("test-trace")
            trace_id = trace.trace_id

            client.finish_trace(trace)

            assert trace.is_finished()
            assert trace_id not in client._active_traces

    def test_finish_trace_with_end_time(self):
        """Test trace finishing with custom end time."""
        config = Config.create()

        with patch("noveum_trace.core.client.HttpTransport"):
            client = NoveumClient(config=config)

            trace = client.start_trace("test-trace")
            end_time = datetime.now()

            client.finish_trace(trace, end_time=end_time)

            assert trace.end_time == end_time

    def test_finish_trace_already_finished(self):
        """Test finishing already finished trace."""
        config = Config.create()

        with patch("noveum_trace.core.client.HttpTransport"):
            client = NoveumClient(config=config)

            trace = client.start_trace("test-trace")
            trace.finish()  # Finish directly

            # Should not raise error
            client.finish_trace(trace)

    def test_finish_trace_clears_current_context(self):
        """Test finishing trace clears current context."""
        config = Config.create()

        with patch("noveum_trace.core.client.HttpTransport"):
            with patch(
                "noveum_trace.core.client.get_current_trace"
            ) as mock_get_current:
                with patch(
                    "noveum_trace.core.client.set_current_trace"
                ) as mock_set_current:
                    with patch(
                        "noveum_trace.core.client.set_current_span"
                    ) as mock_set_span:
                        client = NoveumClient(config=config)

                        trace = client.start_trace("test-trace")
                        mock_get_current.return_value = trace

                        client.finish_trace(trace)

                        mock_set_current.assert_called_once_with(None)
                        mock_set_span.assert_called_once_with(None)

    def test_finish_trace_exports_trace(self):
        """Test finishing trace exports it."""
        config = Config.create()

        with patch("noveum_trace.core.client.HttpTransport"):
            client = NoveumClient(config=config)
            client._export_trace = Mock()

            trace = client.start_trace("test-trace")

            client.finish_trace(trace)

            client._export_trace.assert_called_once_with(trace)

    def test_finish_trace_logs_debug(self, caplog):
        """Test finishing trace logs debug message."""
        config = Config.create()

        with patch("noveum_trace.core.client.HttpTransport"):
            client = NoveumClient(config=config)

            trace = client.start_trace("test-trace")

            with caplog.at_level("DEBUG"):
                client.finish_trace(trace)

            assert f"Finished trace: {trace.trace_id}" in caplog.text


class TestNoveumClientSpanOperations:
    """Test span creation and management operations."""

    def test_start_span_basic(self):
        """Test basic span creation."""
        config = Config.create()

        with patch("noveum_trace.core.client.HttpTransport"):
            with patch("noveum_trace.core.client.get_current_trace") as mock_get_trace:
                client = NoveumClient(config=config)

                # Mock current trace
                mock_trace = Mock(spec=Trace)
                mock_span = Mock(spec=Span)
                mock_trace.create_span.return_value = mock_span
                mock_get_trace.return_value = mock_trace

                span = client.start_span("test-span")

                assert span == mock_span
                mock_trace.create_span.assert_called_once_with(
                    name="test-span",
                    parent_span_id=None,
                    attributes=None,
                    start_time=None,
                )

    def test_start_span_with_attributes(self):
        """Test span creation with attributes."""
        config = Config.create()

        with patch("noveum_trace.core.client.HttpTransport"):
            with patch("noveum_trace.core.client.get_current_trace") as mock_get_trace:
                client = NoveumClient(config=config)

                # Mock current trace
                mock_trace = Mock(spec=Trace)
                mock_span = Mock(spec=Span)
                mock_trace.create_span.return_value = mock_span
                mock_get_trace.return_value = mock_trace

                attributes = {"key": "value"}
                client.start_span("test-span", attributes=attributes)

                mock_trace.create_span.assert_called_once_with(
                    name="test-span",
                    parent_span_id=None,
                    attributes=attributes,
                    start_time=None,
                )

    def test_start_span_with_parent_span_id(self):
        """Test span creation with explicit parent span ID."""
        config = Config.create()

        with patch("noveum_trace.core.client.HttpTransport"):
            with patch("noveum_trace.core.client.get_current_trace") as mock_get_trace:
                client = NoveumClient(config=config)

                # Mock current trace
                mock_trace = Mock(spec=Trace)
                mock_span = Mock(spec=Span)
                mock_trace.create_span.return_value = mock_span
                mock_get_trace.return_value = mock_trace

                client.start_span("test-span", parent_span_id="parent-id")

                mock_trace.create_span.assert_called_once_with(
                    name="test-span",
                    parent_span_id="parent-id",
                    attributes=None,
                    start_time=None,
                )

    def test_start_span_with_current_parent(self):
        """Test span creation with current span as parent."""
        config = Config.create()

        with patch("noveum_trace.core.client.HttpTransport"):
            with patch("noveum_trace.core.client.get_current_trace") as mock_get_trace:
                with patch(
                    "noveum_trace.core.client.get_current_span"
                ) as mock_get_span:
                    client = NoveumClient(config=config)

                    # Mock current trace and span
                    mock_trace = Mock(spec=Trace)
                    mock_span = Mock(spec=Span)
                    mock_current_span = Mock(spec=Span)
                    mock_current_span.span_id = "current-span-id"

                    mock_trace.create_span.return_value = mock_span
                    mock_get_trace.return_value = mock_trace
                    mock_get_span.return_value = mock_current_span

                    client.start_span("test-span")

                    mock_trace.create_span.assert_called_once_with(
                        name="test-span",
                        parent_span_id="current-span-id",
                        attributes=None,
                        start_time=None,
                    )

    def test_start_span_set_as_current(self):
        """Test span creation with setting as current."""
        config = Config.create()

        with patch("noveum_trace.core.client.HttpTransport"):
            with patch("noveum_trace.core.client.get_current_trace") as mock_get_trace:
                with patch(
                    "noveum_trace.core.client.set_current_span"
                ) as mock_set_current:
                    client = NoveumClient(config=config)

                    # Mock current trace
                    mock_trace = Mock(spec=Trace)
                    mock_span = Mock(spec=Span)
                    mock_trace.create_span.return_value = mock_span
                    mock_get_trace.return_value = mock_trace

                    client.start_span("test-span", set_as_current=True)

                    mock_set_current.assert_called_once_with(mock_span)

    def test_start_span_not_set_as_current(self):
        """Test span creation without setting as current."""
        config = Config.create()

        with patch("noveum_trace.core.client.HttpTransport"):
            with patch("noveum_trace.core.client.get_current_trace") as mock_get_trace:
                with patch(
                    "noveum_trace.core.client.set_current_span"
                ) as mock_set_current:
                    client = NoveumClient(config=config)

                    # Mock current trace
                    mock_trace = Mock(spec=Trace)
                    mock_span = Mock(spec=Span)
                    mock_trace.create_span.return_value = mock_span
                    mock_get_trace.return_value = mock_trace

                    client.start_span("test-span", set_as_current=False)

                    mock_set_current.assert_not_called()

    def test_start_span_no_active_trace(self):
        """Test span creation without active trace."""
        config = Config.create()

        with patch("noveum_trace.core.client.HttpTransport"):
            with patch("noveum_trace.core.client.get_current_trace") as mock_get_trace:
                client = NoveumClient(config=config)

                mock_get_trace.return_value = None

                with pytest.raises(NoveumTraceError, match="No active trace found"):
                    client.start_span("test-span")

    def test_start_span_when_shutdown(self):
        """Test span creation when client is shutdown."""
        config = Config.create()

        with patch("noveum_trace.core.client.HttpTransport"):
            client = NoveumClient(config=config)
            client._shutdown = True

            with pytest.raises(NoveumTraceError, match="Client has been shutdown"):
                client.start_span("test-span")

    def test_start_span_logs_debug(self, caplog):
        """Test span creation logs debug message."""
        config = Config.create()

        with patch("noveum_trace.core.client.HttpTransport"):
            with patch("noveum_trace.core.client.get_current_trace") as mock_get_trace:
                client = NoveumClient(config=config)

                # Mock current trace
                mock_trace = Mock(spec=Trace)
                mock_trace.trace_id = "test-trace-id"
                mock_span = Mock(spec=Span)
                mock_span.span_id = "test-span-id"
                mock_trace.create_span.return_value = mock_span
                mock_get_trace.return_value = mock_trace

                with caplog.at_level("DEBUG"):
                    client.start_span("test-span")

                assert (
                    "Started span: test-span-id in trace: test-trace-id" in caplog.text
                )

    def test_finish_span_basic(self):
        """Test basic span finishing."""
        config = Config.create()

        with patch("noveum_trace.core.client.HttpTransport"):
            client = NoveumClient(config=config)

            mock_span = Mock(spec=Span)
            mock_span.is_finished.return_value = False

            client.finish_span(mock_span)

            mock_span.finish.assert_called_once_with(None)

    def test_finish_span_with_end_time(self):
        """Test span finishing with custom end time."""
        config = Config.create()

        with patch("noveum_trace.core.client.HttpTransport"):
            client = NoveumClient(config=config)

            mock_span = Mock(spec=Span)
            mock_span.is_finished.return_value = False
            end_time = datetime.now()

            client.finish_span(mock_span, end_time=end_time)

            mock_span.finish.assert_called_once_with(end_time)

    def test_finish_span_already_finished(self):
        """Test finishing already finished span."""
        config = Config.create()

        with patch("noveum_trace.core.client.HttpTransport"):
            client = NoveumClient(config=config)

            mock_span = Mock(spec=Span)
            mock_span.is_finished.return_value = True

            client.finish_span(mock_span)

            mock_span.finish.assert_not_called()

    def test_finish_span_clears_current_context(self):
        """Test finishing span clears current context."""
        config = Config.create()

        with patch("noveum_trace.core.client.HttpTransport"):
            with patch("noveum_trace.core.client.get_current_span") as mock_get_current:
                with patch(
                    "noveum_trace.core.client.get_current_trace"
                ) as mock_get_trace:
                    with patch(
                        "noveum_trace.core.client.set_current_span"
                    ) as mock_set_current:
                        client = NoveumClient(config=config)

                        mock_span = Mock(spec=Span)
                        mock_span.span_id = "test-span-id"
                        mock_span.parent_span_id = "parent-span-id"
                        mock_span.is_finished.return_value = False

                        # Mock current span
                        mock_current_span = Mock(spec=Span)
                        mock_current_span.span_id = "test-span-id"
                        mock_get_current.return_value = mock_current_span

                        # Mock parent span
                        mock_trace = Mock(spec=Trace)
                        mock_parent_span = Mock(spec=Span)
                        mock_trace.get_span.return_value = mock_parent_span
                        mock_get_trace.return_value = mock_trace

                        client.finish_span(mock_span)

                        mock_set_current.assert_called_once_with(mock_parent_span)

    def test_finish_span_logs_debug(self, caplog):
        """Test finishing span logs debug message."""
        config = Config.create()

        with patch("noveum_trace.core.client.HttpTransport"):
            client = NoveumClient(config=config)

            mock_span = Mock(spec=Span)
            mock_span.span_id = "test-span-id"
            mock_span.is_finished.return_value = False

            with caplog.at_level("DEBUG"):
                client.finish_span(mock_span)

            assert "Finished span: test-span-id" in caplog.text


class TestNoveumClientContextualOperations:
    """Test contextual trace and span operations."""

    def test_create_contextual_trace(self):
        """Test creating contextual trace."""
        config = Config.create()

        with patch("noveum_trace.core.client.HttpTransport"):
            with patch("noveum_trace.core.client.ContextualTrace") as mock_contextual:
                client = NoveumClient(config=config)

                # Mock start_trace
                mock_trace = Mock(spec=Trace)
                client.start_trace = Mock(return_value=mock_trace)

                client.create_contextual_trace("test-trace")

                client.start_trace.assert_called_once_with(
                    name="test-trace",
                    attributes=None,
                    start_time=None,
                    set_as_current=False,
                )
                mock_contextual.assert_called_once_with(mock_trace)

    def test_create_contextual_span(self):
        """Test creating contextual span."""
        config = Config.create()

        with patch("noveum_trace.core.client.HttpTransport"):
            with patch("noveum_trace.core.client.ContextualSpan") as mock_contextual:
                client = NoveumClient(config=config)

                # Mock start_span
                mock_span = Mock(spec=Span)
                client.start_span = Mock(return_value=mock_span)

                client.create_contextual_span("test-span")

                client.start_span.assert_called_once_with(
                    name="test-span",
                    parent_span_id=None,
                    attributes=None,
                    start_time=None,
                    set_as_current=False,
                )
                mock_contextual.assert_called_once_with(mock_span)


class TestNoveumClientTraceManagement:
    """Test trace management operations."""

    def test_get_active_traces(self):
        """Test getting active traces."""
        config = Config.create()

        with patch("noveum_trace.core.client.HttpTransport"):
            client = NoveumClient(config=config)

            trace1 = client.start_trace("trace1")
            trace2 = client.start_trace("trace2")

            active_traces = client.get_active_traces()

            assert len(active_traces) == 2
            assert trace1 in active_traces
            assert trace2 in active_traces

    def test_get_trace_by_id(self):
        """Test getting trace by ID."""
        config = Config.create()

        with patch("noveum_trace.core.client.HttpTransport"):
            client = NoveumClient(config=config)

            trace = client.start_trace("test-trace")

            retrieved_trace = client.get_trace(trace.trace_id)

            assert retrieved_trace == trace

    def test_get_trace_by_nonexistent_id(self):
        """Test getting trace by nonexistent ID."""
        config = Config.create()

        with patch("noveum_trace.core.client.HttpTransport"):
            client = NoveumClient(config=config)

            retrieved_trace = client.get_trace("nonexistent-id")

            assert retrieved_trace is None


class TestNoveumClientFlushAndShutdown:
    """Test flush and shutdown operations."""

    def test_flush_active_traces(self):
        """Test flushing active traces."""
        config = Config.create()

        with patch("noveum_trace.core.client.HttpTransport"):
            client = NoveumClient(config=config)

            trace1 = client.start_trace("trace1")
            trace2 = client.start_trace("trace2")

            # Mock finish_trace
            client.finish_trace = Mock()

            # Mock transport flush
            client.transport.flush = Mock()

            client.flush(timeout=10.0)

            # Verify traces were finished
            client.finish_trace.assert_any_call(trace1)
            client.finish_trace.assert_any_call(trace2)
            assert client.finish_trace.call_count == 2

            # Verify transport was flushed
            client.transport.flush.assert_called_once_with(10.0)

    def test_flush_when_shutdown(self):
        """Test flush when client is shutdown."""
        config = Config.create()

        with patch("noveum_trace.core.client.HttpTransport"):
            client = NoveumClient(config=config)
            client._shutdown = True

            client.transport.flush = Mock()

            client.flush()

            client.transport.flush.assert_not_called()

    def test_flush_logs_message(self, caplog):
        """Test flush logs completion message."""
        config = Config.create()

        with patch("noveum_trace.core.client.HttpTransport"):
            client = NoveumClient(config=config)

            client.transport.flush = Mock()

            client.flush()

            assert "Flushed all pending traces" in caplog.text

    def test_shutdown_full_process(self, caplog):
        """Test complete shutdown process."""
        config = Config.create()

        with patch("noveum_trace.core.client.HttpTransport"):
            client = NoveumClient(config=config)

            # Mock flush and transport shutdown
            client.flush = Mock()
            client.transport.shutdown = Mock()

            client.shutdown()

            # Verify shutdown process
            assert client._shutdown is True
            client.flush.assert_called_once_with(timeout=30.0)
            client.transport.shutdown.assert_called_once()

            # Verify log messages
            assert "Shutting down Noveum Trace client" in caplog.text
            assert "Noveum Trace client shutdown complete" in caplog.text

    def test_shutdown_idempotent(self):
        """Test shutdown is idempotent."""
        config = Config.create()

        with patch("noveum_trace.core.client.HttpTransport"):
            client = NoveumClient(config=config)

            # Mock flush and transport shutdown
            client.flush = Mock()
            client.transport.shutdown = Mock()

            # First shutdown
            client.shutdown()

            # Reset mocks
            client.flush.reset_mock()
            client.transport.shutdown.reset_mock()

            # Second shutdown should do nothing
            client.shutdown()

            client.flush.assert_not_called()
            client.transport.shutdown.assert_not_called()

    def test_is_shutdown(self):
        """Test is_shutdown method."""
        config = Config.create()

        with patch("noveum_trace.core.client.HttpTransport"):
            client = NoveumClient(config=config)

            assert client.is_shutdown() is False

            client.shutdown()

            assert client.is_shutdown() is True


class TestNoveumClientPrivateMethods:
    """Test private methods."""

    def test_export_trace_success(self):
        """Test successful trace export."""
        config = Config.create()

        with patch("noveum_trace.core.client.HttpTransport"):
            client = NoveumClient(config=config)

            client.transport.export_trace = Mock()

            mock_trace = Mock(spec=Trace)

            client._export_trace(mock_trace)

            client.transport.export_trace.assert_called_once_with(mock_trace)

    def test_export_trace_failure(self, caplog):
        """Test trace export failure."""
        config = Config.create()

        with patch("noveum_trace.core.client.HttpTransport"):
            client = NoveumClient(config=config)

            client.transport.export_trace = Mock(side_effect=Exception("Export failed"))

            mock_trace = Mock(spec=Trace)
            mock_trace.trace_id = "test-trace-id"

            client._export_trace(mock_trace)

            assert "Failed to export trace test-trace-id: Export failed" in caplog.text

    def test_create_noop_trace(self):
        """Test creating no-op trace."""
        config = Config.create()

        with patch("noveum_trace.core.client.HttpTransport"):
            client = NoveumClient(config=config)

            noop_trace = client._create_noop_trace("test-trace")

            assert noop_trace.name == "test-trace"
            assert hasattr(noop_trace, "_noop")
            assert noop_trace._noop is True

    def test_repr(self):
        """Test string representation."""
        config = Config.create(project="test-project")

        with patch("noveum_trace.core.client.HttpTransport"):
            client = NoveumClient(config=config)

            # Add a trace to test active count
            client.start_trace("test-trace")

            repr_str = repr(client)

            assert "NoveumClient" in repr_str
            assert "test-project" in repr_str
            assert "active_traces=1" in repr_str


class TestNoveumClientIntegration:
    """Integration tests for NoveumClient."""

    def test_complete_trace_workflow(self):
        """Test complete trace workflow."""
        config = Config.create(project="test-project", environment="test-env")

        with patch("noveum_trace.core.client.HttpTransport"):
            client = NoveumClient(config=config)

            # Mock transport
            client.transport.export_trace = Mock()

            # Create and finish a trace
            trace = client.start_trace("test-trace", attributes={"key": "value"})
            span = client.start_span("test-span", attributes={"span_key": "span_value"})

            client.finish_span(span)
            client.finish_trace(trace)

            # Verify trace was exported
            client.transport.export_trace.assert_called_once_with(trace)

            # Verify trace and span are finished
            assert trace.is_finished()
            assert span.is_finished()

            # Verify trace is no longer active
            assert trace.trace_id not in client._active_traces

    def test_multiple_traces_and_spans(self):
        """Test handling multiple traces and spans."""
        config = Config.create()

        with patch("noveum_trace.core.client.HttpTransport"):
            client = NoveumClient(config=config)

            # Create multiple traces
            trace1 = client.start_trace("trace1")
            trace2 = client.start_trace("trace2")

            # Create spans in each trace
            with patch("noveum_trace.core.client.get_current_trace") as mock_get_trace:
                mock_get_trace.return_value = trace1
                span1 = client.start_span("span1")

                mock_get_trace.return_value = trace2
                span2 = client.start_span("span2")

            # Verify active traces
            active_traces = client.get_active_traces()
            assert len(active_traces) == 2
            assert trace1 in active_traces
            assert trace2 in active_traces

            # Finish everything
            client.finish_span(span1)
            client.finish_span(span2)
            client.finish_trace(trace1)
            client.finish_trace(trace2)

            # Verify no active traces
            assert len(client.get_active_traces()) == 0
