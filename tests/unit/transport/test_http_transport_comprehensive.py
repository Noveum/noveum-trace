"""
Comprehensive tests for HTTP transport implementation.

This module provides extensive test coverage for the HttpTransport class,
including all methods, error conditions, and edge cases.
"""

from unittest.mock import Mock, patch

import pytest
import requests
from requests.adapters import HTTPAdapter

from noveum_trace.core.config import Config
from noveum_trace.core.trace import Trace
from noveum_trace.transport.http_transport import HttpTransport
from noveum_trace.utils.exceptions import TransportError


class TestHttpTransportInitialization:
    """Test HTTP transport initialization and configuration."""

    def test_init_with_config(self):
        """Test initialization with provided config."""
        config = Config.create(
            api_key="test-key", project="test-project", endpoint="https://api.test.com"
        )

        with patch(
            "noveum_trace.transport.http_transport.BatchProcessor"
        ) as mock_batch:
            transport = HttpTransport(config)

            assert transport.config == config
            assert transport._shutdown is False
            assert transport.session is not None
            mock_batch.assert_called_once()

    def test_init_without_config(self):
        """Test initialization without config uses global config."""
        with patch(
            "noveum_trace.transport.http_transport.get_config"
        ) as mock_get_config:
            mock_config = Mock()
            mock_get_config.return_value = mock_config

            with patch("noveum_trace.transport.http_transport.BatchProcessor"):
                transport = HttpTransport()

                assert transport.config == mock_config
                mock_get_config.assert_called_once()

    def test_init_logs_endpoint(self, caplog):
        """Test that initialization logs the endpoint."""
        config = Config.create(endpoint="https://api.test.com")

        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            HttpTransport(config)

            assert (
                "HTTP transport initialized for endpoint: https://api.test.com"
                in caplog.text
            )

    def test_get_sdk_version(self):
        """Test SDK version retrieval."""
        config = Config.create()

        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            transport = HttpTransport(config)

            version = transport._get_sdk_version()
            assert isinstance(version, str)
            assert len(version) > 0


class TestHttpTransportSessionCreation:
    """Test HTTP session creation and configuration."""

    def test_create_session_basic_headers(self):
        """Test session creation with basic headers."""
        config = Config.create()

        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            transport = HttpTransport(config)

            session = transport.session
            assert session.headers["Content-Type"] == "application/json"
            assert "noveum-trace-sdk/" in session.headers["User-Agent"]

    def test_create_session_with_auth(self):
        """Test session creation with authentication."""
        config = Config.create(api_key="test-api-key")

        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            transport = HttpTransport(config)

            session = transport.session
            assert session.headers["Authorization"] == "Bearer test-api-key"

    def test_create_session_without_auth(self):
        """Test session creation without authentication."""
        config = Config.create(api_key=None)

        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            transport = HttpTransport(config)

            session = transport.session
            assert "Authorization" not in session.headers

    def test_create_session_retry_configuration(self):
        """Test session retry configuration."""
        config = Config.create()
        config.transport.retry_attempts = 5
        config.transport.retry_backoff = 2.0

        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            transport = HttpTransport(config)

            # Check that adapters are mounted
            assert "http://" in transport.session.adapters
            assert "https://" in transport.session.adapters

            # Check that adapters are HTTPAdapter instances
            http_adapter = transport.session.adapters["http://"]
            https_adapter = transport.session.adapters["https://"]
            assert isinstance(http_adapter, HTTPAdapter)
            assert isinstance(https_adapter, HTTPAdapter)


class TestHttpTransportURLBuilding:
    """Test URL building functionality."""

    def test_build_api_url_basic(self):
        """Test basic API URL building."""
        config = Config.create(endpoint="https://api.test.com")

        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            transport = HttpTransport(config)

            url = transport._build_api_url("/v1/traces")
            assert url == "https://api.test.com/v1/traces"

    def test_build_api_url_trailing_slash(self):
        """Test API URL building with trailing slash in endpoint."""
        config = Config.create(endpoint="https://api.test.com/")

        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            transport = HttpTransport(config)

            url = transport._build_api_url("/v1/traces")
            assert url == "https://api.test.com/v1/traces"

    def test_build_api_url_no_leading_slash(self):
        """Test API URL building without leading slash in path."""
        config = Config.create(endpoint="https://api.test.com")

        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            transport = HttpTransport(config)

            url = transport._build_api_url("v1/traces")
            assert url == "https://api.test.com/v1/traces"

    def test_build_api_url_complex_endpoint(self):
        """Test API URL building with complex endpoint."""
        config = Config.create(endpoint="https://api.test.com/custom/path")

        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            transport = HttpTransport(config)

            url = transport._build_api_url("/v1/traces")
            assert url == "https://api.test.com/custom/path/v1/traces"


class TestHttpTransportTraceExport:
    """Test trace export functionality."""

    def test_export_trace_success(self):
        """Test successful trace export."""
        config = Config.create()

        with patch(
            "noveum_trace.transport.http_transport.BatchProcessor"
        ) as mock_batch:
            transport = HttpTransport(config)

            # Create a mock trace
            trace = Mock(spec=Trace)
            trace.trace_id = "test-trace-id"

            # Mock the format method
            transport._format_trace_for_export = Mock(
                return_value={"trace_id": "test-trace-id"}
            )

            transport.export_trace(trace)

            # Verify trace was formatted and added to batch
            transport._format_trace_for_export.assert_called_once_with(trace)
            mock_batch.return_value.add_trace.assert_called_once_with(
                {"trace_id": "test-trace-id"}
            )

    def test_export_trace_when_shutdown(self):
        """Test export trace raises error when transport is shutdown."""
        config = Config.create()

        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            transport = HttpTransport(config)
            transport._shutdown = True

            trace = Mock(spec=Trace)

            with pytest.raises(TransportError, match="Transport has been shutdown"):
                transport.export_trace(trace)

    def test_export_trace_noop_trace(self):
        """Test export trace skips no-op traces."""
        config = Config.create()

        with patch(
            "noveum_trace.transport.http_transport.BatchProcessor"
        ) as mock_batch:
            transport = HttpTransport(config)

            # Create a no-op trace
            trace = Mock(spec=Trace)
            trace._noop = True

            transport.export_trace(trace)

            # Verify no processing occurred
            mock_batch.return_value.add_trace.assert_not_called()

    def test_export_trace_logs_debug(self, caplog):
        """Test export trace logs debug message."""
        config = Config.create()

        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            transport = HttpTransport(config)

            trace = Mock(spec=Trace)
            trace.trace_id = "test-trace-id"
            transport._format_trace_for_export = Mock(
                return_value={"trace_id": "test-trace-id"}
            )

            with caplog.at_level("DEBUG"):
                transport.export_trace(trace)

            assert "Trace test-trace-id queued for export" in caplog.text


class TestHttpTransportFlushAndShutdown:
    """Test flush and shutdown functionality."""

    def test_flush_success(self):
        """Test successful flush."""
        config = Config.create()

        with patch(
            "noveum_trace.transport.http_transport.BatchProcessor"
        ) as mock_batch:
            transport = HttpTransport(config)

            transport.flush(timeout=10.0)

            mock_batch.return_value.flush.assert_called_once_with(10.0)

    def test_flush_when_shutdown(self):
        """Test flush does nothing when transport is shutdown."""
        config = Config.create()

        with patch(
            "noveum_trace.transport.http_transport.BatchProcessor"
        ) as mock_batch:
            transport = HttpTransport(config)
            transport._shutdown = True

            transport.flush()

            mock_batch.return_value.flush.assert_not_called()

    def test_flush_logs_completion(self, caplog):
        """Test flush logs completion message."""
        config = Config.create()

        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            transport = HttpTransport(config)

            transport.flush()

            assert "HTTP transport flush completed" in caplog.text

    def test_shutdown_success(self, caplog):
        """Test successful shutdown."""
        config = Config.create()

        with patch(
            "noveum_trace.transport.http_transport.BatchProcessor"
        ) as mock_batch:
            transport = HttpTransport(config)

            # Mock session.close
            transport.session.close = Mock()

            transport.shutdown()

            # Verify shutdown process
            assert transport._shutdown is True
            mock_batch.return_value.flush.assert_called_once_with(30.0)
            mock_batch.return_value.shutdown.assert_called_once()
            transport.session.close.assert_called_once()

            # Verify log messages
            assert "Shutting down HTTP transport" in caplog.text
            assert "HTTP transport shutdown completed" in caplog.text

    def test_shutdown_idempotent(self):
        """Test shutdown is idempotent."""
        config = Config.create()

        with patch(
            "noveum_trace.transport.http_transport.BatchProcessor"
        ) as mock_batch:
            transport = HttpTransport(config)
            transport.session.close = Mock()

            # First shutdown
            transport.shutdown()

            # Reset mocks
            mock_batch.reset_mock()
            transport.session.close.reset_mock()

            # Second shutdown should do nothing
            transport.shutdown()

            mock_batch.return_value.flush.assert_not_called()
            mock_batch.return_value.shutdown.assert_not_called()
            transport.session.close.assert_not_called()


class TestHttpTransportTraceFormatting:
    """Test trace formatting functionality."""

    def test_format_trace_for_export_basic(self):
        """Test basic trace formatting."""
        config = Config.create(project="test-project", environment="test-env")

        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            transport = HttpTransport(config)

            trace = Mock(spec=Trace)
            trace.to_dict.return_value = {"trace_id": "test-id", "spans": []}

            result = transport._format_trace_for_export(trace)

            assert result["trace_id"] == "test-id"
            assert result["spans"] == []
            assert result["sdk"]["name"] == "noveum-trace-python"
            assert result["sdk"]["version"] == transport._get_sdk_version()
            assert result["project"] == "test-project"
            assert result["environment"] == "test-env"

    def test_format_trace_for_export_no_project(self):
        """Test trace formatting without project."""
        config = Config.create(project=None)

        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            transport = HttpTransport(config)

            trace = Mock(spec=Trace)
            trace.to_dict.return_value = {"trace_id": "test-id"}

            result = transport._format_trace_for_export(trace)

            assert "project" not in result
            assert result["sdk"]["name"] == "noveum-trace-python"

    def test_format_trace_for_export_no_environment(self):
        """Test trace formatting without environment."""
        config = Config.create(environment=None)

        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            transport = HttpTransport(config)

            trace = Mock(spec=Trace)
            trace.to_dict.return_value = {"trace_id": "test-id"}

            result = transport._format_trace_for_export(trace)

            assert "environment" not in result
            assert result["sdk"]["name"] == "noveum-trace-python"


class TestHttpTransportSendRequest:
    """Test HTTP request sending functionality."""

    def test_send_request_success(self):
        """Test successful HTTP request."""
        config = Config.create()

        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            transport = HttpTransport(config)

            # Mock successful response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"success": True}

            transport.session.post = Mock(return_value=mock_response)

            trace_data = {"trace_id": "test-id"}
            result = transport._send_request(trace_data)

            assert result == {"success": True}
            transport.session.post.assert_called_once_with(
                "https://api.noveum.ai/v1/trace",
                json=trace_data,
                timeout=transport.config.transport.timeout,
            )

    def test_send_request_auth_error(self):
        """Test HTTP request with authentication error."""
        config = Config.create()

        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            transport = HttpTransport(config)

            # Mock 401 response
            mock_response = Mock()
            mock_response.status_code = 401

            transport.session.post = Mock(return_value=mock_response)

            trace_data = {"trace_id": "test-id"}

            with pytest.raises(
                TransportError, match="Authentication failed - check API key"
            ):
                transport._send_request(trace_data)

    def test_send_request_forbidden_error(self):
        """Test HTTP request with forbidden error."""
        config = Config.create()

        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            transport = HttpTransport(config)

            # Mock 403 response
            mock_response = Mock()
            mock_response.status_code = 403

            transport.session.post = Mock(return_value=mock_response)

            trace_data = {"trace_id": "test-id"}

            with pytest.raises(
                TransportError, match="Access forbidden - check project permissions"
            ):
                transport._send_request(trace_data)

    def test_send_request_rate_limit_error(self):
        """Test HTTP request with rate limit error."""
        config = Config.create()

        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            transport = HttpTransport(config)

            # Mock 429 response
            mock_response = Mock()
            mock_response.status_code = 429

            transport.session.post = Mock(return_value=mock_response)

            trace_data = {"trace_id": "test-id"}

            with pytest.raises(TransportError, match="Rate limit exceeded"):
                transport._send_request(trace_data)

    def test_send_request_other_http_error(self):
        """Test HTTP request with other HTTP error."""
        config = Config.create()

        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            transport = HttpTransport(config)

            # Mock 500 response
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
                "Server Error"
            )
            mock_response.json.return_value = {"error": "Server Error"}

            transport.session.post = Mock(return_value=mock_response)

            trace_data = {"trace_id": "test-id"}

            with pytest.raises(TransportError, match="HTTP request failed"):
                transport._send_request(trace_data)

    def test_send_request_connection_error(self):
        """Test HTTP request with connection error."""
        config = Config.create()

        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            transport = HttpTransport(config)

            # Mock connection error
            transport.session.post = Mock(
                side_effect=requests.exceptions.ConnectionError("Connection failed")
            )

            trace_data = {"trace_id": "test-id"}

            with pytest.raises(TransportError, match="HTTP request failed"):
                transport._send_request(trace_data)

    def test_send_request_logs_debug_on_success(self, caplog):
        """Test send request logs debug message on success."""
        config = Config.create()

        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            transport = HttpTransport(config)

            # Mock successful response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"success": True}

            transport.session.post = Mock(return_value=mock_response)

            trace_data = {"trace_id": "test-trace-id"}

            with caplog.at_level("DEBUG"):
                transport._send_request(trace_data)

            assert "Successfully sent trace: test-trace-id" in caplog.text


class TestHttpTransportSendBatch:
    """Test batch sending functionality."""

    def test_send_batch_success(self):
        """Test successful batch send."""
        config = Config.create()

        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            transport = HttpTransport(config)

            # Mock successful response
            mock_response = Mock()
            mock_response.status_code = 200

            transport.session.post = Mock(return_value=mock_response)

            traces = [{"trace_id": "test-1"}, {"trace_id": "test-2"}]

            transport._send_batch(traces)

            # Verify request was made
            transport.session.post.assert_called_once()
            args, kwargs = transport.session.post.call_args
            assert kwargs["json"]["traces"] == traces
            assert "timestamp" in kwargs["json"]

    def test_send_batch_empty_traces(self):
        """Test batch send with empty traces."""
        config = Config.create()

        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            transport = HttpTransport(config)

            transport.session.post = Mock()

            transport._send_batch([])

            # Verify no request was made
            transport.session.post.assert_not_called()

    def test_send_batch_with_compression(self):
        """Test batch send with compression enabled."""
        config = Config.create()
        config.transport.compression = True

        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            transport = HttpTransport(config)

            # Mock compression method
            transport._compress_payload = Mock(return_value={"compressed": True})

            # Mock successful response
            mock_response = Mock()
            mock_response.status_code = 200

            transport.session.post = Mock(return_value=mock_response)

            traces = [{"trace_id": "test-1"}]

            transport._send_batch(traces)

            # Verify compression was called
            transport._compress_payload.assert_called_once()

    def test_send_batch_timeout_error(self):
        """Test batch send with timeout error."""
        config = Config.create()
        config.transport.timeout = 30.0

        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            transport = HttpTransport(config)

            # Mock timeout error
            transport.session.post = Mock(
                side_effect=requests.exceptions.Timeout("Request timed out")
            )

            traces = [{"trace_id": "test-1"}]

            with pytest.raises(TransportError, match="Request timeout after 30.0s"):
                transport._send_batch(traces)

    def test_send_batch_connection_error(self):
        """Test batch send with connection error."""
        config = Config.create()

        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            transport = HttpTransport(config)

            # Mock connection error
            transport.session.post = Mock(
                side_effect=requests.exceptions.ConnectionError("Connection failed")
            )

            traces = [{"trace_id": "test-1"}]

            with pytest.raises(TransportError, match="Connection error"):
                transport._send_batch(traces)

    def test_send_batch_http_error(self):
        """Test batch send with HTTP error."""
        config = Config.create()

        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            transport = HttpTransport(config)

            # Mock HTTP error
            transport.session.post = Mock(
                side_effect=requests.exceptions.HTTPError("HTTP error")
            )

            traces = [{"trace_id": "test-1"}]

            with pytest.raises(TransportError, match="HTTP error"):
                transport._send_batch(traces)

    def test_send_batch_unexpected_error(self):
        """Test batch send with unexpected error."""
        config = Config.create()

        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            transport = HttpTransport(config)

            # Mock unexpected error
            transport.session.post = Mock(side_effect=ValueError("Unexpected error"))

            traces = [{"trace_id": "test-1"}]

            with pytest.raises(TransportError, match="Unexpected error"):
                transport._send_batch(traces)

    def test_send_batch_logs_debug_on_success(self, caplog):
        """Test send batch logs debug message on success."""
        config = Config.create()

        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            transport = HttpTransport(config)

            # Mock successful response
            mock_response = Mock()
            mock_response.status_code = 200

            transport.session.post = Mock(return_value=mock_response)

            traces = [{"trace_id": "test-1"}, {"trace_id": "test-2"}]

            with caplog.at_level("DEBUG"):
                transport._send_batch(traces)

            assert "Successfully sent batch of 2 traces" in caplog.text


class TestHttpTransportCompressionAndHealth:
    """Test compression and health check functionality."""

    def test_compress_payload(self):
        """Test payload compression (currently pass-through)."""
        config = Config.create()

        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            transport = HttpTransport(config)

            payload = {"test": "data"}
            result = transport._compress_payload(payload)

            # Currently just returns the payload as-is
            assert result == payload

    def test_health_check_success(self):
        """Test successful health check."""
        config = Config.create(endpoint="https://api.test.com")

        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            transport = HttpTransport(config)

            # Mock successful response
            mock_response = Mock()
            mock_response.status_code = 200

            transport.session.get = Mock(return_value=mock_response)

            result = transport.health_check()

            assert result is True
            transport.session.get.assert_called_once_with(
                "https://api.test.com/health", timeout=10
            )

    def test_health_check_failure(self):
        """Test health check failure."""
        config = Config.create(endpoint="https://api.test.com")

        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            transport = HttpTransport(config)

            # Mock failed response
            mock_response = Mock()
            mock_response.status_code = 500

            transport.session.get = Mock(return_value=mock_response)

            result = transport.health_check()

            assert result is False

    def test_health_check_exception(self):
        """Test health check with exception."""
        config = Config.create(endpoint="https://api.test.com")

        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            transport = HttpTransport(config)

            # Mock exception
            transport.session.get = Mock(
                side_effect=requests.exceptions.RequestException("Connection failed")
            )

            result = transport.health_check()

            assert result is False


class TestHttpTransportStringRepresentation:
    """Test string representation functionality."""

    def test_repr(self):
        """Test string representation of transport."""
        config = Config.create(endpoint="https://api.test.com")
        config.transport.batch_size = 100

        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            transport = HttpTransport(config)

            repr_str = repr(transport)

            assert "HttpTransport" in repr_str
            assert "https://api.test.com" in repr_str
            assert "batch_size=100" in repr_str


class TestHttpTransportIntegration:
    """Integration tests for HTTP transport."""

    def test_full_export_workflow(self):
        """Test complete export workflow."""
        config = Config.create(
            api_key="test-key", project="test-project", environment="test-env"
        )

        with patch(
            "noveum_trace.transport.http_transport.BatchProcessor"
        ) as mock_batch:
            transport = HttpTransport(config)

            # Create a real trace
            trace = Trace("test-trace")
            trace.set_attribute("test", "value")

            # Export the trace
            transport.export_trace(trace)

            # Verify the trace was processed
            mock_batch.return_value.add_trace.assert_called_once()

            # Verify the formatted trace data
            call_args = mock_batch.return_value.add_trace.call_args[0][0]
            assert call_args["trace_id"] == trace.trace_id
            assert call_args["name"] == "test-trace"
            assert call_args["sdk"]["name"] == "noveum-trace-python"
            assert call_args["project"] == "test-project"
            assert call_args["environment"] == "test-env"

    def test_shutdown_after_export(self):
        """Test shutdown after exporting traces."""
        config = Config.create()

        with patch(
            "noveum_trace.transport.http_transport.BatchProcessor"
        ) as mock_batch:
            transport = HttpTransport(config)
            transport.session.close = Mock()

            # Export a trace
            trace = Trace("test-trace")
            transport.export_trace(trace)

            # Shutdown
            transport.shutdown()

            # Verify shutdown process
            assert transport._shutdown is True
            mock_batch.return_value.flush.assert_called_once_with(30.0)
            mock_batch.return_value.shutdown.assert_called_once()
            transport.session.close.assert_called_once()

            # Verify subsequent exports are rejected
            with pytest.raises(TransportError, match="Transport has been shutdown"):
                transport.export_trace(trace)
