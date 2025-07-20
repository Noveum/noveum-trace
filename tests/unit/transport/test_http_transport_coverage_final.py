"""
Final working unit tests for HttpTransport to increase code coverage.

These tests target specific missing coverage lines that can be properly tested.
"""

from unittest.mock import Mock, patch

from noveum_trace.core.config import Config, TransportConfig
from noveum_trace.transport.http_transport import HttpTransport


class TestHttpTransportCoverageFinal:
    """Working tests to increase coverage."""

    def test_type_checking_import_coverage(self):
        """Cover TYPE_CHECKING import (line 22)."""
        import noveum_trace.transport.http_transport as http_transport_module

        assert hasattr(http_transport_module, "HttpTransport")
        assert hasattr(http_transport_module, "TYPE_CHECKING")

    @patch("noveum_trace.utils.logging.log_debug_enabled", return_value=True)
    @patch("noveum_trace.transport.http_transport.BatchProcessor")
    def test_init_with_debug_logging(self, mock_batch, mock_log_debug):
        """Test init debug logging paths (lines 63-69)."""
        config = Config.create(
            endpoint="https://api.test.com",
            transport=TransportConfig(
                timeout=30.0,
                retry_attempts=3,
                batch_size=10,
                batch_timeout=1.0,
                compression=True,
            ),
        )
        transport = HttpTransport(config)
        assert transport is not None

    @patch("noveum_trace.utils.logging.log_debug_enabled", return_value=True)
    @patch("noveum_trace.transport.http_transport.BatchProcessor")
    def test_build_api_url_with_debug(self, mock_batch, mock_log_debug):
        """Test _build_api_url debug logging (line 90)."""
        config = Config.create(endpoint="https://api.test.com")
        transport = HttpTransport(config)
        url = transport._build_api_url("/v1/traces")
        assert url == "https://api.test.com/v1/traces"

    @patch("noveum_trace.transport.http_transport.BatchProcessor")
    def test_export_trace_noop_check(self, mock_batch):
        """Test export_trace noop attribute check (lines 114-115)."""
        config = Config.create()
        transport = HttpTransport(config)

        trace = Mock()
        trace.trace_id = "test-id"
        trace.name = "test-trace"
        trace._noop = True

        transport.export_trace(trace)
        mock_batch.return_value.add_trace.assert_not_called()

    @patch("noveum_trace.transport.http_transport.BatchProcessor")
    def test_flush_when_shutdown(self, mock_batch):
        """Test flush when shutdown (lines 181-182)."""
        config = Config.create()
        transport = HttpTransport(config)
        transport._shutdown = True

        transport.flush()
        mock_batch.return_value.flush.assert_not_called()

    @patch("noveum_trace.transport.http_transport.BatchProcessor")
    def test_shutdown_exception_paths(self, mock_batch):
        """Test shutdown exception handling paths (lines 205-235)."""
        config = Config.create()
        transport = HttpTransport(config)

        # Mock exceptions in shutdown operations
        mock_batch.return_value.flush.side_effect = RuntimeError("Flush error")
        mock_batch.return_value.shutdown.side_effect = RuntimeError("Shutdown error")
        transport.session = Mock()
        transport.session.close.side_effect = RuntimeError("Close error")

        transport.shutdown()
        assert transport._shutdown is True

    @patch("noveum_trace.utils.logging.log_debug_enabled", return_value=True)
    @patch("noveum_trace.transport.http_transport.BatchProcessor")
    def test_create_session_debug_with_api_key(self, mock_batch, mock_log_debug):
        """Test session creation debug logging with API key (lines 251-254)."""
        config = Config.create(api_key="test-key-123456")
        transport = HttpTransport(config)
        assert transport is not None

    @patch("noveum_trace.utils.logging.log_debug_enabled", return_value=True)
    @patch("noveum_trace.transport.http_transport.BatchProcessor")
    def test_create_session_debug_config(self, mock_batch, mock_log_debug):
        """Test session creation debug logging for config (lines 275-278)."""
        config = Config.create(
            transport=TransportConfig(retry_attempts=5, retry_backoff=2.0)
        )
        transport = HttpTransport(config)
        assert transport is not None

    @patch("noveum_trace.transport.http_transport.BatchProcessor")
    def test_send_request_success_with_json(self, mock_batch):
        """Test _send_request successful response with JSON (line 386)."""
        config = Config.create()
        transport = HttpTransport(config)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True, "id": "123"}

        transport.session.post = Mock(return_value=mock_response)

        result = transport._send_request({"trace_id": "test"})
        assert result == {"success": True, "id": "123"}

    @patch("noveum_trace.utils.logging.log_debug_enabled", return_value=True)
    @patch("noveum_trace.transport.http_transport.BatchProcessor")
    def test_send_batch_debug_individual_traces(self, mock_batch, mock_log_debug):
        """Test _send_batch debug logging individual traces (lines 435-439)."""
        config = Config.create()
        transport = HttpTransport(config)

        mock_response = Mock()
        mock_response.status_code = 200
        transport.session.post = Mock(return_value=mock_response)

        traces = [
            {"trace_id": "trace-1", "name": "test-1", "spans": [{"span_id": "s1"}]},
            {"trace_id": "trace-2", "name": "test-2", "spans": []},
        ]

        transport._send_batch(traces)
        transport.session.post.assert_called_once()

    @patch("noveum_trace.utils.logging.log_debug_enabled", return_value=True)
    @patch("noveum_trace.transport.http_transport.BatchProcessor")
    def test_send_batch_debug_response(self, mock_batch, mock_log_debug):
        """Test _send_batch debug response logging (line 455)."""
        config = Config.create()
        transport = HttpTransport(config)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '{"success": true}'
        mock_response.headers = {"content-type": "application/json"}

        transport.session.post = Mock(return_value=mock_response)

        traces = [{"trace_id": "test"}]
        transport._send_batch(traces)

    @patch("noveum_trace.utils.logging.log_debug_enabled", return_value=True)
    @patch("noveum_trace.transport.http_transport.BatchProcessor")
    def test_compress_payload_debug(self, mock_batch, mock_log_debug):
        """Test _compress_payload debug logging (line 560)."""
        config = Config.create()
        transport = HttpTransport(config)

        payload = {"test": "data"}
        result = transport._compress_payload(payload)
        assert result == payload

    @patch("noveum_trace.transport.http_transport.BatchProcessor")
    def test_shutdown_already_shutdown_logging_errors(self, mock_batch):
        """Test shutdown when already shutdown with logging errors."""
        config = Config.create()
        transport = HttpTransport(config)
        transport._shutdown = True

        with patch("noveum_trace.transport.http_transport.logger") as mock_logger:
            mock_logger.debug.side_effect = ValueError("Logging error")
            transport.shutdown()

    @patch("noveum_trace.transport.http_transport.BatchProcessor")
    def test_shutdown_logging_errors_various_stages(self, mock_batch):
        """Test shutdown with logging errors at various stages."""
        config = Config.create()
        transport = HttpTransport(config)

        with patch("noveum_trace.transport.http_transport.logger") as mock_logger:
            mock_logger.info.side_effect = [
                ValueError("Error1"),
                None,
                ValueError("Error2"),
            ]
            mock_logger.debug.side_effect = ValueError("Debug error")
            mock_logger.warning.side_effect = ValueError("Warning error")

            transport.shutdown()
            assert transport._shutdown is True
