"""Additional unit tests for HTTP transport to improve coverage."""

import json
import sys
import types
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

from noveum_trace.core.config import Config, SecurityConfig
from noveum_trace.transport.http_transport import HttpTransport
from noveum_trace.utils.exceptions import TransportError


class TestHttpTransportSensitiveDataHandling:
    """Test sensitive data detection and response preview functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        self.config = Config.create(
            api_key="test-key", project="test-project", endpoint="https://api.test.com"
        )
        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            self.transport = HttpTransport(self.config)

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("password=12345", True),
            ("api_key: secret123", True),
            ("token: bearer xyz", True),
            ("secret_key=abcdef", True),
            ("credential: admin", True),
            ("authorization: Bearer token123", True),
            ("access_token=xyz789", True),
            ("private_key: -----BEGIN", True),
            ("certificate data", True),
            ("ssn: 123-45-6789", True),
            ("social_security_number: 999999999", True),
            ("credit_card: 4111111111111111", True),
            ("card_number=5500000000000004", True),
            ("cvv: 123", True),
            ("pin: 1234", True),
            ("account_number: 9876543210", True),
            ("normal response data", False),
            ("user data without sensitive info", False),
            ("", False),
            (None, False),
        ],
    )
    def test_contains_sensitive_data(self, text, expected):
        """Test sensitive data detection."""
        result = self.transport._contains_sensitive_data(text)
        assert result == expected

    def test_get_safe_response_preview_with_sensitive_data(self):
        """Test response preview with sensitive data."""
        response = Mock()
        response.text = "password=secret123 and api_key=xyz789"

        preview = self.transport._get_safe_response_preview(response)

        assert (
            preview
            == f"<Response contains sensitive data, length: {len(response.text)} chars>"
        )

    def test_get_safe_response_preview_normal_data(self):
        """Test response preview with normal data."""
        response = Mock()
        response.text = "Normal response without sensitive information"

        preview = self.transport._get_safe_response_preview(response)

        assert preview == response.text

    def test_get_safe_response_preview_truncation(self):
        """Test response preview truncation."""
        response = Mock()
        response.text = "x" * 2000  # Long response

        preview = self.transport._get_safe_response_preview(response, max_length=100)

        assert preview.startswith("x" * 100)
        assert "truncated" in preview
        assert "2000 chars" in preview

    def test_get_safe_response_preview_custom_max_length(self):
        """Test response preview with custom max length from config."""
        # Update config with custom max_response_preview
        self.config.transport.max_response_preview = 50

        response = Mock()
        response.text = "y" * 100

        preview = self.transport._get_safe_response_preview(response)

        assert len(preview) > 50  # Including truncation message
        assert preview.startswith("y" * 50)
        assert "truncated" in preview

    def test_get_safe_response_preview_empty_response(self):
        """Test response preview with empty response."""
        response = Mock()
        response.text = ""

        preview = self.transport._get_safe_response_preview(response)

        assert preview is None

    def test_get_safe_response_preview_none_response(self):
        """Test response preview with None response text."""
        response = Mock()
        response.text = None

        preview = self.transport._get_safe_response_preview(response)

        assert preview is None


class TestHttpTransportErrorHandling:
    """Test error handling in HTTP transport."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        self.config = Config.create(api_key="test-key", project="test-project")
        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            self.transport = HttpTransport(self.config)
            # Properly mock session with headers that support iteration
            # Note: Content-Type is NOT in session defaults; it's added per-request for JSON
            self.transport.session = Mock()
            self.transport.session.headers = {
                "Authorization": "Bearer test-key",
            }

    @patch("noveum_trace.transport.http_transport.logger")
    @patch("noveum_trace.transport.http_transport.log_error_always")
    def test_make_request_rate_limit_with_retry_after(
        self, mock_log_error, mock_logger
    ):
        """Test rate limit handling with Retry-After header."""
        response = Mock()
        response.status_code = 429
        response.headers = {"Retry-After": "60"}
        response.raise_for_status.side_effect = requests.HTTPError()

        self.transport.session.post.return_value = response

        with pytest.raises(TransportError) as exc_info:
            self.transport._send_request({})

        assert "Rate limit exceeded" in str(exc_info.value)

    @patch("noveum_trace.transport.http_transport.logger")
    @patch("noveum_trace.transport.http_transport.log_error_always")
    def test_make_request_unexpected_status(self, mock_log_error, mock_logger):
        """Test handling of unexpected HTTP status codes."""
        response = Mock()
        response.status_code = 503
        response.text = "Service temporarily unavailable"
        response.raise_for_status.side_effect = requests.HTTPError()

        self.transport.session.post.return_value = response

        with pytest.raises(TransportError):
            self.transport._send_request({})

        # Verify error was logged
        mock_log_error.assert_called()

    @patch("noveum_trace.transport.http_transport.logger")
    @patch("noveum_trace.transport.http_transport.log_error_always")
    def test_export_batch_authentication_error(self, mock_log_error, mock_logger):
        """Test batch export with authentication error."""
        response = Mock()
        response.status_code = 401
        response.text = "Invalid API key"

        self.transport.session.post.return_value = response

        traces = [
            {"trace_data": "test1"},
            {"trace_data": "test2"},
            {"trace_data": "test3"},
        ]

        # Should raise TransportError for authentication failure
        with pytest.raises(TransportError):
            self.transport._send_batch({"type": "traces", "data": traces})

        # Verify error was logged
        mock_log_error.assert_called()

    @patch("noveum_trace.transport.http_transport.logger")
    @patch("noveum_trace.transport.http_transport.log_error_always")
    def test_export_batch_server_error(self, mock_log_error, mock_logger):
        """Test batch export with server error."""
        response = Mock()
        response.status_code = 500
        response.text = "Internal server error"
        response.raise_for_status.side_effect = requests.HTTPError()

        self.transport.session.post.return_value = response

        traces = [{"trace_data": "test1"}, {"trace_data": "test2"}]

        # Should raise TransportError for server error
        with pytest.raises(TransportError):
            self.transport._send_batch({"type": "traces", "data": traces})

        # Verify error was logged
        mock_log_error.assert_called()

    @patch("noveum_trace.transport.http_transport.logger")
    @patch("noveum_trace.transport.http_transport.log_error_always")
    def test_export_batch_timeout(self, mock_log_error, mock_logger):
        """Test batch export with timeout."""
        self.transport.session.post.side_effect = requests.exceptions.Timeout()

        traces = [{"trace_data": "test1"}, {"trace_data": "test2"}]

        # Should raise TransportError for timeout
        with pytest.raises(TransportError):
            self.transport._send_batch({"type": "traces", "data": traces})

        # Verify error was logged
        mock_log_error.assert_called()


class TestHttpTransportBatchExport:
    """Test batch export functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        self.config = Config.create(api_key="test-key", project="test-project")
        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            self.transport = HttpTransport(self.config)
            # Properly mock session with headers that support iteration
            # Note: Content-Type is NOT in session defaults; it's added per-request for JSON
            self.transport.session = Mock()
            self.transport.session.headers = {
                "Authorization": "Bearer test-key",
            }

    @patch("noveum_trace.transport.http_transport.logger")
    def test_export_batch_success_with_response_preview(self, mock_logger):
        """Test successful batch export with response preview in debug mode."""
        response = Mock()
        response.status_code = 200
        response.text = '{"status": "success", "processed": 5}'
        response.headers = {"Content-Type": "application/json", "Content-Length": "35"}

        self.transport.session.post.return_value = response

        traces = [{"trace_data": f"test{i}"} for i in range(5)]

        with patch(
            "noveum_trace.transport.http_transport.log_debug_enabled", return_value=True
        ):
            self.transport._send_batch({"type": "traces", "data": traces})

        # Verify success log
        mock_logger.info.assert_called()

        # Verify debug log with preview
        mock_logger.debug.assert_called()

    @patch("noveum_trace.transport.http_transport.log_error_always")
    def test_export_batch_with_sensitive_response(self, mock_log_error):
        """Test batch export when response contains sensitive data."""
        response = Mock()
        response.status_code = 200
        response.text = '{"status": "success", "api_key": "secret123"}'
        response.headers = {"Content-Type": "application/json"}

        self.transport.session.post.return_value = response

        traces = [{"trace_data": "test"}]

        with patch(
            "noveum_trace.transport.http_transport.log_debug_enabled", return_value=True
        ):
            with patch("noveum_trace.transport.http_transport.logger") as mock_logger:
                # Mock logger.level to avoid comparison issues
                mock_logger.level = 10  # DEBUG level

                self.transport._send_batch({"type": "traces", "data": traces})

                # Verify sensitive data was masked in debug output
                debug_calls = mock_logger.debug.call_args_list
                for call in debug_calls:
                    if "Response preview" in str(call):
                        assert "sensitive data" in str(call)


class TestHttpTransportRequestDetails:
    """Test request construction and details."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        self.config = Config.create(
            api_key="test-key",
            project="test-project",
            endpoint="https://api.test.com/v1",
        )
        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            self.transport = HttpTransport(self.config)

    def test_build_url_with_trailing_slash(self):
        """Test URL building with trailing slashes."""
        # Endpoint with trailing slash
        self.config.endpoint = "https://api.test.com/v1/"

        url = self.transport._build_api_url("/traces")
        assert url == "https://api.test.com/v1/traces"

        url = self.transport._build_api_url("traces")
        assert url == "https://api.test.com/v1/traces"

    def test_build_url_without_trailing_slash(self):
        """Test URL building without trailing slashes."""
        self.config.endpoint = "https://api.test.com/v1"

        url = self.transport._build_api_url("/traces")
        assert url == "https://api.test.com/v1/traces"

        url = self.transport._build_api_url("traces")
        assert url == "https://api.test.com/v1/traces"

    @patch("noveum_trace.transport.http_transport.log_debug_enabled", return_value=True)
    @patch("noveum_trace.transport.http_transport.logger")
    def test_build_url_logging(self, mock_logger, mock_debug_enabled):
        """Test URL building with debug logging."""
        url = self.transport._build_api_url("/test/path")

        assert url == "https://api.test.com/v1/test/path"
        mock_logger.debug.assert_called_with(f"🔗 Built API URL: {url}")


class TestHttpTransportEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        self.config = Config.create()
        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            self.transport = HttpTransport(self.config)

    def test_sensitive_data_patterns_case_insensitive(self):
        """Test that sensitive data detection is case insensitive."""
        test_cases = [
            "PASSWORD=secret",
            "Password=secret",
            "password=secret",
            "PaSsWoRd=secret",
        ]

        for text in test_cases:
            assert self.transport._contains_sensitive_data(text) is True

    def test_sensitive_data_partial_matches(self):
        """Test sensitive data detection with partial word matches."""
        # These should match because they contain sensitive keywords
        assert self.transport._contains_sensitive_data("mypassword123") is True
        assert self.transport._contains_sensitive_data("apikey_value") is True
        assert self.transport._contains_sensitive_data("user_token_here") is True

        # These should not match (avoid words that contain sensitive substrings)
        assert self.transport._contains_sensitive_data("pass the test") is False
        assert self.transport._contains_sensitive_data("feature list") is False

    @pytest.mark.parametrize(
        "max_length,text_length,expected_truncated",
        [
            (100, 50, False),  # Text shorter than max
            (100, 100, False),  # Text exactly at max
            (100, 150, True),  # Text longer than max
            (1000, 2000, True),  # Default scenario
        ],
    )
    def test_response_preview_truncation_boundaries(
        self, max_length, text_length, expected_truncated
    ):
        """Test response preview truncation at boundaries."""
        response = Mock()
        response.text = "a" * text_length

        preview = self.transport._get_safe_response_preview(
            response, max_length=max_length
        )

        if expected_truncated:
            assert "truncated" in preview
            assert f"{text_length} chars" in preview
        else:
            assert preview == response.text


class TestHttpTransportPiiPseudonymization:
    """PII pseudonymization on POST while dev JSON stays raw."""

    @pytest.fixture(autouse=True)
    def stub_spacy(self, monkeypatch):
        class FakeDoc:
            __slots__ = ("ents", "text")

            def __init__(self, text: str) -> None:
                self.text = text
                self.ents: list = []

        class FakeNlp:
            def __call__(self, text: str) -> FakeDoc:
                return FakeDoc(text)

        fake_spacy = types.ModuleType("spacy")
        fake_spacy.load = lambda *_a, **_k: FakeNlp()
        monkeypatch.setitem(sys.modules, "spacy", fake_spacy)

    def test_send_request_dev_raw_post_pseudonymized(self, tmp_path):
        config = Config.create(
            api_key="k",
            project="p",
            endpoint="https://api.test.com",
            dev_mode=True,
            dev_traces_dir=str(tmp_path),
            security=SecurityConfig(pii_enabled=True, pii_salt="salt-for-test"),
        )
        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            transport = HttpTransport(config)
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True}
        transport.session.post = Mock(return_value=mock_response)

        trace_data = {
            "trace_id": "abc123",
            "name": "n",
            "note": "email me@example.com ok",
        }
        transport._send_request(trace_data)

        written = Path(tmp_path) / "abc123.json"
        assert written.exists()
        assert "me@example.com" in written.read_text(encoding="utf-8")

        posted = transport.session.post.call_args.kwargs["json"]
        assert "me@example.com" not in json.dumps(posted)
        assert "EMAIL_" in json.dumps(posted)
        assert trace_data["note"] == "email me@example.com ok"

    def test_send_trace_batch_dev_raw_post_pseudonymized(self, tmp_path):
        config = Config.create(
            api_key="k",
            project="p",
            endpoint="https://api.test.com",
            dev_mode=True,
            dev_traces_dir=str(tmp_path),
            security=SecurityConfig(pii_enabled=True, pii_salt="batch-salt"),
        )
        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            transport = HttpTransport(config)
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "{}"
        mock_response.headers = {}
        transport.session.post = Mock(return_value=mock_response)

        traces = [{"trace_id": "t1", "note": "a@b.co"}]
        with patch(
            "noveum_trace.transport.http_transport.log_debug_enabled",
            return_value=False,
        ):
            transport._send_trace_batch(traces)

        assert "a@b.co" in (Path(tmp_path) / "t1.json").read_text(encoding="utf-8")

        posted = transport.session.post.call_args.kwargs["json"]
        assert "a@b.co" not in json.dumps(posted)
        assert "EMAIL_" in json.dumps(posted)
        assert traces[0]["note"] == "a@b.co"

    def test_pii_disabled_posts_original_body(self):
        config = Config.create(
            api_key="k",
            project="p",
            endpoint="https://api.test.com",
            security=SecurityConfig(pii_enabled=False),
        )
        with patch("noveum_trace.transport.http_transport.BatchProcessor"):
            transport = HttpTransport(config)
        assert transport._pii_pseudonymizer is None

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True}
        transport.session.post = Mock(return_value=mock_response)

        trace_data = {"trace_id": "x", "note": "u@v.co"}
        transport._send_request(trace_data)
        posted = transport.session.post.call_args.kwargs["json"]
        assert posted == trace_data
        assert "u@v.co" in posted["note"]
