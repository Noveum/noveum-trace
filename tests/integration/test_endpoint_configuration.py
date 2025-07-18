"""
Tests for endpoint configuration functionality.

This module tests the critical endpoint configuration features including:
- Custom endpoint configuration via init()
- Environment variable endpoint configuration
- Transport layer using correct endpoints
"""

from unittest.mock import Mock, patch

import pytest

from noveum_trace import configure
from noveum_trace.core.client import NoveumClient
from noveum_trace.core.config import Config
from noveum_trace.transport.http_transport import HttpTransport


@pytest.mark.disable_transport_mocking
class TestEndpointConfiguration:
    """Test endpoint configuration functionality."""

    def setup_method(self):
        """Clear config state before each test."""
        # Clear global config state
        from tests.conftest import reset_noveum_config

        reset_noveum_config()

    def teardown_method(self):
        """Clear config state after each test."""
        # Clear global config state
        from tests.conftest import reset_noveum_config

        reset_noveum_config()

    def test_config_accepts_endpoint_parameter(self):
        """Test that Config.create accepts endpoint parameter."""
        custom_endpoint = "http://localhost:8080/api/v1"
        config = Config.create(endpoint=custom_endpoint)
        assert config.endpoint == custom_endpoint

    def test_config_default_endpoint(self):
        """Test that Config uses default endpoint when none provided."""
        config = Config.create()
        assert config.endpoint == "https://api.noveum.ai/api"

    def test_config_endpoint_from_env(self):
        """Test that Config uses environment variable for endpoint."""
        custom_endpoint = "http://env-endpoint:9000/api/v1"

        with patch.dict("os.environ", {"NOVEUM_ENDPOINT": custom_endpoint}):
            # Force reload from environment
            from noveum_trace.core.config import _load_from_environment

            config = _load_from_environment()
            assert config.endpoint == custom_endpoint

    def test_config_explicit_overrides_env(self):
        """Test that explicit endpoint parameter overrides environment variable."""
        env_endpoint = "http://env-endpoint:9000/api/v1"
        explicit_endpoint = "http://explicit-endpoint:8000/api/v1"

        with patch.dict("os.environ", {"NOVEUM_ENDPOINT": env_endpoint}):
            config = Config.create(endpoint=explicit_endpoint)
            assert config.endpoint == explicit_endpoint

    def test_client_uses_config_endpoint(self):
        """Test that NoveumClient uses endpoint from config."""
        custom_endpoint = "http://localhost:7000/api/v1"
        config = Config.create(endpoint=custom_endpoint)

        # Temporarily override the autouse fixtures for this test
        with patch("noveum_trace.transport.batch_processor.BatchProcessor"):
            client = NoveumClient(config=config)
            assert client.config.endpoint == custom_endpoint
            assert client.transport.config.endpoint == custom_endpoint

    def test_http_transport_uses_config_endpoint(self):
        """Test that HttpTransport uses endpoint from config."""
        custom_endpoint = "http://localhost:6000/api/v1"
        config = Config.create(endpoint=custom_endpoint)

        with patch("noveum_trace.transport.batch_processor.BatchProcessor"):
            transport = HttpTransport(config)
            assert transport.config.endpoint == custom_endpoint

    def test_http_transport_constructs_correct_urls(self):
        """Test that HTTP transport constructs correct URLs from custom endpoint."""
        custom_endpoint = "http://localhost:8081/api/v1"
        config = Config.create(endpoint=custom_endpoint)

        with patch("requests.Session") as mock_session_class:
            mock_session = Mock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"success": True}
            mock_session.post.return_value = mock_response
            mock_session_class.return_value = mock_session

            with patch("noveum_trace.transport.batch_processor.BatchProcessor"):
                transport = HttpTransport(config)

                # Test URL construction directly
                url = transport._build_api_url("/v1/traces")
                expected_url = "http://localhost:8081/api/v1/v1/traces"
                assert url == expected_url

    def test_http_transport_batch_url(self):
        """Test that HTTP transport uses correct URL for batch requests."""
        custom_endpoint = "http://localhost:8082/api/v1"
        config = Config.create(endpoint=custom_endpoint)

        with patch("requests.Session") as mock_session_class:
            mock_session = Mock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"success": True}
            mock_session.post.return_value = mock_response
            mock_session_class.return_value = mock_session

            with patch("noveum_trace.transport.batch_processor.BatchProcessor"):
                transport = HttpTransport(config)

                # Test batch operation by calling _send_batch directly
                traces = [{"trace_id": "test1"}, {"trace_id": "test2"}]
                transport._send_batch(traces)

                # Verify correct URL was called
                expected_url = "http://localhost:8082/api/v1/v1/traces"
                mock_session.post.assert_called_once()
                call_args = mock_session.post.call_args

                # Check both positional and keyword arguments
                if len(call_args.args) > 0:
                    actual_url = call_args.args[0]  # URL as positional argument
                else:
                    actual_url = call_args.kwargs["url"]  # URL as keyword argument

                assert actual_url == expected_url

    def test_configure_function_updates_global_config(self):
        """Test that configure() function updates global configuration."""
        custom_endpoint = "http://localhost:5000/api/v1"
        config = Config.create(endpoint=custom_endpoint)

        # Configure globally
        configure(config)

        # Verify the configuration is applied
        from noveum_trace.core.config import get_config

        current_config = get_config()
        assert current_config.endpoint == custom_endpoint

    def test_endpoint_preserves_path_structure(self):
        """Test that custom endpoints preserve their path structure."""
        test_cases = [
            ("http://localhost:8080", "http://localhost:8080/v1/traces"),
            ("http://localhost:8080/", "http://localhost:8080/v1/traces"),
            ("http://localhost:8080/api", "http://localhost:8080/api/v1/traces"),
            ("http://localhost:8080/api/", "http://localhost:8080/api/v1/traces"),
            (
                "http://localhost:8080/custom/path",
                "http://localhost:8080/custom/path/v1/traces",
            ),
            (
                "https://custom-domain.com/api/v2",
                "https://custom-domain.com/api/v2/v1/traces",
            ),
        ]

        for base_endpoint, expected_url in test_cases:
            config = Config.create(endpoint=base_endpoint)

            with (
                patch("requests.Session"),
                patch("noveum_trace.transport.batch_processor.BatchProcessor"),
            ):
                transport = HttpTransport(config)

                # Test URL construction
                actual_url = transport._build_api_url("/v1/traces")
                assert (
                    actual_url == expected_url
                ), f"Expected {expected_url}, got {actual_url} for base {base_endpoint}"
