"""
Tests for endpoint path preservation functionality.

This module tests that custom endpoint paths (like /beta, /api, etc.)
are preserved when constructing API URLs, instead of being stripped away.
"""

from unittest.mock import MagicMock, patch

import pytest

import noveum_trace
from noveum_trace.core.config import Config, configure, get_config
from noveum_trace.transport.http_transport import HttpTransport


class TestEndpointPathPreservation:
    """Test that endpoint paths are preserved correctly."""

    def setup_method(self):
        """Reset configuration before each test."""
        noveum_trace._client = None
        if hasattr(noveum_trace.core, "config"):
            noveum_trace.core.config._config = None

    def teardown_method(self):
        """Clean up after each test."""
        noveum_trace._client = None
        if hasattr(noveum_trace.core, "config"):
            noveum_trace.core.config._config = None

    def test_default_endpoint_includes_api_path(self):
        """Test that default endpoint includes /api path."""
        # Initialize with just API key
        noveum_trace.init(api_key="test-key", project="test-project")

        config = get_config()
        assert config.transport.endpoint == "https://api.noveum.ai/api"

    @pytest.mark.parametrize(
        "custom_endpoint,expected_trace_url,expected_traces_url",
        [
            # Test preserving /beta path
            (
                "http://localhost:8080/beta",
                "http://localhost:8080/beta/v1/trace",
                "http://localhost:8080/beta/v1/traces",
            ),
            # Test preserving /api path
            (
                "https://endpoint.com/api",
                "https://endpoint.com/api/v1/trace",
                "https://endpoint.com/api/v1/traces",
            ),
            # Test preserving nested paths
            (
                "https://endpoint.com/api/beta",
                "https://endpoint.com/api/beta/v1/trace",
                "https://endpoint.com/api/beta/v1/traces",
            ),
            # Test preserving paths with version
            (
                "https://endpoint.com/v2",
                "https://endpoint.com/v2/v1/trace",
                "https://endpoint.com/v2/v1/traces",
            ),
            # Test without trailing slash
            (
                "https://endpoint.com/beta",
                "https://endpoint.com/beta/v1/trace",
                "https://endpoint.com/beta/v1/traces",
            ),
            # Test with trailing slash
            (
                "https://endpoint.com/beta/",
                "https://endpoint.com/beta/v1/trace",
                "https://endpoint.com/beta/v1/traces",
            ),
            # Test base domain only (no path)
            (
                "https://endpoint.com",
                "https://endpoint.com/v1/trace",
                "https://endpoint.com/v1/traces",
            ),
        ],
    )
    def test_endpoint_path_preservation(
        self, custom_endpoint, expected_trace_url, expected_traces_url
    ):
        """Test that various endpoint paths are preserved correctly."""
        # Configure with custom endpoint
        config = Config.create(
            api_key="test-key", project="test-project", endpoint=custom_endpoint
        )
        configure(config)

        # Create HTTP transport with custom config
        transport = HttpTransport(config)

        # Test single trace URL construction
        single_trace_url = transport._build_api_url("/v1/trace")
        assert single_trace_url == expected_trace_url

        # Test batch traces URL construction
        batch_traces_url = transport._build_api_url("/v1/traces")
        assert batch_traces_url == expected_traces_url

    @pytest.mark.disable_transport_mocking
    @patch("noveum_trace.transport.http_transport.requests.Session")
    def test_http_transport_uses_preserved_paths_for_single_trace(
        self, mock_session_class
    ):
        """Test that HTTP transport uses preserved paths for single trace requests."""
        # Mock session and response
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        mock_session.post.return_value = mock_response
        mock_session.headers = {}
        mock_session_class.return_value = mock_session

        # Configure with custom endpoint containing path
        config = Config.create(
            api_key="test-key",
            project="test-project",
            endpoint="http://localhost:8080/beta",
        )
        configure(config)

        # Create transport and send a trace
        transport = HttpTransport(config)
        trace_data = {
            "trace_id": "test-trace",
            "spans": [],
            "name": "test",
            "project": "test-project",
        }

        transport._send_request(trace_data)

        # Verify the URL used includes the full path
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args

        # Check the URL argument (first positional argument)
        actual_url = call_args[0][0]
        expected_url = "http://localhost:8080/beta/v1/trace"
        assert actual_url == expected_url

    @pytest.mark.disable_transport_mocking
    @patch("noveum_trace.transport.http_transport.requests.Session")
    def test_http_transport_uses_preserved_paths_for_batch_traces(
        self, mock_session_class
    ):
        """Test that HTTP transport uses preserved paths for batch trace requests."""
        # Mock session and response
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        mock_session.post.return_value = mock_response
        mock_session.headers = {}
        mock_session_class.return_value = mock_session

        # Configure with custom endpoint containing path
        config = Config.create(
            api_key="test-key",
            project="test-project",
            endpoint="https://endpoint.com/api",
        )
        configure(config)

        # Create transport and send batch
        transport = HttpTransport(config)
        traces_data = [
            {"trace_id": "trace1", "spans": [], "name": "test1"},
            {"trace_id": "trace2", "spans": [], "name": "test2"},
        ]

        transport._send_batch(traces_data)

        # Verify the URL used includes the full path
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args

        # Check the URL argument (first positional argument)
        actual_url = call_args[0][0]
        expected_url = "https://endpoint.com/api/v1/traces"
        assert actual_url == expected_url

    def test_build_api_url_method_directly(self):
        """Test the _build_api_url method directly with various inputs."""
        # Configure with custom endpoint
        config = Config.create(
            api_key="test-key",
            project="test-project",
            endpoint="https://endpoint.com/beta/v2",
        )
        configure(config)

        transport = HttpTransport(config)

        # Test various path inputs
        assert (
            transport._build_api_url("/v1/traces")
            == "https://endpoint.com/beta/v2/v1/traces"
        )
        assert (
            transport._build_api_url("v1/traces")
            == "https://endpoint.com/beta/v2/v1/traces"
        )
        assert (
            transport._build_api_url("/v1/trace")
            == "https://endpoint.com/beta/v2/v1/trace"
        )
        assert (
            transport._build_api_url("health") == "https://endpoint.com/beta/v2/health"
        )

    def test_endpoint_with_trailing_slash_normalization(self):
        """Test that trailing slashes are handled correctly."""
        # Configure endpoint with trailing slash
        config = Config.create(
            api_key="test-key",
            project="test-project",
            endpoint="https://endpoint.com/beta/",
        )
        configure(config)

        transport = HttpTransport(config)

        # Should normalize and avoid double slashes
        assert (
            transport._build_api_url("/v1/traces")
            == "https://endpoint.com/beta/v1/traces"
        )
        assert (
            transport._build_api_url("v1/traces")
            == "https://endpoint.com/beta/v1/traces"
        )

    def test_endpoint_without_path_still_works(self):
        """Test that endpoints without paths still work correctly."""
        # Configure with just domain
        config = Config.create(
            api_key="test-key",
            project="test-project",
            endpoint="https://api.example.com",
        )
        configure(config)

        transport = HttpTransport(config)

        # Should work without any existing path
        assert (
            transport._build_api_url("/v1/traces")
            == "https://api.example.com/v1/traces"
        )
        assert (
            transport._build_api_url("v1/traces") == "https://api.example.com/v1/traces"
        )

    @pytest.mark.parametrize(
        "endpoint,path,expected",
        [
            # Test edge cases
            ("https://api.com", "/", "https://api.com/"),
            ("https://api.com/", "/", "https://api.com/"),
            ("https://api.com/path", "", "https://api.com/path/"),
            ("https://api.com/path/", "", "https://api.com/path/"),
            # Test complex paths
            (
                "https://api.com/v1/beta",
                "/v2/traces",
                "https://api.com/v1/beta/v2/traces",
            ),
            (
                "https://api.com/service/v1",
                "health/check",
                "https://api.com/service/v1/health/check",
            ),
        ],
    )
    def test_build_api_url_edge_cases(self, endpoint, path, expected):
        """Test edge cases for URL building."""
        config = Config.create(
            api_key="test-key", project="test-project", endpoint=endpoint
        )
        configure(config)

        transport = HttpTransport(config)
        assert transport._build_api_url(path) == expected

    @pytest.mark.disable_transport_mocking
    def test_real_world_beta_endpoint_scenario(self):
        """Test a real-world scenario with a beta endpoint."""
        # Initialize noveum_trace with beta endpoint
        noveum_trace.init(
            api_key="test-key",
            project="test-project",
            endpoint="https://beta.noveum.ai/beta",
        )

        # Create a client and verify URL construction
        client = noveum_trace.get_client()

        # Test that the URL construction is correct for this endpoint
        expected_url = client.transport._build_api_url("/v1/traces")
        assert expected_url == "https://beta.noveum.ai/beta/v1/traces"

        # Test single trace URL as well
        expected_trace_url = client.transport._build_api_url("/v1/trace")
        assert expected_trace_url == "https://beta.noveum.ai/beta/v1/trace"
