"""
Tests for endpoint configuration functionality.

This module tests the critical endpoint configuration features including:
- Custom endpoint configuration via init()
- Environment variable endpoint configuration
- Transport layer using correct endpoints
"""

import os
from unittest.mock import MagicMock, patch

import noveum_trace
from noveum_trace.core.config import Config, configure, get_config
from noveum_trace.transport.http_transport import HttpTransport


class TestEndpointConfiguration:
    """Test endpoint configuration functionality."""

    def setup_method(self):
        """Reset configuration before each test."""
        # Clear any existing configuration
        noveum_trace._client = None
        if hasattr(noveum_trace.core, "config"):
            noveum_trace.core.config._config = None

    def teardown_method(self):
        """Clean up after each test."""
        # Reset configuration and client
        noveum_trace._client = None
        if hasattr(noveum_trace.core, "config"):
            noveum_trace.core.config._config = None

    def test_init_with_custom_endpoint(self):
        """Test that init() correctly configures custom endpoint."""
        custom_endpoint = "http://localhost:8082/api/v1"

        # Initialize with custom endpoint
        noveum_trace.init(
            api_key="test-key", project="test-project", endpoint=custom_endpoint
        )

        # Check that configuration was set correctly
        config = get_config()
        assert config.transport.endpoint == custom_endpoint
        assert config.api_key == "test-key"
        assert config.project == "test-project"

    def test_init_with_environment_variable(self):
        """Test that NOVEUM_ENDPOINT environment variable works."""
        custom_endpoint = "http://localhost:8082/api/v1"

        # Ensure clean state
        noveum_trace._client = None
        if hasattr(noveum_trace.core, "config"):
            noveum_trace.core.config._config = None

        with patch.dict(
            os.environ,
            {
                "NOVEUM_ENDPOINT": custom_endpoint,
                "NOVEUM_API_KEY": "test-key",
                "NOVEUM_PROJECT": "test-project",
            },
            clear=True,
        ):
            # Initialize without explicit endpoint (should use env var)
            noveum_trace.init()

            # Check that configuration was set correctly
            config = get_config()
            assert config.transport.endpoint == custom_endpoint
            assert config.api_key == "test-key"
            assert config.project == "test-project"

    def test_init_endpoint_overrides_environment(self):
        """Test that init() endpoint parameter overrides environment variable."""
        env_endpoint = "http://env.example.com/api/v1"
        init_endpoint = "http://localhost:8082/api/v1"

        with patch.dict(
            os.environ,
            {
                "NOVEUM_ENDPOINT": env_endpoint,
                "NOVEUM_API_KEY": "test-key",
                "NOVEUM_PROJECT": "test-project",
            },
        ):
            # Initialize with explicit endpoint (should override env var)
            noveum_trace.init(endpoint=init_endpoint)

            # Check that init parameter took precedence
            config = get_config()
            assert config.transport.endpoint == init_endpoint

    def test_config_constructor_with_endpoint(self):
        """Test Config.create() accepts endpoint parameter."""
        custom_endpoint = "http://localhost:8082/api/v1"

        # This should not raise an error
        config = Config.create(
            api_key="test-config-key",
            project="config-test-project",
            endpoint=custom_endpoint,
        )

        assert config.api_key == "test-config-key"
        assert config.project == "config-test-project"
        assert config.transport.endpoint == custom_endpoint

    def test_http_transport_uses_configured_endpoint(self):
        """Test that HttpTransport uses the configured endpoint."""
        custom_endpoint = "http://localhost:8082/api/v1"

        # Configure with custom endpoint
        config = Config.create(endpoint=custom_endpoint)
        configure(config)

        # Create transport (should use configured endpoint)
        transport = HttpTransport()
        assert transport.config.transport.endpoint == custom_endpoint

    @patch("requests.Session.post")
    def test_http_transport_constructs_correct_urls(self, mock_post):
        """Test that HTTP transport constructs URLs correctly with custom endpoint."""
        custom_endpoint = "http://localhost:8082/api/v1"

        # Configure with custom endpoint
        config = Config.create(endpoint=custom_endpoint)
        configure(config)

        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True}
        mock_post.return_value = mock_response

        # Create transport and test single trace URL
        transport = HttpTransport()
        trace_data = {"trace_id": "test", "spans": []}

        transport._send_request(trace_data)

        # Check that correct URL was called
        expected_single_url = "http://localhost:8082/v1/trace"
        mock_post.assert_called_once()
        called_url = (
            mock_post.call_args[1]["url"]
            if "url" in mock_post.call_args[1]
            else mock_post.call_args[0][0]
        )
        assert called_url == expected_single_url

    @patch("requests.Session.post")
    def test_http_transport_batch_url(self, mock_post):
        """Test that HTTP transport uses correct URL for batch requests."""
        custom_endpoint = "http://localhost:8082/api/v1"

        # Configure with custom endpoint
        config = Config.create(endpoint=custom_endpoint)
        configure(config)

        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True}
        mock_post.return_value = mock_response

        # Create transport and test batch URL
        transport = HttpTransport()
        traces = [{"trace_id": "test1"}, {"trace_id": "test2"}]

        transport._send_batch(traces)

        # Check that correct batch URL was called
        expected_batch_url = "http://localhost:8082/v1/traces"
        mock_post.assert_called_once()
        called_url = (
            mock_post.call_args[1]["url"]
            if "url" in mock_post.call_args[1]
            else mock_post.call_args[0][0]
        )
        assert called_url == expected_batch_url

    def test_endpoint_configuration_integration(self):
        """Integration test for complete endpoint configuration flow."""
        custom_endpoint = "http://localhost:8082/api/v1"

        # Test the complete flow: init -> config -> transport
        noveum_trace.init(
            api_key="test-key", project="test-project", endpoint=custom_endpoint
        )

        # Get the initialized client and check its transport
        client = noveum_trace._client
        assert client is not None
        assert client.transport.config.transport.endpoint == custom_endpoint

    def test_default_endpoint_when_none_provided(self):
        """Test that default endpoint is used when none provided."""
        noveum_trace.init(api_key="test-key", project="test-project")

        config = get_config()
        assert config.transport.endpoint == "https://api.noveum.ai"

    def test_endpoint_in_config_dict(self):
        """Test endpoint configuration via config dictionary."""
        custom_endpoint = "http://localhost:8082/api/v1"

        config_dict = {
            "api_key": "test-key",
            "project": "test-project",
            "endpoint": custom_endpoint,
        }

        config = Config.from_dict(config_dict)
        assert config.transport.endpoint == custom_endpoint

    def test_nested_transport_endpoint_precedence(self):
        """Test that top-level endpoint takes precedence over nested transport.endpoint."""
        top_level_endpoint = "http://localhost:8082/api/v1"
        transport_endpoint = "http://localhost:9000/api/v1"

        config_dict = {
            "api_key": "test-key",
            "project": "test-project",
            "endpoint": top_level_endpoint,
            "transport": {"endpoint": transport_endpoint},
        }

        config = Config.from_dict(config_dict)
        assert (
            config.transport.endpoint == top_level_endpoint
        )  # top-level endpoint should win
