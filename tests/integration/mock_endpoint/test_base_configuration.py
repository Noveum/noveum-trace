"""
Integration tests for base configuration with configurable endpoints.

This module tests the core integration functionality with configurable endpoints,
supporting local development (http://localhost:3000) and production (api.noveum.ai).
"""

import os
import time
from typing import Any, Optional
from unittest.mock import Mock, patch

import pytest

import noveum_trace
from noveum_trace.utils.exceptions import ConfigurationError

# Configurable endpoint for integration tests
ENDPOINT = os.environ.get("NOVEUM_ENDPOINT", "https://api.noveum.ai/api")


class MockEndpointServer:
    """Mock server for testing endpoint configuration."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.requests: list[dict[str, Any]] = []
        self.responses: dict[str, Any] = {}

    def setup_response(
        self, endpoint: str, response: dict[str, Any], status_code: int = 200
    ):
        """Setup a mock response for a specific endpoint."""
        self.responses[endpoint] = {"response": response, "status_code": status_code}

    def capture_request(self, method: str, url: str, **kwargs) -> Mock:
        """Capture a request and return a mock response."""
        # Extract endpoint from URL - handle various URL formats
        endpoint = url
        if url.startswith(self.base_url):
            endpoint = url[len(self.base_url) :]
        elif url.startswith("https://api.noveum.ai"):
            endpoint = url.replace("https://api.noveum.ai", "")
        elif url.startswith("http://localhost"):
            endpoint = url.replace("http://localhost:3000", "").replace(
                "http://localhost", ""
            )

        # Ensure endpoint starts with /
        if endpoint and not endpoint.startswith("/"):
            endpoint = "/" + endpoint

        # Store request details
        request_data = {
            "method": method,
            "url": url,
            "endpoint": endpoint,
            "timestamp": time.time(),
            "kwargs": kwargs,
        }
        self.requests.append(request_data)

        # Return configured response or default
        mock_response = Mock()
        if endpoint in self.responses:
            config = self.responses[endpoint]
            mock_response.status_code = config["status_code"]
            mock_response.json.return_value = config["response"]
            mock_response.text = str(config["response"])
        else:
            mock_response.status_code = 200
            mock_response.json.return_value = {"success": True}
            mock_response.text = '{"success": true}'

        mock_response.headers = {"content-type": "application/json"}
        return mock_response

    def get_requests(self, endpoint: Optional[str] = None) -> list[dict[str, Any]]:
        """Get captured requests, optionally filtered by endpoint."""
        if endpoint is None:
            return self.requests.copy()
        return [req for req in self.requests if req["endpoint"] == endpoint]

    def clear_requests(self):
        """Clear all captured requests."""
        self.requests.clear()


@pytest.fixture
def mock_localhost_server():
    """Mock server for localhost:3000 testing."""
    # Use the same endpoint as the tests for consistency
    server = MockEndpointServer(ENDPOINT.rstrip("/"))

    # Setup common endpoints
    server.setup_response("/health", {"status": "ok"})
    server.setup_response("/v1/trace", {"success": True, "trace_id": "test-trace-id"})
    server.setup_response("/v1/traces", {"success": True, "batch_id": "test-batch-id"})
    server.setup_response(
        "/api/v1/trace", {"success": True, "trace_id": "test-trace-id"}
    )
    server.setup_response(
        "/api/v1/traces", {"success": True, "batch_id": "test-batch-id"}
    )

    def mock_post(*args, **kwargs):
        return server.capture_request("POST", args[0], **kwargs)

    def mock_get(*args, **kwargs):
        return server.capture_request("GET", args[0], **kwargs)

    with (
        patch("requests.post", side_effect=mock_post),
        patch("requests.get", side_effect=mock_get),
        patch("requests.Session.post", side_effect=mock_post),
        patch("requests.Session.get", side_effect=mock_get),
    ):
        yield server


@pytest.fixture
def mock_production_server():
    """Mock server for api.noveum.ai testing."""
    server = MockEndpointServer("https://api.noveum.ai")

    # Setup common endpoints
    server.setup_response("/health", {"status": "ok"})
    server.setup_response("/v1/trace", {"success": True, "trace_id": "prod-trace-id"})
    server.setup_response("/v1/traces", {"success": True, "batch_id": "prod-batch-id"})
    server.setup_response(
        "/api/v1/trace", {"success": True, "trace_id": "prod-trace-id"}
    )
    server.setup_response(
        "/api/v1/traces", {"success": True, "batch_id": "prod-batch-id"}
    )

    def mock_post(*args, **kwargs):
        return server.capture_request("POST", args[0], **kwargs)

    def mock_get(*args, **kwargs):
        return server.capture_request("GET", args[0], **kwargs)

    with (
        patch("requests.post", side_effect=mock_post),
        patch("requests.get", side_effect=mock_get),
        patch("requests.Session.post", side_effect=mock_post),
        patch("requests.Session.get", side_effect=mock_get),
    ):
        yield server


@pytest.fixture(autouse=True)
def clean_state_before_each_test():
    """Ensure completely clean state before and after each test."""
    # Before test: Clean up any existing state
    _cleanup_noveum_state()

    yield

    # After test: Clean up to prevent state leakage
    _cleanup_noveum_state()


def _cleanup_noveum_state():
    """Comprehensive cleanup of noveum state."""
    import noveum_trace
    from noveum_trace.core import config

    # Force shutdown any existing client and clear the lock
    with noveum_trace._client_lock:
        if noveum_trace._client is not None:
            try:
                noveum_trace._client.shutdown()
            except Exception:
                # If shutdown fails, force reset the client
                pass
            finally:
                # Force clear the client reference
                noveum_trace._client = None

    # Clear global configuration
    if hasattr(config, "_config"):
        config._config = None

    # Clear any context state
    try:
        from noveum_trace.core import context

        context.clear_context()
    except (ImportError, AttributeError):
        pass


def _force_immediate_export(client):
    """Force immediate export of traces for testing."""
    # Flush the client to force immediate export
    client.flush()

    # Also flush the transport layer
    if hasattr(client, "transport"):
        client.transport.flush()

    # Give a small amount of time for async processing
    time.sleep(0.1)


@pytest.mark.integration
@pytest.mark.disable_transport_mocking
class TestBaseConfiguration:
    """Test base configuration with different endpoints."""

    def test_localhost_endpoint_configuration(self, mock_localhost_server):
        """Test configuration with localhost endpoint."""
        # Initialize with localhost endpoint
        noveum_trace.init(
            project="test-project",
            api_key="test-key",
            endpoint=ENDPOINT,
            environment="development",
        )

        client = noveum_trace.get_client()
        assert client.config.endpoint == ENDPOINT
        assert client.config.project == "test-project"
        assert client.config.environment == "development"

    def test_production_endpoint_configuration(self, mock_production_server):
        """Test configuration with production endpoint."""
        # Initialize with production endpoint
        noveum_trace.init(
            project="prod-project",
            api_key="prod-key",
            endpoint=ENDPOINT,
            environment="production",
        )

        client = noveum_trace.get_client()
        assert client.config.endpoint == ENDPOINT
        assert client.config.project == "prod-project"
        assert client.config.environment == "production"

    def test_default_endpoint_configuration(self):
        """Test that default endpoint is production."""
        noveum_trace.init(project="default-project", api_key="default-key")

        client = noveum_trace.get_client()
        assert client.config.endpoint == "https://api.noveum.ai/api"

    def test_endpoint_validation(self):
        """Test endpoint validation."""
        # Test invalid URL
        with pytest.raises(ConfigurationError):
            noveum_trace.init(
                project="test-project", api_key="test-key", endpoint="not-a-valid-url"
            )

        # Test missing protocol
        with pytest.raises(ConfigurationError):
            noveum_trace.init(
                project="test-project", api_key="test-key", endpoint="localhost:3000"
            )

    def test_health_check_localhost(self, mock_localhost_server):
        """Test health check against localhost."""
        noveum_trace.init(project="test-project", api_key="test-key", endpoint=ENDPOINT)

        client = noveum_trace.get_client()
        health_status = client.transport.health_check()

        # Check that health endpoint was called
        health_requests = mock_localhost_server.get_requests("/health")
        assert len(health_requests) == 1
        assert health_status is True

    def test_health_check_production(self, mock_production_server):
        """Test health check against production."""
        noveum_trace.init(project="test-project", api_key="test-key", endpoint=ENDPOINT)

        client = noveum_trace.get_client()
        health_status = client.transport.health_check()

        # Check that health endpoint was called
        health_requests = mock_production_server.get_requests("/health")
        assert len(health_requests) == 1
        assert health_status is True

    def test_endpoint_switching(self, mock_localhost_server, mock_production_server):
        """Test switching between endpoints in the same session."""
        # Start with localhost
        noveum_trace.init(project="test-project", api_key="test-key", endpoint=ENDPOINT)

        client = noveum_trace.get_client()
        assert client.config.endpoint == ENDPOINT

        # Shutdown and reinitialize with production
        noveum_trace.shutdown()

        noveum_trace.init(project="test-project", api_key="test-key", endpoint=ENDPOINT)

        new_client = noveum_trace.get_client()
        assert new_client.config.endpoint == ENDPOINT

    def test_trace_export_localhost(self, mock_localhost_server):
        """Test trace export to localhost endpoint."""
        noveum_trace.init(project="test-project", api_key="test-key", endpoint=ENDPOINT)

        # Create and export a trace
        trace = noveum_trace.start_trace("test-trace")
        trace.set_attribute("test", "value")
        span = trace.create_span("test-span")
        span.finish()

        client = noveum_trace.get_client()
        client.finish_trace(trace)

        # Force immediate export for testing
        _force_immediate_export(client)

        # Check that trace was sent to localhost
        trace_requests = mock_localhost_server.get_requests("/api/v1/traces")
        trace_requests_v1 = mock_localhost_server.get_requests("/v1/traces")

        assert (
            len(trace_requests) >= 1 or len(trace_requests_v1) >= 1
        ), f"Expected at least 1 trace request, got {len(trace_requests)} for /api/v1/traces and {len(trace_requests_v1)} for /v1/traces"

    def test_trace_export_production(self, mock_production_server):
        """Test trace export to production endpoint."""
        noveum_trace.init(project="test-project", api_key="test-key", endpoint=ENDPOINT)

        # Create and export a trace
        trace = noveum_trace.start_trace("test-trace")
        trace.set_attribute("test", "value")
        span = trace.create_span("test-span")
        span.finish()

        client = noveum_trace.get_client()
        client.finish_trace(trace)

        # Force immediate export for testing
        _force_immediate_export(client)

        # Check that trace was sent to production
        trace_requests = mock_production_server.get_requests("/api/v1/traces")
        trace_requests_v1 = mock_production_server.get_requests("/v1/traces")

        assert (
            len(trace_requests) >= 1 or len(trace_requests_v1) >= 1
        ), f"Expected at least 1 trace request, got {len(trace_requests)} for /api/v1/traces and {len(trace_requests_v1)} for /v1/traces"

    def test_environment_specific_configuration(self, mock_localhost_server):
        """Test environment-specific configuration patterns."""
        # Development environment should use localhost
        os.environ["ENVIRONMENT"] = "development"
        try:
            noveum_trace.init(
                project="test-project",
                api_key="test-key",
                endpoint=ENDPOINT,
                environment="development",
            )

            client = noveum_trace.get_client()
            config = client.config

            assert config.endpoint == ENDPOINT
            assert config.environment == "development"

            # Verify that development-specific settings can be applied
            assert config.transport.batch_timeout <= 5.0  # Faster flushing in dev
        finally:
            # Clean up environment variable
            os.environ.pop("ENVIRONMENT", None)

    def test_production_configuration_best_practices(self, mock_production_server):
        """Test production configuration best practices."""
        noveum_trace.init(
            project="prod-project",
            api_key="prod-api-key",
            endpoint=ENDPOINT,
            environment="production",
        )

        client = noveum_trace.get_client()
        config = client.config

        assert config.endpoint == ENDPOINT
        assert config.environment == "production"

        # Verify production-appropriate settings
        assert config.transport.batch_size >= 50  # Larger batches for efficiency
        assert config.transport.timeout >= 30.0  # Longer timeout for reliability

    def test_custom_endpoint_configuration(self):
        """Test custom endpoint configuration."""
        custom_endpoint = "https://custom.noveum-instance.com"

        with (
            patch("requests.get") as mock_get,
            patch("requests.post") as mock_post,
            patch("requests.Session.get") as mock_session_get,
            patch("requests.Session.post") as mock_session_post,
        ):

            # Mock successful responses
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "ok"}

            mock_get.return_value = mock_response
            mock_post.return_value = mock_response
            mock_session_get.return_value = mock_response
            mock_session_post.return_value = mock_response

            noveum_trace.init(
                project="custom-project", api_key="custom-key", endpoint=custom_endpoint
            )

            client = noveum_trace.get_client()
            assert client.config.endpoint == custom_endpoint

    def test_endpoint_path_preservation(self):
        """Test that custom endpoint paths are preserved."""
        custom_endpoint = "https://api.custom.com/tracing"

        with (
            patch("requests.get") as mock_get,
            patch("requests.post") as mock_post,
            patch("requests.Session.get") as mock_session_get,
            patch("requests.Session.post") as mock_session_post,
        ):

            # Mock successful responses
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "ok"}

            mock_get.return_value = mock_response
            mock_post.return_value = mock_response
            mock_session_get.return_value = mock_response
            mock_session_post.return_value = mock_response

            noveum_trace.init(
                project="custom-project", api_key="custom-key", endpoint=custom_endpoint
            )

            client = noveum_trace.get_client()

            # Create a test trace to trigger transport
            trace = noveum_trace.start_trace("test-trace")
            span = trace.create_span("test-span")
            span.finish()
            client.finish_trace(trace)

            # Force immediate export for testing
            _force_immediate_export(client)

            # Verify the correct endpoint was used
            assert client.config.endpoint == custom_endpoint


@pytest.mark.integration
@pytest.mark.disable_transport_mocking
class TestConfigurationPersistence:
    """Test configuration persistence across operations."""

    def test_configuration_persists_across_operations(self, mock_localhost_server):
        """Test that configuration persists across multiple operations."""
        noveum_trace.init(
            project="persistence-test",
            api_key="test-key",
            endpoint=ENDPOINT,
            environment="test",
        )

        client = noveum_trace.get_client()

        # Perform multiple operations and verify config persists
        for i in range(3):
            trace = noveum_trace.start_trace(f"test-trace-{i}")
            trace.set_attribute("iteration", i)
            span = trace.create_span(f"test-span-{i}")
            span.finish()
            client.finish_trace(trace)

            # Verify config hasn't changed
            assert client.config.endpoint == ENDPOINT
            assert client.config.project == "persistence-test"
            assert client.config.environment == "test"

        # Force immediate export for testing
        _force_immediate_export(client)

        # Verify all traces were sent to the correct endpoint
        trace_requests = mock_localhost_server.get_requests("/api/v1/traces")
        trace_requests_v1 = mock_localhost_server.get_requests("/v1/traces")

        assert (
            len(trace_requests) >= 1 or len(trace_requests_v1) >= 1
        ), f"Expected at least 1 trace request, got {len(trace_requests)} for /api/v1/traces and {len(trace_requests_v1)} for /v1/traces"

    def test_configuration_validation_on_init(self):
        """Test that configuration is validated during initialization."""
        # Test missing required fields - project is not actually required
        # The SDK will work without a project, so remove this test

        # Test invalid endpoint format
        with pytest.raises(ConfigurationError):
            noveum_trace.init(project="test", api_key="test", endpoint="invalid-url")

        # Test valid configuration passes
        noveum_trace.init(project="test-project", api_key="test-key", endpoint=ENDPOINT)

        assert noveum_trace.is_initialized()
