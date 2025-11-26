import json
import sys
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

# Ensure we use the source code, not the installed package
# Add src directory to Python path if not already there
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import noveum_trace modules for proper mocking
# Note: These imports must come after sys.path modification
import noveum_trace  # noqa: E402
from noveum_trace.core.client import NoveumClient  # noqa: E402
from noveum_trace.core.config import Config  # noqa: E402
from noveum_trace.transport.batch_processor import BatchProcessor  # noqa: E402
from noveum_trace.transport.http_transport import HttpTransport  # noqa: E402

# Global registry to track all clients created during tests
_test_clients: list[NoveumClient] = []
_original_client_init = None


def track_client(client: NoveumClient):
    """Track a client for cleanup"""
    _test_clients.append(client)


def cleanup_all_clients():
    """Clean up all tracked clients"""
    global _test_clients
    for client in _test_clients:
        try:
            if hasattr(client, "transport") and client.transport:
                # Stop batch processor immediately
                if (
                    hasattr(client.transport, "batch_processor")
                    and client.transport.batch_processor
                ):
                    batch_proc = client.transport.batch_processor
                    if hasattr(batch_proc, "_shutdown"):
                        batch_proc._shutdown = True
                    if hasattr(batch_proc, "_thread"):
                        batch_proc._thread = None

                # Mock transport methods to prevent real operations
                client.transport.export_trace = Mock()
                client.transport.flush = Mock()
                client.transport.shutdown = Mock()

            # Set shutdown flag immediately
            if hasattr(client, "_shutdown"):
                client._shutdown = True
            if hasattr(client, "_active_traces"):
                client._active_traces.clear()

        except Exception:
            pass  # Ignore cleanup errors

    _test_clients.clear()


def reset_noveum_config():
    """Reset global configuration state for tests."""
    import noveum_trace.core.config as config_module

    config_module._config = None


# Add marker support


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "disable_transport_mocking: disable automatic transport mocking for this test",
    )


@pytest.fixture(autouse=True)
def prevent_real_api_calls(request):
    """Prevent any real HTTP calls to noveum.ai or other endpoints"""

    # Skip if test is marked to disable mocking
    if request.node.get_closest_marker("disable_transport_mocking"):
        yield {}
        return

    def mock_requests_post(*args, **kwargs):
        """Mock requests.post to prevent real API calls"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True, "trace_id": "mock-trace-id"}
        mock_response.text = '{"success": true}'
        mock_response.headers = {"content-type": "application/json"}
        return mock_response

    def mock_requests_get(*args, **kwargs):
        """Mock requests.get to prevent real API calls"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_response.text = '{"status": "ok"}'
        return mock_response

    def mock_session_post(*args, **kwargs):
        """Mock requests.Session.post"""
        return mock_requests_post(*args, **kwargs)

    def mock_session_get(*args, **kwargs):
        """Mock requests.Session.get"""
        return mock_requests_get(*args, **kwargs)

    # Patch all HTTP methods
    with (
        patch("requests.post", side_effect=mock_requests_post) as mock_post,
        patch("requests.get", side_effect=mock_requests_get) as mock_get,
        patch(
            "requests.Session.post", side_effect=mock_session_post
        ) as mock_session_post,
        patch("requests.Session.get", side_effect=mock_session_get) as mock_session_get,
        patch("urllib3.PoolManager.request") as mock_urllib3,
        patch("urllib.request.urlopen") as mock_urlopen,
    ):

        # Configure urllib3 mock
        mock_urllib3_response = Mock()
        mock_urllib3_response.status = 200
        mock_urllib3_response.data = b'{"success": true}'
        mock_urllib3.return_value = mock_urllib3_response

        # Configure urlopen mock
        mock_urlopen_response = Mock()
        mock_urlopen_response.read.return_value = b'{"success": true}'
        mock_urlopen_response.getcode.return_value = 200
        mock_urlopen.return_value = mock_urlopen_response

        yield {
            "post": mock_post,
            "get": mock_get,
            "session_post": mock_session_post,
            "session_get": mock_session_get,
            "urllib3": mock_urllib3,
            "urlopen": mock_urlopen,
        }


@pytest.fixture(autouse=True)
def mock_transport_completely(request):
    """Mock all transport operations completely"""

    # Skip if test is marked to disable mocking
    if request.node.get_closest_marker("disable_transport_mocking"):
        yield
        return

    def mock_http_transport_init(self, config=None):
        """Mock HttpTransport.__init__"""
        self.config = config or Config.create()
        self.session = Mock()
        self.batch_processor = Mock()
        self.batch_processor.add_trace = Mock()
        self.batch_processor.flush = Mock()
        self.batch_processor.shutdown = Mock()
        self._shutdown = False

    def mock_http_transport_export(self, trace_data):
        """Mock HttpTransport.export_trace"""
        return {"success": True, "trace_id": "mock-trace-id"}

    def mock_http_transport_flush(self, timeout=None):
        """Mock HttpTransport.flush"""
        return True

    def mock_http_transport_shutdown(self):
        """Mock HttpTransport.shutdown"""
        self._shutdown = True
        return True

    def mock_batch_processor_init(
        self, transport, max_batch_size=100, flush_interval=5.0
    ):
        """Mock BatchProcessor.__init__"""
        self.transport = transport
        self.max_batch_size = max_batch_size
        self.flush_interval = flush_interval
        self._queue = []
        self._shutdown = False
        self._thread = None  # Don't start real thread

    def mock_batch_processor_add(self, trace_data):
        """Mock BatchProcessor.add_trace"""
        if not self._shutdown:
            self._queue.append(trace_data)

    def mock_batch_processor_flush(self):
        """Mock BatchProcessor.flush"""
        self._queue.clear()

    def mock_batch_processor_shutdown(self):
        """Mock BatchProcessor.shutdown"""
        self._shutdown = True
        self._queue.clear()

    with (
        patch.object(HttpTransport, "__init__", mock_http_transport_init),
        patch.object(HttpTransport, "export_trace", mock_http_transport_export),
        patch.object(HttpTransport, "flush", mock_http_transport_flush),
        patch.object(HttpTransport, "shutdown", mock_http_transport_shutdown),
        patch.object(BatchProcessor, "__init__", mock_batch_processor_init),
        patch.object(BatchProcessor, "add_trace", mock_batch_processor_add),
        patch.object(BatchProcessor, "flush", mock_batch_processor_flush),
        patch.object(BatchProcessor, "shutdown", mock_batch_processor_shutdown),
    ):
        yield


@pytest.fixture(autouse=True)
def mock_client_creation(request):
    """Mock client creation to track instances for cleanup"""
    global _original_client_init

    # Skip if test is marked to disable mocking
    if request.node.get_closest_marker("disable_transport_mocking"):
        yield
        return

    if _original_client_init is None:
        _original_client_init = NoveumClient.__init__

    def mock_client_init(self, api_key=None, project=None, config=None, **kwargs):
        """Mock NoveumClient.__init__ to track instances"""
        # Call original init logic but with mocked transport
        self.api_key = api_key or "test-api-key"
        self.project = project or "test-project"
        self.config = config or Config.create(
            api_key=self.api_key, project=self.project
        )
        self._active_traces = {}
        self._shutdown = False
        self._lock = threading.RLock()  # Add the missing _lock attribute

        # Create mocked transport
        self.transport = Mock()
        self.transport.export_trace = Mock(return_value={"success": True})
        self.transport.flush = Mock()
        self.transport.shutdown = Mock()

        # Track this instance
        track_client(self)

    with patch.object(NoveumClient, "__init__", mock_client_init):
        yield


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up after each test"""
    yield
    # Clean up all clients created during the test
    cleanup_all_clients()

    # Clear any module-level state
    try:
        noveum_trace.core.context.clear_context()
    except Exception:
        pass


@pytest.fixture
def mock_http_transport():
    """Provide a properly mocked HttpTransport"""
    transport = Mock(spec=HttpTransport)
    transport.export_trace = Mock(
        return_value={"success": True, "trace_id": "mock-trace-id"}
    )
    transport.flush = Mock()
    transport.shutdown = Mock()
    transport.batch_processor = Mock()
    transport.batch_processor.add_trace = Mock()
    transport.batch_processor.flush = Mock()
    transport.batch_processor.shutdown = Mock()
    return transport


@pytest.fixture
def mock_requests():
    """Provide mocked requests for specific test needs"""
    with patch("requests.Session") as mock_session_class:
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True}
        mock_response.text = '{"success": true}'

        mock_session.post.return_value = mock_response
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        yield {
            "session": mock_session,
            "response": mock_response,
            "session_class": mock_session_class,
        }


@pytest.fixture
def client_with_mocked_transport():
    """Provide a client with completely mocked transport"""
    transport = Mock(spec=HttpTransport)
    transport.export_trace = Mock(return_value={"success": True})
    transport.flush = Mock()
    transport.shutdown = Mock()

    with patch("noveum_trace.core.client.HttpTransport", return_value=transport):
        client = NoveumClient(api_key="test-key", project="test-project")
        client.transport = transport  # Ensure it's set

        # Set as global client for decorator tests
        import noveum_trace

        with noveum_trace._client_lock:
            old_client = noveum_trace._client
            noveum_trace._client = client

        track_client(client)
        try:
            yield client
        finally:
            # Restore old client
            with noveum_trace._client_lock:
                noveum_trace._client = old_client


@pytest.fixture
def mock_config():
    """Provide a mock config for testing"""
    config = Config.create(
        api_key="test-api-key", project="test-project", endpoint="https://api.noveum.ai"
    )
    return config


# Ensure proper cleanup on test session end
def pytest_sessionfinish(session, exitstatus):
    """Clean up at the end of test session"""
    cleanup_all_clients()


# Add endpoint capture functionality
class TestEndpointCapture:
    """Enhanced endpoint capture for validating HTTP requests and trace data"""

    def __init__(self):
        self.requests: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def capture_request(
        self, url: str, method: str, headers: dict[str, str], data: Any
    ) -> Mock:
        """Capture an HTTP request and return a mock response"""
        with self.lock:
            request_data = {
                "url": url,
                "method": method,
                "headers": dict(headers) if headers else {},
                "timestamp": time.time(),
                "raw_data": data,
            }

            # Try to parse JSON data
            if data:
                try:
                    if isinstance(data, (str, bytes)):
                        request_data["json_data"] = json.loads(data)
                    else:
                        request_data["json_data"] = data
                except (json.JSONDecodeError, TypeError):
                    request_data["json_data"] = None

            self.requests.append(request_data)

        # Return a mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "trace_id": f"trace-{len(self.requests)}",
        }
        mock_response.text = '{"success": true}'
        mock_response.headers = {"content-type": "application/json"}
        return mock_response

    def get_trace_requests(self) -> list[dict[str, Any]]:
        """Get all requests that contain trace data"""
        with self.lock:
            return [req for req in self.requests if req.get("json_data")]

    def get_latest_trace(self) -> dict[str, Any]:
        """Get the latest trace data sent to the endpoint"""
        trace_requests = self.get_trace_requests()
        return trace_requests[-1]["json_data"] if trace_requests else {}

    def get_all_requests(self) -> list[dict[str, Any]]:
        """Get all captured requests"""
        with self.lock:
            return self.requests.copy()

    def get_health_check_requests(self) -> list[dict[str, Any]]:
        """Get all health check requests"""
        with self.lock:
            return [req for req in self.requests if "/health" in req.get("url", "")]

    def get_single_trace_requests(self) -> list[dict[str, Any]]:
        """Get all single trace requests (not batch)"""
        with self.lock:
            return [
                req
                for req in self.requests
                if "/v1/trace" in req.get("url", "")
                and "/v1/traces" not in req.get("url", "")
            ]

    def clear(self):
        """Clear all captured requests"""
        with self.lock:
            self.requests.clear()


@pytest.fixture
def endpoint_capture():
    """Provide an endpoint capture instance for validation tests"""
    capture = TestEndpointCapture()

    def mock_requests_post(url, data=None, headers=None, **kwargs):
        return capture.capture_request(url, "POST", headers, data)

    def mock_session_post(url, data=None, headers=None, **kwargs):
        return capture.capture_request(url, "POST", headers, data)

    # Patch HTTP methods to capture requests but only for endpoint validation tests
    with (
        patch("requests.post", side_effect=mock_requests_post),
        patch("requests.Session.post", side_effect=mock_session_post),
    ):
        yield capture


@pytest.fixture
def capturing_http_transport():
    """Provide an HttpTransport that captures requests instead of mocking them away"""
    capture = TestEndpointCapture()

    def mock_export_trace(self, trace_data):
        """Mock export_trace to capture the data"""
        # Simulate sending the trace data to endpoint
        mock_response = capture.capture_request(
            url=self.config.endpoint + "/traces",
            method="POST",
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            data=trace_data,
        )
        return mock_response.json()

    with patch.object(HttpTransport, "export_trace", mock_export_trace):
        yield capture
