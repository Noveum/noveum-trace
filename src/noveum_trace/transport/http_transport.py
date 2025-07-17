"""
HTTP transport implementation for Noveum Trace SDK.

This module handles HTTP communication with the Noveum platform,
including request formatting, authentication, and error handling.
"""

import logging
import time
from typing import Any, Optional
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from noveum_trace import __version__
from noveum_trace.core.config import get_config
from noveum_trace.core.trace import Trace
from noveum_trace.transport.batch_processor import BatchProcessor
from noveum_trace.utils.exceptions import TransportError

logger = logging.getLogger(__name__)


class HttpTransport:
    """
    HTTP transport for sending traces to the Noveum platform.

    This class handles HTTP communication including authentication,
    request formatting, batching, retries, and error handling.
    """

    def __init__(self) -> None:
        """Initialize the HTTP transport."""
        self.config = get_config()
        self.session = self._create_session()
        self.batch_processor = BatchProcessor(self._send_batch)
        self._shutdown = False

        logger.info(
            f"HTTP transport initialized for endpoint: {self.config.transport.endpoint}"
        )

    def _get_sdk_version(self) -> str:
        """Get the SDK version."""
        return __version__

    def export_trace(self, trace: Trace) -> None:
        """
        Export a trace to the Noveum platform.

        Args:
            trace: Trace to export

        Raises:
            TransportError: If transport is shutdown or export fails
        """
        if self._shutdown:
            raise TransportError("Transport has been shutdown")

        # Skip no-op traces
        if hasattr(trace, "_noop") and trace._noop:
            return

        # Convert trace to export format
        trace_data = self._format_trace_for_export(trace)

        # Add to batch processor
        self.batch_processor.add_trace(trace_data)

        logger.debug(f"Trace {trace.trace_id} queued for export")

    def flush(self, timeout: Optional[float] = None) -> None:
        """
        Flush all pending traces.

        Args:
            timeout: Maximum time to wait for flush completion
        """
        if self._shutdown:
            return

        self.batch_processor.flush(timeout)
        logger.info("HTTP transport flush completed")

    def shutdown(self) -> None:
        """Shutdown the transport and flush pending data."""
        if self._shutdown:
            return

        logger.info("Shutting down HTTP transport")
        self._shutdown = True

        # Flush pending data
        self.flush(timeout=30.0)

        # Shutdown batch processor
        self.batch_processor.shutdown()

        # Close session
        self.session.close()

        logger.info("HTTP transport shutdown completed")

    def _create_session(self) -> requests.Session:
        """Create and configure HTTP session."""
        session = requests.Session()

        # Set headers
        session.headers.update(
            {
                "Content-Type": "application/json",
                "User-Agent": f"noveum-trace-sdk/{self._get_sdk_version()}",
            }
        )

        # Add authentication
        if self.config.api_key:
            session.headers["Authorization"] = f"Bearer {self.config.api_key}"

        # Configure retries
        retry_strategy = Retry(
            total=self.config.transport.retry_attempts,
            backoff_factor=self.config.transport.retry_backoff,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def _format_trace_for_export(self, trace: Trace) -> dict[str, Any]:
        """
        Format trace data for export to Noveum platform.

        Args:
            trace: Trace to format

        Returns:
            Formatted trace data
        """
        trace_data = trace.to_dict()

        # Add SDK metadata
        trace_data["sdk"] = {
            "name": "noveum-trace-python",
            "version": self._get_sdk_version(),
        }

        # Add project information
        if self.config.project:
            trace_data["project"] = self.config.project

        if self.config.environment:
            trace_data["environment"] = self.config.environment

        return trace_data

    def _send_request(self, trace_data: dict[str, Any]) -> dict[str, Any]:
        """
        Send a single trace request to the Noveum platform.

        Args:
            trace_data: Trace data to send

        Returns:
            Response data

        Raises:
            TransportError: If the request fails
        """
        try:
            # Send request
            url = urljoin(self.config.transport.endpoint, "/v1/trace")
            response = self.session.post(
                url,
                json=trace_data,
                timeout=self.config.transport.timeout,
            )

            # Check response
            if response.status_code == 200:
                logger.debug(f"Successfully sent trace: {trace_data.get('trace_id')}")
                return response.json()
            elif response.status_code == 401:
                raise TransportError("Authentication failed - check API key")
            elif response.status_code == 403:
                raise TransportError("Access forbidden - check project permissions")
            elif response.status_code == 429:
                raise TransportError("Rate limit exceeded")
            else:
                response.raise_for_status()
                return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send trace: {e}")
            raise TransportError(f"HTTP request failed: {e}") from e

    def _send_batch(self, traces: list[dict[str, Any]]) -> None:
        """
        Send a batch of traces to the Noveum platform.

        Args:
            traces: List of trace data to send

        Raises:
            TransportError: If the request fails
        """
        if not traces:
            return

        # Prepare request payload
        payload = {
            "traces": traces,
            "timestamp": time.time(),
        }

        # Compress payload if enabled
        if self.config.transport.compression:
            payload = self._compress_payload(payload)

        try:
            # Send request
            url = urljoin(self.config.transport.endpoint, "/v1/traces")
            response = self.session.post(
                url,
                json=payload,
                timeout=self.config.transport.timeout,
            )

            # Check response
            if response.status_code == 200:
                logger.debug(f"Successfully sent batch of {len(traces)} traces")
            elif response.status_code == 401:
                raise TransportError("Authentication failed - check API key")
            elif response.status_code == 403:
                raise TransportError("Access forbidden - check project permissions")
            elif response.status_code == 429:
                raise TransportError("Rate limit exceeded")
            else:
                response.raise_for_status()

        except requests.exceptions.Timeout as e:
            raise TransportError(
                f"Request timeout after {self.config.transport.timeout}s"
            ) from e
        except requests.exceptions.ConnectionError as e:
            raise TransportError(f"Connection error: {e}") from e
        except requests.exceptions.HTTPError as e:
            raise TransportError(f"HTTP error: {e}") from e
        except Exception as e:
            raise TransportError(f"Unexpected error: {e}") from e

    def _compress_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Compress payload if beneficial.

        Args:
            payload: Payload to compress

        Returns:
            Potentially compressed payload
        """
        # For now, just return the payload as-is
        # In the future, we could implement gzip compression
        return payload

    def health_check(self) -> bool:
        """
        Perform a health check against the Noveum platform.

        Returns:
            True if the platform is reachable, False otherwise
        """
        try:
            url = urljoin(self.config.transport.endpoint, "/health")
            response = self.session.get(url, timeout=10)
            return response.status_code == 200
        except Exception:
            return False

    def __repr__(self) -> str:
        """String representation of the transport."""
        return (
            f"HttpTransport(endpoint='{self.config.transport.endpoint}', "
            f"batch_size={self.config.transport.batch_size})"
        )
