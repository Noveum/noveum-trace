"""
HTTP transport implementation for Noveum Trace SDK.

This module handles HTTP communication with the Noveum platform,
including request formatting, authentication, and error handling.
"""

import time

# Import for type hints
from typing import TYPE_CHECKING, Any, NoReturn, Optional
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from noveum_trace import __version__
from noveum_trace.core.config import get_config

if TYPE_CHECKING:
    from noveum_trace.core.config import Config

from noveum_trace.core.trace import Trace
from noveum_trace.transport.batch_processor import BatchProcessor
from noveum_trace.utils.exceptions import TransportError
from noveum_trace.utils.logging import (
    get_sdk_logger,
    log_debug_enabled,
    log_error_always,
    log_http_request,
    log_http_response,
    log_trace_flow,
)

_MOCK_TYPES: tuple[type[Any], ...]

try:
    from unittest.mock import (
        AsyncMock,
        MagicMock,
        Mock,
        NonCallableMagicMock,
        NonCallableMock,
    )

    _MOCK_TYPES = (Mock, MagicMock, AsyncMock, NonCallableMagicMock, NonCallableMock)
except ImportError:
    _MOCK_TYPES = ()

logger = get_sdk_logger("transport.http_transport")


class HttpTransport:
    """
    HTTP transport for sending traces to the Noveum platform.

    This class handles HTTP communication including authentication,
    request formatting, batching, retries, and error handling.
    """

    def __init__(self, config: Optional["Config"] = None) -> None:
        """Initialize the HTTP transport.

        Args:
            config: Optional configuration instance. If not provided, uses global config.
        """
        self.config = config if config is not None else get_config()
        self.session = self._create_session()
        self.batch_processor = BatchProcessor(self._send_batch, self.config)
        self._shutdown = False

        logger.info(
            f"HTTP transport initialized for endpoint: {self.config.transport.endpoint}"
        )

        if log_debug_enabled():
            logger.debug("🔧 Transport configuration:")
            logger.debug(f"    endpoint: {self.config.transport.endpoint}")
            logger.debug(f"    timeout: {self.config.transport.timeout}s")
            logger.debug(f"    retry_attempts: {self.config.transport.retry_attempts}")
            logger.debug(f"    batch_size: {self.config.transport.batch_size}")
            logger.debug(f"    batch_timeout: {self.config.transport.batch_timeout}s")
            logger.debug(f"    compression: {self.config.transport.compression}")
            logger.debug(f"    ssl_verify: {self.config.transport.ssl_verify}")
            logger.debug(
                f"    ca_bundle: {self.config.transport.ca_bundle or 'default (certifi)'}"
            )

    def _get_sdk_version(self) -> str:
        """Get the SDK version."""
        return __version__

    def _build_api_url(self, path: str) -> str:
        """
        Build API URL preserving the full endpoint path.

        Args:
            path: API path (e.g., "/v1/traces")

        Returns:
            Complete URL with preserved endpoint path
        """
        endpoint = self.config.transport.endpoint.rstrip("/")
        path = path.lstrip("/")
        url = f"{endpoint}/{path}"

        if log_debug_enabled():
            logger.debug(f"🔗 Built API URL: {url}")

        return url

    def _contains_sensitive_data(self, text: str) -> bool:
        """
        Check if response text contains potentially sensitive data.

        Args:
            text: Response text to check

        Returns:
            True if sensitive patterns are detected
        """
        if not text:
            return False

        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower()

        # Common sensitive data patterns
        sensitive_patterns = [
            "password",
            "token",
            "secret",
            "key",
            "credential",
            "authorization",
            "bearer",
            "api_key",
            "access_token",
            "private_key",
            "certificate",
            "ssn",
            "social_security",
            "credit_card",
            "card_number",
            "cvv",
            "pin",
            "account_number",
        ]

        # Check for sensitive patterns
        for pattern in sensitive_patterns:
            if pattern in text_lower:
                return True

        return False

    def _get_safe_response_preview(
        self, response: requests.Response, max_length: Optional[int] = None
    ) -> Optional[str]:
        """
        Get a safe preview of response text for logging.

        Args:
            response: HTTP response object
            max_length: Maximum length for preview (default: from config or 1000)

        Returns:
            Safe preview string or None if no response text
        """
        if not response.text:
            return None

        # Use provided max_length or get from config, with fallback to 1000
        if max_length is None:
            max_length = getattr(self.config.transport, "max_response_preview", 1000)

        # Check if response contains sensitive patterns
        if self._contains_sensitive_data(response.text):
            return f"<Response contains sensitive data, length: {len(response.text)} chars>"

        # Return truncated version if too long
        if len(response.text) > max_length:
            return f"{response.text[:max_length]}... (truncated, total length: {len(response.text)} chars)"

        return response.text

    def _handle_ssl_error(
        self,
        error: requests.exceptions.SSLError,
        url: str,
        **context: Any,
    ) -> NoReturn:
        """
        Handle SSL errors with helpful troubleshooting information.

        Args:
            error: The SSL error that occurred
            url: The URL that was being accessed
            **context: Additional context for logging (e.g., trace_count)

        Raises:
            TransportError: Always raised with SSL error details
        """
        ssl_error_msg = str(error)
        help_msg = (
            f"SSL error: {ssl_error_msg}\n\n"
            "Possible solutions:\n"
            "1. Upgrade certifi: pip install --upgrade certifi\n"
            "2. If behind corporate proxy (Netskope, Zscaler, etc.): "
            "pip install pip-system-certs\n"
            "3. Set custom CA bundle: export NOVEUM_CA_BUNDLE=/path/to/ca.crt\n"
            "4. Disable SSL verification (debugging only): "
            "export NOVEUM_SSL_VERIFY=false"
        )
        log_error_always(logger, help_msg, exc_info=True, url=url, **context)
        raise TransportError(f"SSL error: {ssl_error_msg}") from error

    def export_trace(self, trace: Trace) -> None:
        """
        Export a trace to the Noveum platform.

        Args:
            trace: Trace to export

        Raises:
            TransportError: If transport is shutdown or export fails
        """
        if self._shutdown:
            log_error_always(
                logger,
                f"Cannot export trace {trace.trace_id} - transport has been shutdown",
                trace_id=trace.trace_id,
            )
            raise TransportError("Transport has been shutdown")

        # Skip no-op traces
        if hasattr(trace, "_noop") and trace._noop:
            logger.debug(f"⏭️  Skipping no-op trace {trace.trace_id}")
            return

        # Log trace export details
        span_count = len(trace.spans) if hasattr(trace, "spans") else 0
        logger.info(
            f"📤 EXPORTING TRACE: {trace.name} (ID: {trace.trace_id}) - {span_count} spans"
        )

        if log_debug_enabled():
            log_trace_flow(
                logger,
                "Exporting trace to transport",
                trace_id=trace.trace_id,
                trace_name=trace.name,
                span_count=span_count,
                trace_status=getattr(trace, "status", "unknown"),
                trace_finished=getattr(trace, "_finished", "unknown"),
            )

        # Convert trace to export format
        try:
            trace_data = self._format_trace_for_export(trace)
        except Exception as e:
            log_error_always(
                logger,
                f"Failed to format trace {trace.trace_id} for export",
                exc_info=True,
                trace_id=trace.trace_id,
                error=str(e),
            )
            raise TransportError(f"Failed to format trace: {e}") from e

        # Log formatted trace details
        if log_debug_enabled():
            formatted_span_count = len(trace_data.get("spans", []))
            logger.debug("📋 Formatted trace details:")
            logger.debug(f"    trace_id: {trace_data.get('trace_id')}")
            logger.debug(f"    name: {trace_data.get('name')}")
            logger.debug(f"    spans: {formatted_span_count}")
            logger.debug(f"    keys: {list(trace_data.keys())}")
            logger.debug(f"    sdk_info: {trace_data.get('sdk', {})}")
            logger.debug(f"    project: {trace_data.get('project', 'None')}")
            logger.debug(f"    environment: {trace_data.get('environment', 'None')}")

        # Add to batch processor
        try:
            self.batch_processor.add_trace(trace_data)
            logger.info(f"✅ Trace {trace.trace_id} successfully queued for export")
        except Exception as e:
            log_error_always(
                logger,
                f"Failed to queue trace {trace.trace_id} for export",
                exc_info=True,
                trace_id=trace.trace_id,
                error=str(e),
            )
            raise

    def flush(self, timeout: Optional[float] = None) -> None:
        """
        Flush all pending traces.

        Args:
            timeout: Maximum time to wait for flush completion
        """
        if self._shutdown:
            logger.debug("Transport already shutdown, skipping flush")
            return

        log_trace_flow(logger, "Starting transport flush", timeout=timeout)

        try:
            self.batch_processor.flush(timeout)
            try:
                logger.info("HTTP transport flush completed")
            except (ValueError, OSError, RuntimeError, Exception):
                # Logger may be closed during shutdown
                pass
        except Exception as e:
            log_error_always(
                logger,
                "Transport flush failed",
                exc_info=True,
                timeout=timeout,
                error=str(e),
            )
            raise

    def shutdown(self) -> None:
        """Shutdown the transport and flush pending data."""
        if self._shutdown:
            try:
                logger.debug("Transport already shutdown")
            except (ValueError, OSError, RuntimeError, Exception):
                pass
            return

        try:
            logger.info("Shutting down HTTP transport")
        except (ValueError, OSError, RuntimeError, Exception):
            # Logger may be closed during shutdown
            pass
        self._shutdown = True

        try:
            # Flush pending data
            self.flush(timeout=30.0)

            # Shutdown batch processor
            self.batch_processor.shutdown()

            # Close session
            self.session.close()

            try:
                logger.info("HTTP transport shutdown completed")
            except (ValueError, OSError, RuntimeError, Exception):
                # Logger may be closed during shutdown
                pass
        except Exception as e:
            log_error_always(
                logger, "Error during transport shutdown", exc_info=True, error=str(e)
            )

    def _create_session(self) -> requests.Session:
        """Create and configure HTTP session."""
        session = requests.Session()

        # Set headers
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"noveum-trace-sdk/{self._get_sdk_version()}",
        }

        # Add authentication
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
            if log_debug_enabled():
                # Don't log the actual key, just indicate it's present
                logger.debug(
                    f"🔐 API key configured (length: {len(self.config.api_key)})"
                )
        else:
            logger.warning("⚠️  No API key configured - requests may fail")

        session.headers.update(headers)

        # Configure SSL verification
        ssl_verify = getattr(self.config.transport, "ssl_verify", True)
        ca_bundle = getattr(self.config.transport, "ca_bundle", None)

        if ca_bundle:
            # Use custom CA bundle (for corporate proxies)
            session.verify = ca_bundle
            logger.info(f"🔒 Using custom CA bundle: {ca_bundle}")
        elif not ssl_verify:
            # Disable SSL verification (NOT recommended for production!)
            session.verify = False
            logger.warning(
                "⚠️  SSL verification DISABLED - this is insecure and should only "
                "be used for debugging. Set ssl_verify=True for production."
            )
            # Suppress InsecureRequestWarning
            import urllib3

            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        else:
            # Default: use certifi bundle
            session.verify = True
            if log_debug_enabled():
                try:
                    import certifi

                    logger.debug(f"🔒 Using certifi CA bundle: {certifi.where()}")
                    logger.debug(f"    certifi version: {certifi.__version__}")
                except ImportError:
                    logger.debug("🔒 Using system CA bundle (certifi not installed)")

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

        if log_debug_enabled():
            logger.debug("🔄 HTTP session configured:")
            logger.debug(f"    retry_attempts: {self.config.transport.retry_attempts}")
            logger.debug(f"    retry_backoff: {self.config.transport.retry_backoff}")
            logger.debug(f"    ssl_verify: {ssl_verify}")
            logger.debug(f"    ca_bundle: {ca_bundle or 'default'}")
            logger.debug(f"    headers: {dict(session.headers)}")

        return session

    def trace_to_dict(self, obj: Any) -> Any:
        """
        Recursively convert objects to JSON-serializable dictionaries.

        Args:
            obj: Object to convert

        Returns:
            JSON-serializable representation of the object
        """
        # Safeguard: Detect Mock objects to prevent infinite recursion
        if _MOCK_TYPES and isinstance(obj, _MOCK_TYPES):
            return "<Mock object>"
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, dict):
            return {key: self.trace_to_dict(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self.trace_to_dict(item) for item in obj]
        elif hasattr(obj, "to_dict") and callable(obj.to_dict):
            try:
                return self.trace_to_dict(obj.to_dict())
            except Exception:
                return "Non-serializable object, issue with tracing SDK"
        elif hasattr(obj, "__dict__"):
            try:
                # Extract attributes from object
                attrs = {}
                for key, value in obj.__dict__.items():
                    if not key.startswith("_"):  # Skip private attributes
                        attrs[key] = self.trace_to_dict(value)
                return attrs
            except Exception:
                return "Non-serializable object, issue with tracing SDK"
        else:
            try:
                # Try to convert to string representation
                return str(obj)
            except Exception:
                return "Non-serializable object, issue with tracing SDK"

    def _format_trace_for_export(self, trace: Trace) -> dict[str, Any]:
        """
        Format trace data for export to Noveum platform.

        Args:
            trace: Trace to format

        Returns:
            Formatted trace data
        """
        log_trace_flow(logger, "Formatting trace for export", trace_id=trace.trace_id)

        trace_data = self.trace_to_dict(trace)

        # Handle case where trace_to_dict returns a string due to serialization errors
        if isinstance(trace_data, str):
            # Create a minimal trace structure with the error message and all required fields
            from datetime import datetime, timezone

            current_time = datetime.now(timezone.utc)
            trace_data = {
                "trace_id": trace.trace_id,
                "name": getattr(trace, "name", "unknown"),
                "start_time": getattr(trace, "start_time", current_time).isoformat(),
                "end_time": (
                    getattr(trace, "end_time", current_time).isoformat()
                    if getattr(trace, "end_time", None)
                    else current_time.isoformat()
                ),
                "duration_ms": getattr(trace, "duration_ms", 0.0) or 0.0,
                "status": "error",
                "status_message": trace_data,  # The error message from trace_to_dict
                "span_count": getattr(trace, "span_count", 0),
                "error_count": getattr(trace, "error_count", 1),
                "attributes": {},
                "metadata": {
                    "user_id": None,
                    "session_id": None,
                    "request_id": None,
                    "tags": {},
                    "custom_attributes": {},
                },
                "spans": [],
            }

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
        url = self._build_api_url("/v1/trace")

        log_http_request(
            logger,
            "POST",
            url,
            trace_id=trace_data.get("trace_id"),
            content_type="application/json",
            timeout=self.config.transport.timeout,
        )

        try:
            # Send request
            response = self.session.post(
                url,
                json=trace_data,
                timeout=self.config.transport.timeout,
            )

            log_http_response(
                logger,
                response.status_code,
                url,
                response_headers=(
                    dict(response.headers) if log_debug_enabled() else None
                ),
            )

            # Check response
            if response.status_code in [200, 201]:
                logger.debug(f"Successfully sent trace: {trace_data.get('trace_id')}")
                return response.json()
            elif response.status_code == 401:
                log_error_always(
                    logger,
                    "Authentication failed - check API key",
                    status=response.status_code,
                    url=url,
                )
                raise TransportError("Authentication failed - check API key")
            elif response.status_code == 403:
                log_error_always(
                    logger,
                    "Access forbidden - check project permissions",
                    status=response.status_code,
                    url=url,
                )
                raise TransportError("Access forbidden - check project permissions")
            elif response.status_code == 429:
                log_error_always(
                    logger, "Rate limit exceeded", status=response.status_code, url=url
                )
                raise TransportError("Rate limit exceeded")
            else:
                log_error_always(
                    logger,
                    f"Unexpected HTTP status code: {response.status_code}",
                    status=response.status_code,
                    url=url,
                    response_text=self._get_safe_response_preview(response),
                )
                response.raise_for_status()
                return response.json()

        except requests.exceptions.SSLError as e:
            self._handle_ssl_error(e, url)
        except requests.exceptions.RequestException as e:
            log_error_always(
                logger, f"HTTP request failed: {e}", exc_info=True, url=url
            )
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
            logger.debug("_send_batch called with empty traces list, skipping")
            return

        # Prepare request payload
        payload = {
            "traces": traces,
            "timestamp": time.time(),
        }

        # Compress payload if enabled
        if self.config.transport.compression:
            payload = self._compress_payload(payload)

        # Log detailed request information
        url = self._build_api_url("/v1/traces")
        logger.info(f"🚀 SENDING BATCH: {len(traces)} traces to {url}")

        if log_debug_enabled():
            log_http_request(
                logger,
                "POST",
                url,
                trace_count=len(traces),
                payload_keys=list(payload.keys()),
                payload_size_chars=len(str(payload)),
                headers=dict(self.session.headers),
                compression_enabled=self.config.transport.compression,
            )

            # Log individual trace info
            for i, trace in enumerate(traces):
                trace_id = trace.get("trace_id", "unknown")
                trace_name = trace.get("name", "unnamed")
                span_count = len(trace.get("spans", []))
                logger.debug(
                    f"    [{i+1}] {trace_name} (ID: {trace_id}) - {span_count} spans"
                )

        try:
            # Send request
            response = self.session.post(
                url,
                json=payload,
                timeout=self.config.transport.timeout,
            )

            # Log response details
            logger.info(f"📡 HTTP RESPONSE: Status {response.status_code} from {url}")

            if log_debug_enabled():
                log_http_response(
                    logger,
                    response.status_code,
                    url,
                    response_headers=dict(response.headers),
                    response_size=len(response.text) if response.text else 0,
                    response_preview=self._get_safe_response_preview(response),
                )

            # Check response
            if response.status_code in [200, 201]:
                logger.info(f"✅ Successfully sent batch of {len(traces)} traces")
                if log_debug_enabled():
                    safe_preview = self._get_safe_response_preview(
                        response, max_length=2000
                    )
                    logger.debug(f"📋 Response preview: {safe_preview}")
            elif response.status_code == 401:
                log_error_always(
                    logger,
                    "Authentication failed - check API key",
                    status=response.status_code,
                    url=url,
                    trace_count=len(traces),
                )
                raise TransportError("Authentication failed - check API key")
            elif response.status_code == 403:
                log_error_always(
                    logger,
                    "Access forbidden - check project permissions",
                    status=response.status_code,
                    url=url,
                    trace_count=len(traces),
                )
                raise TransportError("Access forbidden - check project permissions")
            elif response.status_code == 429:
                log_error_always(
                    logger,
                    "Rate limit exceeded",
                    status=response.status_code,
                    url=url,
                    trace_count=len(traces),
                )
                raise TransportError("Rate limit exceeded")
            else:
                log_error_always(
                    logger,
                    f"Unexpected status code: {response.status_code}",
                    status=response.status_code,
                    url=url,
                    trace_count=len(traces),
                    response_text=self._get_safe_response_preview(response),
                )
                response.raise_for_status()

        except requests.exceptions.Timeout as e:
            log_error_always(
                logger,
                f"Request timeout after {self.config.transport.timeout}s",
                exc_info=True,
                url=url,
                timeout=self.config.transport.timeout,
                trace_count=len(traces),
            )
            raise TransportError(
                f"Request timeout after {self.config.transport.timeout}s"
            ) from e
        except requests.exceptions.SSLError as e:
            self._handle_ssl_error(e, url, trace_count=len(traces))
        except requests.exceptions.ConnectionError as e:
            log_error_always(
                logger,
                f"Connection error: {e}",
                exc_info=True,
                url=url,
                trace_count=len(traces),
            )
            raise TransportError(f"Connection error: {e}") from e
        except requests.exceptions.HTTPError as e:
            log_error_always(
                logger,
                f"HTTP error: {e}",
                exc_info=True,
                url=url,
                trace_count=len(traces),
            )
            raise TransportError(f"HTTP error: {e}") from e
        except Exception as e:
            log_error_always(
                logger,
                f"Unexpected error: {e}",
                exc_info=True,
                url=url,
                trace_count=len(traces),
            )
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
        if log_debug_enabled():
            logger.debug("🗜️  Payload compression requested but not implemented yet")
        return payload

    def health_check(self) -> bool:
        """
        Perform a health check against the Noveum platform.

        Returns:
            True if the platform is reachable, False otherwise
        """
        try:
            url = urljoin(self.config.transport.endpoint, "/health")

            log_trace_flow(logger, "Performing health check", url=url)

            response = self.session.get(url, timeout=10)
            is_healthy = response.status_code == 200

            if is_healthy:
                logger.debug(f"✅ Health check passed for {url}")
            else:
                logger.warning(
                    f"⚠️  Health check failed for {url} (status: {response.status_code})"
                )

            return is_healthy
        except Exception as e:
            logger.warning(
                f"⚠️  Health check failed for {self.config.transport.endpoint}: {e}"
            )
            return False

    def __repr__(self) -> str:
        """String representation of the transport."""
        return (
            f"HttpTransport(endpoint='{self.config.transport.endpoint}', "
            f"batch_size={self.config.transport.batch_size})"
        )
