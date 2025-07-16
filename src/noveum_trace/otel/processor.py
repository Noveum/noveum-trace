"""
OpenTelemetry simple span processor for Noveum Trace SDK.
"""

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from opentelemetry.context import Context
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter

try:
    from opentelemetry.context import Context
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    # Type aliases for when OpenTelemetry is not available
    SimpleSpanProcessor = Any  # type: ignore
    SpanExporter = Any  # type: ignore
    ReadableSpan = Any  # type: ignore
    Context = Any  # type: ignore

from noveum_trace.utils.exceptions import ConfigurationError

from .exporter import NoveumSpanExporter

logger = logging.getLogger(__name__)


class NoveumSpanProcessor(SimpleSpanProcessor):
    """OpenTelemetry span processor that uses Noveum exporter."""

    def __init__(self, exporter: NoveumSpanExporter):
        """Initialize the processor with a Noveum exporter."""
        if not OPENTELEMETRY_AVAILABLE:
            raise ConfigurationError(
                "OpenTelemetry is not available. Install with: pip install opentelemetry-api opentelemetry-sdk"
            )

        super().__init__(exporter)
        self._shutdown = False

        logger.info("NoveumSpanProcessor initialized")

    def shutdown(self) -> None:
        """Shutdown the processor."""
        self._shutdown = True
        super().shutdown()
        logger.info("NoveumSpanProcessor shutdown")
