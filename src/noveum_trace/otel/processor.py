"""
OpenTelemetry span processor for Noveum Trace SDK.
"""

from typing import Optional
import logging

try:
    from opentelemetry.sdk.trace import SpanProcessor, ReadableSpan
    from opentelemetry.context import Context
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    SpanProcessor = object
    ReadableSpan = object
    Context = object

from .exporter import NoveumSpanExporter
from ..utils.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class NoveumSpanProcessor(SpanProcessor):
    """OpenTelemetry span processor that uses Noveum exporter."""
    
    def __init__(self, exporter: NoveumSpanExporter):
        """Initialize the processor with a Noveum exporter."""
        if not OPENTELEMETRY_AVAILABLE:
            raise ConfigurationError(
                "OpenTelemetry is not available. Install with: pip install opentelemetry-api opentelemetry-sdk"
            )
        
        self._exporter = exporter
        self._shutdown = False
        
        logger.info("NoveumSpanProcessor initialized")
    
    def on_start(self, span: ReadableSpan, parent_context: Optional[Context] = None) -> None:
        """Called when a span is started."""
        # No action needed on start
        pass
    
    def on_end(self, span: ReadableSpan) -> None:
        """Called when a span is ended."""
        if self._shutdown:
            return
        
        try:
            # Export the span immediately (simple processor)
            self._exporter.export([span])
        except Exception as e:
            logger.error(f"Failed to process span on end: {e}")
    
    def shutdown(self) -> None:
        """Shutdown the processor."""
        self._shutdown = True
        if self._exporter:
            self._exporter.shutdown()
        logger.info("NoveumSpanProcessor shutdown")
    
    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush any pending spans."""
        if self._exporter:
            return self._exporter.force_flush(timeout_millis)
        return True

