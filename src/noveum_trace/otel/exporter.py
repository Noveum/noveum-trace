"""
OpenTelemetry span exporter for Noveum Trace SDK.
"""

from typing import Sequence, Optional
import logging

try:
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
    from opentelemetry.trace import Span as OTelSpan
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    SpanExporter = object
    SpanExportResult = object
    OTelSpan = object

from ..core.tracer import NoveumTracer
from ..types import SpanData, SpanKind, SpanStatus
from ..utils.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class NoveumSpanExporter(SpanExporter):
    """OpenTelemetry span exporter that forwards spans to Noveum Trace SDK."""
    
    def __init__(self, tracer: NoveumTracer):
        """Initialize the exporter with a Noveum tracer."""
        if not OPENTELEMETRY_AVAILABLE:
            raise ConfigurationError(
                "OpenTelemetry is not available. Install with: pip install opentelemetry-api opentelemetry-sdk"
            )
        
        self._tracer = tracer
        self._shutdown = False
        
        logger.info("NoveumSpanExporter initialized")
    
    def export(self, spans: Sequence[OTelSpan]) -> SpanExportResult:
        """Export OpenTelemetry spans to Noveum Trace SDK."""
        if self._shutdown:
            return SpanExportResult.FAILURE
        
        try:
            # Convert OpenTelemetry spans to Noveum SpanData
            noveum_spans = []
            for otel_span in spans:
                span_data = self._convert_otel_span(otel_span)
                if span_data:
                    noveum_spans.append(span_data)
            
            # Export to Noveum sinks
            if noveum_spans:
                self._tracer._export_spans(noveum_spans)
            
            return SpanExportResult.SUCCESS
            
        except Exception as e:
            logger.error(f"Failed to export spans: {e}")
            return SpanExportResult.FAILURE
    
    def shutdown(self) -> None:
        """Shutdown the exporter."""
        self._shutdown = True
        logger.info("NoveumSpanExporter shutdown")
    
    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush any pending spans."""
        try:
            return self._tracer.flush(timeout_ms=timeout_millis)
        except Exception as e:
            logger.error(f"Failed to flush spans: {e}")
            return False
    
    def _convert_otel_span(self, otel_span) -> Optional[SpanData]:
        """Convert OpenTelemetry span to Noveum SpanData."""
        try:
            # Get span context
            span_context = otel_span.get_span_context()
            
            # Convert span kind
            kind_mapping = {
                1: SpanKind.INTERNAL,  # INTERNAL
                2: SpanKind.SERVER,    # SERVER
                3: SpanKind.CLIENT,    # CLIENT
                4: SpanKind.PRODUCER,  # PRODUCER
                5: SpanKind.CONSUMER,  # CONSUMER
            }
            kind = kind_mapping.get(otel_span.kind, SpanKind.INTERNAL)
            
            # Convert span status
            status_mapping = {
                0: SpanStatus.UNSET,  # UNSET
                1: SpanStatus.OK,     # OK
                2: SpanStatus.ERROR,  # ERROR
            }
            status = status_mapping.get(otel_span.status.status_code, SpanStatus.UNSET)
            
            # Extract attributes
            attributes = {}
            if hasattr(otel_span, 'attributes') and otel_span.attributes:
                attributes = dict(otel_span.attributes)
            
            # Extract events
            events = []
            if hasattr(otel_span, 'events') and otel_span.events:
                for event in otel_span.events:
                    event_data = {
                        "name": event.name,
                        "timestamp": event.timestamp,
                        "attributes": dict(event.attributes) if event.attributes else {},
                    }
                    events.append(event_data)
            
            # Extract links
            links = []
            if hasattr(otel_span, 'links') and otel_span.links:
                for link in otel_span.links:
                    link_context = link.context
                    link_data = {
                        "span_id": f"{link_context.span_id:016x}",
                        "trace_id": f"{link_context.trace_id:032x}",
                        "attributes": dict(link.attributes) if link.attributes else {},
                    }
                    links.append(link_data)
            
            # Create SpanData
            span_data = SpanData(
                span_id=f"{span_context.span_id:016x}",
                trace_id=f"{span_context.trace_id:032x}",
                parent_span_id=f"{otel_span.parent.span_id:016x}" if otel_span.parent else None,
                name=otel_span.name,
                kind=kind,
                status=status,
                start_time=otel_span.start_time,
                end_time=otel_span.end_time,
                attributes=attributes,
                events=events,
                links=links,
            )
            
            return span_data
            
        except Exception as e:
            logger.error(f"Failed to convert OpenTelemetry span: {e}")
            return None

