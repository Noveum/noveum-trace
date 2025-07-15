"""
OpenTelemetry tracer provider for Noveum Trace SDK.
"""

from typing import Optional
import logging

try:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.resources import Resource
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    TracerProvider = object
    Resource = object

from .processor import NoveumSpanProcessor
from .exporter import NoveumSpanExporter
from ..core.tracer import NoveumTracer
from ..utils.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class NoveumTracerProvider(TracerProvider):
    """OpenTelemetry tracer provider that integrates with Noveum Trace SDK."""
    
    def __init__(
        self,
        noveum_tracer: NoveumTracer,
        resource: Optional[Resource] = None,
        shutdown_on_exit: bool = True,
    ):
        """Initialize the tracer provider."""
        if not OPENTELEMETRY_AVAILABLE:
            raise ConfigurationError(
                "OpenTelemetry is not available. Install with: pip install opentelemetry-api opentelemetry-sdk"
            )
        
        # Create resource from Noveum tracer config if not provided
        if resource is None:
            resource = Resource.create({
                "service.name": noveum_tracer.config.service_name,
                "service.version": noveum_tracer.config.service_version,
                "deployment.environment": noveum_tracer.config.environment,
            })
        
        super().__init__(resource=resource, shutdown_on_exit=shutdown_on_exit)
        
        # Set up Noveum integration
        self._noveum_tracer = noveum_tracer
        self._exporter = NoveumSpanExporter(noveum_tracer)
        self._processor = NoveumSpanProcessor(self._exporter)
        
        # Add the processor to this provider
        self.add_span_processor(self._processor)
        
        logger.info("NoveumTracerProvider initialized with Noveum integration")
    
    def shutdown(self) -> None:
        """Shutdown the tracer provider."""
        if self._processor:
            self._processor.shutdown()
        if self._noveum_tracer:
            self._noveum_tracer.shutdown()
        super().shutdown()
        logger.info("NoveumTracerProvider shutdown")


def setup_opentelemetry_integration(noveum_tracer: NoveumTracer) -> NoveumTracerProvider:
    """Set up OpenTelemetry integration with Noveum Trace SDK."""
    if not OPENTELEMETRY_AVAILABLE:
        raise ConfigurationError(
            "OpenTelemetry is not available. Install with: pip install opentelemetry-api opentelemetry-sdk"
        )
    
    # Create and configure the tracer provider
    provider = NoveumTracerProvider(noveum_tracer)
    
    # Set as global tracer provider
    try:
        from opentelemetry import trace
        trace.set_tracer_provider(provider)
        logger.info("OpenTelemetry integration configured successfully")
    except Exception as e:
        logger.error(f"Failed to set global tracer provider: {e}")
        raise
    
    return provider

