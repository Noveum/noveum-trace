"""
Main client for Noveum Trace SDK.

The NoveumClient class orchestrates all tracing functionality including
trace creation, span management, and data export to the Noveum platform.
"""

import atexit
import logging
import random
import threading
from datetime import datetime
from enum import Enum

# Import for type hints
from typing import TYPE_CHECKING, Any, Optional

from noveum_trace import __version__
from noveum_trace.core.config import get_config

if TYPE_CHECKING:
    from noveum_trace.core.config import Config

from noveum_trace.core.context import (
    ContextualSpan,
    ContextualTrace,
    get_current_span,
    get_current_trace,
    set_current_span,
    set_current_trace,
)
from noveum_trace.core.span import Span
from noveum_trace.core.trace import Trace
from noveum_trace.transport.http_transport import HttpTransport
from noveum_trace.utils.exceptions import NoveumTraceError

logger = logging.getLogger(__name__)


class SamplingDecision(Enum):
    """Sampling decision enumeration."""

    RECORD = "record"
    DROP = "drop"


def should_sample(
    name: str,
    attributes: Optional[dict[str, Any]] = None,
    sample_rate: float = 1.0,
) -> SamplingDecision:
    """
    Simple sampling decision function.

    Args:
        name: Trace/span name
        attributes: Attributes (unused in simple implementation)
        sample_rate: Sampling rate (0.0 to 1.0)

    Returns:
        Sampling decision
    """
    if sample_rate >= 1.0:
        return SamplingDecision.RECORD
    elif sample_rate <= 0.0:
        return SamplingDecision.DROP
    else:
        return (
            SamplingDecision.RECORD
            if random.random() < sample_rate
            else SamplingDecision.DROP
        )


class NoveumClient:
    """
    Main client for the Noveum Trace SDK.

    This class provides the primary interface for creating traces and spans,
    managing context, and exporting data to the Noveum platform.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        project: Optional[str] = None,
        config: Optional["Config"] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Noveum client.

        Args:
            api_key: API key for authentication
            project: Project name
            config: Optional configuration instance
            **kwargs: Additional configuration options
        """

        if config is not None:
            self.config = config
        elif any([api_key, project, kwargs]):
            # Create config from provided parameters
            config_dict = {}
            if api_key is not None:
                config_dict["api_key"] = api_key
            if project is not None:
                config_dict["project"] = project
            config_dict.update(kwargs)

            from noveum_trace.core.config import configure

            configure(config_dict)
            self.config = get_config()
        else:
            self.config = get_config()

        self.transport = HttpTransport(self.config)

        # State management
        self._active_traces: dict[str, Trace] = {}
        self._lock = threading.RLock()
        self._shutdown = False

        # Register shutdown handler
        atexit.register(self.shutdown)

        logger.info("Noveum Trace client initialized")

    def _get_sdk_version(self) -> str:
        """Get the SDK version."""
        return __version__

    def start_trace(
        self,
        name: str,
        attributes: Optional[dict[str, Any]] = None,
        start_time: Optional[datetime] = None,
        set_as_current: bool = True,
        **kwargs: Any,
    ) -> Trace:
        """
        Start a new trace.

        Args:
            name: Trace name
            attributes: Initial trace attributes
            start_time: Start time (defaults to current time)
            set_as_current: Whether to set as current trace
            **kwargs: Additional keyword arguments added to trace attributes

        Returns:
            New Trace instance

        Raises:
            NoveumTraceError: If tracing is disabled or client is shutdown
        """
        if self._shutdown:
            raise NoveumTraceError("Client has been shutdown")

        if not self.config.tracing.enabled:
            logger.debug("Tracing is disabled, returning no-op trace")
            return self._create_noop_trace(name)

        # Merge kwargs into attributes
        merged_attributes = attributes.copy() if attributes else {}
        merged_attributes.update(kwargs)

        # Check sampling decision
        sampling_decision = should_sample(
            name=name,
            attributes=merged_attributes,
            sample_rate=self.config.tracing.sample_rate,
        )

        if sampling_decision == SamplingDecision.DROP:
            logger.debug(f"Trace '{name}' dropped by sampling")
            return self._create_noop_trace(name)

        # Create the trace
        trace = Trace(
            name=name,
            attributes=merged_attributes,
            start_time=start_time,
        )

        # Add configuration attributes
        trace.set_attributes(
            {
                "noveum.project": self.config.project,
                "noveum.environment": self.config.environment,
                "noveum.sdk.version": self._get_sdk_version(),
                "noveum.sampling.decision": sampling_decision.value,
            }
        )

        # Register the trace
        with self._lock:
            self._active_traces[trace.trace_id] = trace

        # Set as current trace if requested
        if set_as_current:
            set_current_trace(trace)

        logger.debug(f"Started trace: {trace.trace_id}")
        return trace

    def start_span(
        self,
        name: str,
        parent_span_id: Optional[str] = None,
        attributes: Optional[dict[str, Any]] = None,
        start_time: Optional[datetime] = None,
        set_as_current: bool = True,
    ) -> Span:
        """
        Start a new span.

        Args:
            name: Span name
            parent_span_id: Parent span ID (defaults to current span)
            attributes: Initial span attributes
            start_time: Start time (defaults to current time)
            set_as_current: Whether to set as current span

        Returns:
            New Span instance

        Raises:
            NoveumTraceError: If no active trace or client is shutdown
        """
        if self._shutdown:
            raise NoveumTraceError("Client has been shutdown")

        # Get current trace
        trace = get_current_trace()
        if trace is None:
            raise NoveumTraceError(
                "No active trace found. Please start a trace before creating spans."
            )

        # Determine parent span
        if parent_span_id is None:
            current_span = get_current_span()
            if current_span:
                parent_span_id = current_span.span_id

        # Create the span
        span = trace.create_span(
            name=name,
            parent_span_id=parent_span_id,
            attributes=attributes,
            start_time=start_time,
        )

        # Set as current span if requested
        if set_as_current:
            set_current_span(span)

        logger.debug(f"Started span: {span.span_id} in trace: {trace.trace_id}")
        return span

    def finish_span(
        self,
        span: Span,
        end_time: Optional[datetime] = None,
    ) -> None:
        """
        Finish a span.

        Args:
            span: Span to finish
            end_time: End time (defaults to current time)
        """
        if span.is_finished():
            return

        span.finish(end_time)

        # Clear from current context if it's the current span
        current_span = get_current_span()
        if current_span and current_span.span_id == span.span_id:
            # Set parent span as current, or None if no parent
            parent_span = None
            if span.parent_span_id:
                trace = get_current_trace()
                if trace:
                    parent_span = trace.get_span(span.parent_span_id)
            set_current_span(parent_span)

        logger.debug(f"Finished span: {span.span_id}")

    def finish_trace(
        self,
        trace: Trace,
        end_time: Optional[datetime] = None,
    ) -> None:
        """
        Finish a trace.

        Args:
            trace: Trace to finish
            end_time: End time (defaults to current time)
        """
        if trace.is_finished():
            return

        trace.finish(end_time)

        # Remove from active traces
        with self._lock:
            self._active_traces.pop(trace.trace_id, None)

        # Clear from current context if it's the current trace
        current_trace = get_current_trace()
        if current_trace and current_trace.trace_id == trace.trace_id:
            set_current_trace(None)
            set_current_span(None)

        # Export the trace
        self._export_trace(trace)

        logger.debug(f"Finished trace: {trace.trace_id}")

    def create_contextual_trace(
        self,
        name: str,
        attributes: Optional[dict[str, Any]] = None,
        start_time: Optional[datetime] = None,
        **kwargs: Any,
    ) -> ContextualTrace:
        """
        Create a contextual trace that manages context automatically.

        Args:
            name: Trace name
            attributes: Initial trace attributes
            start_time: Start time (defaults to current time)
            **kwargs: Additional keyword arguments passed to start_trace

        Returns:
            ContextualTrace instance
        """
        trace = self.start_trace(
            name=name,
            attributes=attributes,
            start_time=start_time,
            set_as_current=False,
            **kwargs,
        )
        return ContextualTrace(trace)

    def create_contextual_span(
        self,
        name: str,
        parent_span_id: Optional[str] = None,
        attributes: Optional[dict[str, Any]] = None,
        start_time: Optional[datetime] = None,
    ) -> ContextualSpan:
        """
        Create a contextual span that manages context automatically.

        Args:
            name: Span name
            parent_span_id: Parent span ID (defaults to current span)
            attributes: Initial span attributes
            start_time: Start time (defaults to current time)

        Returns:
            ContextualSpan instance
        """
        span = self.start_span(
            name=name,
            parent_span_id=parent_span_id,
            attributes=attributes,
            start_time=start_time,
            set_as_current=False,
        )
        return ContextualSpan(span)

    def get_active_traces(self) -> list[Trace]:
        """
        Get all active traces.

        Returns:
            List of active traces
        """
        with self._lock:
            return list(self._active_traces.values())

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """
        Get a trace by ID.

        Args:
            trace_id: Trace ID

        Returns:
            Trace instance or None if not found
        """
        with self._lock:
            return self._active_traces.get(trace_id)

    def flush(self, timeout: Optional[float] = None) -> None:
        """
        Flush all pending traces.

        Args:
            timeout: Maximum time to wait for flush completion
        """
        if self._shutdown:
            return

        # Finish all active traces
        active_traces = self.get_active_traces()
        for trace in active_traces:
            if not trace.is_finished():
                self.finish_trace(trace)

        # Flush transport
        self.transport.flush(timeout)

        try:
            logger.info("Flushed all pending traces")
        except (ValueError, OSError, RuntimeError, Exception):
            # Logger may be closed during shutdown
            pass

    def shutdown(self) -> None:
        """Shutdown the client and flush all pending data."""
        if self._shutdown:
            return

        try:
            logger.info("Shutting down Noveum Trace client")
        except (ValueError, OSError, RuntimeError, Exception):
            # Logger may be closed during shutdown
            pass

        # Flush all pending data BEFORE setting shutdown flag
        self.flush(timeout=30.0)

        # Now set shutdown flag
        self._shutdown = True

        # Shutdown transport
        self.transport.shutdown()

        try:
            logger.info("Noveum Trace client shutdown complete")
        except (ValueError, OSError, RuntimeError):
            # Logger may be closed during shutdown
            pass

    def is_shutdown(self) -> bool:
        """Check if the client has been shutdown."""
        return self._shutdown

    def _export_trace(self, trace: Trace) -> None:
        """
        Export a trace to the Noveum platform.

        Args:
            trace: Trace to export
        """
        try:
            self.transport.export_trace(trace)
        except Exception as e:
            logger.error(f"Failed to export trace {trace.trace_id}: {e}")

    def export_audio(
        self,
        audio_data: bytes,
        trace_id: str,
        span_id: str,
        audio_uuid: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Export audio to the Noveum platform.

        Args:
            audio_data: Audio file bytes (WAV format)
            trace_id: Associated trace ID
            span_id: Associated span ID
            audio_uuid: Unique identifier for this audio file
            metadata: Additional metadata (duration, format, etc.)
        """
        if self._shutdown:
            raise NoveumTraceError("Client has been shutdown")

        try:
            self.transport.export_audio(
                audio_data=audio_data,
                trace_id=trace_id,
                span_id=span_id,
                audio_uuid=audio_uuid,
                metadata=metadata,
            )
        except Exception as e:
            logger.error(f"Failed to export audio {audio_uuid}: {e}")
            # Don't raise - audio export failure shouldn't break tracing

    def export_image(
        self,
        image_data: bytes,
        trace_id: str,
        span_id: str,
        image_uuid: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Export image to the Noveum platform.

        Args:
            image_data: Image file bytes
            trace_id: Associated trace ID
            span_id: Associated span ID
            image_uuid: Unique identifier for this image file
            metadata: Additional metadata (format, size, etc.)
        """
        if self._shutdown:
            raise NoveumTraceError("Client has been shutdown")

        try:
            self.transport.export_image(
                image_data=image_data,
                trace_id=trace_id,
                span_id=span_id,
                image_uuid=image_uuid,
                metadata=metadata,
            )
        except Exception as e:
            logger.error(f"Failed to export image {image_uuid}: {e}")
            # Don't raise - image export failure shouldn't break tracing

    def _create_noop_trace(self, name: str) -> Trace:
        """
        Create a no-op trace for when tracing is disabled or sampled out.

        Args:
            name: Trace name

        Returns:
            No-op Trace instance
        """
        # Create a minimal trace that doesn't get exported
        trace = Trace(name=name)
        trace._noop = True  # Mark as no-op
        return trace

    def __repr__(self) -> str:
        """String representation of the client."""
        return (
            f"NoveumClient(project='{self.config.project}', "
            f"active_traces={len(self._active_traces)})"
        )
