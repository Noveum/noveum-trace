"""
Core Span implementation for representing individual operations in a trace.
"""

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
from dataclasses import asdict

from ..types import (
    SpanData, SpanKind, SpanStatus, LLMRequest, LLMResponse, TraceID, SpanID
)
from ..utils.exceptions import SpanError, ValidationError

if TYPE_CHECKING:
    from .context import TraceContext


class Span:
    """
    Represents a single operation within a trace.
    
    A span represents a single operation within a trace. Spans can be nested
    to form a trace tree. Each span contains operation name, start/end times,
    and key-value attributes.
    """
    
    def __init__(
        self,
        name_or_span_data: Union[str, SpanData],
        tracer: Optional['NoveumTracer'] = None,
        attributes: Optional[Dict[str, Any]] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        parent_span_id: Optional[SpanID] = None,
        trace_id: Optional[TraceID] = None,
    ):
        """Initialize a new span."""
        # Handle both string name and SpanData object
        if isinstance(name_or_span_data, str):
            # Create SpanData from string name
            from ..types import SpanData, TraceID, SpanID
            import uuid
            from datetime import datetime, timezone
            
            self.span_data = SpanData(
                span_id=SpanID(str(uuid.uuid4())),
                trace_id=trace_id or TraceID(str(uuid.uuid4())),
                parent_span_id=parent_span_id,
                name=name_or_span_data,
                kind=kind,
                status=SpanStatus.UNSET,
                start_time=datetime.now(timezone.utc),
                end_time=None,
                duration_ms=None,
                attributes=attributes or {},
                events=[],
                links=[],
            )
        else:
            # Use provided SpanData
            self.span_data = name_or_span_data
            if attributes:
                self.span_data.attributes.update(attributes)
        
        self._tracer = tracer
        self._is_recording = True
        self._ended = False
    
    @property
    def span_id(self) -> SpanID:
        """Get the span ID."""
        return self.span_data.span_id
    
    @property
    def trace_id(self) -> TraceID:
        """Get the trace ID."""
        return self.span_data.trace_id

    @property
    def parent_span_id(self) -> Optional[SpanID]:
        """Get the parent span ID."""
        return self.span_data.parent_span_id

    @property
    def name(self) -> str:
        """Get the span name."""
        return self.span_data.name

    @property
    def kind(self) -> SpanKind:
        """Get the span kind."""
        return self.span_data.kind

    @property
    def status(self) -> SpanStatus:
        """Get the span status."""
        return self.span_data.status

    @property
    def is_recording(self) -> bool:
        """Check if the span is recording."""
        return self._is_recording and not self._ended

    @property
    def is_ended(self) -> bool:
        """Check if the span has ended."""
        return self._ended

    @property
    def duration_ms(self) -> Optional[int]:
        """Get the span duration in milliseconds."""
        return self.span_data.duration_ms

    @property
    def end_time(self) -> Optional[datetime]:
        """Get the span end time."""
        return self.span_data.end_time

    def get_attribute(self, key: str) -> Any:
        """Get an attribute value by key."""
        return self.span_data.attributes.get(key)
    
    def set_attribute(self, key: str, value: Any) -> 'Span':
        """
        Set a single attribute on the span.
        
        Args:
            key: Attribute key
            value: Attribute value
        
        Returns:
            Self for method chaining
        """
        if not self.is_recording:
            return self
        
        self.span_data.set_attribute(key, value)
        return self
    
    def set_attributes(self, attributes: Dict[str, Any]) -> 'Span':
        """
        Set multiple attributes on the span.
        
        Args:
            attributes: Dictionary of attributes to set
        
        Returns:
            Self for method chaining
        """
        if not self.is_recording:
            return self
        
        for key, value in attributes.items():
            self.span_data.set_attribute(key, value)
        return self
    
    def add_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> 'Span':
        """
        Add an event to the span.
        
        Args:
            name: Event name
            attributes: Event attributes
            timestamp: Event timestamp (defaults to now)
        
        Returns:
            Self for method chaining
        """
        if not self.is_recording:
            return self
        
        self.span_data.add_event(name, attributes)
        return self
    
    def record_exception(
        self,
        exception: Exception,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> 'Span':
        """
        Record an exception as a span event.
        
        Args:
            exception: Exception to record
            attributes: Additional attributes
        
        Returns:
            Self for method chaining
        """
        if not self.is_recording:
            return self
        
        event_attributes = {
            "exception.type": type(exception).__name__,
            "exception.message": str(exception),
        }
        
        if attributes:
            event_attributes.update(attributes)
        
        self.add_event("exception", event_attributes)
        self.set_status(SpanStatus.ERROR, str(exception))
        
        return self
    
    def set_status(self, status: Union[SpanStatus, str], description: Optional[str] = None) -> 'Span':
        """
        Set the span status.
        
        Args:
            status: Span status (enum or string)
            description: Status description
        
        Returns:
            Self for method chaining
        """
        if not self.is_recording:
            return self
        
        # Convert string to SpanStatus enum if needed
        if isinstance(status, str):
            status_mapping = {
                "ok": SpanStatus.OK,
                "error": SpanStatus.ERROR,
                "unset": SpanStatus.UNSET,
            }
            status = status_mapping.get(status.lower(), SpanStatus.UNSET)
        
        self.span_data.set_status(status, description)
        
        # Also set the description as an attribute for backward compatibility
        if description:
            self.span_data.set_attribute("status.description", description)
        
        return self
    
    def add_link(
        self,
        trace_id: TraceID,
        span_id: SpanID,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> 'Span':
        """
        Add a link to another span.
        
        Args:
            trace_id: Target trace ID
            span_id: Target span ID
            attributes: Link attributes
        
        Returns:
            Self for method chaining
        """
        if not self.is_recording:
            return self
        
        link = {
            "trace_id": str(trace_id),
            "span_id": str(span_id),
            "attributes": attributes or {}
        }
        self.span_data.links.append(link)
        
        return self
    
    def set_llm_request(self, request: LLMRequest) -> 'Span':
        """
        Set LLM request data on the span.
        
        Args:
            request: LLM request data
        
        Returns:
            Self for method chaining
        """
        if not self.is_recording:
            return self
        
        self.span_data.llm_request = request
        
        # Add LLM-specific attributes
        self.set_attribute("llm.request.model", request.model)
        self.set_attribute("gen_ai.request.model", request.model)
        self.set_attribute("llm.request.messages", len(request.messages))
        
        # Set operation type attributes
        if request.operation_type:
            self.set_attribute("gen_ai.operation.name", request.operation_type.value.split(".")[-1])
        
        # Set AI system attributes
        if request.ai_system:
            self.set_attribute("gen_ai.system", request.ai_system.value)
        
        if request.temperature is not None:
            self.set_attribute("llm.request.temperature", request.temperature)
        if request.max_tokens is not None:
            self.set_attribute("llm.request.max_tokens", request.max_tokens)
        
        return self
    
    def set_llm_response(self, response: LLMResponse) -> 'Span':
        """
        Set LLM response data on the span.
        
        Args:
            response: LLM response data
        
        Returns:
            Self for method chaining
        """
        if not self.is_recording:
            return self
        
        self.span_data.llm_response = response
        
        # Add LLM-specific attributes
        self.set_attribute("llm.response.model", response.model)
        self.set_attribute("llm.response.id", response.id)
        
        if response.usage:
            self.set_attribute("llm.usage.prompt_tokens", response.usage.prompt_tokens)
            self.set_attribute("llm.usage.completion_tokens", response.usage.completion_tokens)
            self.set_attribute("llm.usage.total_tokens", response.usage.total_tokens)
        
        return self
    
    def end(self, end_time: Optional[datetime] = None) -> None:
        """
        End the span.
        
        Args:
            end_time: End time (defaults to now)
        """
        if self._ended:
            return
        
        self._ended = True
        self.span_data.end_time = end_time or datetime.now(timezone.utc)
        
        # Calculate duration
        if self.span_data.start_time and self.span_data.end_time:
            duration = self.span_data.end_time - self.span_data.start_time
            self.span_data.duration_ms = duration.total_seconds() * 1000
        
        # Export to tracer
        if self._tracer:
            self._tracer.export_span(self.span_data)
    
    def __enter__(self) -> 'Span':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        if exc_type is not None:
            self.record_exception(exc_val)
        self.end()
    
    @classmethod
    def create_no_op_span(cls) -> 'Span':
        """Create a no-op span that doesn't record anything."""
        span_data = SpanData(
            trace_id=TraceID.generate(),
            span_id=SpanID.generate(),
            name="no-op"
        )
        span = cls(span_data)
        span._is_recording = False
        return span

    def get_span_data(self) -> SpanData:
        """Get the span data."""
        return self.span_data


# Export main class
__all__ = ["Span"]

