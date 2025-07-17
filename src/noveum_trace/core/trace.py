"""
Trace implementation for Noveum Trace SDK.

A trace represents a complete user interaction or system operation,
containing multiple related spans that form a hierarchical execution tree.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from noveum_trace.core.span import Span, SpanStatus
from noveum_trace.utils.exceptions import NoveumTraceError


@dataclass
class TraceMetadata:
    """Metadata associated with a trace."""

    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    tags: dict[str, str] = field(default_factory=dict)
    custom_attributes: dict[str, Any] = field(default_factory=dict)


class Trace:
    """
    Represents a complete execution trace.

    A trace contains multiple spans that represent individual operations
    within a larger workflow or user interaction.
    """

    def __init__(
        self,
        name: str,
        trace_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        metadata: Optional[TraceMetadata] = None,
        attributes: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize a new trace.

        Args:
            name: Human-readable name for the trace
            trace_id: Unique ID for this trace (generated if not provided)
            start_time: Start time (defaults to current time)
            metadata: Trace metadata
            attributes: Initial trace attributes
        """
        self.name = name
        self.trace_id = trace_id or self._generate_trace_id()
        self.start_time = start_time or datetime.now(timezone.utc)
        self.end_time: Optional[datetime] = None
        self.metadata = metadata or TraceMetadata()
        self.attributes: dict[str, Any] = attributes or {}

        # Span management
        self.spans: list[Span] = []
        self.root_span: Optional[Span] = None
        self.active_spans: dict[str, Span] = {}

        # Trace status
        self.status = SpanStatus.UNSET
        self.status_message: Optional[str] = None

        # Performance metrics
        self.duration_ms: Optional[float] = None
        self.span_count = 0
        self.error_count = 0

        # Flags
        self._finished = False

    def __enter__(self) -> "Trace":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        if exc_type is not None:
            self.set_status(SpanStatus.ERROR, str(exc_val))
        self.finish()

    def create_span(
        self,
        name: str,
        parent_span_id: Optional[str] = None,
        attributes: Optional[dict[str, Any]] = None,
        start_time: Optional[datetime] = None,
    ) -> Span:
        """
        Create a new span within this trace.

        Args:
            name: Span name
            parent_span_id: Parent span ID (if any)
            attributes: Initial span attributes
            start_time: Start time (defaults to current time)

        Returns:
            New Span instance
        """
        if self._finished:
            raise NoveumTraceError("Cannot create span in finished trace")

        span = Span(
            name=name,
            trace_id=self.trace_id,
            parent_span_id=parent_span_id,
            attributes=attributes,
            start_time=start_time,
        )

        self.spans.append(span)
        self.active_spans[span.span_id] = span
        self.span_count += 1

        # Set as root span if it's the first span
        if self.root_span is None and parent_span_id is None:
            self.root_span = span

        return span

    def get_span(self, span_id: str) -> Optional[Span]:
        """
        Get a span by ID.

        Args:
            span_id: Span ID to look up

        Returns:
            Span instance or None if not found
        """
        return self.active_spans.get(span_id)

    def finish_span(self, span_id: str, end_time: Optional[datetime] = None) -> None:
        """
        Finish a span.

        Args:
            span_id: ID of span to finish
            end_time: End time (defaults to current time)
        """
        span = self.active_spans.get(span_id)
        if span and not span.is_finished():
            span.finish(end_time)

            # Update trace statistics
            if span.status == SpanStatus.ERROR:
                self.error_count += 1

            # Remove from active spans
            del self.active_spans[span_id]

    def get_active_spans(self) -> list[Span]:
        """
        Get all currently active (unfinished) spans.

        Returns:
            List of active spans
        """
        return list(self.active_spans.values())

    def get_child_spans(self, parent_span_id: str) -> list[Span]:
        """
        Get all child spans of a given parent span.

        Args:
            parent_span_id: Parent span ID

        Returns:
            List of child spans
        """
        return [span for span in self.spans if span.parent_span_id == parent_span_id]

    def set_attribute(self, key: str, value: Any) -> "Trace":
        """
        Set a trace attribute.

        Args:
            key: Attribute key
            value: Attribute value

        Returns:
            Self for method chaining
        """
        if not self._finished:
            self.attributes[key] = value
        return self

    def set_attributes(self, attributes: dict[str, Any]) -> "Trace":
        """
        Set multiple trace attributes.

        Args:
            attributes: Dictionary of attributes to set

        Returns:
            Self for method chaining
        """
        if not self._finished:
            self.attributes.update(attributes)
        return self

    def set_metadata(self, **kwargs: Any) -> "Trace":
        """
        Set trace metadata.

        Args:
            **kwargs: Metadata fields to set

        Returns:
            Self for method chaining
        """
        if not self._finished:
            for key, value in kwargs.items():
                if hasattr(self.metadata, key):
                    setattr(self.metadata, key, value)
                else:
                    self.metadata.custom_attributes[key] = value
        return self

    def add_tag(self, key: str, value: str) -> "Trace":
        """
        Add a tag to the trace.

        Args:
            key: Tag key
            value: Tag value

        Returns:
            Self for method chaining
        """
        if not self._finished:
            self.metadata.tags[key] = value
        return self

    def set_status(self, status: SpanStatus, message: Optional[str] = None) -> "Trace":
        """
        Set the trace status.

        Args:
            status: Trace status
            message: Optional status message

        Returns:
            Self for method chaining
        """
        if not self._finished:
            self.status = status
            self.status_message = message
        return self

    def finish(self, end_time: Optional[datetime] = None) -> None:
        """
        Finish the trace.

        Args:
            end_time: End time (defaults to current time)
        """
        if self._finished:
            return

        # Finish any remaining active spans
        for span in list(self.active_spans.values()):
            span.finish(end_time)
        self.active_spans.clear()

        self.end_time = end_time or datetime.now(timezone.utc)
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000

        # Set default status if not set
        if self.status == SpanStatus.UNSET:
            if self.error_count > 0:
                self.status = SpanStatus.ERROR
            else:
                self.status = SpanStatus.OK

        self._finished = True

    def is_finished(self) -> bool:
        """Check if the trace is finished."""
        return self._finished

    def get_summary(self) -> dict[str, Any]:
        """
        Get a summary of the trace.

        Returns:
            Dictionary containing trace summary information
        """
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "span_count": self.span_count,
            "error_count": self.error_count,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }

    def to_dict(self) -> dict[str, Any]:
        """
        Convert trace to dictionary representation.

        Returns:
            Dictionary representation of the trace
        """
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "status_message": self.status_message,
            "span_count": self.span_count,
            "error_count": self.error_count,
            "attributes": self.attributes,
            "metadata": {
                "user_id": self.metadata.user_id,
                "session_id": self.metadata.session_id,
                "request_id": self.metadata.request_id,
                "tags": self.metadata.tags,
                "custom_attributes": self.metadata.custom_attributes,
            },
            "spans": [span.to_dict() for span in self.spans],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Trace":
        """
        Create trace from dictionary representation.

        Args:
            data: Dictionary representation

        Returns:
            Trace instance
        """
        metadata_data = data.get("metadata", {})
        metadata = TraceMetadata(
            user_id=metadata_data.get("user_id"),
            session_id=metadata_data.get("session_id"),
            request_id=metadata_data.get("request_id"),
            tags=metadata_data.get("tags", {}),
            custom_attributes=metadata_data.get("custom_attributes", {}),
        )

        trace = cls(
            name=data["name"],
            trace_id=data["trace_id"],
            start_time=datetime.fromisoformat(data["start_time"]),
            metadata=metadata,
            attributes=data.get("attributes", {}),
        )

        if data.get("end_time"):
            trace.end_time = datetime.fromisoformat(data["end_time"])
            trace._finished = True

        trace.duration_ms = data.get("duration_ms")
        trace.status = SpanStatus(data.get("status", "unset"))
        trace.status_message = data.get("status_message")
        trace.span_count = data.get("span_count", 0)
        trace.error_count = data.get("error_count", 0)

        # Restore spans
        for span_data in data.get("spans", []):
            span = Span.from_dict(span_data)
            trace.spans.append(span)

            # Set root span
            if span.parent_span_id is None and trace.root_span is None:
                trace.root_span = span

            # Add to active spans if not finished
            if not span.is_finished():
                trace.active_spans[span.span_id] = span

        return trace

    def _generate_trace_id(self) -> str:
        """Generate a unique trace ID."""
        return str(uuid.uuid4())

    def __repr__(self) -> str:
        """String representation of the trace."""
        return (
            f"Trace(name='{self.name}', trace_id='{self.trace_id}', "
            f"span_count={self.span_count}, status='{self.status.value}')"
        )
