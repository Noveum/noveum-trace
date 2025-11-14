"""
Span implementation for Noveum Trace SDK.

A span represents a single operation within a trace, such as a function call,
LLM interaction, or agent decision.
"""

import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from noveum_trace.utils.serialization import convert_to_json_string


class SpanStatus(Enum):
    """Enumeration of possible span statuses."""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class SpanEvent:
    """Represents an event within a span."""

    name: str
    timestamp: datetime
    attributes: dict[str, Any] = field(default_factory=dict)


class Span:
    """
    Represents a single operation within a trace.

    A span captures the execution of a discrete operation including
    timing, inputs, outputs, errors, and custom metadata.
    """

    def __init__(
        self,
        name: str,
        trace_id: str,
        span_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        attributes: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize a new span.

        Args:
            name: Human-readable name for the span
            trace_id: ID of the parent trace
            span_id: Unique ID for this span (generated if not provided)
            parent_span_id: ID of the parent span (if any)
            start_time: Start time (defaults to current time)
            attributes: Initial span attributes
        """
        self.name = name
        self.trace_id = trace_id
        self.span_id = span_id or self._generate_span_id()
        self.parent_span_id = parent_span_id
        self.start_time = start_time or datetime.now(timezone.utc)
        self.end_time: Optional[datetime] = None
        self.status = SpanStatus.UNSET
        self.status_message: Optional[str] = None

        # Span data
        # Convert any dicts in initial attributes to JSON strings
        initial_attrs = attributes or {}
        self.attributes: dict[str, Any] = {
            key: convert_to_json_string(value) for key, value in initial_attrs.items()
        }
        self.events: list[SpanEvent] = []
        self.links: list[dict[str, Any]] = []

        # Performance data
        self.duration_ms: Optional[float] = None

        # Error information
        self.exception: Optional[Exception] = None
        self.stack_trace: Optional[str] = None

        # Flags
        self._finished = False

    def __enter__(self) -> "Span":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        if exc_type is not None:
            self.record_exception(exc_val)
            self.set_status(SpanStatus.ERROR, str(exc_val))
        self.finish()

    def set_attribute(self, key: str, value: Any) -> "Span":
        """
        Set a span attribute.

        Args:
            key: Attribute key
            value: Attribute value (dicts are automatically converted to JSON strings)

        Returns:
            Self for method chaining
        """
        if self._finished:
            return self

        # Convert dicts to JSON strings for safe serialization
        self.attributes[key] = convert_to_json_string(value)
        return self

    def set_attributes(self, attributes: dict[str, Any]) -> "Span":
        """
        Set multiple span attributes.

        Args:
            attributes: Dictionary of attributes to set (dicts are automatically converted to JSON strings)

        Returns:
            Self for method chaining
        """
        if self._finished:
            return self

        # Convert dicts to JSON strings for safe serialization
        converted_attributes = {
            key: convert_to_json_string(value) for key, value in attributes.items()
        }
        self.attributes.update(converted_attributes)
        return self

    def add_event(
        self,
        name: str,
        attributes: Optional[dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> "Span":
        """
        Add an event to the span.

        Args:
            name: Event name
            attributes: Event attributes (dicts are automatically converted to JSON strings)
            timestamp: Event timestamp (defaults to current time)

        Returns:
            Self for method chaining
        """
        if self._finished:
            return self

        # Convert dicts in event attributes to JSON strings
        event_attrs = attributes or {}
        converted_event_attrs = {
            key: convert_to_json_string(value) for key, value in event_attrs.items()
        }

        event = SpanEvent(
            name=name,
            timestamp=timestamp or datetime.now(timezone.utc),
            attributes=converted_event_attrs,
        )
        self.events.append(event)
        return self

    def add_link(
        self, trace_id: str, span_id: str, attributes: Optional[dict[str, Any]] = None
    ) -> "Span":
        """
        Add a link to another span.

        Args:
            trace_id: Linked trace ID
            span_id: Linked span ID
            attributes: Link attributes (dicts are automatically converted to JSON strings)

        Returns:
            Self for method chaining
        """
        if self._finished:
            return self

        # Convert dicts in link attributes to JSON strings
        link_attrs = attributes or {}
        converted_link_attrs = {
            key: convert_to_json_string(value) for key, value in link_attrs.items()
        }

        link = {
            "trace_id": trace_id,
            "span_id": span_id,
            "attributes": converted_link_attrs,
        }
        self.links.append(link)
        return self

    def set_status(self, status: SpanStatus, message: Optional[str] = None) -> "Span":
        """
        Set the span status.

        Args:
            status: Span status
            message: Optional status message

        Returns:
            Self for method chaining
        """
        if self._finished:
            return self

        self.status = status
        self.status_message = message
        return self

    def record_exception(
        self, exception: Exception, capture_stack_trace: bool = True
    ) -> "Span":
        """
        Record an exception in the span.

        Args:
            exception: Exception to record
            capture_stack_trace: Whether to capture the stack trace

        Returns:
            Self for method chaining
        """
        if self._finished:
            return self

        self.exception = exception

        # Add exception attributes
        self.set_attributes(
            {
                "exception.type": type(exception).__name__,
                "exception.message": str(exception),
                "exception.escaped": False,
            }
        )

        # Capture stack trace if requested
        if capture_stack_trace:
            self.stack_trace = traceback.format_exc()
            self.set_attribute("exception.stacktrace", self.stack_trace)

        # Add exception event
        self.add_event(
            "exception",
            {
                "exception.type": type(exception).__name__,
                "exception.message": str(exception),
            },
        )

        return self

    def finish(self, end_time: Optional[datetime] = None) -> None:
        """
        Finish the span.

        Args:
            end_time: End time (defaults to current time)
        """
        if self._finished:
            return

        self.end_time = end_time or datetime.now(timezone.utc)
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000

        # Set default status if not set
        if self.status == SpanStatus.UNSET:
            self.status = SpanStatus.OK

        self._finished = True

    def is_finished(self) -> bool:
        """Check if the span is finished."""
        return self._finished

    def to_dict(self) -> dict[str, Any]:
        """
        Convert span to dictionary representation.

        Returns:
            Dictionary representation of the span
        """
        data = {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "status_message": self.status_message,
            "attributes": self.attributes,
            "events": [
                {
                    "name": event.name,
                    "timestamp": event.timestamp.isoformat(),
                    "attributes": event.attributes,
                }
                for event in self.events
            ],
            "links": self.links,
        }

        # Add exception information if present
        if self.exception:
            data["exception"] = {
                "type": type(self.exception).__name__,
                "message": str(self.exception),
                "stack_trace": self.stack_trace,
            }

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Span":
        """
        Create span from dictionary representation.

        Args:
            data: Dictionary representation

        Returns:
            Span instance
        """
        span = cls(
            name=data["name"],
            trace_id=data["trace_id"],
            span_id=data["span_id"],
            parent_span_id=data.get("parent_span_id"),
            start_time=datetime.fromisoformat(data["start_time"]),
            attributes=data.get("attributes", {}),
        )

        if data.get("end_time"):
            span.end_time = datetime.fromisoformat(data["end_time"])
            span._finished = True

        span.duration_ms = data.get("duration_ms")
        span.status = SpanStatus(data.get("status", "unset"))
        span.status_message = data.get("status_message")

        # Restore events
        for event_data in data.get("events", []):
            event = SpanEvent(
                name=event_data["name"],
                timestamp=datetime.fromisoformat(event_data["timestamp"]),
                attributes=event_data.get("attributes", {}),
            )
            span.events.append(event)

        # Restore links
        span.links = data.get("links", [])

        return span

    def _generate_span_id(self) -> str:
        """Generate a unique span ID."""
        return str(uuid.uuid4())

    def __repr__(self) -> str:
        """String representation of the span."""
        return (
            f"Span(name='{self.name}', span_id='{self.span_id}', "
            f"trace_id='{self.trace_id}', status='{self.status.value}')"
        )
