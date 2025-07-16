"""
Context management for trace propagation and span relationships.
"""

import threading
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .span import Span

# Context variables for async support
_current_span: ContextVar[Optional["Span"]] = ContextVar("current_span", default=None)
_current_trace: ContextVar[Optional["TraceContext"]] = ContextVar(
    "current_trace", default=None
)

# Thread-local storage for sync support
_thread_local = threading.local()


@dataclass
class TraceContext:
    """Represents a trace context with metadata and span hierarchy."""

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    baggage: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    spans: Dict[str, "Span"] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize context after creation."""
        if not hasattr(_thread_local, "contexts"):
            _thread_local.contexts = {}

    def add_span(self, span: "Span") -> None:
        """Add a span to this trace context."""
        self.spans[str(span.span_id)] = span

    def get_span(self, span_id: str) -> Optional["Span"]:
        """Get a span by ID."""
        return self.spans.get(span_id)

    def set_baggage(self, key: str, value: Any) -> None:
        """Set a baggage item that will be propagated to child spans."""
        self.baggage[key] = value

    def get_baggage(self, key: str, default: Any = None) -> Any:
        """Get a baggage item."""
        return self.baggage.get(key, default)

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata for this trace context."""
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata from this trace context."""
        return self.metadata.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary representation."""
        return {
            "trace_id": self.trace_id,
            "baggage": self.baggage,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceContext":
        """Create context from dictionary representation."""
        return cls(
            trace_id=data.get("trace_id", str(uuid.uuid4())),
            baggage=data.get("baggage", {}),
            metadata=data.get("metadata", {}),
        )


class ContextManager:
    """Manages trace context and span relationships."""

    def __init__(self) -> None:
        """Initialize the context manager."""
        self._span_stack: List[Span] = []

    def get_current_span(self) -> Optional["Span"]:
        """Get the currently active span."""
        # Try async context first
        try:
            span = _current_span.get()
            if span is not None:
                return span
        except LookupError:
            pass

        # Fall back to thread-local storage
        if hasattr(_thread_local, "current_span"):
            return _thread_local.current_span

        return None

    def set_current_span(self, span: Optional["Span"]) -> None:
        """Set the currently active span."""
        # Set in async context
        _current_span.set(span)

        # Also set in thread-local for sync compatibility
        _thread_local.current_span = span

    def get_current_trace(self) -> Optional[TraceContext]:
        """Get the current trace context."""
        # Try async context first
        try:
            trace = _current_trace.get()
            if trace is not None:
                return trace
        except LookupError:
            pass

        # Fall back to thread-local storage
        if hasattr(_thread_local, "current_trace"):
            return _thread_local.current_trace

        return None

    def set_current_trace(self, trace: Optional[TraceContext]) -> None:
        """Set the current trace context."""
        # Set in async context
        _current_trace.set(trace)

        # Also set in thread-local for sync compatibility
        _thread_local.current_trace = trace

    def start_trace(self, trace_id: Optional[str] = None) -> TraceContext:
        """Start a new trace context."""
        if trace_id is None:
            trace_id = str(uuid.uuid4())

        trace = TraceContext(trace_id=trace_id)
        self.set_current_trace(trace)
        return trace

    def end_trace(self) -> None:
        """End the current trace context."""
        self.set_current_trace(None)
        self.set_current_span(None)

    def push_span(self, span: "Span") -> None:
        """Push a span onto the context stack."""
        self._span_stack.append(span)
        self.set_current_span(span)

    def pop_span(self) -> Optional["Span"]:
        """Pop the current span from the context stack."""
        if not self._span_stack:
            return None

        span = self._span_stack.pop()

        # Set the previous span as current (if any)
        if self._span_stack:
            self.set_current_span(self._span_stack[-1])
        else:
            self.set_current_span(None)

        return span

    def get_span_stack(self) -> list:
        """Get a copy of the current span stack."""
        return self._span_stack.copy()


# Global context manager instance
_context_manager = ContextManager()


def get_current_span() -> Optional["Span"]:
    """Get the currently active span."""
    return _context_manager.get_current_span()


def set_current_span(span: Optional["Span"]) -> None:
    """Set the currently active span."""
    _context_manager.set_current_span(span)


def get_current_trace() -> Optional[TraceContext]:
    """Get the current trace context."""
    return _context_manager.get_current_trace()


def set_current_trace(trace: Optional[TraceContext]) -> None:
    """Set the current trace context."""
    _context_manager.set_current_trace(trace)


def start_trace(trace_id: Optional[str] = None) -> TraceContext:
    """Start a new trace context."""
    return _context_manager.start_trace(trace_id)


def end_trace() -> None:
    """End the current trace context."""
    _context_manager.end_trace()


class SpanContext:
    """Context manager for automatic span lifecycle management."""

    def __init__(self, span: "Span"):
        """Initialize span context."""
        self.span = span
        self.previous_span: Optional[Span] = None

    def __enter__(self) -> "Span":
        """Enter the span context."""
        self.previous_span = get_current_span()
        _context_manager.push_span(self.span)
        return self.span

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the span context."""
        _context_manager.pop_span()

        # Mark span as error if exception occurred
        if exc_type is not None:
            self.span.set_status("error")
            self.span.set_attribute("error.type", exc_type.__name__)
            self.span.set_attribute("error.message", str(exc_val))

        # End the span
        self.span.end()


class AsyncSpanContext:
    """Async context manager for automatic span lifecycle management."""

    def __init__(self, span: "Span"):
        """Initialize async span context."""
        self.span = span
        self.previous_span: Optional[Span] = None

    async def __aenter__(self) -> "Span":
        """Enter the async span context."""
        self.previous_span = get_current_span()
        _context_manager.push_span(self.span)
        return self.span

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the async span context."""
        _context_manager.pop_span()

        # Mark span as error if exception occurred
        if exc_type is not None:
            self.span.set_status("error")
            self.span.set_attribute("error.type", exc_type.__name__)
            self.span.set_attribute("error.message", str(exc_val))

        # End the span
        self.span.end()
