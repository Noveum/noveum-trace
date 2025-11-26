"""
Comprehensive unit tests for Span class.

This module tests all aspects of the Span class including creation, attribute management,
events, links, error handling, serialization, and complex scenarios.
"""

from datetime import datetime, timezone

import pytest

from noveum_trace.core.span import Span, SpanEvent, SpanStatus


class TestSpanEvent:
    """Test SpanEvent dataclass functionality."""

    def test_span_event_creation_basic(self):
        """Test creating basic span event."""
        timestamp = datetime.now(timezone.utc)
        event = SpanEvent("test_event", timestamp)

        assert event.name == "test_event"
        assert event.timestamp == timestamp
        assert event.attributes == {}

    def test_span_event_creation_with_attributes(self):
        """Test creating span event with attributes."""
        timestamp = datetime.now(timezone.utc)
        attributes = {"key1": "value1", "key2": 42}
        event = SpanEvent("test_event", timestamp, attributes)

        assert event.name == "test_event"
        assert event.timestamp == timestamp
        assert event.attributes == attributes

    @pytest.mark.parametrize(
        "event_name,attributes",
        [
            ("simple_event", {"type": "info"}),
            ("complex_event", {"data": {"nested": "value"}, "count": 100}),
            ("error_event", {"error": True, "message": "Something failed"}),
        ],
    )
    def test_span_event_different_types(self, event_name, attributes):
        """Test span events with different types and attributes."""
        timestamp = datetime.now(timezone.utc)
        event = SpanEvent(event_name, timestamp, attributes)

        assert event.name == event_name
        assert event.attributes == attributes


class TestSpanStatus:
    """Test SpanStatus enum functionality."""

    def test_span_status_values(self):
        """Test all span status enum values."""
        assert SpanStatus.UNSET.value == "unset"
        assert SpanStatus.OK.value == "ok"
        assert SpanStatus.ERROR.value == "error"
        assert SpanStatus.TIMEOUT.value == "timeout"
        assert SpanStatus.CANCELLED.value == "cancelled"

    @pytest.mark.parametrize(
        "status",
        [
            SpanStatus.UNSET,
            SpanStatus.OK,
            SpanStatus.ERROR,
            SpanStatus.TIMEOUT,
            SpanStatus.CANCELLED,
        ],
    )
    def test_span_status_enum_roundtrip(self, status):
        """Test span status enum serialization roundtrip."""
        value = status.value
        reconstructed = SpanStatus(value)
        assert reconstructed == status


class TestSpanInitialization:
    """Test Span class initialization and basic properties."""

    def test_span_creation_minimal(self):
        """Test creating span with minimal parameters."""
        span = Span("test_span", "trace-123")

        assert span.name == "test_span"
        assert span.trace_id == "trace-123"
        assert span.span_id is not None
        assert len(span.span_id) > 0
        assert span.parent_span_id is None
        assert isinstance(span.start_time, datetime)
        assert span.end_time is None
        assert span.status == SpanStatus.UNSET
        assert span.status_message is None
        assert span.attributes == {}
        assert span.events == []
        assert span.links == []
        assert span.duration_ms is None
        assert span.exception is None
        assert span.stack_trace is None
        assert span._finished is False

    @pytest.mark.parametrize(
        "span_name,trace_id,span_id",
        [
            ("simple_span", "trace-123", "span-456"),
            ("complex-span-name", "trace_with_underscores", "span_with_underscores"),
            ("span with spaces", "trace-with-dashes", "span-with-dashes"),
        ],
    )
    def test_span_creation_with_custom_ids(self, span_name, trace_id, span_id):
        """Test creating span with custom IDs."""
        span = Span(span_name, trace_id, span_id=span_id)

        assert span.name == span_name
        assert span.trace_id == trace_id
        assert span.span_id == span_id

    def test_span_creation_with_parent(self):
        """Test creating span with parent span ID."""
        span = Span("child_span", "trace-123", parent_span_id="parent-456")

        assert span.parent_span_id == "parent-456"

    def test_span_creation_with_custom_time(self):
        """Test creating span with custom start time."""
        custom_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        span = Span("test_span", "trace-123", start_time=custom_time)

        assert span.start_time == custom_time

    def test_span_creation_with_attributes(self):
        """Test creating span with initial attributes."""
        attributes = {"key1": "value1", "key2": 42, "key3": True}
        span = Span("test_span", "trace-123", attributes=attributes)

        assert span.attributes == attributes

    def test_span_id_generation_uniqueness(self):
        """Test that generated span IDs are unique."""
        span_ids = set()

        for _ in range(100):
            span = Span("test_span", "trace-123")
            assert span.span_id not in span_ids
            span_ids.add(span.span_id)

        assert len(span_ids) == 100


class TestSpanAttributeManagement:
    """Test span attribute management functionality."""

    def test_set_attribute_basic(self):
        """Test setting individual attributes."""
        span = Span("test_span", "trace-123")

        result = span.set_attribute("key1", "value1")

        assert result == span  # Should return self for chaining
        assert span.attributes["key1"] == "value1"

    @pytest.mark.parametrize(
        "key,value",
        [
            ("string_key", "string_value"),
            ("int_key", 42),
            ("float_key", 3.14),
            ("bool_key", True),
            ("list_key", [1, 2, 3]),
            ("dict_key", {"nested": "value"}),
            ("none_key", None),
        ],
    )
    def test_set_attribute_different_types(self, key, value):
        """Test setting attributes with different data types."""

        span = Span("test_span", "trace-123")

        span.set_attribute(key, value)

        # Dicts and lists are preserved as native types (not converted to JSON strings)
        if isinstance(value, (dict, list)):
            assert isinstance(span.attributes[key], type(value))
            assert span.attributes[key] == value
        else:
            # Other types remain unchanged
            assert span.attributes[key] == value

    def test_set_attributes_multiple(self):
        """Test setting multiple attributes."""
        span = Span("test_span", "trace-123")

        attributes = {"key1": "value1", "key2": 42, "key3": True}
        result = span.set_attributes(attributes)

        assert result == span  # Should return self for chaining
        for key, value in attributes.items():
            assert span.attributes[key] == value

    def test_set_attributes_update_existing(self):
        """Test that set_attributes updates existing attributes."""
        span = Span("test_span", "trace-123")

        # Set initial attributes
        span.set_attribute("key1", "old_value")
        span.set_attribute("key2", "keep_value")

        # Update with new attributes
        new_attributes = {"key1": "new_value", "key3": "added_value"}
        span.set_attributes(new_attributes)

        assert span.attributes["key1"] == "new_value"  # Updated
        assert span.attributes["key2"] == "keep_value"  # Kept
        assert span.attributes["key3"] == "added_value"  # Added

    def test_attribute_operations_on_finished_span(self):
        """Test that attribute operations are ignored on finished spans."""
        span = Span("test_span", "trace-123")
        span.finish()

        span.set_attribute("key1", "value1")
        span.set_attributes({"key2": "value2"})

        assert "key1" not in span.attributes
        assert "key2" not in span.attributes


class TestSpanEventManagement:
    """Test span event management functionality."""

    def test_add_event_basic(self):
        """Test adding basic event."""
        span = Span("test_span", "trace-123")

        result = span.add_event("test_event")

        assert result == span  # Should return self for chaining
        assert len(span.events) == 1
        event = span.events[0]
        assert event.name == "test_event"
        assert isinstance(event.timestamp, datetime)
        assert event.attributes == {}

    def test_add_event_with_attributes(self):
        """Test adding event with attributes."""
        span = Span("test_span", "trace-123")

        attributes = {"key1": "value1", "key2": 42}
        span.add_event("test_event", attributes)

        event = span.events[0]
        assert event.name == "test_event"
        assert event.attributes == attributes

    def test_add_event_with_custom_timestamp(self):
        """Test adding event with custom timestamp."""
        span = Span("test_span", "trace-123")

        custom_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        span.add_event("test_event", timestamp=custom_time)

        event = span.events[0]
        assert event.timestamp == custom_time

    @pytest.mark.parametrize(
        "event_name,attributes",
        [
            ("user_action", {"action": "click", "element": "button"}),
            ("llm_call", {"model": "gpt-4", "tokens": 100}),
            ("database_query", {"query": "SELECT * FROM users", "duration_ms": 50}),
            (
                "error_occurred",
                {"error_type": "ValueError", "message": "Invalid input"},
            ),
        ],
    )
    def test_add_different_event_types(self, event_name, attributes):
        """Test adding different types of events."""
        span = Span("test_span", "trace-123")

        span.add_event(event_name, attributes)

        event = span.events[0]
        assert event.name == event_name
        assert event.attributes == attributes

    def test_add_multiple_events(self):
        """Test adding multiple events maintains order."""
        span = Span("test_span", "trace-123")

        span.add_event("event1", {"order": 1})
        span.add_event("event2", {"order": 2})
        span.add_event("event3", {"order": 3})

        assert len(span.events) == 3
        assert span.events[0].name == "event1"
        assert span.events[1].name == "event2"
        assert span.events[2].name == "event3"
        assert span.events[0].attributes["order"] == 1
        assert span.events[1].attributes["order"] == 2
        assert span.events[2].attributes["order"] == 3

    def test_add_event_on_finished_span(self):
        """Test that adding events to finished span is ignored."""
        span = Span("test_span", "trace-123")
        span.finish()

        span.add_event("test_event")

        assert len(span.events) == 0


class TestSpanLinkManagement:
    """Test span link management functionality."""

    def test_add_link_basic(self):
        """Test adding basic link."""
        span = Span("test_span", "trace-123")

        result = span.add_link("other-trace-id", "other-span-id")

        assert result == span  # Should return self for chaining
        assert len(span.links) == 1
        link = span.links[0]
        assert link["trace_id"] == "other-trace-id"
        assert link["span_id"] == "other-span-id"
        assert link["attributes"] == {}

    def test_add_link_with_attributes(self):
        """Test adding link with attributes."""
        span = Span("test_span", "trace-123")

        attributes = {"relationship": "follows", "correlation_id": "abc123"}
        span.add_link("other-trace-id", "other-span-id", attributes)

        link = span.links[0]
        assert link["trace_id"] == "other-trace-id"
        assert link["span_id"] == "other-span-id"
        assert link["attributes"] == attributes

    @pytest.mark.parametrize(
        "trace_id,span_id,relationship",
        [
            ("trace-1", "span-1", "parent"),
            ("trace-2", "span-2", "child"),
            ("trace-3", "span-3", "follows"),
            ("trace-4", "span-4", "causes"),
        ],
    )
    def test_add_different_link_types(self, trace_id, span_id, relationship):
        """Test adding different types of links."""
        span = Span("test_span", "trace-123")

        span.add_link(trace_id, span_id, {"relationship": relationship})

        link = span.links[0]
        assert link["trace_id"] == trace_id
        assert link["span_id"] == span_id
        assert link["attributes"]["relationship"] == relationship

    def test_add_multiple_links(self):
        """Test adding multiple links."""
        span = Span("test_span", "trace-123")

        span.add_link("trace-1", "span-1", {"order": 1})
        span.add_link("trace-2", "span-2", {"order": 2})
        span.add_link("trace-3", "span-3", {"order": 3})

        assert len(span.links) == 3
        assert span.links[0]["trace_id"] == "trace-1"
        assert span.links[1]["trace_id"] == "trace-2"
        assert span.links[2]["trace_id"] == "trace-3"

    def test_add_link_on_finished_span(self):
        """Test that adding links to finished span is ignored."""
        span = Span("test_span", "trace-123")
        span.finish()

        span.add_link("other-trace-id", "other-span-id")

        assert len(span.links) == 0


class TestSpanStatusAndErrorHandling:
    """Test span status management and error handling."""

    def test_set_status_basic(self):
        """Test setting span status."""
        span = Span("test_span", "trace-123")

        result = span.set_status(SpanStatus.OK, "All good")

        assert result == span  # Should return self for chaining
        assert span.status == SpanStatus.OK
        assert span.status_message == "All good"

    @pytest.mark.parametrize(
        "status,message",
        [
            (SpanStatus.OK, None),
            (SpanStatus.ERROR, "Something went wrong"),
            (SpanStatus.TIMEOUT, "Operation timed out"),
            (SpanStatus.CANCELLED, "Operation was cancelled"),
        ],
    )
    def test_set_different_statuses(self, status, message):
        """Test setting different status values."""
        span = Span("test_span", "trace-123")

        span.set_status(status, message)

        assert span.status == status
        assert span.status_message == message

    def test_status_operations_on_finished_span(self):
        """Test that status operations are ignored on finished spans."""
        span = Span("test_span", "trace-123")
        span.finish()

        original_status = span.status
        span.set_status(SpanStatus.ERROR, "Error after finish")

        assert span.status == original_status
        assert span.status_message != "Error after finish"

    def test_record_exception_basic(self):
        """Test recording basic exception."""
        span = Span("test_span", "trace-123")

        test_exception = ValueError("Test error message")
        result = span.record_exception(test_exception)

        assert result == span  # Should return self for chaining
        assert span.exception == test_exception

        # Should add exception attributes
        assert span.attributes["exception.type"] == "ValueError"
        assert span.attributes["exception.message"] == "Test error message"
        assert span.attributes["exception.escaped"] is False

        # Should add exception event
        assert len(span.events) == 1
        event = span.events[0]
        assert event.name == "exception"
        assert event.attributes["exception.type"] == "ValueError"
        assert event.attributes["exception.message"] == "Test error message"

    def test_record_exception_with_stack_trace(self):
        """Test recording exception with stack trace."""
        span = Span("test_span", "trace-123")

        try:
            raise ValueError("Test error")
        except ValueError as e:
            span.record_exception(e, capture_stack_trace=True)

        assert span.exception is not None
        assert span.stack_trace is not None
        assert "ValueError: Test error" in span.stack_trace
        assert "exception.stacktrace" in span.attributes
        assert span.attributes["exception.stacktrace"] == span.stack_trace

    def test_record_exception_without_stack_trace(self):
        """Test recording exception without stack trace."""
        span = Span("test_span", "trace-123")

        test_exception = ValueError("Test error")
        span.record_exception(test_exception, capture_stack_trace=False)

        assert span.exception == test_exception
        assert span.stack_trace is None
        assert "exception.stacktrace" not in span.attributes

    @pytest.mark.parametrize(
        "exception_type,message,expected_message",
        [
            (ValueError, "Invalid value", "Invalid value"),
            (TypeError, "Wrong type", "Wrong type"),
            (RuntimeError, "Runtime issue", "Runtime issue"),
            (
                KeyError,
                "Missing key",
                "'Missing key'",
            ),  # KeyError wraps message in quotes
        ],
    )
    def test_record_different_exception_types(
        self, exception_type, message, expected_message
    ):
        """Test recording different exception types."""
        span = Span("test_span", "trace-123")

        test_exception = exception_type(message)
        span.record_exception(test_exception)

        assert span.attributes["exception.type"] == exception_type.__name__
        assert span.attributes["exception.message"] == expected_message

    def test_record_exception_on_finished_span(self):
        """Test that recording exception on finished span is ignored."""
        span = Span("test_span", "trace-123")
        span.finish()

        test_exception = ValueError("Test error")
        span.record_exception(test_exception)

        assert span.exception is None
        assert len(span.events) == 0


class TestSpanLifecycle:
    """Test span lifecycle operations."""

    def test_finish_basic(self):
        """Test basic span finishing."""
        span = Span("test_span", "trace-123")

        assert not span.is_finished()

        span.finish()

        assert span.is_finished()
        assert span.end_time is not None
        assert span.duration_ms is not None
        assert span.duration_ms >= 0
        assert span._finished is True

    def test_finish_with_custom_time(self):
        """Test finishing span with custom end time."""
        start_time = datetime(2023, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        end_time = datetime(2023, 1, 1, 10, 5, 30, tzinfo=timezone.utc)

        span = Span("test_span", "trace-123", start_time=start_time)
        span.finish(end_time)

        assert span.end_time == end_time
        expected_duration = (end_time - start_time).total_seconds() * 1000
        assert span.duration_ms == expected_duration

    def test_finish_sets_default_status(self):
        """Test that finish sets default status if unset."""
        span = Span("test_span", "trace-123")

        assert span.status == SpanStatus.UNSET

        span.finish()

        assert span.status == SpanStatus.OK

    def test_finish_preserves_existing_status(self):
        """Test that finish preserves existing status."""
        span = Span("test_span", "trace-123")
        span.set_status(SpanStatus.ERROR, "Custom error")

        span.finish()

        assert span.status == SpanStatus.ERROR
        assert span.status_message == "Custom error"

    def test_finish_idempotent(self):
        """Test that finishing a span multiple times is safe."""
        span = Span("test_span", "trace-123")

        span.finish()
        original_end_time = span.end_time
        original_duration = span.duration_ms

        # Finish again - should be no-op
        span.finish()

        assert span.end_time == original_end_time
        assert span.duration_ms == original_duration


class TestSpanContextManager:
    """Test span context manager functionality."""

    def test_context_manager_success(self):
        """Test span context manager for successful execution."""
        with Span("test_span", "trace-123") as span:
            assert isinstance(span, Span)
            assert span.name == "test_span"
            assert not span.is_finished()

        # Should be finished after context exit
        assert span.is_finished()
        assert span.status == SpanStatus.OK

    def test_context_manager_with_exception(self):
        """Test span context manager with exception."""
        test_exception = ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            with Span("test_span", "trace-123") as span:
                raise test_exception

        # Should be finished with error status and recorded exception
        assert span.is_finished()
        assert span.status == SpanStatus.ERROR
        assert span.status_message == "Test error"
        assert span.exception == test_exception
        assert len(span.events) == 1
        assert span.events[0].name == "exception"

    def test_context_manager_with_operations(self):
        """Test span context manager with operations."""
        with Span("test_span", "trace-123") as span:
            span.set_attribute("key", "value")
            span.add_event("operation_start")
            span.add_link("other-trace", "other-span")

        # Operations should persist after context exit
        assert span.attributes["key"] == "value"
        assert len(span.events) == 1
        assert len(span.links) == 1
        assert span.is_finished()


class TestSpanSerialization:
    """Test span serialization and deserialization functionality."""

    def test_to_dict_basic(self):
        """Test basic span serialization to dictionary."""
        span = Span("test_span", "trace-123", span_id="span-456")
        span.set_attribute("key", "value")
        span.finish()

        span_dict = span.to_dict()

        assert span_dict["span_id"] == "span-456"
        assert span_dict["trace_id"] == "trace-123"
        assert span_dict["name"] == "test_span"
        assert span_dict["attributes"]["key"] == "value"
        assert span_dict["status"] == "ok"
        assert "start_time" in span_dict
        assert "end_time" in span_dict
        assert "duration_ms" in span_dict

    def test_to_dict_with_parent(self):
        """Test serialization with parent span ID."""
        span = Span("child_span", "trace-123", parent_span_id="parent-456")
        span.finish()

        span_dict = span.to_dict()

        assert span_dict["parent_span_id"] == "parent-456"

    def test_to_dict_with_events_and_links(self):
        """Test serialization with events and links."""
        span = Span("test_span", "trace-123")

        span.add_event("test_event", {"key": "value"})
        span.add_link("other-trace", "other-span", {"relationship": "follows"})
        span.finish()

        span_dict = span.to_dict()

        assert len(span_dict["events"]) == 1
        event = span_dict["events"][0]
        assert event["name"] == "test_event"
        assert event["attributes"]["key"] == "value"
        assert "timestamp" in event

        assert len(span_dict["links"]) == 1
        link = span_dict["links"][0]
        assert link["trace_id"] == "other-trace"
        assert link["span_id"] == "other-span"
        assert link["attributes"]["relationship"] == "follows"

    def test_to_dict_with_exception(self):
        """Test serialization with exception information."""
        span = Span("test_span", "trace-123")

        test_exception = ValueError("Test error")
        span.record_exception(test_exception)
        span.finish()

        span_dict = span.to_dict()

        assert "exception" in span_dict
        exception_data = span_dict["exception"]
        assert exception_data["type"] == "ValueError"
        assert exception_data["message"] == "Test error"
        assert exception_data["stack_trace"] is not None

    def test_from_dict_reconstruction(self):
        """Test reconstructing span from dictionary."""
        # Create original span
        original_span = Span(
            "test_span", "trace-123", span_id="span-456", parent_span_id="parent-789"
        )
        original_span.set_attribute("key", "value")
        original_span.add_event("test_event", {"event_key": "event_value"})
        original_span.add_link("other-trace", "other-span")
        original_span.finish()

        # Serialize and reconstruct
        span_dict = original_span.to_dict()
        reconstructed_span = Span.from_dict(span_dict)

        # Verify reconstruction
        assert reconstructed_span.name == original_span.name
        assert reconstructed_span.trace_id == original_span.trace_id
        assert reconstructed_span.span_id == original_span.span_id
        assert reconstructed_span.parent_span_id == original_span.parent_span_id
        assert reconstructed_span.status == original_span.status
        assert reconstructed_span.attributes == original_span.attributes
        assert len(reconstructed_span.events) == len(original_span.events)
        assert len(reconstructed_span.links) == len(original_span.links)
        assert reconstructed_span.is_finished() == original_span.is_finished()

    @pytest.mark.parametrize("include_finished", [True, False])
    def test_serialization_roundtrip(self, include_finished):
        """Test complete serialization roundtrip with different states."""
        span = Span("test_span", "trace-123")
        span.set_attribute("test_key", "test_value")
        span.add_event("test_event")

        if include_finished:
            span.finish()

        # Serialize and reconstruct
        span_dict = span.to_dict()
        reconstructed = Span.from_dict(span_dict)

        # Should be identical
        assert reconstructed.name == span.name
        assert reconstructed.trace_id == span.trace_id
        assert reconstructed.span_id == span.span_id
        assert reconstructed.is_finished() == span.is_finished()
        assert reconstructed.attributes == span.attributes


class TestSpanEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    def test_span_with_empty_name(self):
        """Test span with empty name."""
        span = Span("", "trace-123")

        assert span.name == ""
        assert span.trace_id == "trace-123"

    def test_span_with_very_long_name(self):
        """Test span with very long name."""
        long_name = "a" * 1000
        span = Span(long_name, "trace-123")

        assert span.name == long_name

    def test_span_repr(self):
        """Test span string representation."""
        span = Span("test_span", "trace-123", span_id="span-456")

        repr_str = repr(span)

        assert "Span" in repr_str
        assert "test_span" in repr_str
        assert "span-456" in repr_str
        assert "trace-123" in repr_str
        assert "unset" in repr_str  # Initial status

    def test_span_with_none_values(self):
        """Test span behavior with None values."""
        span = Span("test_span", "trace-123")

        # Test setting None values
        span.set_attribute("null_key", None)
        span.set_status(SpanStatus.OK, None)
        span.add_event("test_event", None)
        span.add_link("trace", "span", None)

        assert span.attributes["null_key"] is None
        assert span.status_message is None
        assert span.events[0].attributes == {}
        assert span.links[0]["attributes"] == {}

    def test_concurrent_operations(self):
        """Test that operations maintain consistency."""
        span = Span("test_span", "trace-123")

        # Simulate concurrent operations
        for i in range(10):
            span.set_attribute(f"key_{i}", f"value_{i}")
            span.add_event(f"event_{i}", {"index": i})
            span.add_link(f"trace_{i}", f"span_{i}")

        assert len(span.attributes) >= 10  # At least the added attributes
        assert len(span.events) == 10
        assert len(span.links) == 10

    def test_method_chaining(self):
        """Test that methods can be chained."""
        span = Span("test_span", "trace-123")

        result = (
            span.set_attribute("key1", "value1")
            .set_attributes({"key2": "value2"})
            .add_event("event1")
            .add_link("trace", "span")
            .set_status(SpanStatus.OK)
        )

        assert result == span
        assert span.attributes["key1"] == "value1"
        assert span.attributes["key2"] == "value2"
        assert len(span.events) == 1
        assert len(span.links) == 1
        assert span.status == SpanStatus.OK

    def test_duration_calculation_precision(self):
        """Test duration calculation precision."""
        start_time = datetime(
            2023, 1, 1, 10, 0, 0, 500000, tzinfo=timezone.utc
        )  # .5 seconds
        end_time = datetime(
            2023, 1, 1, 10, 0, 1, 750000, tzinfo=timezone.utc
        )  # 1.75 seconds

        span = Span("test_span", "trace-123", start_time=start_time)
        span.finish(end_time)

        expected_duration = (end_time - start_time).total_seconds() * 1000
        assert span.duration_ms == expected_duration
        assert span.duration_ms == 1250.0  # 1.25 seconds in milliseconds
