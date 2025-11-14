"""
Comprehensive unit tests for Trace class.

This module tests all aspects of the Trace class including creation, span management,
serialization, error handling, and complex hierarchical scenarios.
"""

from datetime import datetime, timezone

import pytest

from noveum_trace.core.span import Span, SpanStatus
from noveum_trace.core.trace import Trace, TraceMetadata
from noveum_trace.utils.exceptions import NoveumTraceError


class TestTraceMetadata:
    """Test TraceMetadata dataclass functionality."""

    def test_trace_metadata_creation_empty(self):
        """Test creating empty trace metadata."""
        metadata = TraceMetadata()

        assert metadata.user_id is None
        assert metadata.session_id is None
        assert metadata.request_id is None
        assert metadata.tags == {}
        assert metadata.custom_attributes == {}

    @pytest.mark.parametrize(
        "user_id,session_id,request_id",
        [
            ("user123", "session456", "req789"),
            ("admin", "admin-session", "admin-req-001"),
            (None, "guest-session", "guest-req"),
        ],
    )
    def test_trace_metadata_creation_with_data(self, user_id, session_id, request_id):
        """Test creating trace metadata with different data."""
        tags = {"environment": "test", "version": "1.0"}
        custom_attrs = {"custom": "value", "number": 42}

        metadata = TraceMetadata(
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            tags=tags,
            custom_attributes=custom_attrs,
        )

        assert metadata.user_id == user_id
        assert metadata.session_id == session_id
        assert metadata.request_id == request_id
        assert metadata.tags == tags
        assert metadata.custom_attributes == custom_attrs


class TestTraceInitialization:
    """Test Trace class initialization and basic properties."""

    def test_trace_creation_minimal(self):
        """Test creating trace with minimal parameters."""
        trace = Trace("test_trace")

        assert trace.name == "test_trace"
        assert trace.trace_id is not None
        assert len(trace.trace_id) > 0
        assert isinstance(trace.start_time, datetime)
        assert trace.end_time is None
        assert isinstance(trace.metadata, TraceMetadata)
        assert trace.attributes == {}
        assert trace.spans == []
        assert trace.root_span is None
        assert trace.active_spans == {}
        assert trace.status == SpanStatus.UNSET
        assert trace.status_message is None
        assert trace.duration_ms is None
        assert trace.span_count == 0
        assert trace.error_count == 0
        assert trace._finished is False
        assert trace._noop is False

    @pytest.mark.parametrize(
        "trace_name,trace_id",
        [
            ("simple_trace", "custom-trace-id-123"),
            ("complex-trace-name", "trace_456_789"),
            ("trace with spaces", "trace-with-dashes"),
        ],
    )
    def test_trace_creation_with_custom_id(self, trace_name, trace_id):
        """Test creating trace with custom ID."""
        trace = Trace(trace_name, trace_id=trace_id)

        assert trace.name == trace_name
        assert trace.trace_id == trace_id

    def test_trace_creation_with_custom_time(self):
        """Test creating trace with custom start time."""
        custom_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        trace = Trace("test_trace", start_time=custom_time)

        assert trace.start_time == custom_time

    def test_trace_creation_with_metadata(self):
        """Test creating trace with custom metadata."""
        metadata = TraceMetadata(
            user_id="test_user", session_id="test_session", tags={"env": "test"}
        )
        trace = Trace("test_trace", metadata=metadata)

        assert trace.metadata == metadata
        assert trace.metadata.user_id == "test_user"

    def test_trace_creation_with_attributes(self):
        """Test creating trace with initial attributes."""
        attributes = {"key1": "value1", "key2": 42, "key3": True}
        trace = Trace("test_trace", attributes=attributes)

        assert trace.attributes == attributes

    def test_trace_id_generation_uniqueness(self):
        """Test that generated trace IDs are unique."""
        trace_ids = set()

        for _ in range(100):
            trace = Trace("test_trace")
            assert trace.trace_id not in trace_ids
            trace_ids.add(trace.trace_id)

        assert len(trace_ids) == 100


class TestTraceSpanManagement:
    """Test trace span creation and management functionality."""

    def test_create_span_basic(self):
        """Test basic span creation."""
        trace = Trace("test_trace")

        span = trace.create_span("test_span")

        assert isinstance(span, Span)
        assert span.name == "test_span"
        assert span.trace_id == trace.trace_id
        assert span.parent_span_id is None
        assert span in trace.spans
        assert span.span_id in trace.active_spans
        assert trace.active_spans[span.span_id] == span
        assert trace.span_count == 1
        assert trace.root_span == span

    @pytest.mark.parametrize(
        "span_name,attributes",
        [
            ("simple_span", None),
            ("llm_span", {"llm.model": "gpt-4", "llm.tokens": 100}),
            ("agent_span", {"agent.action": "search", "agent.tool": "web_search"}),
            ("complex_span", {"custom": {"nested": "data"}, "list": [1, 2, 3]}),
        ],
    )
    def test_create_span_with_attributes(self, span_name, attributes):
        """Test span creation with different attributes."""
        import json

        trace = Trace("test_trace")

        span = trace.create_span(span_name, attributes=attributes)

        assert span.name == span_name
        if attributes:
            for key, value in attributes.items():
                # Dicts and lists are converted to JSON strings
                if isinstance(value, (dict, list)):
                    assert isinstance(span.attributes[key], str)
                    # Verify it's valid JSON and can be parsed back
                    parsed = json.loads(span.attributes[key])
                    assert parsed == value
                else:
                    # Other types remain unchanged
                    assert span.attributes[key] == value

    def test_create_span_with_custom_time(self):
        """Test span creation with custom start time."""
        trace = Trace("test_trace")
        custom_time = datetime(2023, 6, 1, 10, 0, 0, tzinfo=timezone.utc)

        span = trace.create_span("test_span", start_time=custom_time)

        assert span.start_time == custom_time

    def test_create_span_with_parent(self):
        """Test creating child spans with parent relationships."""
        trace = Trace("test_trace")

        # Create parent span
        parent_span = trace.create_span("parent_span")

        # Create child span
        child_span = trace.create_span("child_span", parent_span_id=parent_span.span_id)

        assert child_span.parent_span_id == parent_span.span_id
        assert child_span.trace_id == trace.trace_id
        assert trace.span_count == 2
        assert trace.root_span == parent_span  # First span without parent is root

    def test_create_multiple_spans_hierarchy(self):
        """Test creating complex span hierarchies."""
        trace = Trace("test_trace")

        # Create root span
        root = trace.create_span("root")

        # Create children
        child1 = trace.create_span("child1", parent_span_id=root.span_id)
        child2 = trace.create_span("child2", parent_span_id=root.span_id)

        # Create grandchildren
        _grandchild1 = trace.create_span("grandchild1", parent_span_id=child1.span_id)
        _grandchild2 = trace.create_span("grandchild2", parent_span_id=child1.span_id)

        assert trace.span_count == 5
        assert trace.root_span == root
        assert len(trace.get_child_spans(root.span_id)) == 2
        assert len(trace.get_child_spans(child1.span_id)) == 2
        assert len(trace.get_child_spans(child2.span_id)) == 0

    def test_create_span_in_finished_trace(self):
        """Test that span creation fails in finished trace."""
        trace = Trace("test_trace")
        trace.finish()

        with pytest.raises(
            NoveumTraceError, match="Cannot create span in finished trace"
        ):
            trace.create_span("test_span")

    def test_get_span(self):
        """Test retrieving spans by ID."""
        trace = Trace("test_trace")

        span1 = trace.create_span("span1")
        span2 = trace.create_span("span2")

        # Test getting existing spans
        assert trace.get_span(span1.span_id) == span1
        assert trace.get_span(span2.span_id) == span2

        # Test getting non-existent span
        assert trace.get_span("nonexistent-id") is None

    def test_finish_span(self):
        """Test finishing spans and updating trace statistics."""
        trace = Trace("test_trace")

        span1 = trace.create_span("span1")
        span2 = trace.create_span("span2")

        custom_end_time = datetime(2023, 6, 1, 10, 0, 0, tzinfo=timezone.utc)

        # Finish span1 with custom time
        trace.finish_span(span1.span_id, custom_end_time)

        assert span1.is_finished()
        assert span1.end_time == custom_end_time
        assert span1.span_id not in trace.active_spans
        assert span2.span_id in trace.active_spans

    def test_finish_span_with_error(self):
        """Test finishing spans with error status updates trace error count."""
        trace = Trace("test_trace")

        span = trace.create_span("span1")
        span.set_status(SpanStatus.ERROR, "Test error")

        trace.finish_span(span.span_id)

        assert trace.error_count == 1

    def test_finish_nonexistent_span(self):
        """Test finishing non-existent span is safe."""
        trace = Trace("test_trace")

        # Should not raise exception
        trace.finish_span("nonexistent-id")

    def test_get_active_spans(self):
        """Test getting all active spans."""
        trace = Trace("test_trace")

        span1 = trace.create_span("span1")
        span2 = trace.create_span("span2")
        span3 = trace.create_span("span3")

        active_spans = trace.get_active_spans()
        assert len(active_spans) == 3
        assert span1 in active_spans
        assert span2 in active_spans
        assert span3 in active_spans

        # Finish one span
        trace.finish_span(span2.span_id)

        active_spans = trace.get_active_spans()
        assert len(active_spans) == 2
        assert span1 in active_spans
        assert span2 not in active_spans
        assert span3 in active_spans

    def test_get_child_spans(self):
        """Test getting child spans of a parent."""
        trace = Trace("test_trace")

        parent = trace.create_span("parent")
        child1 = trace.create_span("child1", parent_span_id=parent.span_id)
        child2 = trace.create_span("child2", parent_span_id=parent.span_id)
        unrelated = trace.create_span("unrelated")

        children = trace.get_child_spans(parent.span_id)
        assert len(children) == 2
        assert child1 in children
        assert child2 in children
        assert unrelated not in children

        # Test parent with no children
        no_children = trace.get_child_spans(unrelated.span_id)
        assert len(no_children) == 0


class TestTraceAttributesAndMetadata:
    """Test trace attribute and metadata management."""

    def test_set_attribute(self):
        """Test setting individual attributes."""
        trace = Trace("test_trace")

        result = trace.set_attribute("key1", "value1")

        assert result == trace  # Should return self for chaining
        assert trace.attributes["key1"] == "value1"

    def test_set_attributes(self):
        """Test setting multiple attributes."""
        trace = Trace("test_trace")

        attributes = {"key1": "value1", "key2": 42, "key3": True}
        result = trace.set_attributes(attributes)

        assert result == trace  # Should return self for chaining
        for key, value in attributes.items():
            assert trace.attributes[key] == value

    def test_set_attributes_update_existing(self):
        """Test that set_attributes updates existing attributes."""
        trace = Trace("test_trace")

        # Set initial attributes
        trace.set_attribute("key1", "old_value")
        trace.set_attribute("key2", "keep_value")

        # Update with new attributes
        new_attributes = {"key1": "new_value", "key3": "added_value"}
        trace.set_attributes(new_attributes)

        assert trace.attributes["key1"] == "new_value"  # Updated
        assert trace.attributes["key2"] == "keep_value"  # Kept
        assert trace.attributes["key3"] == "added_value"  # Added

    def test_attribute_operations_on_finished_trace(self):
        """Test that attribute operations are ignored on finished traces."""
        trace = Trace("test_trace")
        trace.finish()

        trace.set_attribute("key1", "value1")
        trace.set_attributes({"key2": "value2"})

        assert "key1" not in trace.attributes
        assert "key2" not in trace.attributes

    def test_set_metadata(self):
        """Test setting metadata fields."""
        trace = Trace("test_trace")

        result = trace.set_metadata(user_id="test_user", session_id="test_session")

        assert result == trace  # Should return self for chaining
        assert trace.metadata.user_id == "test_user"
        assert trace.metadata.session_id == "test_session"

    def test_set_metadata_custom_attributes(self):
        """Test setting custom metadata attributes."""
        trace = Trace("test_trace")

        trace.set_metadata(custom_field="custom_value", another_field=123)

        assert trace.metadata.custom_attributes["custom_field"] == "custom_value"
        assert trace.metadata.custom_attributes["another_field"] == 123

    def test_add_tag(self):
        """Test adding tags to trace."""
        trace = Trace("test_trace")

        result = trace.add_tag("environment", "test")

        assert result == trace  # Should return self for chaining
        assert trace.metadata.tags["environment"] == "test"

    @pytest.mark.parametrize(
        "tag_key,tag_value",
        [
            ("env", "production"),
            ("version", "1.2.3"),
            ("team", "ml-team"),
            ("priority", "high"),
        ],
    )
    def test_add_multiple_tags(self, tag_key, tag_value):
        """Test adding multiple tags."""
        trace = Trace("test_trace")

        trace.add_tag(tag_key, tag_value)

        assert trace.metadata.tags[tag_key] == tag_value

    def test_metadata_operations_on_finished_trace(self):
        """Test that metadata operations are ignored on finished traces."""
        trace = Trace("test_trace")
        trace.finish()

        trace.set_metadata(user_id="test_user")
        trace.add_tag("env", "test")

        assert trace.metadata.user_id is None
        assert "env" not in trace.metadata.tags


class TestTraceStatusAndLifecycle:
    """Test trace status management and lifecycle operations."""

    def test_set_status(self):
        """Test setting trace status."""
        trace = Trace("test_trace")

        result = trace.set_status(SpanStatus.OK, "All good")

        assert result == trace  # Should return self for chaining
        assert trace.status == SpanStatus.OK
        assert trace.status_message == "All good"

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
        trace = Trace("test_trace")

        trace.set_status(status, message)

        assert trace.status == status
        assert trace.status_message == message

    def test_status_operations_on_finished_trace(self):
        """Test that status operations are ignored on finished traces."""
        trace = Trace("test_trace")
        trace.finish()

        original_status = trace.status
        trace.set_status(SpanStatus.ERROR, "Error after finish")

        assert trace.status == original_status
        assert trace.status_message != "Error after finish"

    def test_finish_basic(self):
        """Test basic trace finishing."""
        trace = Trace("test_trace")

        assert not trace.is_finished()

        trace.finish()

        assert trace.is_finished()
        assert trace.end_time is not None
        assert trace.duration_ms is not None
        assert trace.duration_ms >= 0
        assert trace._finished is True

    def test_finish_with_custom_time(self):
        """Test finishing trace with custom end time."""
        start_time = datetime(2023, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        end_time = datetime(2023, 1, 1, 10, 5, 30, tzinfo=timezone.utc)

        trace = Trace("test_trace", start_time=start_time)
        trace.finish(end_time)

        assert trace.end_time == end_time
        expected_duration = (end_time - start_time).total_seconds() * 1000
        assert trace.duration_ms == expected_duration

    def test_finish_finishes_active_spans(self):
        """Test that finishing trace finishes all active spans."""
        trace = Trace("test_trace")

        span1 = trace.create_span("span1")
        span2 = trace.create_span("span2")

        assert not span1.is_finished()
        assert not span2.is_finished()
        assert len(trace.active_spans) == 2

        trace.finish()

        assert span1.is_finished()
        assert span2.is_finished()
        assert len(trace.active_spans) == 0

    def test_finish_sets_default_status(self):
        """Test that finish sets default status based on errors."""
        # Test default OK status
        trace1 = Trace("test_trace1")
        trace1.finish()
        assert trace1.status == SpanStatus.OK

        # Test ERROR status when errors present
        trace2 = Trace("test_trace2")
        span = trace2.create_span("error_span")
        span.set_status(SpanStatus.ERROR)
        trace2.finish_span(span.span_id)
        trace2.finish()
        assert trace2.status == SpanStatus.ERROR

    def test_finish_idempotent(self):
        """Test that finishing a trace multiple times is safe."""
        trace = Trace("test_trace")

        trace.finish()
        original_end_time = trace.end_time
        original_duration = trace.duration_ms

        # Finish again - should be no-op
        trace.finish()

        assert trace.end_time == original_end_time
        assert trace.duration_ms == original_duration


class TestTraceContextManager:
    """Test trace context manager functionality."""

    def test_context_manager_success(self):
        """Test trace context manager for successful execution."""
        with Trace("test_trace") as trace:
            assert isinstance(trace, Trace)
            assert trace.name == "test_trace"
            assert not trace.is_finished()

        # Should be finished after context exit
        assert trace.is_finished()
        assert trace.status == SpanStatus.OK

    def test_context_manager_with_exception(self):
        """Test trace context manager with exception."""
        test_exception = ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            with Trace("test_trace") as trace:
                raise test_exception

        # Should be finished with error status
        assert trace.is_finished()
        assert trace.status == SpanStatus.ERROR
        assert trace.status_message == "Test error"

    def test_context_manager_with_spans(self):
        """Test trace context manager with spans."""
        with Trace("test_trace") as trace:
            span1 = trace.create_span("span1")
            span2 = trace.create_span("span2")

        # All spans should be finished
        assert span1.is_finished()
        assert span2.is_finished()
        assert trace.is_finished()


class TestTraceSummaryAndSerialization:
    """Test trace summary and serialization functionality."""

    def test_get_summary(self):
        """Test getting trace summary."""
        trace = Trace("test_trace")
        span = trace.create_span("test_span")
        trace.finish_span(span.span_id)
        trace.finish()

        summary = trace.get_summary()

        assert summary["trace_id"] == trace.trace_id
        assert summary["name"] == "test_trace"
        assert summary["status"] == trace.status.value
        assert summary["duration_ms"] == trace.duration_ms
        assert summary["span_count"] == 1
        assert summary["error_count"] == 0
        assert "start_time" in summary
        assert "end_time" in summary

    def test_to_dict_complete(self):
        """Test complete trace serialization to dictionary."""
        # Create trace with metadata and attributes
        metadata = TraceMetadata(
            user_id="test_user", session_id="test_session", tags={"env": "test"}
        )
        trace = Trace("test_trace", metadata=metadata, attributes={"key": "value"})

        # Add span
        span = trace.create_span("test_span", attributes={"span_key": "span_value"})
        trace.finish_span(span.span_id)
        trace.finish()

        trace_dict = trace.to_dict()

        # Verify all fields are present
        assert trace_dict["trace_id"] == trace.trace_id
        assert trace_dict["name"] == "test_trace"
        assert trace_dict["status"] == trace.status.value
        assert trace_dict["attributes"]["key"] == "value"
        assert trace_dict["metadata"]["user_id"] == "test_user"
        assert trace_dict["metadata"]["tags"]["env"] == "test"
        assert len(trace_dict["spans"]) == 1
        assert trace_dict["spans"][0]["name"] == "test_span"

    def test_from_dict_reconstruction(self):
        """Test reconstructing trace from dictionary."""
        # Create original trace
        original_metadata = TraceMetadata(user_id="test_user", tags={"env": "test"})
        original_trace = Trace(
            "test_trace",
            trace_id="custom-trace-id",
            metadata=original_metadata,
            attributes={"key": "value"},
        )
        span = original_trace.create_span("test_span")
        original_trace.finish_span(span.span_id)
        original_trace.finish()

        # Serialize and reconstruct
        trace_dict = original_trace.to_dict()
        reconstructed_trace = Trace.from_dict(trace_dict)

        # Verify reconstruction
        assert reconstructed_trace.name == original_trace.name
        assert reconstructed_trace.trace_id == original_trace.trace_id
        assert reconstructed_trace.status == original_trace.status
        assert reconstructed_trace.attributes == original_trace.attributes
        assert reconstructed_trace.metadata.user_id == original_trace.metadata.user_id
        assert reconstructed_trace.metadata.tags == original_trace.metadata.tags
        assert len(reconstructed_trace.spans) == len(original_trace.spans)
        assert reconstructed_trace.is_finished() == original_trace.is_finished()

    @pytest.mark.parametrize("include_finished", [True, False])
    def test_serialization_roundtrip(self, include_finished):
        """Test complete serialization roundtrip with different states."""
        trace = Trace("test_trace")
        span1 = trace.create_span("span1")
        _span2 = trace.create_span("span2", parent_span_id=span1.span_id)

        if include_finished:
            trace.finish()

        # Serialize and reconstruct
        trace_dict = trace.to_dict()
        reconstructed = Trace.from_dict(trace_dict)

        # Should be identical
        assert reconstructed.name == trace.name
        assert reconstructed.trace_id == trace.trace_id
        assert reconstructed.is_finished() == trace.is_finished()
        assert len(reconstructed.spans) == len(trace.spans)


class TestTraceEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    def test_trace_with_empty_name(self):
        """Test trace with empty name."""
        trace = Trace("")

        assert trace.name == ""
        assert trace.trace_id is not None

    def test_trace_with_very_long_name(self):
        """Test trace with very long name."""
        long_name = "a" * 1000
        trace = Trace(long_name)

        assert trace.name == long_name

    def test_trace_repr(self):
        """Test trace string representation."""
        trace = Trace("test_trace")
        trace.create_span("span1")

        repr_str = repr(trace)

        assert "Trace" in repr_str
        assert "test_trace" in repr_str
        assert trace.trace_id in repr_str
        assert "span_count=1" in repr_str
        assert "unset" in repr_str  # Initial status

    def test_concurrent_span_operations(self):
        """Test that span operations maintain consistency."""
        trace = Trace("test_trace")

        # Simulate concurrent operations
        spans = []
        for i in range(10):
            span = trace.create_span(f"span_{i}")
            spans.append(span)

        assert trace.span_count == 10
        assert len(trace.active_spans) == 10

        # Finish half the spans
        for i in range(0, 10, 2):
            trace.finish_span(spans[i].span_id)

        assert len(trace.active_spans) == 5
        assert trace.error_count == 0

    def test_trace_with_none_values(self):
        """Test trace behavior with None values."""
        trace = Trace("test_trace")

        # Test setting None values
        trace.set_attribute("null_key", None)
        trace.set_status(SpanStatus.OK, None)

        assert trace.attributes["null_key"] is None
        assert trace.status_message is None
