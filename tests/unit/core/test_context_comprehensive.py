"""
Comprehensive tests for context management functionality.

This module provides extensive test coverage for the context management
system, including all context operations, contextual managers, and edge cases.
"""

import asyncio
import contextvars
import threading
from unittest.mock import Mock, call

import pytest

from noveum_trace.core.context import (
    ContextualSpan,
    ContextualTrace,
    TraceContext,
    attach_context_to_span,
    clear_context,
    copy_context,
    get_current_context,
    get_current_span,
    get_current_trace,
    inherit_context_attributes,
    propagate_context_to_thread,
    set_current_context,
    set_current_span,
    set_current_trace,
    span_context,
    trace_context,
)
from noveum_trace.core.span import Span, SpanStatus
from noveum_trace.core.trace import Trace


class TestTraceContext:
    """Test TraceContext class functionality."""

    def test_trace_context_default_initialization(self):
        """Test TraceContext initialization with defaults."""
        context = TraceContext()

        assert context.trace is None
        assert context.span is None
        assert context.attributes == {}

    def test_trace_context_initialization_with_values(self):
        """Test TraceContext initialization with values."""
        trace = Mock(spec=Trace)
        span = Mock(spec=Span)
        attributes = {"key": "value"}

        context = TraceContext(trace=trace, span=span, attributes=attributes)

        assert context.trace == trace
        assert context.span == span
        assert context.attributes == attributes

    def test_trace_context_initialization_with_none_attributes(self):
        """Test TraceContext initialization with None attributes."""
        context = TraceContext(attributes=None)

        assert context.attributes == {}

    def test_trace_context_repr(self):
        """Test TraceContext string representation."""
        trace = Mock(spec=Trace)
        trace.trace_id = "test-trace-id"
        span = Mock(spec=Span)
        span.span_id = "test-span-id"

        context = TraceContext(trace=trace, span=span, attributes={"key": "value"})

        repr_str = repr(context)
        assert "TraceContext" in repr_str
        # The repr shows the Mock object, not the actual IDs
        assert "Mock" in repr_str


class TestContextManagement:
    """Test basic context management operations."""

    def test_get_current_context_default(self):
        """Test getting current context returns default."""
        # Clear context first
        clear_context()

        context = get_current_context()

        assert isinstance(context, TraceContext)
        assert context.trace is None
        assert context.span is None
        assert context.attributes == {}

    def test_set_and_get_current_context(self):
        """Test setting and getting current context."""
        trace = Mock(spec=Trace)
        span = Mock(spec=Span)
        attributes = {"key": "value"}

        context = TraceContext(trace=trace, span=span, attributes=attributes)
        set_current_context(context)

        retrieved_context = get_current_context()

        assert retrieved_context.trace == trace
        assert retrieved_context.span == span
        assert retrieved_context.attributes == attributes

    def test_get_current_trace_none(self):
        """Test getting current trace when none is set."""
        clear_context()

        current_trace = get_current_trace()

        assert current_trace is None

    def test_set_and_get_current_trace(self):
        """Test setting and getting current trace."""
        trace = Mock(spec=Trace)

        set_current_trace(trace)

        current_trace = get_current_trace()

        assert current_trace == trace

    def test_get_current_span_none(self):
        """Test getting current span when none is set."""
        clear_context()

        current_span = get_current_span()

        assert current_span is None

    def test_set_and_get_current_span(self):
        """Test setting and getting current span."""
        span = Mock(spec=Span)

        set_current_span(span)

        current_span = get_current_span()

        assert current_span == span

    def test_set_current_trace_preserves_span(self):
        """Test that setting current trace preserves existing span."""
        original_span = Mock(spec=Span)
        set_current_span(original_span)

        new_trace = Mock(spec=Trace)
        set_current_trace(new_trace)

        context = get_current_context()
        assert context.trace == new_trace
        assert context.span == original_span

    def test_set_current_span_preserves_trace(self):
        """Test that setting current span preserves existing trace."""
        original_trace = Mock(spec=Trace)
        set_current_trace(original_trace)

        new_span = Mock(spec=Span)
        set_current_span(new_span)

        context = get_current_context()
        assert context.trace == original_trace
        assert context.span == new_span

    def test_clear_context(self):
        """Test clearing context."""
        # Set some context
        trace = Mock(spec=Trace)
        span = Mock(spec=Span)
        set_current_trace(trace)
        set_current_span(span)

        clear_context()

        context = get_current_context()
        assert context.trace is None
        assert context.span is None
        assert context.attributes == {}

    def test_copy_context(self):
        """Test copying context."""
        trace = Mock(spec=Trace)
        span = Mock(spec=Span)
        attributes = {"key": "value"}

        original_context = TraceContext(trace=trace, span=span, attributes=attributes)
        set_current_context(original_context)

        copied_context = copy_context()

        assert copied_context.trace == trace
        assert copied_context.span == span
        assert copied_context.attributes == attributes
        assert (
            copied_context.attributes is not original_context.attributes
        )  # Should be a copy

    def test_propagate_context_to_thread_with_context(self):
        """Test propagating specific context to thread."""
        trace = Mock(spec=Trace)
        span = Mock(spec=Span)
        attributes = {"key": "value"}

        context = TraceContext(trace=trace, span=span, attributes=attributes)

        propagated_context = propagate_context_to_thread(context)

        assert propagated_context.trace == trace
        assert propagated_context.span == span
        assert propagated_context.attributes == attributes
        assert (
            propagated_context.attributes is not context.attributes
        )  # Should be a copy

    def test_propagate_context_to_thread_current_context(self):
        """Test propagating current context to thread."""
        trace = Mock(spec=Trace)
        span = Mock(spec=Span)
        attributes = {"key": "value"}

        set_current_context(TraceContext(trace=trace, span=span, attributes=attributes))

        propagated_context = propagate_context_to_thread()

        assert propagated_context.trace == trace
        assert propagated_context.span == span
        assert propagated_context.attributes == attributes


class TestContextManagers:
    """Test context manager functionality."""

    def test_with_trace_context_basic(self):
        """Test basic trace_context usage."""
        trace = Mock(spec=Trace)

        with trace_context(trace=trace) as context:
            assert context.trace == trace
            assert get_current_trace() == trace

    def test_with_trace_context_with_attributes(self):
        """Test trace_context with attributes."""
        trace = Mock(spec=Trace)
        attributes = {"key": "value"}

        with trace_context(trace=trace, **attributes) as context:
            assert context.trace == trace
            assert get_current_trace() == trace

            current_context = get_current_context()
            assert current_context.attributes["key"] == "value"

    def test_with_trace_context_restores_previous(self):
        """Test that trace_context restores previous context."""
        original_trace = Mock(spec=Trace)
        set_current_trace(original_trace)

        new_trace = Mock(spec=Trace)

        with trace_context(trace=new_trace):
            assert get_current_trace() == new_trace

        # Should restore original trace
        assert get_current_trace() == original_trace

    def test_with_trace_context_exception_handling(self):
        """Test trace_context handles exceptions."""
        original_trace = Mock(spec=Trace)
        set_current_trace(original_trace)

        new_trace = Mock(spec=Trace)

        with pytest.raises(ValueError):
            with trace_context(trace=new_trace):
                assert get_current_trace() == new_trace
                raise ValueError("Test exception")

        # Should restore original trace even after exception
        assert get_current_trace() == original_trace

    def test_with_span_context_basic(self):
        """Test basic span_context usage."""
        # Set a trace first
        trace = Mock(spec=Trace)
        set_current_trace(trace)

        span = Mock(spec=Span)

        with span_context(span) as context_span:
            assert context_span == span
            assert get_current_span() == span
            assert get_current_trace() == trace  # Should preserve trace

    def test_with_span_context_restores_previous(self):
        """Test that span_context restores previous context."""
        trace = Mock(spec=Trace)
        original_span = Mock(spec=Span)
        set_current_trace(trace)
        set_current_span(original_span)

        new_span = Mock(spec=Span)

        with span_context(new_span):
            assert get_current_span() == new_span

        # Should restore original span
        assert get_current_span() == original_span

    def test_with_span_context_exception_handling(self):
        """Test span_context handles exceptions."""
        trace = Mock(spec=Trace)
        original_span = Mock(spec=Span)
        set_current_trace(trace)
        set_current_span(original_span)

        new_span = Mock(spec=Span)

        with pytest.raises(ValueError):
            with span_context(new_span):
                assert get_current_span() == new_span
                raise ValueError("Test exception")

        # Should restore original span even after exception
        assert get_current_span() == original_span


class TestContextualSpan:
    """Test ContextualSpan functionality."""

    def test_contextual_span_basic_usage(self):
        """Test basic ContextualSpan usage."""
        span = Mock(spec=Span)

        contextual_span = ContextualSpan(span)

        with contextual_span as ctx_span:
            assert ctx_span == span
            assert get_current_span() == span

    def test_contextual_span_restores_previous(self):
        """Test ContextualSpan restores previous span."""
        original_span = Mock(spec=Span)
        set_current_span(original_span)

        new_span = Mock(spec=Span)
        contextual_span = ContextualSpan(new_span)

        with contextual_span:
            assert get_current_span() == new_span

        assert get_current_span() == original_span

    def test_contextual_span_handles_exceptions(self):
        """Test ContextualSpan handles exceptions."""
        original_span = Mock(spec=Span)
        set_current_span(original_span)

        new_span = Mock(spec=Span)
        contextual_span = ContextualSpan(new_span)

        with pytest.raises(ValueError):
            with contextual_span:
                raise ValueError("Test exception")

        # Should have recorded exception and set error status
        new_span.record_exception.assert_called_once()
        new_span.set_status.assert_called_once_with(SpanStatus.ERROR, "Test exception")

        # Should have finished span and restored previous
        new_span.finish.assert_called_once()
        assert get_current_span() == original_span

    def test_contextual_span_finishes_span(self):
        """Test ContextualSpan finishes span on exit."""
        span = Mock(spec=Span)
        contextual_span = ContextualSpan(span)

        with contextual_span:
            pass

        span.finish.assert_called_once()

    def test_contextual_span_attribute_delegation(self):
        """Test ContextualSpan delegates attributes to span."""
        span = Mock(spec=Span)
        span.span_id = "test-span-id"
        span.name = "test-span"

        contextual_span = ContextualSpan(span)

        assert contextual_span.span_id == "test-span-id"
        assert contextual_span.name == "test-span"

    @pytest.mark.asyncio
    async def test_contextual_span_async_usage(self):
        """Test ContextualSpan async context manager."""
        span = Mock(spec=Span)
        contextual_span = ContextualSpan(span)

        async with contextual_span as ctx_span:
            assert ctx_span == span
            assert get_current_span() == span

    @pytest.mark.asyncio
    async def test_contextual_span_async_restores_previous(self):
        """Test ContextualSpan async restores previous span."""
        original_span = Mock(spec=Span)
        set_current_span(original_span)

        new_span = Mock(spec=Span)
        contextual_span = ContextualSpan(new_span)

        async with contextual_span:
            assert get_current_span() == new_span

        assert get_current_span() == original_span

    @pytest.mark.asyncio
    async def test_contextual_span_async_handles_exceptions(self):
        """Test ContextualSpan async handles exceptions."""
        original_span = Mock(spec=Span)
        set_current_span(original_span)

        new_span = Mock(spec=Span)
        contextual_span = ContextualSpan(new_span)

        with pytest.raises(ValueError):
            async with contextual_span:
                raise ValueError("Test exception")

        # Should have recorded exception and set error status
        new_span.record_exception.assert_called_once()
        new_span.set_status.assert_called_once_with(SpanStatus.ERROR, "Test exception")

        # Should have finished span and restored previous
        new_span.finish.assert_called_once()
        assert get_current_span() == original_span


class TestContextualTrace:
    """Test ContextualTrace functionality."""

    def test_contextual_trace_basic_usage(self):
        """Test basic ContextualTrace usage."""
        trace = Mock(spec=Trace)

        contextual_trace = ContextualTrace(trace)

        with contextual_trace as ctx_trace:
            assert ctx_trace == trace
            assert get_current_trace() == trace

    def test_contextual_trace_restores_previous(self):
        """Test ContextualTrace restores previous trace."""
        original_trace = Mock(spec=Trace)
        set_current_trace(original_trace)

        new_trace = Mock(spec=Trace)
        contextual_trace = ContextualTrace(new_trace)

        with contextual_trace:
            assert get_current_trace() == new_trace

        assert get_current_trace() == original_trace

    def test_contextual_trace_handles_exceptions(self):
        """Test ContextualTrace handles exceptions."""
        original_trace = Mock(spec=Trace)
        set_current_trace(original_trace)

        new_trace = Mock(spec=Trace)
        contextual_trace = ContextualTrace(new_trace)

        with pytest.raises(ValueError):
            with contextual_trace:
                raise ValueError("Test exception")

        # Should have set error status
        new_trace.set_status.assert_called_once_with(SpanStatus.ERROR, "Test exception")

        # Should have finished trace and restored previous
        new_trace.finish.assert_called_once()
        assert get_current_trace() == original_trace

    def test_contextual_trace_finishes_trace(self):
        """Test ContextualTrace finishes trace on exit."""
        trace = Mock(spec=Trace)
        contextual_trace = ContextualTrace(trace)

        with contextual_trace:
            pass

        trace.finish.assert_called_once()

    def test_contextual_trace_attribute_delegation(self):
        """Test ContextualTrace delegates attributes to trace."""
        trace = Mock(spec=Trace)
        trace.trace_id = "test-trace-id"
        trace.name = "test-trace"

        contextual_trace = ContextualTrace(trace)

        assert contextual_trace.trace_id == "test-trace-id"
        assert contextual_trace.name == "test-trace"

    @pytest.mark.asyncio
    async def test_contextual_trace_async_usage(self):
        """Test ContextualTrace async context manager."""
        trace = Mock(spec=Trace)
        contextual_trace = ContextualTrace(trace)

        async with contextual_trace as ctx_trace:
            assert ctx_trace == trace
            assert get_current_trace() == trace

    @pytest.mark.asyncio
    async def test_contextual_trace_async_restores_previous(self):
        """Test ContextualTrace async restores previous trace."""
        original_trace = Mock(spec=Trace)
        set_current_trace(original_trace)

        new_trace = Mock(spec=Trace)
        contextual_trace = ContextualTrace(new_trace)

        async with contextual_trace:
            assert get_current_trace() == new_trace

        assert get_current_trace() == original_trace

    @pytest.mark.asyncio
    async def test_contextual_trace_async_handles_exceptions(self):
        """Test ContextualTrace async handles exceptions."""
        original_trace = Mock(spec=Trace)
        set_current_trace(original_trace)

        new_trace = Mock(spec=Trace)
        contextual_trace = ContextualTrace(new_trace)

        with pytest.raises(ValueError):
            async with contextual_trace:
                raise ValueError("Test exception")

        # Should have set error status
        new_trace.set_status.assert_called_once_with(SpanStatus.ERROR, "Test exception")

        # Should have finished trace and restored previous
        new_trace.finish.assert_called_once()
        assert get_current_trace() == original_trace


class TestContextAttachment:
    """Test context attachment functionality."""

    def test_attach_context_to_span_with_attributes(self):
        """Test attaching context with attributes to span."""
        span = Mock(spec=Span)

        # Set context with attributes
        context = TraceContext(attributes={"user_id": "123", "session_id": "abc"})
        set_current_context(context)

        attach_context_to_span(span)

        # Should have called set_attributes with prefixed keys
        span.set_attributes.assert_called_once_with(
            {"context.user_id": "123", "context.session_id": "abc"}
        )

    def test_attach_context_to_span_without_attributes(self):
        """Test attaching context without attributes to span."""
        span = Mock(spec=Span)

        # Set context without attributes
        clear_context()

        attach_context_to_span(span)

        # Should still call set_attributes with empty dict
        span.set_attributes.assert_called_once_with({})

    def test_inherit_context_attributes_from_parent(self):
        """Test inheriting context attributes from parent span."""
        parent_span = Mock(spec=Span)
        parent_span.attributes = {
            "context.user_id": "123",
            "context.session_id": "abc",
            "context.agent_id": "agent-001",
            "other.attribute": "value",
        }

        child_span = Mock(spec=Span)

        inherit_context_attributes(child_span, parent_span)

        # Should inherit only the inheritable attributes
        expected_calls = [
            (("context.user_id", "123"), {}),
            (("context.session_id", "abc"), {}),
            (("context.agent_id", "agent-001"), {}),
        ]

        child_span.set_attribute.assert_has_calls(
            [call(*args, **kwargs) for args, kwargs in expected_calls], any_order=True
        )

    def test_inherit_context_attributes_from_current_span(self):
        """Test inheriting context attributes from current span."""
        current_span = Mock(spec=Span)
        current_span.attributes = {
            "context.user_id": "123",
            "context.request_id": "req-456",
        }

        set_current_span(current_span)

        child_span = Mock(spec=Span)

        inherit_context_attributes(child_span)

        # Should inherit from current span
        expected_calls = [
            (("context.user_id", "123"), {}),
            (("context.request_id", "req-456"), {}),
        ]

        child_span.set_attribute.assert_has_calls(
            [call(*args, **kwargs) for args, kwargs in expected_calls], any_order=True
        )

    def test_inherit_context_attributes_no_parent(self):
        """Test inheriting context attributes with no parent."""
        clear_context()

        child_span = Mock(spec=Span)

        inherit_context_attributes(child_span)

        # Should not call set_attribute
        child_span.set_attribute.assert_not_called()

    def test_inherit_context_attributes_no_inheritable_attributes(self):
        """Test inheriting context attributes with no inheritable attributes."""
        parent_span = Mock(spec=Span)
        parent_span.attributes = {"other.attribute": "value", "span.name": "test-span"}

        child_span = Mock(spec=Span)

        inherit_context_attributes(child_span, parent_span)

        # Should not call set_attribute
        child_span.set_attribute.assert_not_called()


class TestThreadSafety:
    """Test thread safety of context operations."""

    def test_context_isolated_between_threads(self):
        """Test that context is isolated between threads."""
        main_trace = Mock(spec=Trace)
        main_trace.trace_id = "main-trace"

        thread_trace = Mock(spec=Trace)
        thread_trace.trace_id = "thread-trace"

        # Set context in main thread
        set_current_trace(main_trace)

        thread_context = None

        def thread_worker():
            nonlocal thread_context
            # Set different context in thread
            set_current_trace(thread_trace)
            thread_context = get_current_context()

        thread = threading.Thread(target=thread_worker)
        thread.start()
        thread.join()

        # Main thread should still have its context
        assert get_current_trace() == main_trace

        # Thread should have had its own context
        assert thread_context.trace == thread_trace

    def test_propagate_context_between_threads(self):
        """Test propagating context between threads."""
        main_trace = Mock(spec=Trace)
        main_span = Mock(spec=Span)
        main_attributes = {"user_id": "123"}

        # Set context in main thread
        set_current_trace(main_trace)
        set_current_span(main_span)
        context = get_current_context()
        context.attributes.update(main_attributes)

        # Propagate to thread
        propagated_context = propagate_context_to_thread()

        thread_context = None

        def thread_worker():
            nonlocal thread_context
            # Set propagated context in thread
            set_current_context(propagated_context)
            thread_context = get_current_context()

        thread = threading.Thread(target=thread_worker)
        thread.start()
        thread.join()

        # Thread should have the propagated context
        assert thread_context.trace == main_trace
        assert thread_context.span == main_span
        assert thread_context.attributes == main_attributes

    def test_concurrent_context_operations(self):
        """Test concurrent context operations."""
        num_threads = 10
        results = []

        def worker(thread_id):
            # Each thread sets its own context
            trace = Mock(spec=Trace)
            trace.trace_id = f"trace-{thread_id}"

            span = Mock(spec=Span)
            span.span_id = f"span-{thread_id}"

            set_current_trace(trace)
            set_current_span(span)

            # Verify context is correct
            current_trace = get_current_trace()
            current_span = get_current_span()

            results.append((thread_id, current_trace.trace_id, current_span.span_id))

        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Each thread should have had its own context
        assert len(results) == num_threads
        for thread_id, trace_id, span_id in results:
            assert trace_id == f"trace-{thread_id}"
            assert span_id == f"span-{thread_id}"


class TestAsyncContextSupport:
    """Test async context support."""

    @pytest.mark.asyncio
    async def test_async_context_preservation(self):
        """Test that context is preserved across async operations."""
        trace = Mock(spec=Trace)
        span = Mock(spec=Span)

        set_current_trace(trace)
        set_current_span(span)

        # Context should be preserved across async operations
        await asyncio.sleep(0.001)

        assert get_current_trace() == trace
        assert get_current_span() == span

    @pytest.mark.asyncio
    async def test_async_context_managers(self):
        """Test async context managers work correctly."""
        original_trace = Mock(spec=Trace)
        set_current_trace(original_trace)

        new_trace = Mock(spec=Trace)

        async with ContextualTrace(new_trace):
            assert get_current_trace() == new_trace

            # Should work across async operations
            await asyncio.sleep(0.001)
            assert get_current_trace() == new_trace

        # Should restore original
        assert get_current_trace() == original_trace

    @pytest.mark.asyncio
    async def test_async_concurrent_contexts(self):
        """Test concurrent async contexts."""
        # Clear any existing context to prevent interference
        clear_context()

        # Ensure we start with no context
        assert get_current_trace() is None
        assert get_current_span() is None

        async def async_worker(worker_id):
            # Create a new context for this worker
            ctx = contextvars.copy_context()

            def run_in_context():
                trace = Mock(spec=Trace)
                trace.trace_id = f"async-trace-{worker_id}"

                # Set the context for this specific worker
                set_current_trace(trace)
                current_trace = get_current_trace()
                trace_id = current_trace.trace_id if current_trace else None
                return (worker_id, trace_id)

            # Run in the copied context to ensure isolation
            result = ctx.run(run_in_context)
            await asyncio.sleep(0.001)  # Simulate async work
            return result

        # Run multiple concurrent async tasks
        tasks = [async_worker(i) for i in range(5)]
        results = await asyncio.gather(*tasks)

        # Each task should have had its own context
        # Check that we got the right number of results
        assert len(results) == 5

        # Check that each result matches its worker_id
        results_dict = dict(results)
        for worker_id in range(5):
            expected_trace_id = f"async-trace-{worker_id}"
            actual_trace_id = results_dict[worker_id]
            assert (
                actual_trace_id == expected_trace_id
            ), f"Worker {worker_id}: expected {expected_trace_id}, got {actual_trace_id}"

        # Final cleanup
        clear_context()


class TestContextIntegration:
    """Integration tests for context functionality."""

    def test_complete_context_workflow(self):
        """Test complete context workflow."""
        # Start with empty context
        clear_context()

        # Create trace and span
        trace = Mock(spec=Trace)
        trace.trace_id = "integration-trace"

        span = Mock(spec=Span)
        span.span_id = "integration-span"
        span.attributes = {}

        # Set context with attributes
        context = TraceContext(
            trace=trace, span=span, attributes={"user_id": "123", "session_id": "abc"}
        )
        set_current_context(context)

        # Verify context is set
        assert get_current_trace() == trace
        assert get_current_span() == span

        # Attach context to span
        attach_context_to_span(span)

        # Create child span and inherit attributes
        child_span = Mock(spec=Span)
        child_span.attributes = {}

        inherit_context_attributes(child_span, span)

        # Use contextual managers
        with ContextualSpan(child_span):
            assert get_current_span() == child_span

            # Copy and propagate context
            copied_context = copy_context()
            propagated_context = propagate_context_to_thread()

            assert copied_context.trace == trace
            assert propagated_context.trace == trace

        # Should restore original span
        assert get_current_span() == span

    def test_nested_context_managers(self):
        """Test nested context managers."""
        trace1 = Mock(spec=Trace)
        trace1.trace_id = "trace1"

        trace2 = Mock(spec=Trace)
        trace2.trace_id = "trace2"

        span1 = Mock(spec=Span)
        span1.span_id = "span1"

        span2 = Mock(spec=Span)
        span2.span_id = "span2"

        with ContextualTrace(trace1):
            assert get_current_trace() == trace1

            with ContextualSpan(span1):
                assert get_current_span() == span1

                with ContextualTrace(trace2):
                    assert get_current_trace() == trace2

                    with ContextualSpan(span2):
                        assert get_current_span() == span2

                    # Should restore span1
                    assert get_current_span() == span1

                # Should restore trace1
                assert get_current_trace() == trace1

            # Should restore no span
            assert get_current_span() is None

        # Should restore no trace
        assert get_current_trace() is None

    def test_context_with_exception_handling(self):
        """Test context handling with exceptions."""
        original_trace = Mock(spec=Trace)
        original_span = Mock(spec=Span)

        set_current_trace(original_trace)
        set_current_span(original_span)

        new_trace = Mock(spec=Trace)
        new_span = Mock(spec=Span)

        with pytest.raises(ValueError):
            with ContextualTrace(new_trace):
                with ContextualSpan(new_span):
                    raise ValueError("Test exception")

        # Should restore original context
        assert get_current_trace() == original_trace
        assert get_current_span() == original_span

        # Should have recorded exception and finished
        new_trace.set_status.assert_called_once()
        new_trace.finish.assert_called_once()
        new_span.record_exception.assert_called_once()
        new_span.set_status.assert_called_once()
        new_span.finish.assert_called_once()
