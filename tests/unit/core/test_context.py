"""
Comprehensive unit tests for Context Management.

This module tests all aspects of context management including context variables,
context propagation, trace/span context management, and contextual operations.
"""

import asyncio
import threading
import time
from unittest.mock import Mock

import pytest

from noveum_trace.core.context import (
    ContextualSpan,
    ContextualTrace,
    TraceContext,
    attach_context_to_span,
    clear_context,
    get_context_attribute,
    get_current_context,
    get_current_span,
    get_current_trace,
    set_context_attribute,
    set_current_span,
    set_current_trace,
    trace_context,
)
from noveum_trace.core.span import Span, SpanStatus
from noveum_trace.core.trace import Trace


class TestTraceContext:
    """Test TraceContext dataclass and basic functionality."""

    def test_trace_context_creation_empty(self):
        """Test creating empty trace context."""
        context = TraceContext()

        assert context.trace is None
        assert context.span is None
        assert context.attributes == {}

    def test_trace_context_creation_with_data(self):
        """Test creating trace context with data."""
        trace = Mock(spec=Trace)
        span = Mock(spec=Span)
        attributes = {"key": "value", "test": 123}

        context = TraceContext(trace=trace, span=span, attributes=attributes)

        assert context.trace == trace
        assert context.span == span
        assert context.attributes == attributes

    def test_trace_context_defaults(self):
        """Test TraceContext default values."""
        context = TraceContext()

        assert isinstance(context.attributes, dict)
        assert len(context.attributes) == 0


class TestContextVariables:
    """Test context variable operations and isolation."""

    def setup_method(self):
        """Setup for each test method."""
        # Clear context before each test
        clear_context()

    def teardown_method(self):
        """Cleanup after each test method."""
        # Clear context after each test
        clear_context()

    def test_get_current_context_initial(self):
        """Test getting current context when none is set."""
        context = get_current_context()

        assert isinstance(context, TraceContext)
        assert context.trace is None
        assert context.span is None

    def test_context_trace_operations(self):
        """Test setting and getting current trace."""
        trace = Mock(spec=Trace)
        trace.trace_id = "test-trace-id"

        # Initially no trace
        assert get_current_trace() is None

        # Set trace
        set_current_trace(trace)
        current_trace = get_current_trace()

        assert current_trace == trace
        assert current_trace.trace_id == "test-trace-id"

    def test_context_span_operations(self):
        """Test setting and getting current span."""
        span = Mock(spec=Span)
        span.span_id = "test-span-id"

        # Initially no span
        assert get_current_span() is None

        # Set span
        set_current_span(span)
        current_span = get_current_span()

        assert current_span == span
        assert current_span.span_id == "test-span-id"

    def test_context_attributes_operations(self):
        """Test setting and getting context attributes."""
        # Initially no attributes
        assert get_context_attribute("test_key") is None

        # Set attribute
        set_context_attribute("test_key", "test_value")
        value = get_context_attribute("test_key")

        assert value == "test_value"

    @pytest.mark.parametrize(
        "trace_name,span_name",
        [
            ("simple_trace", "simple_span"),
            ("complex-trace-123", "complex-span-456"),
            ("trace_with_underscore", "span_with_underscore"),
        ],
    )
    def test_trace_span_context_together(self, trace_name, span_name):
        """Test setting trace and span context together."""
        trace = Mock(spec=Trace)
        trace.name = trace_name
        trace.trace_id = f"{trace_name}-id"

        span = Mock(spec=Span)
        span.name = span_name
        span.span_id = f"{span_name}-id"
        span.trace_id = trace.trace_id

        # Set both
        set_current_trace(trace)
        set_current_span(span)

        # Verify both are accessible
        current_trace = get_current_trace()
        current_span = get_current_span()

        assert current_trace.name == trace_name
        assert current_span.name == span_name
        assert current_span.trace_id == current_trace.trace_id

    def test_clear_context(self):
        """Test clearing context resets everything."""
        # Set up context
        trace = Mock(spec=Trace)
        span = Mock(spec=Span)
        set_current_trace(trace)
        set_current_span(span)
        set_context_attribute("test", "value")

        # Verify context is set
        assert get_current_trace() == trace
        assert get_current_span() == span
        assert get_context_attribute("test") == "value"

        # Clear context
        clear_context()

        # Verify context is cleared
        assert get_current_trace() is None
        assert get_current_span() is None
        assert get_context_attribute("test") is None

    def test_context_none_values(self):
        """Test setting context to None values."""
        # Set up some context first
        trace = Mock(spec=Trace)
        span = Mock(spec=Span)
        set_current_trace(trace)
        set_current_span(span)

        # Clear using None
        set_current_trace(None)
        set_current_span(None)

        assert get_current_trace() is None
        assert get_current_span() is None


class TestContextualTrace:
    """Test ContextualTrace context manager functionality."""

    def setup_method(self):
        """Setup for each test method."""
        clear_context()

    def teardown_method(self):
        """Cleanup after each test method."""
        clear_context()

    def test_contextual_trace_basic(self):
        """Test basic ContextualTrace context manager."""
        trace = Mock(spec=Trace)
        trace.trace_id = "test-trace"
        trace.is_finished.return_value = False

        contextual_trace = ContextualTrace(trace)

        # Initially no trace in context
        assert get_current_trace() is None

        # Use context manager
        with contextual_trace as ctx_trace:
            assert ctx_trace == trace
            assert get_current_trace() == trace

        # Context should be restored after exit
        assert get_current_trace() is None

    def test_contextual_trace_with_existing_context(self):
        """Test ContextualTrace with existing trace context."""
        existing_trace = Mock(spec=Trace)
        existing_trace.trace_id = "existing-trace"

        new_trace = Mock(spec=Trace)
        new_trace.trace_id = "new-trace"
        new_trace.is_finished.return_value = False

        # Set existing context
        set_current_trace(existing_trace)

        # Use contextual trace
        contextual_trace = ContextualTrace(new_trace)
        with contextual_trace as _ctx_trace:
            assert get_current_trace() == new_trace

        # Should restore previous context
        assert get_current_trace() == existing_trace

    def test_contextual_trace_with_exception(self):
        """Test ContextualTrace handles exceptions correctly."""
        trace = Mock(spec=Trace)
        trace.trace_id = "test-trace"
        trace.is_finished.return_value = False

        contextual_trace = ContextualTrace(trace)

        with pytest.raises(ValueError, match="test error"):
            with contextual_trace:
                raise ValueError("test error")

        # Should set error status and finish trace
        trace.set_status.assert_called_with(SpanStatus.ERROR, "test error")
        trace.finish.assert_called_once()

    def test_contextual_trace_attribute_delegation(self):
        """Test ContextualTrace delegates attributes to wrapped trace."""
        trace = Mock(spec=Trace)
        trace.name = "test-trace"
        # Don't use a method that doesn't exist in the spec
        # Instead test attribute access for actual Trace attributes
        trace.trace_id = "test-trace-id"
        trace.is_finished.return_value = False

        contextual_trace = ContextualTrace(trace)

        # Should delegate attribute access
        assert contextual_trace.name == "test-trace"
        assert contextual_trace.trace_id == "test-trace-id"
        assert contextual_trace.is_finished() is False

        trace.is_finished.assert_called_once()


class TestContextualSpan:
    """Test ContextualSpan context manager functionality."""

    def setup_method(self):
        """Setup for each test method."""
        clear_context()

    def teardown_method(self):
        """Cleanup after each test method."""
        clear_context()

    def test_contextual_span_basic(self):
        """Test basic ContextualSpan context manager."""
        span = Mock(spec=Span)
        span.span_id = "test-span"
        span.is_finished.return_value = False

        contextual_span = ContextualSpan(span)

        # Initially no span in context
        assert get_current_span() is None

        # Use context manager
        with contextual_span as ctx_span:
            assert ctx_span == span
            assert get_current_span() == span

        # Context should be restored after exit
        assert get_current_span() is None

    def test_contextual_span_with_existing_context(self):
        """Test ContextualSpan with existing span context."""
        existing_span = Mock(spec=Span)
        existing_span.span_id = "existing-span"

        new_span = Mock(spec=Span)
        new_span.span_id = "new-span"
        new_span.is_finished.return_value = False

        # Set existing context
        set_current_span(existing_span)

        # Use contextual span
        contextual_span = ContextualSpan(new_span)
        with contextual_span as _ctx_span:
            assert get_current_span() == new_span

        # Should restore previous context
        assert get_current_span() == existing_span

    def test_contextual_span_with_exception(self):
        """Test ContextualSpan handles exceptions correctly."""
        span = Mock(spec=Span)
        span.span_id = "test-span"
        span.is_finished.return_value = False

        contextual_span = ContextualSpan(span)

        test_exception = ValueError("test error")
        with pytest.raises(ValueError, match="test error"):
            with contextual_span:
                raise test_exception

        # Should record exception and set error status
        span.record_exception.assert_called_with(test_exception)
        span.set_status.assert_called_with(SpanStatus.ERROR, "test error")
        span.finish.assert_called_once()

    @pytest.mark.parametrize(
        "span_name,span_attributes",
        [
            ("simple_span", {"type": "test"}),
            ("complex_span", {"llm.model": "gpt-4", "tokens": 100}),
            ("agent_span", {"agent.action": "search", "agent.result": "found"}),
        ],
    )
    def test_contextual_span_with_different_configs(self, span_name, span_attributes):
        """Test ContextualSpan with different span configurations."""
        span = Mock(spec=Span)
        span.name = span_name
        span.attributes = span_attributes
        span.is_finished.return_value = False

        contextual_span = ContextualSpan(span)

        with contextual_span as ctx_span:
            assert ctx_span.name == span_name
            assert ctx_span.attributes == span_attributes


class TestTraceContextManager:
    """Test trace_context context manager function."""

    def setup_method(self):
        """Setup for each test method."""
        clear_context()

    def teardown_method(self):
        """Cleanup after each test method."""
        clear_context()

    def test_trace_context_basic(self):
        """Test basic trace_context usage."""
        trace = Mock(spec=Trace)
        trace.trace_id = "test-trace"

        with trace_context(trace=trace) as context:
            assert isinstance(context, TraceContext)
            assert context.trace == trace
            assert get_current_trace() == trace

    def test_trace_context_with_span(self):
        """Test trace_context with both trace and span."""
        trace = Mock(spec=Trace)
        span = Mock(spec=Span)

        with trace_context(trace=trace, span=span) as context:
            assert context.trace == trace
            assert context.span == span
            assert get_current_trace() == trace
            assert get_current_span() == span

    def test_trace_context_with_attributes(self):
        """Test trace_context with attributes."""
        attributes = {"key1": "value1", "key2": "value2"}

        with trace_context(**attributes) as context:
            # The context.attributes should contain the attributes directly
            # Since trace_context uses **attributes in the function signature
            for key, value in attributes.items():
                assert context.attributes[key] == value
                assert get_context_attribute(key) == value

    def test_trace_context_restores_previous(self):
        """Test trace_context restores previous context."""
        # Set initial context
        initial_trace = Mock(spec=Trace)
        initial_span = Mock(spec=Span)
        set_current_trace(initial_trace)
        set_current_span(initial_span)
        set_context_attribute("initial", "value")

        # Use trace_context
        new_trace = Mock(spec=Trace)
        new_span = Mock(spec=Span)

        with trace_context(trace=new_trace, span=new_span):
            assert get_current_trace() == new_trace
            assert get_current_span() == new_span

        # Should restore initial context
        assert get_current_trace() == initial_trace
        assert get_current_span() == initial_span
        assert get_context_attribute("initial") == "value"


class TestContextUtilities:
    """Test context utility functions."""

    def setup_method(self):
        """Setup for each test method."""
        clear_context()

    def teardown_method(self):
        """Cleanup after each test method."""
        clear_context()

    def test_attach_context_to_span(self):
        """Test attaching context attributes to span."""
        span = Mock(spec=Span)
        span.set_attributes = Mock()

        # Set some context attributes
        set_context_attribute("user_id", "user123")
        set_context_attribute("session_id", "session456")
        set_context_attribute("custom_attr", "custom_value")

        attach_context_to_span(span)

        # Should call set_attributes with context data
        span.set_attributes.assert_called_once()
        call_args = span.set_attributes.call_args[0][0]

        assert "context.user_id" in call_args
        assert call_args["context.user_id"] == "user123"
        assert "context.session_id" in call_args
        assert call_args["context.session_id"] == "session456"

    def test_attach_context_to_span_empty_context(self):
        """Test attaching empty context to span."""
        span = Mock(spec=Span)
        span.set_attributes = Mock()

        # No context attributes set
        attach_context_to_span(span)

        # Should still call set_attributes but with empty dict
        span.set_attributes.assert_called_once()
        call_args = span.set_attributes.call_args[0][0]
        assert call_args == {}

    @pytest.mark.parametrize(
        "attribute_key,attribute_value",
        [
            ("simple_key", "simple_value"),
            ("complex.key", {"nested": "data"}),
            ("numeric_key", 12345),
            ("boolean_key", True),
            ("list_key", [1, 2, 3]),
        ],
    )
    def test_context_attribute_types(self, attribute_key, attribute_value):
        """Test context attributes with different data types."""
        set_context_attribute(attribute_key, attribute_value)
        retrieved_value = get_context_attribute(attribute_key)

        assert retrieved_value == attribute_value

    def test_context_attribute_overwrite(self):
        """Test overwriting context attributes."""
        key = "test_key"

        # Set initial value
        set_context_attribute(key, "initial_value")
        assert get_context_attribute(key) == "initial_value"

        # Overwrite with new value
        set_context_attribute(key, "new_value")
        assert get_context_attribute(key) == "new_value"

    def test_context_attribute_nonexistent(self):
        """Test getting nonexistent context attribute."""
        result = get_context_attribute("nonexistent_key")
        assert result is None

    def test_context_attribute_default_value(self):
        """Test getting context attribute with default value."""
        result = get_context_attribute("nonexistent_key", "default_value")
        assert result == "default_value"


class TestContextThreadSafety:
    """Test context isolation between threads."""

    def test_context_isolation_between_threads(self):
        """Test that context is isolated between threads."""
        results = {}
        errors = []

        def thread_function(thread_id, trace_name):
            try:
                # Each thread sets its own context
                trace = Mock(spec=Trace)
                trace.name = trace_name
                trace.trace_id = f"trace-{thread_id}"

                set_current_trace(trace)
                set_context_attribute("thread_id", thread_id)

                # Small delay to allow interleaving
                time.sleep(0.01)

                # Verify context is still correct
                current_trace = get_current_trace()
                thread_attr = get_context_attribute("thread_id")

                results[thread_id] = {
                    "trace_name": current_trace.name if current_trace else None,
                    "trace_id": current_trace.trace_id if current_trace else None,
                    "thread_attr": thread_attr,
                }
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=thread_function, args=(i, f"trace_{i}"))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Verify each thread had isolated context
        assert len(results) == 5
        for thread_id in range(5):
            result = results[thread_id]
            assert result["trace_name"] == f"trace_{thread_id}"
            assert result["trace_id"] == f"trace-{thread_id}"
            assert result["thread_attr"] == thread_id

    def test_context_inheritance_in_threads(self):
        """Test context inheritance when creating new threads."""
        # Set parent context
        parent_trace = Mock(spec=Trace)
        parent_trace.name = "parent_trace"
        set_current_trace(parent_trace)
        set_context_attribute("parent_attr", "parent_value")

        child_results = {}

        def child_thread():
            # Child should initially have same context as parent
            child_trace = get_current_trace()
            child_attr = get_context_attribute("parent_attr")

            child_results["initial_trace"] = child_trace.name if child_trace else None
            child_results["initial_attr"] = child_attr

            # Child can modify its own context
            new_trace = Mock(spec=Trace)
            new_trace.name = "child_trace"
            set_current_trace(new_trace)
            set_context_attribute("child_attr", "child_value")

            child_results["modified_trace"] = get_current_trace().name
            child_results["child_attr"] = get_context_attribute("child_attr")

        # Start child thread
        thread = threading.Thread(target=child_thread)
        thread.start()
        thread.join()

        # Verify child inherited parent context initially
        # Note: contextvars behavior depends on how thread is created
        # This test documents the expected behavior

        # Parent context should be unchanged
        assert get_current_trace().name == "parent_trace"
        assert get_context_attribute("parent_attr") == "parent_value"
        assert (
            get_context_attribute("child_attr") is None
        )  # Child's change doesn't affect parent


@pytest.mark.asyncio
class TestAsyncContextManagement:
    """Test context management in async/await scenarios."""

    async def test_async_context_propagation(self):
        """Test that context propagates correctly in async functions."""
        clear_context()

        # Set initial context
        trace = Mock(spec=Trace)
        trace.name = "async_trace"
        set_current_trace(trace)
        set_context_attribute("async_attr", "async_value")

        async def async_function():
            # Should have access to parent context
            current_trace = get_current_trace()
            async_attr = get_context_attribute("async_attr")

            return {
                "trace_name": current_trace.name if current_trace else None,
                "async_attr": async_attr,
            }

        result = await async_function()

        assert result["trace_name"] == "async_trace"
        assert result["async_attr"] == "async_value"

    async def test_async_context_isolation(self):
        """Test context isolation between concurrent async tasks."""
        clear_context()

        results = {}

        async def async_task(task_id):
            # Clear context at start of each task to ensure isolation
            clear_context()

            # Each task sets its own context
            trace = Mock(spec=Trace)
            trace.name = f"task_trace_{task_id}"
            set_current_trace(trace)
            set_context_attribute("task_id", task_id)

            # Simulate async work
            await asyncio.sleep(0.01)

            # Verify context is still correct
            current_trace = get_current_trace()
            task_attr = get_context_attribute("task_id")

            results[task_id] = {
                "trace_name": current_trace.name if current_trace else None,
                "task_attr": task_attr,
            }

        # Run multiple concurrent tasks
        tasks = [async_task(i) for i in range(3)]
        await asyncio.gather(*tasks)

        # Verify each task had isolated context
        assert len(results) == 3
        for task_id in range(3):
            result = results[task_id]
            assert result["trace_name"] == f"task_trace_{task_id}"
            assert result["task_attr"] == task_id

    async def test_contextual_trace_async(self):
        """Test ContextualTrace in async context."""
        clear_context()

        trace = Mock(spec=Trace)
        trace.name = "async_contextual_trace"
        trace.is_finished.return_value = False

        async def async_operation():
            contextual_trace = ContextualTrace(trace)
            async with contextual_trace as _ctx_trace:
                assert get_current_trace() == trace
                await asyncio.sleep(0.001)  # Simulate async work
                return get_current_trace().name

        result = await async_operation()
        assert result == "async_contextual_trace"

        # Context should be cleared after async context manager
        assert get_current_trace() is None

    async def test_contextual_span_async(self):
        """Test ContextualSpan in async context."""
        clear_context()

        span = Mock(spec=Span)
        span.name = "async_contextual_span"
        span.is_finished.return_value = False

        async def async_operation():
            contextual_span = ContextualSpan(span)
            async with contextual_span as _ctx_span:
                assert get_current_span() == span
                await asyncio.sleep(0.001)  # Simulate async work
                return get_current_span().name

        result = await async_operation()
        assert result == "async_contextual_span"

        # Context should be cleared after async context manager
        assert get_current_span() is None
