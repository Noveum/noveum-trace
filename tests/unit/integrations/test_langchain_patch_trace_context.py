"""
Unit tests for LangChain patch trace context management.

Tests the enhanced _get_or_create_trace_context method with:
- Multi-layered trace lookup strategy
- Root vs child operation handling
- Fallback trace creation scenarios
- Manual trace control interactions
"""

from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

# Skip all tests if LangChain is not available
try:
    from noveum_trace.integrations.langchain import NoveumTraceCallbackHandler

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not available")
class TestGetOrCreateTraceContext:
    """Test enhanced _get_or_create_trace_context method."""

    @pytest.fixture
    def handler(self):
        """Create a callback handler for testing."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            return NoveumTraceCallbackHandler()

    @pytest.fixture
    def mock_context(self):
        """Mock the context module."""
        with (
            patch("noveum_trace.core.context.get_current_trace") as mock_get_trace,
            patch("noveum_trace.core.context.set_current_trace") as mock_set_trace,
        ):
            return {
                "get_current_trace": mock_get_trace,
                "set_current_trace": mock_set_trace,
            }

    def test_get_or_create_trace_context_existing_root_trace(
        self, handler, mock_context
    ):
        """Test using existing root trace from cache."""
        run_id = uuid4()
        parent_run_id = uuid4()
        operation_name = "test_operation"

        # Set up existing root trace
        mock_trace = Mock()
        mock_trace.trace_id = "existing_trace"

        # Mock _find_root_run_id to return a specific root
        root_run_id = uuid4()
        with patch.object(handler, "_find_root_run_id", return_value=root_run_id):
            handler._set_root_trace(root_run_id, mock_trace)

            # Call method
            trace, should_manage = handler._get_or_create_trace_context(
                operation_name, run_id, parent_run_id
            )

        # Should return existing trace and not manage lifecycle
        assert trace == mock_trace
        assert should_manage is False

    def test_get_or_create_trace_context_global_fallback(self, handler, mock_context):
        """Test fallback to global context when no root trace exists."""
        run_id = uuid4()
        parent_run_id = None  # Root call to trigger global fallback path
        operation_name = "test_operation"

        # Mock the actual method that gets called within _get_or_create_trace_context
        mock_global_trace = Mock()
        mock_global_trace.trace_id = "global_trace"

        # Mock _find_root_run_id to return a root with no stored trace
        root_run_id = uuid4()
        with (
            patch.object(handler, "_find_root_run_id", return_value=root_run_id),
            patch(
                "noveum_trace.core.context.get_current_trace",
                return_value=mock_global_trace,
            ),
        ):

            # Ensure no root trace exists for this root_run_id
            assert handler._get_root_trace(root_run_id) is None

            trace, should_manage = handler._get_or_create_trace_context(
                operation_name, run_id, parent_run_id
            )

        # Should return global trace and not manage lifecycle
        assert trace == mock_global_trace
        assert should_manage is False

    def test_get_or_create_trace_context_root_call_creation(
        self, handler, mock_context
    ):
        """Test creating new trace for root call (parent_run_id is None)."""
        run_id = uuid4()
        parent_run_id = None  # Root call
        operation_name = "test_operation"

        # No existing traces
        mock_context["get_current_trace"].return_value = None

        # Mock client to return new trace
        mock_new_trace = Mock()
        mock_new_trace.trace_id = "new_trace"
        handler._client.start_trace.return_value = mock_new_trace

        # Mock _find_root_run_id to return the run_id itself (root)
        with patch.object(handler, "_find_root_run_id", return_value=run_id):
            trace, should_manage = handler._get_or_create_trace_context(
                operation_name, run_id, parent_run_id
            )

        # Should create new trace and manage lifecycle
        assert trace == mock_new_trace
        assert should_manage is True

        # Should have called start_trace
        handler._client.start_trace.assert_called_once_with(operation_name)

        # Should have stored root trace
        assert handler._get_root_trace(run_id) == mock_new_trace

    def test_get_or_create_trace_context_child_call_parent_discovery(
        self, handler, mock_context
    ):
        """Test child call discovering parent's trace."""
        run_id = uuid4()
        parent_run_id = uuid4()
        operation_name = "test_operation"

        # No existing traces in cache or global context
        mock_context["get_current_trace"].return_value = None

        # Set up parent span with trace
        mock_parent_span = Mock()
        mock_parent_trace = Mock()
        mock_parent_trace.trace_id = "parent_trace"
        mock_parent_span.trace = mock_parent_trace

        handler._set_run(parent_run_id, mock_parent_span)

        # Mock _find_root_run_id to return a root
        root_run_id = uuid4()
        with patch.object(handler, "_find_root_run_id", return_value=root_run_id):
            trace, should_manage = handler._get_or_create_trace_context(
                operation_name, run_id, parent_run_id
            )

        # Should return parent's trace and not manage lifecycle
        assert trace == mock_parent_trace
        assert should_manage is False

        # Should have stored root trace
        assert handler._get_root_trace(root_run_id) == mock_parent_trace

    def test_get_or_create_trace_context_child_call_parent_root_discovery(
        self, handler, mock_context
    ):
        """Test child call discovering parent's trace through root trace lookup."""
        run_id = uuid4()
        parent_run_id = uuid4()
        operation_name = "test_operation"

        # No existing traces in cache or global context
        mock_context["get_current_trace"].return_value = None

        # Parent span exists but has no trace attribute
        mock_parent_span = Mock(spec=[])  # No trace attribute
        handler._set_run(parent_run_id, mock_parent_span)

        # But parent has a root trace
        parent_root_id = uuid4()
        mock_parent_trace = Mock()
        mock_parent_trace.trace_id = "parent_root_trace"
        handler._set_root_trace(parent_root_id, mock_parent_trace)

        # Mock _find_root_run_id calls
        root_run_id = uuid4()
        with patch.object(
            handler, "_find_root_run_id", side_effect=[root_run_id, parent_root_id]
        ):
            trace, should_manage = handler._get_or_create_trace_context(
                operation_name, run_id, parent_run_id
            )

        # Should return parent's root trace and not manage lifecycle
        assert trace == mock_parent_trace
        assert should_manage is False

    def test_get_or_create_trace_context_child_call_fallback(
        self, handler, mock_context
    ):
        """Test child call fallback when no parent trace found."""
        run_id = uuid4()
        parent_run_id = uuid4()
        operation_name = "test_operation"

        # No existing traces anywhere
        mock_context["get_current_trace"].return_value = None

        # No parent span exists
        # (handler.runs is empty)

        # Mock client to return fallback trace
        mock_fallback_trace = Mock()
        mock_fallback_trace.trace_id = "fallback_trace"
        handler._client.start_trace.return_value = mock_fallback_trace

        # Mock _find_root_run_id to return a root
        root_run_id = uuid4()
        with patch.object(handler, "_find_root_run_id", return_value=root_run_id):
            trace, should_manage = handler._get_or_create_trace_context(
                operation_name, run_id, parent_run_id
            )

        # Should create fallback trace and manage lifecycle
        assert trace == mock_fallback_trace
        assert should_manage is True

        # Should have called start_trace
        handler._client.start_trace.assert_called_once_with(operation_name)

    def test_get_or_create_trace_context_manual_control_warning(
        self, handler, mock_context
    ):
        """Test warning when manual trace control is enabled but no trace found."""
        run_id = uuid4()
        parent_run_id = None  # Root call
        operation_name = "test_operation"

        # Enable manual trace control
        handler._manual_trace_control = True

        # No existing traces
        mock_context["get_current_trace"].return_value = None

        # Mock client to return new trace
        mock_new_trace = Mock()
        handler._client.start_trace.return_value = mock_new_trace

        # Mock _find_root_run_id to return the run_id itself
        with (
            patch.object(handler, "_find_root_run_id", return_value=run_id),
            patch("noveum_trace.integrations.langchain.logger") as mock_logger,
        ):

            trace, should_manage = handler._get_or_create_trace_context(
                operation_name, run_id, parent_run_id
            )

        # Should have logged warning
        mock_logger.warning.assert_called_once()
        warning_call = mock_logger.warning.call_args[0][0]
        assert "Manual trace control enabled but no trace found" in warning_call

    def test_get_or_create_trace_context_parent_span_trace_extraction(
        self, handler, mock_context
    ):
        """Test extracting trace from parent span with hasattr check."""
        run_id = uuid4()
        parent_run_id = uuid4()
        operation_name = "test_operation"

        # No existing traces in cache or global context
        mock_context["get_current_trace"].return_value = None

        # Test case where parent span exists but has no trace attribute
        mock_parent_span_no_trace = Mock(spec=[])  # No trace attribute
        handler._set_run(parent_run_id, mock_parent_span_no_trace)

        # Mock fallback trace creation
        mock_fallback_trace = Mock()
        handler._client.start_trace.return_value = mock_fallback_trace

        # Mock _find_root_run_id calls (first for child, second for parent lookup)
        root_run_id = uuid4()
        parent_root_id = uuid4()
        with patch.object(
            handler, "_find_root_run_id", side_effect=[root_run_id, parent_root_id]
        ):
            trace, should_manage = handler._get_or_create_trace_context(
                operation_name, run_id, parent_run_id
            )

        # Should create fallback trace since no parent trace found
        assert trace == mock_fallback_trace
        assert should_manage is True

    def test_get_or_create_trace_context_with_warning_logging(
        self, handler, mock_context
    ):
        """Test warning logging for child operation with no parent trace."""
        run_id = uuid4()
        parent_run_id = uuid4()
        operation_name = "test_operation"

        # No existing traces
        mock_context["get_current_trace"].return_value = None

        # Mock fallback trace creation
        mock_fallback_trace = Mock()
        handler._client.start_trace.return_value = mock_fallback_trace

        # Mock _find_root_run_id to return a root
        root_run_id = uuid4()
        with (
            patch.object(handler, "_find_root_run_id", return_value=root_run_id),
            patch("noveum_trace.integrations.langchain.logger") as mock_logger,
        ):

            trace, should_manage = handler._get_or_create_trace_context(
                operation_name, run_id, parent_run_id
            )

        # Should have logged warning about child operation with no parent trace
        mock_logger.warning.assert_called_once()
        warning_call = mock_logger.warning.call_args[0][0]
        assert "Child operation" in warning_call
        assert "has no parent trace" in warning_call
        assert operation_name in warning_call

    def test_get_or_create_trace_context_method_signature(self, handler, mock_context):
        """Test that method accepts the new signature with optional parent_run_id."""
        run_id = uuid4()
        operation_name = "test_operation"

        # Test with no parent_run_id (should default to None)
        mock_trace = Mock()
        handler._client.start_trace.return_value = mock_trace
        mock_context["get_current_trace"].return_value = None

        with patch.object(handler, "_find_root_run_id", return_value=run_id):
            trace, should_manage = handler._get_or_create_trace_context(
                operation_name, run_id
            )

        # Should work with default parent_run_id=None
        assert trace == mock_trace
        assert should_manage is True

    def test_get_or_create_trace_context_return_types(self, handler, mock_context):
        """Test that method returns correct types."""
        run_id = uuid4()
        operation_name = "test_operation"

        mock_trace = Mock()
        handler._client.start_trace.return_value = mock_trace
        mock_context["get_current_trace"].return_value = None

        with patch.object(handler, "_find_root_run_id", return_value=run_id):
            result = handler._get_or_create_trace_context(operation_name, run_id)

        # Should return tuple of (trace, bool)
        assert isinstance(result, tuple)
        assert len(result) == 2

        trace, should_manage = result
        assert trace is not None  # Should be a trace object
        assert isinstance(should_manage, bool)  # Should be boolean
