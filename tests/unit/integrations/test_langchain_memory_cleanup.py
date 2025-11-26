"""
Unit tests for LangChain callback handler memory cleanup functionality.

Tests the memory leak prevention features:
- _find_root_run_id_for_trace() method
- _is_descendant_of() method
- _cleanup_trace_tracking() method
- Integration with _finish_trace_if_needed() and end_trace()
- Memory cleanup in long-running applications
"""

import threading
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

# Skip all tests if LangChain is not available
try:
    # Import directly from the module to avoid issues with other integrations
    from noveum_trace.integrations.langchain.langchain import NoveumTraceCallbackHandler

    LANGCHAIN_AVAILABLE = True
except (ImportError, NameError, AttributeError):
    LANGCHAIN_AVAILABLE = False


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not available")
class TestFindRootRunIdForTrace:
    """Test _find_root_run_id_for_trace method functionality."""

    @pytest.fixture
    def handler(self):
        """Create a callback handler for testing."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            return NoveumTraceCallbackHandler()

    def test_find_root_run_id_for_trace_exists(self, handler):
        """Test finding root_run_id when trace exists in root_traces."""
        root_run_id = uuid4()
        mock_trace = Mock()
        mock_trace.trace_id = "test_trace_123"

        # Store trace in root_traces
        handler._set_root_trace(root_run_id, mock_trace)

        # Should find the root_run_id
        found_root = handler._find_root_run_id_for_trace(mock_trace)
        assert found_root == root_run_id

    def test_find_root_run_id_for_trace_not_exists(self, handler):
        """Test finding root_run_id when trace doesn't exist in root_traces."""
        mock_trace = Mock()
        mock_trace.trace_id = "non_existent_trace"

        # Should return None when trace not found
        found_root = handler._find_root_run_id_for_trace(mock_trace)
        assert found_root is None

    def test_find_root_run_id_for_trace_multiple_traces(self, handler):
        """Test finding correct root_run_id when multiple traces exist."""
        root_run_id_1 = uuid4()
        root_run_id_2 = uuid4()
        root_run_id_3 = uuid4()

        mock_trace_1 = Mock()
        mock_trace_1.trace_id = "trace_1"
        mock_trace_2 = Mock()
        mock_trace_2.trace_id = "trace_2"
        mock_trace_3 = Mock()
        mock_trace_3.trace_id = "trace_3"

        # Store multiple traces
        handler._set_root_trace(root_run_id_1, mock_trace_1)
        handler._set_root_trace(root_run_id_2, mock_trace_2)
        handler._set_root_trace(root_run_id_3, mock_trace_3)

        # Should find the correct root_run_id for trace_2
        found_root = handler._find_root_run_id_for_trace(mock_trace_2)
        assert found_root == root_run_id_2

    def test_find_root_run_id_for_trace_with_string_ids(self, handler):
        """Test finding root_run_id with string IDs."""
        root_run_id = "string_root_id"
        mock_trace = Mock()
        mock_trace.trace_id = "string_trace"

        handler._set_root_trace(root_run_id, mock_trace)
        found_root = handler._find_root_run_id_for_trace(mock_trace)

        assert found_root == root_run_id

    def test_find_root_run_id_for_trace_thread_safety(self, handler):
        """Test thread safety of _find_root_run_id_for_trace."""
        traces_and_roots = []
        for i in range(10):
            root_run_id = f"root_{i}"
            mock_trace = Mock()
            mock_trace.trace_id = f"trace_{i}"
            traces_and_roots.append((root_run_id, mock_trace))
            handler._set_root_trace(root_run_id, mock_trace)

        results = []
        errors = []

        def find_trace_root(root_id, trace):
            try:
                found = handler._find_root_run_id_for_trace(trace)
                results.append((root_id, found))
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for root_id, trace in traces_and_roots:
            thread = threading.Thread(target=find_trace_root, args=(root_id, trace))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Should have no errors
        assert len(errors) == 0

        # Should have 10 results, all correct
        assert len(results) == 10
        for expected_root, found_root in results:
            assert found_root == expected_root


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not available")
class TestIsDescendantOf:
    """Test _is_descendant_of method functionality."""

    @pytest.fixture
    def handler(self):
        """Create a callback handler for testing."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            return NoveumTraceCallbackHandler()

    def test_is_descendant_of_direct_child(self, handler):
        """Test descendant check for direct child."""
        parent_id = uuid4()
        child_id = uuid4()

        # Set up parent relationship
        handler._set_parent(child_id, parent_id)

        # Child should be descendant of parent
        assert handler._is_descendant_of(child_id, parent_id)
        # Parent should not be descendant of child
        assert not handler._is_descendant_of(parent_id, child_id)

    def test_is_descendant_of_deep_chain(self, handler):
        """Test descendant check through deep parent chain."""
        # Create chain: great_grandchild -> grandchild -> child -> parent
        parent_id = uuid4()
        child_id = uuid4()
        grandchild_id = uuid4()
        great_grandchild_id = uuid4()

        # Set up parent relationships
        handler._set_parent(child_id, parent_id)
        handler._set_parent(grandchild_id, child_id)
        handler._set_parent(great_grandchild_id, grandchild_id)

        # All should be descendants of parent
        assert handler._is_descendant_of(child_id, parent_id)
        assert handler._is_descendant_of(grandchild_id, parent_id)
        assert handler._is_descendant_of(great_grandchild_id, parent_id)

        # Intermediate descendants
        assert handler._is_descendant_of(grandchild_id, child_id)
        assert handler._is_descendant_of(great_grandchild_id, child_id)
        assert handler._is_descendant_of(great_grandchild_id, grandchild_id)

    def test_is_descendant_of_no_relationship(self, handler):
        """Test descendant check when no relationship exists."""
        run_id_1 = uuid4()
        run_id_2 = uuid4()

        # No parent relationships set up
        assert not handler._is_descendant_of(run_id_1, run_id_2)
        assert not handler._is_descendant_of(run_id_2, run_id_1)

    def test_is_descendant_of_self_check(self, handler):
        """Test descendant check for self (should be False)."""
        run_id = uuid4()

        # A run should not be a descendant of itself
        assert not handler._is_descendant_of(run_id, run_id)

    def test_is_descendant_of_with_string_ids(self, handler):
        """Test descendant check with string IDs."""
        parent_id = "parent_run"
        child_id = "child_run"

        handler._set_parent(child_id, parent_id)

        assert handler._is_descendant_of(child_id, parent_id)
        assert not handler._is_descendant_of(parent_id, child_id)

    def test_is_descendant_of_orphaned_child(self, handler):
        """Test descendant check when child has no parent."""
        run_id_1 = uuid4()
        run_id_2 = uuid4()

        # Set parent to None (orphaned)
        handler._set_parent(run_id_1, None)

        assert not handler._is_descendant_of(run_id_1, run_id_2)
