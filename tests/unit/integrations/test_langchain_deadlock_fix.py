"""
Unit tests for LangChain callback handler deadlock fix.

Tests the deadlock fix in _cleanup_trace_tracking:
- _is_descendant_of_unlocked() method (lock-free version)
- _cleanup_trace_tracking() uses unlocked version to avoid nested lock acquisition
- Verifies no deadlock occurs during concurrent cleanup operations
"""

import threading
import time
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


@pytest.mark.unit
@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not available")
class TestIsDescendantOfUnlocked:
    """Test _is_descendant_of_unlocked method (lock-free version)."""

    @pytest.fixture
    def handler(self):
        """Create a callback handler for testing."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            return NoveumTraceCallbackHandler()

    def test_direct_child_relationship(self, handler):
        """Test identifying direct parent-child relationship."""
        parent_id = uuid4()
        child_id = uuid4()

        # Build parent map: child -> parent
        parent_map = {child_id: parent_id, parent_id: None}

        # child_id is descendant of parent_id
        assert (
            handler._is_descendant_of_unlocked(child_id, parent_id, parent_map) is True
        )

        # parent_id is NOT descendant of child_id
        assert (
            handler._is_descendant_of_unlocked(parent_id, child_id, parent_map) is False
        )

    def test_deep_descendant_chain(self, handler):
        """Test identifying descendants in a deep chain."""
        root_id = uuid4()
        level1_id = uuid4()
        level2_id = uuid4()
        level3_id = uuid4()

        # Build chain: level3 -> level2 -> level1 -> root
        parent_map = {
            level3_id: level2_id,
            level2_id: level1_id,
            level1_id: root_id,
            root_id: None,
        }

        # All descendants should be identified
        assert (
            handler._is_descendant_of_unlocked(level1_id, root_id, parent_map) is True
        )
        assert (
            handler._is_descendant_of_unlocked(level2_id, root_id, parent_map) is True
        )
        assert (
            handler._is_descendant_of_unlocked(level3_id, root_id, parent_map) is True
        )

        # Also check intermediate relationships
        assert (
            handler._is_descendant_of_unlocked(level3_id, level1_id, parent_map) is True
        )
        assert (
            handler._is_descendant_of_unlocked(level3_id, level2_id, parent_map) is True
        )

    def test_not_descendant_different_branches(self, handler):
        """Test that nodes in different branches are not descendants."""
        root_id = uuid4()
        branch1_id = uuid4()
        branch2_id = uuid4()

        # Two separate branches from root
        parent_map = {
            branch1_id: root_id,
            branch2_id: root_id,
            root_id: None,
        }

        # Siblings are not descendants of each other
        assert (
            handler._is_descendant_of_unlocked(branch1_id, branch2_id, parent_map)
            is False
        )
        assert (
            handler._is_descendant_of_unlocked(branch2_id, branch1_id, parent_map)
            is False
        )

    def test_nonexistent_run_id(self, handler):
        """Test handling of non-existent run IDs."""
        existing_id = uuid4()
        nonexistent_id = uuid4()

        parent_map = {existing_id: None}

        # Checking non-existent ID should not crash
        assert (
            handler._is_descendant_of_unlocked(nonexistent_id, existing_id, parent_map)
            is False
        )


@pytest.mark.unit
@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not available")
class TestCleanupTraceTrackingDeadlockFix:
    """Test that _cleanup_trace_tracking doesn't deadlock."""

    @pytest.fixture
    def handler(self):
        """Create a callback handler for testing."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            return NoveumTraceCallbackHandler()

    @pytest.mark.timeout(5)
    def test_cleanup_uses_unlocked_version(self, handler):
        """Test that cleanup uses _is_descendant_of_unlocked to avoid deadlock."""
        root_id = uuid4()
        child1_id = uuid4()
        child2_id = uuid4()
        grandchild_id = uuid4()

        mock_trace = Mock()
        mock_trace.trace_id = "test_trace"

        # Set up hierarchy: root -> child1 -> grandchild, root -> child2
        handler._set_root_trace(root_id, mock_trace)
        handler._set_parent(child1_id, root_id)
        handler._set_parent(child2_id, root_id)
        handler._set_parent(grandchild_id, child1_id)

        # Run cleanup - should NOT deadlock
        handler._cleanup_trace_tracking(root_id)

        # Verify all entries were cleaned up
        assert len(handler.root_traces) == 0
        assert len(handler.parent_map) == 0

    def test_cleanup_with_deep_hierarchy(self, handler):
        """Test cleanup with deep hierarchy doesn't deadlock."""
        root_id = uuid4()

        # Create a deep hierarchy
        ids = [root_id]
        for _i in range(10):
            child_id = uuid4()
            handler._set_parent(child_id, ids[-1])
            ids.append(child_id)

        mock_trace = Mock()
        handler._set_root_trace(root_id, mock_trace)

        # Cleanup should handle deep hierarchy without deadlock
        handler._cleanup_trace_tracking(root_id)

        # All entries should be cleaned
        assert len(handler.parent_map) == 0

    def test_concurrent_cleanup_no_deadlock(self, handler):
        """Test that concurrent cleanup operations don't deadlock."""
        root1_id = uuid4()
        root2_id = uuid4()

        # Set up two separate trace hierarchies
        for root_id in [root1_id, root2_id]:
            mock_trace = Mock()
            handler._set_root_trace(root_id, mock_trace)

            for _i in range(5):
                child_id = uuid4()
                handler._set_parent(child_id, root_id)

        # Track completion
        completed = []
        errors = []

        def cleanup_task(root_id):
            try:
                handler._cleanup_trace_tracking(root_id)
                completed.append(root_id)
            except Exception as e:
                errors.append(e)

        # Run cleanup concurrently
        thread1 = threading.Thread(target=cleanup_task, args=(root1_id,))
        thread2 = threading.Thread(target=cleanup_task, args=(root2_id,))

        thread1.start()
        thread2.start()

        # Wait with timeout - should complete quickly
        thread1.join(timeout=5.0)
        thread2.join(timeout=5.0)

        # Verify both completed without deadlock
        assert len(completed) == 2
        assert len(errors) == 0
        assert not thread1.is_alive()
        assert not thread2.is_alive()

    def test_cleanup_during_active_queries(self, handler):
        """Test cleanup doesn't deadlock when other threads are querying parent_map."""
        root_id = uuid4()
        mock_trace = Mock()
        handler._set_root_trace(root_id, mock_trace)

        # Create hierarchy
        child_ids = []
        for _i in range(20):
            child_id = uuid4()
            handler._set_parent(child_id, root_id)
            child_ids.append(child_id)

        query_running = threading.Event()
        cleanup_complete = threading.Event()
        errors = []

        def query_parents():
            """Continuously query parent relationships."""
            query_running.set()
            try:
                while not cleanup_complete.is_set():
                    for child_id in child_ids[:10]:  # Only query first 10
                        try:
                            handler._get_parent(child_id)
                        except KeyError:
                            pass  # Expected if already cleaned up
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def cleanup_task():
            """Run cleanup."""
            try:
                query_running.wait(timeout=1.0)
                handler._cleanup_trace_tracking(root_id)
                cleanup_complete.set()
            except Exception as e:
                errors.append(e)
                cleanup_complete.set()

        # Start query thread
        query_thread = threading.Thread(target=query_parents)
        cleanup_thread = threading.Thread(target=cleanup_task)

        query_thread.start()
        cleanup_thread.start()

        # Wait for completion with timeout
        cleanup_thread.join(timeout=5.0)
        cleanup_complete.set()
        query_thread.join(timeout=5.0)

        # Should complete without deadlock or errors
        assert not cleanup_thread.is_alive()
        assert not query_thread.is_alive()
        assert len(errors) == 0


@pytest.mark.unit
@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not available")
class TestCleanupCorrectness:
    """Test that cleanup correctly removes all descendants."""

    @pytest.fixture
    def handler(self):
        """Create a callback handler for testing."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            return NoveumTraceCallbackHandler()

    def test_cleanup_removes_all_descendants(self, handler):
        """Test that cleanup removes root and all descendants from parent_map."""
        root_id = uuid4()
        child1_id = uuid4()
        child2_id = uuid4()
        grandchild1_id = uuid4()
        grandchild2_id = uuid4()

        # Unrelated tree that should NOT be cleaned up
        other_root_id = uuid4()
        other_child_id = uuid4()

        mock_trace = Mock()
        handler._set_root_trace(root_id, mock_trace)

        # Set up first tree
        handler._set_parent(child1_id, root_id)
        handler._set_parent(child2_id, root_id)
        handler._set_parent(grandchild1_id, child1_id)
        handler._set_parent(grandchild2_id, child2_id)

        # Set up separate tree
        handler._set_parent(other_child_id, other_root_id)

        # Cleanup first tree
        handler._cleanup_trace_tracking(root_id)

        # First tree should be completely removed
        assert root_id not in handler.root_traces
        assert root_id not in handler.parent_map
        assert child1_id not in handler.parent_map
        assert child2_id not in handler.parent_map
        assert grandchild1_id not in handler.parent_map
        assert grandchild2_id not in handler.parent_map

        # Other tree should remain
        assert other_child_id in handler.parent_map
        assert handler._get_parent(other_child_id) == other_root_id

    def test_cleanup_logs_correct_count(self, handler):
        """Test that cleanup logs the correct number of cleaned entries."""
        root_id = uuid4()
        mock_trace = Mock()
        mock_trace.trace_id = "test_trace_123"

        handler._set_root_trace(root_id, mock_trace)

        # Create 5 children
        for _i in range(5):
            child_id = uuid4()
            handler._set_parent(child_id, root_id)

        # Cleanup should remove 5 entries (not counting root since it has no parent entry)
        with patch(
            "noveum_trace.integrations.langchain.langchain.logger"
        ) as mock_logger:
            handler._cleanup_trace_tracking(root_id)

            # Check that debug was called with correct count
            # Should be 5 children entries
            assert mock_logger.debug.called
            call_args = mock_logger.debug.call_args[0][0]
            assert "cleaned 5 parent_map entries" in call_args
