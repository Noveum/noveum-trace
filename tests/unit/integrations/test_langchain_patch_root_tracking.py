"""
Unit tests for LangChain patch root trace tracking functionality.

Tests the new root trace tracking and parent relationship management:
- Root trace storage and retrieval
- Parent relationship mapping
- Thread safety of tracking operations
- Parent chain traversal and cycle detection
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


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not available")
class TestRootTraceTracking:
    """Test root trace tracking functionality."""

    @pytest.fixture
    def handler(self):
        """Create a callback handler for testing."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            return NoveumTraceCallbackHandler()

    def test_set_get_root_trace(self, handler):
        """Test basic root trace storage and retrieval."""
        root_run_id = uuid4()
        mock_trace = Mock()
        mock_trace.trace_id = "test_trace_id"

        # Initially should return None
        assert handler._get_root_trace(root_run_id) is None

        # Set root trace
        handler._set_root_trace(root_run_id, mock_trace)

        # Should now return the trace
        retrieved_trace = handler._get_root_trace(root_run_id)
        assert retrieved_trace == mock_trace
        assert retrieved_trace.trace_id == "test_trace_id"

    def test_root_trace_with_string_id(self, handler):
        """Test root trace storage with string ID."""
        root_run_id = "string_run_id"
        mock_trace = Mock()

        handler._set_root_trace(root_run_id, mock_trace)
        retrieved_trace = handler._get_root_trace(root_run_id)

        assert retrieved_trace == mock_trace

    def test_root_trace_overwrite(self, handler):
        """Test overwriting existing root trace."""
        root_run_id = uuid4()
        mock_trace1 = Mock()
        mock_trace1.trace_id = "trace1"
        mock_trace2 = Mock()
        mock_trace2.trace_id = "trace2"

        # Set first trace
        handler._set_root_trace(root_run_id, mock_trace1)
        assert handler._get_root_trace(root_run_id).trace_id == "trace1"

        # Overwrite with second trace
        handler._set_root_trace(root_run_id, mock_trace2)
        assert handler._get_root_trace(root_run_id).trace_id == "trace2"

    def test_root_trace_thread_safety(self, handler):
        """Test thread safety of root trace operations."""
        root_run_id = uuid4()
        results = []
        errors = []

        def set_and_get_trace(trace_id):
            try:
                mock_trace = Mock()
                mock_trace.trace_id = trace_id

                # Set trace
                handler._set_root_trace(root_run_id, mock_trace)

                # Small delay to increase chance of race conditions
                time.sleep(0.001)

                # Get trace
                retrieved = handler._get_root_trace(root_run_id)
                results.append(retrieved.trace_id if retrieved else None)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=set_and_get_trace, args=(f"trace_{i}",))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Should have no errors
        assert len(errors) == 0

        # Should have 10 results
        assert len(results) == 10

        # All results should be valid trace IDs (not None)
        assert all(result is not None for result in results)


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not available")
class TestParentRelationshipMapping:
    """Test parent relationship mapping functionality."""

    @pytest.fixture
    def handler(self):
        """Create a callback handler for testing."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            return NoveumTraceCallbackHandler()

    def test_set_get_parent_relationship(self, handler):
        """Test basic parent relationship storage and retrieval."""
        run_id = uuid4()
        parent_run_id = uuid4()

        # Initially should return None
        assert handler._get_parent(run_id) is None

        # Set parent relationship
        handler._set_parent(run_id, parent_run_id)

        # Should now return the parent
        retrieved_parent = handler._get_parent(run_id)
        assert retrieved_parent == parent_run_id

    def test_parent_relationship_with_none_parent(self, handler):
        """Test parent relationship with None parent (root node)."""
        run_id = uuid4()

        # Set parent as None (root node)
        handler._set_parent(run_id, None)

        # Should return None
        assert handler._get_parent(run_id) is None

    def test_parent_relationship_with_string_ids(self, handler):
        """Test parent relationship with string IDs."""
        run_id = "child_run"
        parent_run_id = "parent_run"

        handler._set_parent(run_id, parent_run_id)
        retrieved_parent = handler._get_parent(run_id)

        assert retrieved_parent == parent_run_id

    def test_parent_map_thread_safety(self, handler):
        """Test thread safety of parent map operations."""
        base_run_id = "run_"
        base_parent_id = "parent_"
        results = []
        errors = []

        def set_and_get_parent(index):
            try:
                run_id = f"{base_run_id}{index}"
                parent_id = f"{base_parent_id}{index}"

                # Set parent
                handler._set_parent(run_id, parent_id)

                # Small delay to increase chance of race conditions
                time.sleep(0.001)

                # Get parent
                retrieved = handler._get_parent(run_id)
                results.append((run_id, retrieved))
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=set_and_get_parent, args=(i,))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Should have no errors
        assert len(errors) == 0

        # Should have 10 results
        assert len(results) == 10

        # Each result should match expected parent
        for run_id, retrieved_parent in results:
            index = run_id.split("_")[1]
            expected_parent = f"{base_parent_id}{index}"
            assert retrieved_parent == expected_parent


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not available")
class TestFindRootRunId:
    """Test _find_root_run_id method functionality."""

    @pytest.fixture
    def handler(self):
        """Create a callback handler for testing."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            return NoveumTraceCallbackHandler()

    def test_find_root_run_id_no_parent(self, handler):
        """Test finding root when there's no parent (root node)."""
        run_id = uuid4()

        root_id = handler._find_root_run_id(run_id, None)

        # Should return the run_id itself as it's the root
        assert root_id == run_id

        # Should have stored the parent relationship
        assert handler._get_parent(run_id) is None

    def test_find_root_run_id_with_parent_chain(self, handler):
        """Test finding root through a parent chain."""
        # Create a chain: child -> parent -> grandparent (root)
        child_id = uuid4()
        parent_id = uuid4()
        grandparent_id = uuid4()

        # Set up existing parent relationships
        handler._set_parent(parent_id, grandparent_id)
        handler._set_parent(grandparent_id, None)  # grandparent is root

        # Find root for child
        root_id = handler._find_root_run_id(child_id, parent_id)

        # Should return grandparent as root
        assert root_id == grandparent_id

        # Should have stored child's parent relationship
        assert handler._get_parent(child_id) == parent_id

    def test_find_root_run_id_existing_root_trace(self, handler):
        """Test finding root when a root trace already exists in the chain."""
        child_id = uuid4()
        parent_id = uuid4()
        grandparent_id = uuid4()

        # Set up parent chain
        handler._set_parent(parent_id, grandparent_id)

        # Set up existing root trace for parent
        mock_trace = Mock()
        handler._set_root_trace(parent_id, mock_trace)

        # Find root for child
        root_id = handler._find_root_run_id(child_id, parent_id)

        # Should return parent_id as it has a root trace
        assert root_id == parent_id

    def test_find_root_run_id_cycle_detection(self, handler):
        """Test cycle detection in parent chain traversal."""
        run_a = uuid4()
        run_b = uuid4()
        run_c = uuid4()

        # Create a cycle: A -> B -> C -> A
        handler._set_parent(run_b, run_c)
        handler._set_parent(run_c, run_a)

        # This should not cause infinite loop
        root_id = handler._find_root_run_id(run_a, run_b)

        # Should return one of the nodes in the cycle (the algorithm stops when it detects the cycle)
        assert root_id in [run_a, run_b, run_c]

        # Should have stored the parent relationship
        assert handler._get_parent(run_a) == run_b

    def test_find_root_run_id_deep_chain(self, handler):
        """Test finding root in a deep parent chain."""
        # Create a deep chain: node0 -> node1 -> ... -> node9 -> root
        nodes = [uuid4() for _ in range(11)]  # 0-9 are chain, 10 is root

        # Set up parent relationships (each points to next, last points to None)
        for i in range(10):
            handler._set_parent(nodes[i + 1], nodes[i + 2] if i < 9 else None)

        # Find root for first node
        root_id = handler._find_root_run_id(nodes[0], nodes[1])

        # Should return the last node (root)
        assert root_id == nodes[10]

    def test_find_root_run_id_with_string_ids(self, handler):
        """Test finding root with string IDs."""
        child_id = "child"
        parent_id = "parent"
        root_id = "root"

        # Set up parent chain
        handler._set_parent(parent_id, root_id)
        handler._set_parent(root_id, None)

        # Find root
        found_root = handler._find_root_run_id(child_id, parent_id)

        assert found_root == root_id

    def test_find_root_run_id_orphaned_parent(self, handler):
        """Test finding root when parent has no further relationships."""
        child_id = uuid4()
        parent_id = uuid4()

        # Parent has no parent relationship stored
        root_id = handler._find_root_run_id(child_id, parent_id)

        # Should return parent_id as it's the highest we can find
        assert root_id == parent_id

        # Should have stored child's parent relationship
        assert handler._get_parent(child_id) == parent_id

    def test_find_root_run_id_none_current_handling(self, handler):
        """Test handling when current becomes None during traversal."""
        child_id = uuid4()
        parent_id = None  # This will cause current to be None immediately

        root_id = handler._find_root_run_id(child_id, parent_id)

        # Should return child_id as it's the root when parent is None
        assert root_id == child_id
