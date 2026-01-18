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


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not available")
class TestTTFTCleanup:
    """Test TTFT (Time To First Token) tracking cleanup to prevent memory leaks."""

    @pytest.fixture
    def handler(self):
        """Create a callback handler for testing."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            return NoveumTraceCallbackHandler()

    def test_on_llm_end_cleans_up_ttft_when_span_is_none(self, handler):
        """Test that on_llm_end cleans up TTFT tracking even when span is None."""
        run_id = uuid4()

        # Simulate first token received (adds run_id to _first_token_received)
        with handler._first_token_lock:
            handler._first_token_received.add(run_id)

        # Verify run_id is in the set
        assert run_id in handler._first_token_received

        # Call on_llm_end with no span stored (span will be None)
        mock_response = Mock()
        mock_response.generations = []
        mock_response.llm_output = {}

        handler.on_llm_end(mock_response, run_id=run_id)

        # Verify run_id was cleaned up even though span was None
        assert run_id not in handler._first_token_received

    def test_on_llm_end_cleans_up_ttft_with_valid_span(self, handler):
        """Test that on_llm_end cleans up TTFT tracking with a valid span."""
        run_id = uuid4()

        # Create a mock span and store it
        mock_span = Mock()
        mock_span.attributes = {}
        mock_span.start_time = None
        handler._set_run(run_id, mock_span)

        # Simulate first token received
        with handler._first_token_lock:
            handler._first_token_received.add(run_id)

        # Verify run_id is in the set
        assert run_id in handler._first_token_received

        # Call on_llm_end
        mock_response = Mock()
        mock_response.generations = []
        mock_response.llm_output = {}

        handler.on_llm_end(mock_response, run_id=run_id)

        # Verify run_id was cleaned up
        assert run_id not in handler._first_token_received

    def test_on_llm_error_cleans_up_ttft_when_span_is_none(self, handler):
        """Test that on_llm_error cleans up TTFT tracking even when span is None."""
        run_id = uuid4()

        # Simulate first token received before error
        with handler._first_token_lock:
            handler._first_token_received.add(run_id)

        # Verify run_id is in the set
        assert run_id in handler._first_token_received

        # Call on_llm_error with no span stored
        error = Exception("Test error")
        handler.on_llm_error(error, run_id=run_id)

        # Verify run_id was cleaned up even though span was None
        assert run_id not in handler._first_token_received

    def test_on_llm_error_cleans_up_ttft_with_valid_span(self, handler):
        """Test that on_llm_error cleans up TTFT tracking with a valid span."""
        run_id = uuid4()

        # Create a mock span and store it
        mock_span = Mock()
        handler._set_run(run_id, mock_span)

        # Simulate first token received before error
        with handler._first_token_lock:
            handler._first_token_received.add(run_id)

        # Verify run_id is in the set
        assert run_id in handler._first_token_received

        # Call on_llm_error
        error = Exception("Test error")
        handler.on_llm_error(error, run_id=run_id)

        # Verify run_id was cleaned up
        assert run_id not in handler._first_token_received

    def test_on_llm_new_token_cleans_up_when_span_missing(self, handler):
        """Test that on_llm_new_token cleans up TTFT tracking when span is missing."""
        run_id = uuid4()

        # Call on_llm_new_token without storing a span first
        # This simulates a race condition or failed span creation
        handler.on_llm_new_token("test token", run_id=run_id)

        # Verify run_id was cleaned up because span was not found
        assert run_id not in handler._first_token_received

    def test_on_llm_new_token_records_ttft_with_valid_span(self, handler):
        """Test that on_llm_new_token records TTFT with a valid span."""
        from datetime import datetime, timezone

        run_id = uuid4()

        # Create a mock span with start_time and store it
        mock_span = Mock()
        mock_span.start_time = datetime.now(timezone.utc)
        handler._set_run(run_id, mock_span)

        # Call on_llm_new_token
        handler.on_llm_new_token("test token", run_id=run_id)

        # Verify run_id is in the set (not cleaned up because span exists)
        assert run_id in handler._first_token_received

        # Verify TTFT metrics were recorded
        mock_span.set_attribute.assert_any_call("llm.streaming", True)

    def test_ttft_cleanup_multiple_runs_no_leak(self, handler):
        """Test that multiple streaming runs don't leak memory in _first_token_received."""
        run_ids = [uuid4() for _ in range(10)]

        # Simulate multiple streaming runs with various outcomes
        for i, run_id in enumerate(run_ids):
            # Add to first token received set
            with handler._first_token_lock:
                handler._first_token_received.add(run_id)

            if i % 3 == 0:
                # Simulate successful completion (no span)
                mock_response = Mock()
                mock_response.generations = []
                mock_response.llm_output = {}
                handler.on_llm_end(mock_response, run_id=run_id)
            elif i % 3 == 1:
                # Simulate error (no span)
                handler.on_llm_error(Exception("test"), run_id=run_id)
            else:
                # Simulate new token with missing span
                # First, ensure the run_id is added by on_llm_new_token itself
                handler._first_token_received.discard(run_id)  # Remove first
                handler.on_llm_new_token("token", run_id=run_id)

        # Verify all run_ids were cleaned up - no memory leak
        assert len(handler._first_token_received) == 0

    def test_ttft_cleanup_thread_safety(self, handler):
        """Test thread safety of TTFT cleanup operations."""
        import threading

        run_ids = [uuid4() for _ in range(50)]
        errors = []

        def simulate_streaming(run_id, outcome):
            try:
                # Add to first token set
                with handler._first_token_lock:
                    handler._first_token_received.add(run_id)

                if outcome == "end":
                    mock_response = Mock()
                    mock_response.generations = []
                    mock_response.llm_output = {}
                    handler.on_llm_end(mock_response, run_id=run_id)
                elif outcome == "error":
                    handler.on_llm_error(Exception("test"), run_id=run_id)
                else:
                    handler._first_token_received.discard(run_id)
                    handler.on_llm_new_token("token", run_id=run_id)
            except Exception as e:
                errors.append(e)

        # Create threads with different outcomes
        threads = []
        outcomes = ["end", "error", "token"]
        for i, run_id in enumerate(run_ids):
            outcome = outcomes[i % 3]
            thread = threading.Thread(target=simulate_streaming, args=(run_id, outcome))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Should have no errors
        assert len(errors) == 0

        # Should have no memory leak
        assert len(handler._first_token_received) == 0
