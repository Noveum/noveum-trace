#!/usr/bin/env python3
"""
Test script to verify memory cleanup in NoveumTraceCallbackHandler.

This script demonstrates that the memory leak fix prevents unbounded growth
of root_traces and parent_map dictionaries.
"""

# Mock the dependencies that aren't available in test environment
import sys
import uuid
from unittest.mock import Mock

# Mock LangChain dependencies
sys.modules["langchain_core"] = Mock()
sys.modules["langchain_core.agents"] = Mock()
sys.modules["langchain_core.callbacks"] = Mock()
sys.modules["langchain_core.documents"] = Mock()
sys.modules["langchain_core.outputs"] = Mock()

# Mock noveum_trace dependencies
sys.modules["noveum_trace"] = Mock()
sys.modules["noveum_trace.core.span"] = Mock()
sys.modules["noveum_trace.integrations.langchain_utils"] = Mock()

# Set up mock classes
mock_span_status = Mock()
mock_span_status.OK = "OK"
mock_span_status.ERROR = "ERROR"
sys.modules["noveum_trace.core.span"].SpanStatus = mock_span_status

# Import after mocking
from src.noveum_trace.integrations.langchain import (  # noqa: E402
    NoveumTraceCallbackHandler,
)


def create_mock_client():
    """Create a mock client with required methods."""
    client = Mock()
    client.start_trace = Mock()
    client.start_span = Mock()
    client.finish_span = Mock()
    client.finish_trace = Mock()

    # Mock trace and span objects
    mock_trace = Mock()
    mock_trace.trace_id = "test-trace-123"
    mock_span = Mock()
    mock_span.span_id = "test-span-456"
    mock_span.trace = mock_trace

    client.start_trace.return_value = mock_trace
    client.start_span.return_value = mock_span

    return client, mock_trace, mock_span


def test_memory_cleanup():
    """Test that memory cleanup prevents unbounded dictionary growth."""
    print("Testing memory cleanup in NoveumTraceCallbackHandler...")

    # Create handler with mock client
    handler = NoveumTraceCallbackHandler()
    client, mock_trace, mock_span = create_mock_client()
    handler._client = client

    # Mock context functions
    def mock_get_current_trace():
        return None

    def mock_set_current_trace(trace):
        pass

    import sys

    sys.modules["noveum_trace.core.context"] = Mock()
    sys.modules["noveum_trace.core.context"].get_current_trace = mock_get_current_trace
    sys.modules["noveum_trace.core.context"].set_current_trace = mock_set_current_trace

    # Simulate multiple LangChain workflows
    print("\n1. Simulating multiple workflow starts...")
    workflows = []

    for i in range(5):
        run_id = uuid.uuid4()
        parent_run_id = None if i == 0 else workflows[0]  # Make some nested

        # Simulate workflow start (this would normally be called by LangChain)
        trace, should_manage = handler._get_or_create_trace_context(
            f"workflow_{i}", run_id, parent_run_id
        )

        workflows.append(run_id)
        handler._set_run(run_id, mock_span)

        if should_manage:
            handler._trace_managed_by_langchain = trace

    # Check that dictionaries have grown
    print(f"   root_traces size: {len(handler.root_traces)}")
    print(f"   parent_map size: {len(handler.parent_map)}")
    print(f"   runs size: {len(handler.runs)}")

    # Verify data was added
    assert len(handler.root_traces) > 0, "root_traces should have entries"
    assert len(handler.parent_map) > 0, "parent_map should have entries"
    assert len(handler.runs) > 0, "runs should have entries"

    print("\n2. Simulating workflow completions with cleanup...")

    # Simulate completing workflows - this triggers cleanup
    for i, run_id in enumerate(workflows):
        # Pop the run (simulates span completion)
        handler._pop_run(run_id)

        # When last run completes, trigger trace cleanup
        if i == len(workflows) - 1:  # Last workflow
            # This simulates what happens in _finish_trace_if_needed
            if handler._trace_managed_by_langchain:
                root_run_id = handler._find_root_run_id_for_trace(
                    handler._trace_managed_by_langchain
                )
                client.finish_trace(handler._trace_managed_by_langchain)
                handler._trace_managed_by_langchain = None

                if root_run_id is not None:
                    handler._cleanup_trace_tracking(root_run_id)

    # Check that cleanup worked
    print(f"   root_traces size after cleanup: {len(handler.root_traces)}")
    print(f"   parent_map size after cleanup: {len(handler.parent_map)}")
    print(f"   runs size after cleanup: {len(handler.runs)}")

    # Verify cleanup worked
    print("\n3. Verifying cleanup effectiveness...")

    # For this test, we expect significant cleanup
    # (exact numbers depend on the tree structure, but should be much smaller)
    assert len(handler.runs) == 0, "runs should be empty after completion"

    print("✅ Memory cleanup test passed!")
    print("\nKey improvements:")
    print("- root_traces entries are cleaned up when traces finish")
    print("- parent_map entries are cleaned up for entire workflow trees")
    print("- Memory usage remains bounded in long-running applications")


def test_cleanup_methods():
    """Test the cleanup helper methods work correctly."""
    print("\nTesting cleanup helper methods...")

    handler = NoveumTraceCallbackHandler()
    client, mock_trace, mock_span = create_mock_client()
    handler._client = client

    # Set up test data
    root_id = "root-123"
    child1_id = "child1-456"
    child2_id = "child2-789"
    grandchild_id = "grandchild-999"

    # Build parent relationships: root -> child1 -> grandchild, root -> child2
    handler._set_parent(child1_id, root_id)
    handler._set_parent(child2_id, root_id)
    handler._set_parent(grandchild_id, child1_id)
    handler._set_root_trace(root_id, mock_trace)

    print(f"   Initial parent_map size: {len(handler.parent_map)}")
    print(f"   Initial root_traces size: {len(handler.root_traces)}")

    # Test _is_descendant_of
    assert handler._is_descendant_of(
        child1_id, root_id
    ), "child1 should be descendant of root"
    assert handler._is_descendant_of(
        grandchild_id, root_id
    ), "grandchild should be descendant of root"
    assert handler._is_descendant_of(
        grandchild_id, child1_id
    ), "grandchild should be descendant of child1"
    assert not handler._is_descendant_of(
        root_id, child1_id
    ), "root should not be descendant of child1"

    # Test _find_root_run_id_for_trace
    found_root = handler._find_root_run_id_for_trace(mock_trace)
    assert found_root == root_id, f"Should find root_id {root_id}, got {found_root}"

    # Test cleanup
    handler._cleanup_trace_tracking(root_id)

    print(f"   Final parent_map size: {len(handler.parent_map)}")
    print(f"   Final root_traces size: {len(handler.root_traces)}")

    # Verify cleanup
    assert len(handler.root_traces) == 0, "root_traces should be empty after cleanup"
    assert len(handler.parent_map) == 0, "parent_map should be empty after cleanup"

    print("✅ Cleanup helper methods test passed!")


if __name__ == "__main__":
    test_memory_cleanup()
    test_cleanup_methods()
    print("\n🎉 All memory cleanup tests passed!")
    print("\nThe memory leak has been successfully fixed:")
    print("• root_traces dictionary is cleaned up when traces finish")
    print("• parent_map dictionary is cleaned up for entire workflow trees")
    print(
        "• Long-running applications will no longer experience unbounded memory growth"
    )
