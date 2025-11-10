"""
Integration tests for LangChain patch async/parallel execution scenarios.

Tests the enhanced async/parallel capabilities including:
- Concurrent callback thread safety
- Parallel workflow trace grouping
- Out-of-order callback handling
- Race condition prevention in trace creation
- Complex parent-child relationships in concurrent scenarios
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
class TestConcurrentCallbackThreadSafety:
    """Test thread safety of callback operations in concurrent scenarios."""

    @pytest.fixture
    def handler(self):
        """Create a callback handler for testing."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            return NoveumTraceCallbackHandler()

    def test_concurrent_callback_thread_safety(self, handler):
        """Test multiple threads calling callbacks simultaneously."""
        results = []
        errors = []
        results_lock = threading.Lock()
        errors_lock = threading.Lock()

        # Use thread-local storage to track expected span_id per thread
        thread_local = threading.local()

        # Create a single shared side_effect that uses thread-local storage
        def start_span_side_effect(*args, **kwargs):
            # Get the expected span_id from thread-local storage
            expected_id = getattr(thread_local, "expected_span_id", None)
            if expected_id is None:
                # Fallback if thread-local not set
                expected_id = "span_unknown"

            # Create a new mock span for this call
            mock_span = Mock()
            mock_span.span_id = expected_id
            return mock_span

        # Set the side_effect once before all threads start
        handler._client.start_span.side_effect = start_span_side_effect

        def run_llm_callback(thread_id):
            try:
                run_id = uuid4()
                serialized = {"name": f"llm_{thread_id}"}
                prompts = [f"prompt_{thread_id}"]

                # Store the expected span_id in thread-local storage
                expected_span_id = f"span_{thread_id}"
                thread_local.expected_span_id = expected_span_id

                with (
                    patch.object(
                        handler,
                        "_get_or_create_trace_context",
                        return_value=(Mock(), True),
                    ),
                    patch.object(handler, "_resolve_parent_span_id", return_value=None),
                ):
                    # Call callback
                    handler.on_llm_start(
                        serialized=serialized, prompts=prompts, run_id=run_id
                    )

                    # Verify span was stored
                    stored_span = handler._get_run(run_id)
                    if stored_span:
                        with results_lock:
                            results.append((thread_id, run_id, stored_span.span_id))
                    else:
                        with errors_lock:
                            errors.append((thread_id, "Span not found in runs dict"))

            except Exception as e:
                with errors_lock:
                    errors.append((thread_id, str(e)))

        # Run multiple threads concurrently
        threads = []
        for i in range(20):
            thread = threading.Thread(target=run_llm_callback, args=(i,))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 20

        # Verify all spans were stored correctly
        for thread_id, _run_id, span_id in results:
            assert span_id == f"span_{thread_id}"

    def test_concurrent_root_trace_operations(self, handler):
        """Test concurrent root trace storage and retrieval."""
        results = []
        errors = []

        def set_and_get_root_trace(thread_id):
            try:
                root_run_id = uuid4()
                mock_trace = Mock()
                mock_trace.trace_id = f"trace_{thread_id}"

                # Set root trace
                handler._set_root_trace(root_run_id, mock_trace)

                # Small delay to increase race condition chances
                time.sleep(0.001)

                # Get root trace
                retrieved = handler._get_root_trace(root_run_id)
                results.append((thread_id, retrieved.trace_id if retrieved else None))

            except Exception as e:
                errors.append((thread_id, e))

        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(set_and_get_root_trace, i) for i in range(50)]

            for future in as_completed(futures):
                future.result()  # Wait for completion and raise any exceptions

        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 50

        # All operations should have succeeded
        for thread_id, trace_id in results:
            assert trace_id == f"trace_{thread_id}"

    def test_concurrent_parent_chain_traversal(self, handler):
        """Test concurrent parent chain traversal operations."""
        # Set up a complex parent chain
        chain_length = 10
        chain_ids = [uuid4() for _ in range(chain_length)]

        # Set up parent relationships: 0->1->2->...->9->None
        for i in range(chain_length - 1):
            handler._set_parent(chain_ids[i], chain_ids[i + 1])
        handler._set_parent(chain_ids[-1], None)  # Last is root

        results = []
        errors = []

        def find_root_concurrent(thread_id):
            try:
                # Each thread tries to find root for the first node
                root_id = handler._find_root_run_id(chain_ids[0], chain_ids[1])
                results.append((thread_id, root_id))
            except Exception as e:
                errors.append((thread_id, e))

        # Run multiple threads
        threads = []
        for i in range(15):
            thread = threading.Thread(target=find_root_concurrent, args=(i,))
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 15

        # All should find the same root
        expected_root = chain_ids[-1]
        for _thread_id, found_root in results:
            assert found_root == expected_root


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not available")
class TestParallelWorkflowTraceGrouping:
    """Test trace grouping in parallel workflow scenarios."""

    @pytest.fixture
    def handler(self):
        """Create a callback handler for testing."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            return NoveumTraceCallbackHandler()

    def test_parallel_workflow_trace_grouping(self, handler):
        """Test that parallel operations in same workflow share trace."""
        # Simulate LangGraph workflow with parallel branches
        root_run_id = uuid4()
        branch_1_id = uuid4()
        branch_2_id = uuid4()

        # Mock trace for the workflow
        mock_workflow_trace = Mock()
        mock_workflow_trace.trace_id = "workflow_trace"

        results = []
        errors = []

        def run_branch_operation(branch_id, parent_id):
            try:
                with patch.object(
                    handler, "_get_or_create_trace_context"
                ) as mock_context:
                    mock_context.return_value = (mock_workflow_trace, False)

                    # Mock span creation
                    mock_span = Mock()
                    handler._client.start_span.return_value = mock_span

                    # Simulate chain start for branch
                    handler.on_chain_start(
                        serialized={"name": f"branch_{branch_id}"},
                        inputs={"input": f"branch_{branch_id}_input"},
                        run_id=branch_id,
                        parent_run_id=parent_id,
                    )

                    # Verify trace context was called with correct parameters
                    call_args = mock_context.call_args[0]
                    results.append(
                        (branch_id, call_args[1], call_args[2])
                    )  # run_id, parent_run_id

            except Exception as e:
                errors.append((branch_id, e))

        # Run branches in parallel
        threads = [
            threading.Thread(
                target=run_branch_operation, args=(branch_1_id, root_run_id)
            ),
            threading.Thread(
                target=run_branch_operation, args=(branch_2_id, root_run_id)
            ),
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 2

        # Both branches should have been called with correct parent
        branch_results = {result[0]: (result[1], result[2]) for result in results}
        assert branch_results[branch_1_id] == (branch_1_id, root_run_id)
        assert branch_results[branch_2_id] == (branch_2_id, root_run_id)

    def test_race_condition_trace_creation(self, handler):
        """Test prevention of race conditions in trace creation."""
        # Simulate multiple root operations starting simultaneously
        root_operations = [uuid4() for _ in range(10)]

        created_traces = []
        errors = []

        def create_root_operation(run_id):
            try:
                # Mock trace creation
                mock_trace = Mock()
                mock_trace.trace_id = f"trace_{run_id}"
                handler._client.start_trace.return_value = mock_trace

                with (
                    patch(
                        "noveum_trace.core.context.get_current_trace", return_value=None
                    ),
                    patch("noveum_trace.core.context.set_current_trace"),
                ):

                    # Call trace context method (simulating root call)
                    trace, should_manage = handler._get_or_create_trace_context(
                        f"operation_{run_id}", run_id, None
                    )

                    created_traces.append((run_id, trace.trace_id, should_manage))

            except Exception as e:
                errors.append((run_id, e))

        # Run operations concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(create_root_operation, run_id)
                for run_id in root_operations
            ]

            for future in as_completed(futures):
                future.result()

        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(created_traces) == 10

        # Each operation should have created its own trace
        trace_ids = [trace_id for _, trace_id, _ in created_traces]
        assert len(set(trace_ids)) == 10  # All unique


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not available")
class TestOutOfOrderCallbackHandling:
    """Test handling of callbacks arriving out of order."""

    @pytest.fixture
    def handler(self):
        """Create a callback handler for testing."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            return NoveumTraceCallbackHandler()

    def test_out_of_order_callback_handling(self, handler):
        """Test callbacks arriving in different order than expected."""
        # Simulate scenario where child callback arrives before parent
        parent_run_id = uuid4()
        child_run_id = uuid4()

        # Mock traces
        mock_parent_trace = Mock()
        mock_parent_trace.trace_id = "parent_trace"

        results = []

        def child_callback():
            """Child callback that might arrive first."""
            try:
                with patch.object(
                    handler, "_get_or_create_trace_context"
                ) as mock_context:
                    mock_context.return_value = (mock_parent_trace, False)

                    mock_span = Mock()
                    handler._client.start_span.return_value = mock_span

                    handler.on_llm_start(
                        serialized={"name": "child_llm"},
                        prompts=["child prompt"],
                        run_id=child_run_id,
                        parent_run_id=parent_run_id,
                    )

                    results.append(("child", mock_context.call_args))
            except Exception as e:
                results.append(("child_error", e))

        def parent_callback():
            """Parent callback that might arrive second."""
            try:
                # Small delay to simulate parent arriving after child
                time.sleep(0.01)

                with patch.object(
                    handler, "_get_or_create_trace_context"
                ) as mock_context:
                    mock_context.return_value = (mock_parent_trace, True)

                    mock_span = Mock()
                    handler._client.start_span.return_value = mock_span

                    handler.on_chain_start(
                        serialized={"name": "parent_chain"},
                        inputs={"input": "parent input"},
                        run_id=parent_run_id,
                        parent_run_id=None,  # Root operation
                    )

                    results.append(("parent", mock_context.call_args))
            except Exception as e:
                results.append(("parent_error", e))

        # Start child first, then parent
        child_thread = threading.Thread(target=child_callback)
        parent_thread = threading.Thread(target=parent_callback)

        child_thread.start()
        parent_thread.start()

        child_thread.join()
        parent_thread.join()

        # Both should have completed successfully
        assert len(results) == 2
        assert not any("error" in result[0] for result in results)

        # Verify both callbacks were handled properly
        result_dict = {result[0]: result[1] for result in results}
        assert "child" in result_dict
        assert "parent" in result_dict


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not available")
class TestComplexParentChildRelationships:
    """Test complex parent-child relationships in concurrent scenarios."""

    @pytest.fixture
    def handler(self):
        """Create a callback handler for testing."""
        with patch("noveum_trace.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            return NoveumTraceCallbackHandler()

    def test_complex_parent_child_relationships(self, handler):
        """Test complex nested parent-child relationships with concurrency."""
        # Create a complex hierarchy:
        # root -> agent -> [tool1, tool2] (parallel tools)
        #      -> chain -> llm

        root_id = uuid4()
        agent_id = uuid4()
        tool1_id = uuid4()
        tool2_id = uuid4()
        chain_id = uuid4()
        llm_id = uuid4()

        # Mock shared trace
        mock_trace = Mock()
        mock_trace.trace_id = "shared_trace"

        operations = []
        errors = []

        def run_operation(op_type, run_id, parent_id, delay=0):
            try:
                if delay:
                    time.sleep(delay)

                with patch.object(
                    handler, "_get_or_create_trace_context"
                ) as mock_context:
                    mock_context.return_value = (mock_trace, parent_id is None)

                    mock_span = Mock()
                    mock_span.span_id = f"span_{run_id}"
                    handler._client.start_span.return_value = mock_span

                    if op_type == "agent":
                        handler.on_agent_start(
                            serialized={"name": "test_agent"},
                            inputs={"input": "test"},
                            run_id=run_id,
                            parent_run_id=parent_id,
                        )
                    elif op_type == "tool":
                        handler.on_tool_start(
                            serialized={"name": f"tool_{run_id}"},
                            input_str="tool input",
                            run_id=run_id,
                            parent_run_id=parent_id,
                        )
                    elif op_type == "chain":
                        handler.on_chain_start(
                            serialized={"name": "test_chain"},
                            inputs={"input": "chain input"},
                            run_id=run_id,
                            parent_run_id=parent_id,
                        )
                    elif op_type == "llm":
                        handler.on_llm_start(
                            serialized={"name": "test_llm"},
                            prompts=["test prompt"],
                            run_id=run_id,
                            parent_run_id=parent_id,
                        )

                    operations.append((op_type, run_id, parent_id))

            except Exception as e:
                errors.append((op_type, run_id, e))

        # Start operations with different delays to simulate real-world timing
        threads = [
            threading.Thread(
                target=run_operation, args=("agent", agent_id, root_id, 0.01)
            ),
            threading.Thread(
                target=run_operation, args=("tool", tool1_id, agent_id, 0.02)
            ),
            threading.Thread(
                target=run_operation, args=("tool", tool2_id, agent_id, 0.02)
            ),
            threading.Thread(
                target=run_operation, args=("chain", chain_id, root_id, 0.015)
            ),
            threading.Thread(
                target=run_operation, args=("llm", llm_id, chain_id, 0.03)
            ),
        ]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(operations) == 5

        # Verify all operations were handled
        op_types = [op[0] for op in operations]
        assert "agent" in op_types
        assert "tool" in op_types  # Should appear twice
        assert "chain" in op_types
        assert "llm" in op_types
        assert op_types.count("tool") == 2

    def test_concurrent_span_lifecycle_management(self, handler):
        """Test concurrent span creation and cleanup."""
        num_operations = 20
        run_ids = [uuid4() for _ in range(num_operations)]

        created_spans = []
        completed_spans = []
        errors = []

        def create_and_complete_span(run_id, delay):
            try:
                # Create span
                with patch.object(
                    handler, "_get_or_create_trace_context", return_value=(Mock(), True)
                ):
                    mock_span = Mock()
                    mock_span.span_id = f"span_{run_id}"
                    handler._client.start_span.return_value = mock_span

                    handler.on_llm_start(
                        serialized={"name": "test_llm"}, prompts=["test"], run_id=run_id
                    )

                    created_spans.append(run_id)

                    # Simulate some processing time
                    time.sleep(delay)

                    # Complete span
                    mock_result = Mock()
                    mock_result.generations = [[Mock(text="response")]]
                    mock_result.llm_output = {"token_usage": {"total_tokens": 100}}

                    handler.on_llm_end(response=mock_result, run_id=run_id)
                    completed_spans.append(run_id)

            except Exception as e:
                errors.append((run_id, e))

        # Run operations with random delays
        import random

        threads = []
        for run_id in run_ids:
            delay = random.uniform(0.001, 0.01)
            thread = threading.Thread(
                target=create_and_complete_span, args=(run_id, delay)
            )
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(created_spans) == num_operations
        assert len(completed_spans) == num_operations

        # All spans should have been created and completed
        assert set(created_spans) == set(run_ids)
        assert set(completed_spans) == set(run_ids)
