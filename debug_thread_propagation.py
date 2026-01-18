"""
Debug script to understand thread context propagation issues with LangChain callbacks.

This script demonstrates and diagnoses the issue where the noveum-trace callback handler
doesn't propagate correctly when:
- The graph runs in one thread
- LLM calls are made in another thread

The root cause is typically that Python's contextvars.ContextVar doesn't automatically
propagate to threads created by ThreadPoolExecutor or when using concurrent.futures.
"""

import contextvars
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

# Create a simple context var to test propagation
_test_context: contextvars.ContextVar[str] = contextvars.ContextVar(
    "test_context", default="NOT_SET"
)


def check_context_in_thread(thread_name: str, expected: str) -> dict:
    """Check what context value is visible in this thread."""
    actual = _test_context.get()
    return {
        "thread_name": thread_name,
        "thread_id": threading.current_thread().name,
        "expected": expected,
        "actual": actual,
        "match": actual == expected,
    }


def test_direct_thread_propagation():
    """Test 1: Direct threading.Thread propagation."""
    print("\n" + "=" * 60)
    print("TEST 1: Direct threading.Thread propagation")
    print("=" * 60)

    # Set context in main thread
    _test_context.set("MAIN_THREAD_VALUE")
    print(f"Main thread context: {_test_context.get()}")

    results = []

    def worker():
        result = check_context_in_thread("direct_thread", "MAIN_THREAD_VALUE")
        results.append(result)

    # Create and run thread
    t = threading.Thread(target=worker)
    t.start()
    t.join()

    for r in results:
        status = "✅ PASS" if r["match"] else "❌ FAIL"
        print(f"{status}: {r['thread_name']} - expected={r['expected']}, got={r['actual']}")

    return all(r["match"] for r in results)


def test_threadpool_propagation():
    """Test 2: ThreadPoolExecutor propagation (what LangChain often uses)."""
    print("\n" + "=" * 60)
    print("TEST 2: ThreadPoolExecutor propagation (WITHOUT copy_context)")
    print("=" * 60)

    # Set context in main thread
    _test_context.set("THREADPOOL_VALUE")
    print(f"Main thread context: {_test_context.get()}")

    results = []

    def worker(task_id):
        result = check_context_in_thread(f"pool_thread_{task_id}", "THREADPOOL_VALUE")
        return result

    # Use ThreadPoolExecutor (default behavior)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(worker, i) for i in range(3)]
        for future in futures:
            results.append(future.result())

    for r in results:
        status = "✅ PASS" if r["match"] else "❌ FAIL"
        print(f"{status}: {r['thread_name']} - expected={r['expected']}, got={r['actual']}")

    return all(r["match"] for r in results)


def test_threadpool_with_copy_context():
    """Test 3: ThreadPoolExecutor with explicit copy_context."""
    print("\n" + "=" * 60)
    print("TEST 3: ThreadPoolExecutor WITH copy_context")
    print("=" * 60)

    # Set context in main thread
    _test_context.set("COPIED_CONTEXT_VALUE")
    print(f"Main thread context: {_test_context.get()}")

    results = []

    def worker(task_id):
        result = check_context_in_thread(
            f"pool_thread_copied_{task_id}", "COPIED_CONTEXT_VALUE"
        )
        return result

    # Capture context BEFORE submitting to thread pool
    ctx = contextvars.copy_context()

    # Use ThreadPoolExecutor with context.run()
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(ctx.run, worker, i) for i in range(3)]
        for future in futures:
            results.append(future.result())

    for r in results:
        status = "✅ PASS" if r["match"] else "❌ FAIL"
        print(f"{status}: {r['thread_name']} - expected={r['expected']}, got={r['actual']}")

    return all(r["match"] for r in results)


def test_nested_thread_pool():
    """Test 4: Nested thread pool (simulates LangGraph calling LLM in different thread)."""
    print("\n" + "=" * 60)
    print("TEST 4: Nested ThreadPool (simulates graph thread -> LLM thread)")
    print("=" * 60)

    # Set context in main thread
    _test_context.set("NESTED_VALUE")
    print(f"Main thread context: {_test_context.get()}")

    results = []

    def outer_worker(task_id):
        """Simulates graph node execution."""
        outer_result = check_context_in_thread(f"outer_{task_id}", "NESTED_VALUE")
        results.append(outer_result)

        # Now simulate LLM call in ANOTHER thread (inner pool)
        def inner_worker():
            return check_context_in_thread(f"inner_llm_{task_id}", "NESTED_VALUE")

        with ThreadPoolExecutor(max_workers=1) as inner_executor:
            inner_future = inner_executor.submit(inner_worker)
            inner_result = inner_future.result()
            results.append(inner_result)

    # Outer thread pool (graph execution)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(outer_worker, i) for i in range(2)]
        for future in futures:
            future.result()

    for r in results:
        status = "✅ PASS" if r["match"] else "❌ FAIL"
        print(f"{status}: {r['thread_name']} - expected={r['expected']}, got={r['actual']}")

    return all(r["match"] for r in results)


def test_nested_with_context_propagation():
    """Test 5: Nested thread pool WITH proper context propagation."""
    print("\n" + "=" * 60)
    print("TEST 5: Nested ThreadPool WITH context propagation")
    print("=" * 60)

    # Set context in main thread
    _test_context.set("NESTED_PROPAGATED_VALUE")
    print(f"Main thread context: {_test_context.get()}")

    results = []

    def outer_worker(task_id):
        """Outer worker that first restores context, then passes it to inner."""
        outer_result = check_context_in_thread(
            f"outer_{task_id}", "NESTED_PROPAGATED_VALUE"
        )
        results.append(outer_result)

        # Capture context in THIS thread before spawning inner thread
        # This is the key: we copy context INSIDE the worker, not from main
        current_ctx = contextvars.copy_context()

        def inner_worker():
            return check_context_in_thread(
                f"inner_llm_{task_id}", "NESTED_PROPAGATED_VALUE"
            )

        with ThreadPoolExecutor(max_workers=1) as inner_executor:
            inner_future = inner_executor.submit(current_ctx.run, inner_worker)
            inner_result = inner_future.result()
            results.append(inner_result)

    # For proper context propagation, we need to use a wrapper approach
    main_ctx = contextvars.copy_context()
    
    def context_wrapper(func, *args):
        """Wrapper that runs function in copied context."""
        return func(*args)

    # Outer thread pool with context propagation
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Use main_ctx.run to execute each outer_worker call
        futures = []
        for i in range(2):
            # Create a new context copy for each submission
            ctx = contextvars.copy_context()
            futures.append(executor.submit(ctx.run, outer_worker, i))
        for future in futures:
            future.result()

    for r in results:
        status = "✅ PASS" if r["match"] else "❌ FAIL"
        print(f"{status}: {r['thread_name']} - expected={r['expected']}, got={r['actual']}")

    return all(r["match"] for r in results)


def simulate_langchain_callback_issue():
    """
    Test 6: Simulate the exact issue with LangChain callback handler.
    
    This simulates:
    1. Main thread sets up trace context and callback handler
    2. Graph execution runs in thread pool
    3. LLM callback is invoked but context is lost
    """
    print("\n" + "=" * 60)
    print("TEST 6: Simulate LangChain Callback Handler Issue")
    print("=" * 60)

    from noveum_trace.core.context import (
        get_current_trace,
        set_current_trace,
        TraceContext,
        _trace_context,
    )

    # Mock trace object
    class MockTrace:
        def __init__(self, trace_id):
            self.trace_id = trace_id

    # Create trace in main thread
    mock_trace = MockTrace(trace_id="trace-123")
    set_current_trace(mock_trace)
    print(f"Main thread trace: {get_current_trace()}")

    results = []

    def simulate_graph_node():
        """Simulates a LangGraph node."""
        trace = get_current_trace()
        results.append({
            "location": "graph_node",
            "trace_found": trace is not None,
            "trace_id": trace.trace_id if trace else None,
        })

        # Now simulate LLM call which might happen in another thread
        def simulate_llm_callback():
            trace = get_current_trace()
            return {
                "location": "llm_callback",
                "trace_found": trace is not None,
                "trace_id": trace.trace_id if trace else None,
            }

        # This is what happens internally in LangChain
        with ThreadPoolExecutor(max_workers=1) as llm_executor:
            future = llm_executor.submit(simulate_llm_callback)
            llm_result = future.result()
            results.append(llm_result)

    # Graph runs in thread pool
    with ThreadPoolExecutor(max_workers=1) as graph_executor:
        future = graph_executor.submit(simulate_graph_node)
        future.result()

    print("\nResults (WITHOUT context propagation):")
    for r in results:
        status = "✅" if r["trace_found"] else "❌"
        print(f"  {status} {r['location']}: trace_found={r['trace_found']}, id={r['trace_id']}")

    return all(r["trace_found"] for r in results)


def simulate_langchain_callback_fixed():
    """
    Test 7: Demonstrate the fix using contextvars.copy_context().
    """
    print("\n" + "=" * 60)
    print("TEST 7: Simulate LangChain Callback Handler FIX")
    print("=" * 60)

    from noveum_trace.core.context import (
        get_current_trace,
        set_current_trace,
    )

    # Mock trace object
    class MockTrace:
        def __init__(self, trace_id):
            self.trace_id = trace_id

    # Create trace in main thread
    mock_trace = MockTrace(trace_id="trace-456")
    set_current_trace(mock_trace)
    print(f"Main thread trace: {get_current_trace()}")

    results = []

    def simulate_graph_node():
        """Simulates graph node with proper context handling."""
        trace = get_current_trace()
        results.append({
            "location": "graph_node_fixed",
            "trace_found": trace is not None,
            "trace_id": trace.trace_id if trace else None,
        })

        # Capture context before spawning LLM thread
        current_ctx = contextvars.copy_context()

        def simulate_llm_callback():
            trace = get_current_trace()
            return {
                "location": "llm_callback_fixed",
                "trace_found": trace is not None,
                "trace_id": trace.trace_id if trace else None,
            }

        # Run LLM callback with propagated context
        with ThreadPoolExecutor(max_workers=1) as llm_executor:
            future = llm_executor.submit(current_ctx.run, simulate_llm_callback)
            llm_result = future.result()
            results.append(llm_result)

    # Graph runs in thread pool with context propagation
    main_ctx = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=1) as graph_executor:
        future = graph_executor.submit(main_ctx.run, simulate_graph_node)
        future.result()

    print("\nResults (WITH context propagation):")
    for r in results:
        status = "✅" if r["trace_found"] else "❌"
        print(f"  {status} {r['location']}: trace_found={r['trace_found']}, id={r['trace_id']}")

    return all(r["trace_found"] for r in results)


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("DEBUGGING THREAD CONTEXT PROPAGATION")
    print("=" * 60)
    print("""
This debug script investigates why NoveumTraceCallbackHandler
doesn't propagate correctly when:
- The graph runs in one thread  
- LLM calls are made in another thread

Root cause: Python's contextvars.ContextVar doesn't automatically
propagate to threads created by ThreadPoolExecutor.
    """)

    results = {}

    # Run tests
    results["direct_thread"] = test_direct_thread_propagation()
    results["threadpool_no_copy"] = test_threadpool_propagation()
    results["threadpool_with_copy"] = test_threadpool_with_copy_context()
    results["nested_no_propagation"] = test_nested_thread_pool()
    results["nested_with_propagation"] = test_nested_with_context_propagation()
    results["langchain_issue"] = simulate_langchain_callback_issue()
    results["langchain_fix"] = simulate_langchain_callback_fixed()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")

    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    print("""
1. Direct threading.Thread: Context DOES propagate (Python 3.7+)
2. ThreadPoolExecutor: Context does NOT automatically propagate
3. nested thread pools: Each level loses context if not explicitly copied

The FIX requires using contextvars.copy_context().run() when:
- Submitting tasks to ThreadPoolExecutor
- Creating new threads that need access to trace context

For LangChain/LangGraph, the callback handler needs to ensure
that when callbacks are invoked, the trace context is properly
copied and restored in the callback thread.
    """)


if __name__ == "__main__":
    main()
