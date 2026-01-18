"""
Debug script to understand the exact issue with NoveumTraceCallbackHandler
when callbacks are invoked from different threads.

This script simulates the problematic scenario:
1. Main thread: Start trace, set context, invoke graph
2. Worker thread: LangGraph node executes, invokes LLM
3. Worker thread: LLM callback is invoked but context is lost
4. Handler tries to get trace context -> fails -> creates new trace
"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
import contextvars
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test_handler_context_resolution():
    """Test how NoveumTraceCallbackHandler resolves trace context from different threads."""
    print("\n" + "=" * 80)
    print("TEST: Handler Context Resolution in Multi-threaded Scenario")
    print("=" * 80)

    from noveum_trace import init, get_client
    from noveum_trace.core.context import (
        set_current_trace, 
        get_current_trace,
        get_current_context,
        set_current_context,
        TraceContext,
    )
    from noveum_trace.integrations.langchain import NoveumTraceCallbackHandler

    # Initialize
    init(api_key="test", endpoint="http://localhost:4318/v1/traces", debug=True)

    # Create trace in main thread
    client = get_client()
    main_trace = client.start_trace("main-thread-trace")
    set_current_trace(main_trace)
    print(f"Main thread trace: {main_trace.trace_id}")

    # Create handler in main thread
    handler = NoveumTraceCallbackHandler()
    print(f"Handler created: {handler}")

    results = []

    def simulate_callback_in_worker_thread(test_name: str, use_parent_run_id: bool = True):
        """Simulate what happens when a callback is invoked in a worker thread."""
        
        # Check what trace context is visible
        current_trace = get_current_trace()
        thread = threading.current_thread()
        
        result = {
            "test": test_name,
            "thread": thread.name,
            "trace_visible": current_trace is not None,
            "trace_id_visible": getattr(current_trace, "trace_id", None) if current_trace else None,
        }
        
        # Simulate on_chain_start callback
        run_id = uuid4()
        parent_run_id = uuid4() if use_parent_run_id else None  # Simulating a child call
        
        # This is what the handler does internally
        try:
            # Get or create trace context
            trace, should_manage = handler._get_or_create_trace_context(
                "test.operation",
                run_id=run_id,
                parent_run_id=parent_run_id
            )
            
            result["trace_obtained"] = trace is not None
            result["trace_id_obtained"] = getattr(trace, "trace_id", None) if trace else None
            result["should_manage"] = should_manage
            result["is_same_trace"] = (
                result["trace_id_obtained"] == main_trace.trace_id if trace else False
            )
            
        except Exception as e:
            result["error"] = str(e)
        
        results.append(result)
        return result

    print("\n--- Scenario 1: Direct thread without context ---")
    t = threading.Thread(
        target=simulate_callback_in_worker_thread, 
        args=("direct_thread", False)
    )
    t.start()
    t.join()

    print("\n--- Scenario 2: ThreadPoolExecutor without context ---")
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(simulate_callback_in_worker_thread, "pool_no_context", False)
        future.result()

    print("\n--- Scenario 3: ThreadPoolExecutor WITH context ---")
    ctx = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(ctx.run, simulate_callback_in_worker_thread, "pool_with_context", False)
        future.result()

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    for r in results:
        print(f"\nTest: {r['test']}")
        print(f"  Thread: {r['thread']}")
        print(f"  Trace visible in thread: {r['trace_visible']} ({r['trace_id_visible']})")
        print(f"  Trace obtained by handler: {r['trace_obtained']} ({r['trace_id_obtained']})")
        if "is_same_trace" in r:
            status = "✅ SAME" if r["is_same_trace"] else "❌ DIFFERENT"
            print(f"  Same as main trace: {status}")
        if "should_manage" in r:
            print(f"  Handler managing trace: {r['should_manage']}")
        if "error" in r:
            print(f"  Error: {r['error']}")


def test_handler_run_id_tracking():
    """Test how handler tracks runs across threads."""
    print("\n" + "=" * 80)
    print("TEST: Handler Run ID Tracking Across Threads")
    print("=" * 80)

    from noveum_trace import init, get_client
    from noveum_trace.core.context import set_current_trace, get_current_trace
    from noveum_trace.integrations.langchain import NoveumTraceCallbackHandler

    # Initialize
    init(api_key="test", endpoint="http://localhost:4318/v1/traces", debug=True)

    client = get_client()
    main_trace = client.start_trace("run-tracking-test")
    set_current_trace(main_trace)
    print(f"Main thread trace: {main_trace.trace_id}")

    handler = NoveumTraceCallbackHandler()

    # Simulate a parent call in main thread
    parent_run_id = uuid4()
    handler.on_chain_start(
        serialized={"name": "parent_chain"},
        inputs={"input": "test"},
        run_id=parent_run_id,
        parent_run_id=None  # Root call
    )
    print(f"\nStarted parent chain with run_id: {parent_run_id}")
    print(f"Handler runs after parent: {list(handler.runs.keys())}")

    results = []

    def simulate_child_callback():
        """Simulate child callback in worker thread."""
        child_run_id = uuid4()
        thread = threading.current_thread()
        
        # Check what trace context is visible
        current_trace = get_current_trace()
        
        result = {
            "thread": thread.name,
            "trace_visible": current_trace is not None,
        }
        
        # Check if handler can find the parent span
        parent_span = handler._get_run(parent_run_id)
        result["parent_span_found"] = parent_span is not None
        result["parent_span_id"] = getattr(parent_span, "span_id", None) if parent_span else None
        
        # Simulate child call
        try:
            handler.on_llm_start(
                serialized={"name": "child_llm"},
                prompts=["test prompt"],
                run_id=child_run_id,
                parent_run_id=parent_run_id  # Reference parent
            )
            
            child_span = handler._get_run(child_run_id)
            result["child_span_created"] = child_span is not None
            result["child_span_id"] = getattr(child_span, "span_id", None) if child_span else None
            result["child_span_parent"] = getattr(child_span, "parent_span_id", None) if child_span else None
            
        except Exception as e:
            result["error"] = str(e)
            import traceback
            traceback.print_exc()
        
        results.append(result)
        return result

    # Test in worker thread WITHOUT context
    print("\n--- Child callback in worker thread (no context) ---")
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(simulate_child_callback)
        future.result()

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    for r in results:
        print(f"\nThread: {r['thread']}")
        print(f"  Trace visible: {r['trace_visible']}")
        print(f"  Parent span found via run_id: {r['parent_span_found']} ({r['parent_span_id']})")
        print(f"  Child span created: {r.get('child_span_created', False)}")
        if r.get("child_span_id"):
            print(f"  Child span ID: {r['child_span_id']}")
            print(f"  Child span parent: {r['child_span_parent']}")
        if "error" in r:
            print(f"  Error: {r['error']}")

    # Finish parent
    handler.on_chain_end(
        outputs={"output": "done"},
        run_id=parent_run_id
    )
    
    print(f"\nFinal handler state: {handler}")


def test_handler_parent_resolution():
    """Test how handler resolves parent span ID when context is lost."""
    print("\n" + "=" * 80)
    print("TEST: Handler Parent Resolution Without Context")
    print("=" * 80)

    from noveum_trace import init, get_client
    from noveum_trace.core.context import set_current_trace, get_current_trace
    from noveum_trace.integrations.langchain import NoveumTraceCallbackHandler

    init(api_key="test", endpoint="http://localhost:4318/v1/traces", debug=True)

    client = get_client()
    main_trace = client.start_trace("parent-resolution-test")
    set_current_trace(main_trace)

    handler = NoveumTraceCallbackHandler(use_langchain_assigned_parent=True)

    # Create root span in main thread
    root_run_id = uuid4()
    handler.on_chain_start(
        serialized={"name": "root"},
        inputs={},
        run_id=root_run_id,
        parent_run_id=None
    )
    root_span = handler._get_run(root_run_id)
    print(f"Root span: {root_span}")
    print(f"Root span ID: {root_span.span_id if root_span else None}")

    results = []

    def create_child_in_worker():
        """Create child span in worker thread."""
        child_run_id = uuid4()
        
        # Test _resolve_parent_span_id
        parent_span_id = handler._resolve_parent_span_id(root_run_id, None)
        
        results.append({
            "parent_span_id_resolved": parent_span_id,
            "expected_parent_span_id": root_span.span_id if root_span else None,
            "match": parent_span_id == (root_span.span_id if root_span else None)
        })

    # Run in worker thread
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(create_child_in_worker)
        future.result()

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    for r in results:
        status = "✅" if r["match"] else "❌"
        print(f"{status} Parent span ID resolved: {r['parent_span_id_resolved']}")
        print(f"   Expected: {r['expected_parent_span_id']}")

    handler.on_chain_end(outputs={}, run_id=root_run_id)


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("HANDLER CONTEXT ISSUE DEBUG")
    print("=" * 80)
    print("""
This debug script tests exactly how NoveumTraceCallbackHandler behaves
when callbacks are invoked from different threads:

1. Does the handler properly resolve parent span IDs?
2. Does the handler fall back correctly when context is lost?
3. Does the run_id -> span mapping work across threads?
    """)

    test_handler_context_resolution()
    test_handler_run_id_tracking()
    test_handler_parent_resolution()

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("""
The NoveumTraceCallbackHandler design has these safeguards:

1. **Thread-safe run tracking**: Uses locks for self.runs dictionary
2. **Parent run ID resolution**: Uses parent_run_id to find parent span
   in self.runs, regardless of thread context
3. **Root trace tracking**: Stores root traces by root_run_id

HOWEVER, there may still be issues when:

A) The first callback in a thread has no parent_run_id:
   - Handler tries to get_current_trace() from context
   - Context is empty -> creates NEW trace instead of using existing

B) The handler._get_or_create_trace_context relies on context:
   - If context is lost, it may create orphan traces

POTENTIAL FIXES:
1. Store the "current working trace" in the handler itself
2. Use a fallback mechanism when context is lost
3. Ensure root trace is always available via run_id chain
    """)


if __name__ == "__main__":
    main()
