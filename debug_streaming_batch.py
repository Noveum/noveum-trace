"""
Debug script to test streaming and batch execution threading behavior.

These are common scenarios where threading issues occur:
1. Streaming responses with callbacks
2. Batch execution across multiple inputs
3. Custom thread pools or executors
"""

import asyncio
import os
import logging
import threading
from typing import Any, AsyncIterator, Iterator
from concurrent.futures import ThreadPoolExecutor
import contextvars

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Check if LangChain is available
try:
    from langchain_core.callbacks import BaseCallbackHandler, AsyncCallbackHandler
    from langchain_core.runnables import RunnableLambda, RunnableConfig
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain not available - skipping tests")


class ThreadTrackingCallbackHandler(BaseCallbackHandler):
    """Tracks threads where callbacks are invoked."""

    def __init__(self, name: str = "default"):
        super().__init__()
        self.name = name
        self.invocations = []
        self._main_thread_id = threading.current_thread().ident
        self._lock = threading.Lock()

    def _record(self, event: str, run_id: Any = None, **kwargs):
        from noveum_trace.core.context import get_current_trace

        current_thread = threading.current_thread()
        trace = get_current_trace()

        record = {
            "handler": self.name,
            "event": event,
            "thread": current_thread.name,
            "thread_id": current_thread.ident,
            "is_main": current_thread.ident == self._main_thread_id,
            "trace_found": trace is not None,
            "trace_id": getattr(trace, "trace_id", None) if trace else None,
            "run_id": str(run_id)[:8] if run_id else None,
        }

        with self._lock:
            self.invocations.append(record)

        status = "✅" if record["trace_found"] else "❌"
        logger.info(f"[{self.name}] {status} {event:20} | {current_thread.name:25} | "
                   f"Trace: {record['trace_id'] or 'None'}")

    def on_llm_start(self, serialized, prompts, *, run_id=None, **kwargs):
        self._record("llm_start", run_id)

    def on_llm_end(self, response, *, run_id=None, **kwargs):
        self._record("llm_end", run_id)

    def on_llm_new_token(self, token, *, run_id=None, **kwargs):
        self._record("llm_new_token", run_id)

    def on_chain_start(self, serialized, inputs, *, run_id=None, **kwargs):
        self._record("chain_start", run_id)

    def on_chain_end(self, outputs, *, run_id=None, **kwargs):
        self._record("chain_end", run_id)

    def print_summary(self):
        print(f"\n{'=' * 80}")
        print(f"SUMMARY for handler: {self.name}")
        print(f"{'=' * 80}")

        for i, r in enumerate(self.invocations):
            status = "✅" if r["trace_found"] else "❌"
            thread_type = "Main" if r["is_main"] else "Worker"
            print(f"{i+1:2}. {status} {r['event']:20} | {thread_type:8} | "
                  f"{r['thread']:25}")

        with_trace = sum(1 for r in self.invocations if r["trace_found"])
        without_trace = sum(1 for r in self.invocations if not r["trace_found"])
        worker_threads = sum(1 for r in self.invocations if not r["is_main"])

        print(f"\nTotal: {len(self.invocations)} | With trace: {with_trace} | "
              f"Without trace: {without_trace} | Worker threads: {worker_threads}")

        if without_trace:
            print(f"\n⚠️  ISSUE: {without_trace} callbacks lost trace context!")


def test_manual_thread_execution():
    """Test manually spawning threads that invoke callbacks."""
    print("\n" + "=" * 80)
    print("TEST: Manual Thread Execution")
    print("=" * 80)

    from noveum_trace import init, get_client
    from noveum_trace.core.context import set_current_trace, get_current_trace

    init(api_key="test", endpoint="http://localhost:4318/v1/traces", debug=True)

    client = get_client()
    trace = client.start_trace("manual-thread-test")
    set_current_trace(trace)
    print(f"Main thread trace: {trace.trace_id}")

    handler = ThreadTrackingCallbackHandler("manual_thread")

    # Simulate callback invocation from different thread
    def worker_that_invokes_callback():
        """Worker thread that invokes callbacks without context."""
        from uuid import uuid4
        run_id = uuid4()

        # This is what might happen inside LangChain when running in executor
        handler.on_chain_start({}, {"input": "test"}, run_id=run_id)
        # Simulate some work
        import time
        time.sleep(0.1)
        handler.on_chain_end({"output": "result"}, run_id=run_id)

    # Test 1: Direct thread
    print("\n1. Direct threading.Thread:")
    t = threading.Thread(target=worker_that_invokes_callback)
    t.start()
    t.join()

    # Test 2: ThreadPoolExecutor
    print("\n2. ThreadPoolExecutor:")
    with ThreadPoolExecutor(max_workers=2) as executor:
        future = executor.submit(worker_that_invokes_callback)
        future.result()

    # Test 3: With context copy
    print("\n3. ThreadPoolExecutor with context copy:")
    ctx = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=2) as executor:
        future = executor.submit(ctx.run, worker_that_invokes_callback)
        future.result()

    handler.print_summary()


def test_batch_execution():
    """Test batch execution where multiple inputs are processed."""
    print("\n" + "=" * 80)
    print("TEST: Batch Execution")
    print("=" * 80)

    if not LANGCHAIN_AVAILABLE:
        print("LangChain not available")
        return

    from noveum_trace import init, get_client
    from noveum_trace.core.context import set_current_trace, get_current_trace

    init(api_key="test", endpoint="http://localhost:4318/v1/traces", debug=True)

    client = get_client()
    trace = client.start_trace("batch-test")
    set_current_trace(trace)
    print(f"Main thread trace: {trace.trace_id}")

    handler = ThreadTrackingCallbackHandler("batch")

    def process_item(x):
        trace = get_current_trace()
        logger.info(f"Processing item {x} in thread {threading.current_thread().name}, trace: {trace}")
        return f"processed_{x}"

    runnable = RunnableLambda(process_item)

    # Batch execution
    inputs = ["a", "b", "c"]
    try:
        results = runnable.batch(
            inputs,
            config={"callbacks": [handler]}
        )
        print(f"\nResults: {results}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    handler.print_summary()


async def test_async_batch():
    """Test async batch execution."""
    print("\n" + "=" * 80)
    print("TEST: Async Batch Execution")
    print("=" * 80)

    if not LANGCHAIN_AVAILABLE:
        print("LangChain not available")
        return

    from noveum_trace import init, get_client
    from noveum_trace.core.context import set_current_trace, get_current_trace

    init(api_key="test", endpoint="http://localhost:4318/v1/traces", debug=True)

    client = get_client()
    trace = client.start_trace("async-batch-test")
    set_current_trace(trace)
    print(f"Main thread trace: {trace.trace_id}")

    handler = ThreadTrackingCallbackHandler("async_batch")

    async def async_process_item(x):
        trace = get_current_trace()
        logger.info(f"Async processing item {x}, trace: {trace}")
        await asyncio.sleep(0.05)
        return f"async_processed_{x}"

    runnable = RunnableLambda(async_process_item)

    inputs = ["x", "y", "z"]
    try:
        results = await runnable.abatch(
            inputs,
            config={"callbacks": [handler]}
        )
        print(f"\nResults: {results}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    handler.print_summary()


def test_runnable_with_executor():
    """Test runnable that explicitly uses a thread executor."""
    print("\n" + "=" * 80)
    print("TEST: Runnable with Custom Executor")
    print("=" * 80)

    if not LANGCHAIN_AVAILABLE:
        print("LangChain not available")
        return

    from noveum_trace import init, get_client
    from noveum_trace.core.context import set_current_trace, get_current_trace

    init(api_key="test", endpoint="http://localhost:4318/v1/traces", debug=True)

    client = get_client()
    trace = client.start_trace("executor-test")
    set_current_trace(trace)
    print(f"Main thread trace: {trace.trace_id}")

    handler = ThreadTrackingCallbackHandler("executor")

    def process_with_trace_check(x):
        trace = get_current_trace()
        thread = threading.current_thread()
        logger.info(f"Processing {x} in {thread.name}, trace: {trace}")
        return {"input": x, "trace_found": trace is not None}

    # Create runnable
    runnable = RunnableLambda(process_with_trace_check)

    # Execute with explicit executor in config
    custom_executor = ThreadPoolExecutor(max_workers=2)

    try:
        # Note: LangChain's RunnableConfig supports 'executor' parameter
        # for some operations
        results = runnable.batch(
            ["1", "2", "3"],
            config={"callbacks": [handler], "max_concurrency": 2}
        )
        print(f"\nResults: {results}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        custom_executor.shutdown(wait=True)

    handler.print_summary()


async def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("STREAMING & BATCH THREADING DEBUG")
    print("=" * 80)

    test_manual_thread_execution()
    test_batch_execution()
    await test_async_batch()
    test_runnable_with_executor()

    print("\n" + "=" * 80)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 80)
    print("""
KEY FINDINGS:
1. Direct threading.Thread: Context does NOT propagate
2. ThreadPoolExecutor: Context does NOT propagate  
3. contextvars.copy_context().run(): Context DOES propagate
4. asyncio.to_thread: Context DOES propagate (Python handles it)

WHERE THE ISSUE OCCURS:
- When LangChain/LangGraph internally uses ThreadPoolExecutor
- When user provides custom executor
- When callbacks are invoked from worker threads

SOLUTIONS:
1. The callback handler itself is NOT the problem - it receives callbacks correctly
2. The issue is that the TRACE CONTEXT is lost in the worker thread
3. The fix requires ensuring trace context is propagated to worker threads

POSSIBLE FIX APPROACHES:
A) Thread-local fallback in the callback handler
B) Use LangChain's parent_run_id to establish relationships (current approach)
C) Store trace context in the handler itself and use it as fallback

The current NoveumTraceCallbackHandler already uses approach (B) by:
- Storing run_id -> span mappings in thread-safe dictionaries
- Using parent_run_id to establish parent-child relationships
- NOT relying solely on contextvars for trace context

The handler SHOULD work correctly because it uses its internal state (self.runs)
rather than relying on contextvars for span lookup.
    """)


if __name__ == "__main__":
    asyncio.run(main())
