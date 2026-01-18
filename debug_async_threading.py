"""
Debug script to test ASYNC LangChain/LangGraph threading behavior.

The threading issue typically occurs when:
1. Using async execution (ainvoke, astream)
2. Using parallel nodes in LangGraph
3. Using batch execution
4. Using streaming with background threads
"""

import asyncio
import os
import logging
import threading
from typing import Any
from concurrent.futures import ThreadPoolExecutor
import contextvars

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Check if LangChain is available
try:
    from langchain_core.callbacks import BaseCallbackHandler, AsyncCallbackHandler
    from langchain_core.messages import HumanMessage, AIMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain not available - skipping tests")


class DebugAsyncCallbackHandler(AsyncCallbackHandler):
    """
    Async callback handler that tracks thread context.
    """

    def __init__(self):
        super().__init__()
        self.callback_invocations = []
        self._main_thread_id = threading.current_thread().ident
        self._lock = threading.Lock()

    def _record_callback(self, callback_name: str, run_id: Any, parent_run_id: Any = None):
        """Record callback invocation details."""
        from noveum_trace.core.context import get_current_trace, get_current_span

        current_thread = threading.current_thread()
        is_main_thread = current_thread.ident == self._main_thread_id

        trace = get_current_trace()
        span = get_current_span()

        record = {
            "callback": callback_name,
            "run_id": str(run_id) if run_id else None,
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "thread_name": current_thread.name,
            "thread_id": current_thread.ident,
            "is_main_thread": is_main_thread,
            "trace_found": trace is not None,
            "trace_id": getattr(trace, "trace_id", None) if trace else None,
        }
        
        with self._lock:
            self.callback_invocations.append(record)
        
        status = "✅" if record["trace_found"] else "❌"
        logger.info(f"{status} {callback_name:25} | {current_thread.name:20} | "
                   f"Trace: {record['trace_id'] or 'None'}")
        return record

    async def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs):
        self._record_callback("on_llm_start", run_id, parent_run_id)

    async def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        self._record_callback("on_llm_end", run_id, parent_run_id)

    async def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kwargs):
        self._record_callback("on_chat_model_start", run_id, parent_run_id)

    async def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        self._record_callback("on_chain_start", run_id, parent_run_id)

    async def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kwargs):
        self._record_callback("on_chain_end", run_id, parent_run_id)

    async def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kwargs):
        self._record_callback("on_tool_start", run_id, parent_run_id)

    async def on_tool_end(self, output, *, run_id, parent_run_id=None, **kwargs):
        self._record_callback("on_tool_end", run_id, parent_run_id)

    def print_summary(self):
        """Print a summary of all callback invocations."""
        print("\n" + "=" * 80)
        print("CALLBACK INVOCATION SUMMARY")
        print("=" * 80)

        for i, record in enumerate(self.callback_invocations):
            trace_status = "✅" if record["trace_found"] else "❌"
            thread_status = "Main" if record["is_main_thread"] else "Worker"
            print(f"{i+1:2}. {record['callback']:25} | {thread_status:8} | "
                  f"Thread: {record['thread_name']:20} | Trace: {trace_status}")

        main_thread_callbacks = [r for r in self.callback_invocations if r["is_main_thread"]]
        worker_thread_callbacks = [r for r in self.callback_invocations if not r["is_main_thread"]]
        callbacks_with_trace = [r for r in self.callback_invocations if r["trace_found"]]
        callbacks_without_trace = [r for r in self.callback_invocations if not r["trace_found"]]

        print("\n" + "-" * 80)
        print("ANALYSIS")
        print("-" * 80)
        print(f"Total callbacks: {len(self.callback_invocations)}")
        print(f"Main thread callbacks: {len(main_thread_callbacks)}")
        print(f"Worker thread callbacks: {len(worker_thread_callbacks)}")
        print(f"Callbacks WITH trace context: {len(callbacks_with_trace)}")
        print(f"Callbacks WITHOUT trace context: {len(callbacks_without_trace)}")

        if callbacks_without_trace:
            print("\n⚠️  ISSUE DETECTED: Some callbacks lost trace context!")
            for r in callbacks_without_trace:
                print(f"  - {r['callback']} (Thread: {r['thread_name']})")
        else:
            print("\n✅ All callbacks have trace context!")


async def test_async_langgraph():
    """Test async LangGraph execution."""
    print("\n" + "=" * 80)
    print("TEST: ASYNC LangGraph Execution")
    print("=" * 80)

    if not LANGCHAIN_AVAILABLE:
        print("LangChain not available - skipping test")
        return

    try:
        from langgraph.graph import StateGraph, END
        from langchain_community.llms.fake import FakeListLLM
        from typing import TypedDict
    except ImportError as e:
        print(f"Dependencies not available: {e}")
        return

    from noveum_trace import init, get_client
    from noveum_trace.core.context import set_current_trace, get_current_trace

    # Initialize
    init(api_key="test-key", endpoint="http://localhost:4318/v1/traces", debug=True)

    client = get_client()
    trace = client.start_trace("async-langgraph-test")
    set_current_trace(trace)
    print(f"Set up trace: {trace.trace_id}")

    debug_handler = DebugAsyncCallbackHandler()

    class GraphState(TypedDict):
        messages: list[str]
        result: str

    fake_llm = FakeListLLM(responses=["Async response"])

    async def async_node_a(state: GraphState) -> GraphState:
        logger.info(f"Async Node A - Trace: {get_current_trace()}")
        # Simulate async LLM call
        response = await asyncio.to_thread(fake_llm.invoke, "Process async")
        return {"messages": state["messages"] + ["A async"], "result": response}

    async def async_node_b(state: GraphState) -> GraphState:
        logger.info(f"Async Node B - Trace: {get_current_trace()}")
        return {"messages": state["messages"] + ["B async"], "result": state["result"]}

    workflow = StateGraph(GraphState)
    workflow.add_node("node_a", async_node_a)
    workflow.add_node("node_b", async_node_b)
    workflow.set_entry_point("node_a")
    workflow.add_edge("node_a", "node_b")
    workflow.add_edge("node_b", END)

    app = workflow.compile()

    try:
        result = await app.ainvoke(
            {"messages": ["start"], "result": ""},
            config={"callbacks": [debug_handler]}
        )
        print(f"\nResult: {result}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    debug_handler.print_summary()


async def test_parallel_nodes():
    """Test parallel node execution in LangGraph."""
    print("\n" + "=" * 80)
    print("TEST: Parallel Nodes in LangGraph")
    print("=" * 80)

    if not LANGCHAIN_AVAILABLE:
        print("LangChain not available - skipping test")
        return

    try:
        from langgraph.graph import StateGraph, END
        from langchain_community.llms.fake import FakeListLLM
        from typing import TypedDict, Annotated
        import operator
    except ImportError as e:
        print(f"Dependencies not available: {e}")
        return

    from noveum_trace import init, get_client
    from noveum_trace.core.context import set_current_trace, get_current_trace

    # Initialize
    init(api_key="test-key", endpoint="http://localhost:4318/v1/traces", debug=True)

    client = get_client()
    trace = client.start_trace("parallel-nodes-test")
    set_current_trace(trace)
    print(f"Set up trace: {trace.trace_id}")

    debug_handler = DebugAsyncCallbackHandler()

    class GraphState(TypedDict):
        results: Annotated[list[str], operator.add]

    fake_llm_1 = FakeListLLM(responses=["Response from branch 1"])
    fake_llm_2 = FakeListLLM(responses=["Response from branch 2"])

    async def branch_1(state: GraphState) -> GraphState:
        logger.info(f"Branch 1 - Trace: {get_current_trace()}")
        await asyncio.sleep(0.1)  # Simulate async work
        response = await asyncio.to_thread(fake_llm_1.invoke, "Branch 1 query")
        return {"results": [f"B1: {response}"]}

    async def branch_2(state: GraphState) -> GraphState:
        logger.info(f"Branch 2 - Trace: {get_current_trace()}")
        await asyncio.sleep(0.1)  # Simulate async work
        response = await asyncio.to_thread(fake_llm_2.invoke, "Branch 2 query")
        return {"results": [f"B2: {response}"]}

    def merge_results(state: GraphState) -> GraphState:
        logger.info(f"Merge - Trace: {get_current_trace()}")
        return state

    workflow = StateGraph(GraphState)
    workflow.add_node("branch_1", branch_1)
    workflow.add_node("branch_2", branch_2)
    workflow.add_node("merge", merge_results)

    # Parallel execution - both branches run from start
    workflow.set_entry_point("branch_1")
    # Note: LangGraph doesn't have direct parallel execution syntax in basic form
    # This tests sequential with async behavior
    workflow.add_edge("branch_1", "branch_2")
    workflow.add_edge("branch_2", "merge")
    workflow.add_edge("merge", END)

    app = workflow.compile()

    try:
        result = await app.ainvoke(
            {"results": []},
            config={"callbacks": [debug_handler]}
        )
        print(f"\nResult: {result}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    debug_handler.print_summary()


async def test_thread_executor_simulation():
    """Simulate what happens when LLM runs in thread executor."""
    print("\n" + "=" * 80)
    print("TEST: Thread Executor Simulation (Direct)")
    print("=" * 80)

    from noveum_trace import init, get_client
    from noveum_trace.core.context import set_current_trace, get_current_trace

    # Initialize
    init(api_key="test-key", endpoint="http://localhost:4318/v1/traces", debug=True)

    client = get_client()
    trace = client.start_trace("thread-executor-test")
    set_current_trace(trace)
    print(f"Main thread trace: {get_current_trace()}")

    results = []

    def check_trace_in_thread(name: str):
        current_trace = get_current_trace()
        thread = threading.current_thread()
        result = {
            "name": name,
            "thread": thread.name,
            "trace_found": current_trace is not None,
            "trace_id": getattr(current_trace, "trace_id", None) if current_trace else None,
        }
        results.append(result)
        return result

    # Test 1: Direct thread - NO context propagation
    print("\n1. Direct threading.Thread (no context propagation):")
    t = threading.Thread(target=check_trace_in_thread, args=("direct_thread",))
    t.start()
    t.join()

    # Test 2: ThreadPoolExecutor - NO context propagation
    print("\n2. ThreadPoolExecutor (no context propagation):")
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(check_trace_in_thread, f"pool_thread_{i}") for i in range(2)]
        for f in futures:
            f.result()

    # Test 3: ThreadPoolExecutor WITH context copy
    print("\n3. ThreadPoolExecutor WITH contextvars.copy_context():")
    ctx = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(ctx.run, check_trace_in_thread, f"pool_thread_ctx_{i}") for i in range(2)]
        for f in futures:
            f.result()

    # Test 4: asyncio.to_thread (should preserve context in Python 3.9+)
    print("\n4. asyncio.to_thread:")
    await asyncio.to_thread(check_trace_in_thread, "asyncio_to_thread")

    # Summary
    print("\n" + "-" * 80)
    print("RESULTS")
    print("-" * 80)
    for r in results:
        status = "✅" if r["trace_found"] else "❌"
        print(f"{status} {r['name']:25} | {r['thread']:20} | Trace: {r['trace_id'] or 'None'}")


async def main():
    """Run all async tests."""
    print("\n" + "=" * 80)
    print("ASYNC THREADING DEBUG TESTS")
    print("=" * 80)

    await test_thread_executor_simulation()
    await test_async_langgraph()
    await test_parallel_nodes()

    print("\n" + "=" * 80)
    print("ASYNC DEBUG COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
