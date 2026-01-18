"""
Debug script to investigate why LLM callbacks aren't firing when LLM runs in separate thread.

The issue: When the graph runs in one thread and LLM calls are made in another thread,
the callback handler's methods (on_llm_start, on_llm_end, etc.) are NOT being called at all.
"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.runnables import RunnableConfig
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain not available")


class VerboseCallbackHandler(BaseCallbackHandler):
    """Callback handler that loudly announces every callback."""
    
    def __init__(self, name: str = "VerboseHandler"):
        super().__init__()
        self.name = name
        self.call_count = 0
        self._lock = threading.Lock()
    
    def _log_callback(self, method_name: str, **kwargs):
        with self._lock:
            self.call_count += 1
            thread = threading.current_thread()
            print(f"\n{'='*60}")
            print(f"🔔 CALLBACK FIRED: {method_name}")
            print(f"   Handler: {self.name}")
            print(f"   Thread: {thread.name} (ID: {thread.ident})")
            print(f"   Call #: {self.call_count}")
            for k, v in kwargs.items():
                print(f"   {k}: {str(v)[:100]}")
            print(f"{'='*60}\n")
    
    def on_llm_start(self, serialized, prompts, *, run_id=None, **kwargs):
        self._log_callback("on_llm_start", run_id=run_id, prompts=prompts[:1] if prompts else None)
    
    def on_llm_end(self, response, *, run_id=None, **kwargs):
        self._log_callback("on_llm_end", run_id=run_id)
    
    def on_llm_error(self, error, *, run_id=None, **kwargs):
        self._log_callback("on_llm_error", run_id=run_id, error=error)
    
    def on_chat_model_start(self, serialized, messages, *, run_id=None, **kwargs):
        self._log_callback("on_chat_model_start", run_id=run_id, message_count=len(messages) if messages else 0)
    
    def on_chain_start(self, serialized, inputs, *, run_id=None, **kwargs):
        self._log_callback("on_chain_start", run_id=run_id, name=serialized.get("name") if serialized else "?")
    
    def on_chain_end(self, outputs, *, run_id=None, **kwargs):
        self._log_callback("on_chain_end", run_id=run_id)
    
    def on_tool_start(self, serialized, input_str, *, run_id=None, **kwargs):
        self._log_callback("on_tool_start", run_id=run_id)
    
    def on_tool_end(self, output, *, run_id=None, **kwargs):
        self._log_callback("on_tool_end", run_id=run_id)


def test_llm_in_main_thread():
    """Test 1: LLM call in main thread - callbacks should fire."""
    print("\n" + "=" * 80)
    print("TEST 1: LLM in MAIN thread")
    print("=" * 80)
    
    try:
        from langchain_community.llms.fake import FakeListLLM
    except ImportError:
        print("FakeListLLM not available")
        return
    
    handler = VerboseCallbackHandler("MainThread-Handler")
    llm = FakeListLLM(responses=["Response from main thread"], callbacks=[handler])
    
    print(f"Main thread: {threading.current_thread().name}")
    print("Invoking LLM...")
    
    result = llm.invoke("test prompt")
    print(f"Result: {result}")
    print(f"Callbacks fired: {handler.call_count}")


def test_llm_in_worker_thread_no_callbacks():
    """Test 2: LLM call in worker thread WITHOUT passing callbacks - should NOT fire."""
    print("\n" + "=" * 80)
    print("TEST 2: LLM in WORKER thread (callbacks NOT passed to invoke)")
    print("=" * 80)
    
    try:
        from langchain_community.llms.fake import FakeListLLM
    except ImportError:
        print("FakeListLLM not available")
        return
    
    handler = VerboseCallbackHandler("WorkerThread-Handler")
    # Callbacks attached to LLM at construction
    llm = FakeListLLM(responses=["Response from worker"], callbacks=[handler])
    
    def worker():
        print(f"Worker thread: {threading.current_thread().name}")
        print("Invoking LLM from worker...")
        result = llm.invoke("test prompt")  # No config passed
        print(f"Result: {result}")
    
    t = threading.Thread(target=worker)
    t.start()
    t.join()
    
    print(f"Callbacks fired: {handler.call_count}")


def test_llm_in_worker_thread_with_callbacks():
    """Test 3: LLM call in worker thread WITH passing callbacks via config."""
    print("\n" + "=" * 80)
    print("TEST 3: LLM in WORKER thread (callbacks PASSED via config)")
    print("=" * 80)
    
    try:
        from langchain_community.llms.fake import FakeListLLM
    except ImportError:
        print("FakeListLLM not available")
        return
    
    handler = VerboseCallbackHandler("WorkerThread-Config-Handler")
    llm = FakeListLLM(responses=["Response from worker with config"])
    
    def worker():
        print(f"Worker thread: {threading.current_thread().name}")
        print("Invoking LLM with callbacks in config...")
        result = llm.invoke("test prompt", config={"callbacks": [handler]})
        print(f"Result: {result}")
    
    t = threading.Thread(target=worker)
    t.start()
    t.join()
    
    print(f"Callbacks fired: {handler.call_count}")


def test_llm_in_threadpool():
    """Test 4: LLM in ThreadPoolExecutor."""
    print("\n" + "=" * 80)
    print("TEST 4: LLM in ThreadPoolExecutor")
    print("=" * 80)
    
    try:
        from langchain_community.llms.fake import FakeListLLM
    except ImportError:
        print("FakeListLLM not available")
        return
    
    handler = VerboseCallbackHandler("ThreadPool-Handler")
    llm = FakeListLLM(responses=["Response from threadpool"], callbacks=[handler])
    
    def worker():
        print(f"Pool thread: {threading.current_thread().name}")
        return llm.invoke("test prompt", config={"callbacks": [handler]})
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(worker)
        result = future.result()
        print(f"Result: {result}")
    
    print(f"Callbacks fired: {handler.call_count}")


def test_langgraph_node_in_separate_thread():
    """Test 5: LangGraph where node runs LLM - simulating the user's scenario."""
    print("\n" + "=" * 80)
    print("TEST 5: LangGraph node running LLM (simulating threading issue)")
    print("=" * 80)
    
    try:
        from langgraph.graph import StateGraph, END
        from langchain_community.llms.fake import FakeListLLM
        from typing import TypedDict
    except ImportError as e:
        print(f"Dependencies not available: {e}")
        return
    
    handler = VerboseCallbackHandler("LangGraph-Handler")
    
    class State(TypedDict):
        result: str
    
    # Create LLM with callbacks attached
    llm = FakeListLLM(responses=["LangGraph LLM response"], callbacks=[handler])
    
    def node_that_calls_llm(state: State) -> State:
        print(f"\n>>> Node executing in thread: {threading.current_thread().name}")
        
        # This is what might happen in user's code
        # The LLM is invoked from within a node
        response = llm.invoke("process this")
        
        return {"result": response}
    
    workflow = StateGraph(State)
    workflow.add_node("llm_node", node_that_calls_llm)
    workflow.set_entry_point("llm_node")
    workflow.add_edge("llm_node", END)
    
    app = workflow.compile()
    
    print(f"Main thread: {threading.current_thread().name}")
    print("Invoking graph...")
    
    result = app.invoke(
        {"result": ""},
        config={"callbacks": [handler]}
    )
    
    print(f"Result: {result}")
    print(f"Callbacks fired: {handler.call_count}")


def test_langgraph_node_in_threadpool():
    """Test 6: Simulating LangGraph running nodes in ThreadPool (user's scenario)."""
    print("\n" + "=" * 80)
    print("TEST 6: LangGraph-like execution in ThreadPool")
    print("=" * 80)
    
    try:
        from langchain_community.llms.fake import FakeListLLM
    except ImportError:
        print("FakeListLLM not available")
        return
    
    handler = VerboseCallbackHandler("ThreadPoolNode-Handler")
    llm = FakeListLLM(responses=["ThreadPool node response"], callbacks=[handler])
    
    def graph_node_in_threadpool(config):
        """Simulates a LangGraph node running in a separate thread."""
        print(f"\n>>> Node running in thread: {threading.current_thread().name}")
        
        # Does the config/callbacks propagate?
        print(f">>> Config received: {config}")
        callbacks = config.get("callbacks", []) if config else []
        print(f">>> Callbacks in config: {len(callbacks)}")
        
        # Invoke LLM - does it use the callbacks?
        response = llm.invoke("process", config=config)
        return response
    
    print(f"Main thread: {threading.current_thread().name}")
    
    # Simulate how LangGraph might execute nodes
    config = {"callbacks": [handler]}
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(graph_node_in_threadpool, config)
        result = future.result()
    
    print(f"Result: {result}")
    print(f"Callbacks fired: {handler.call_count}")


def test_callback_manager_propagation():
    """Test 7: Check LangChain's callback manager propagation."""
    print("\n" + "=" * 80)
    print("TEST 7: Callback Manager Propagation")
    print("=" * 80)
    
    try:
        from langchain_core.callbacks.manager import CallbackManager
        from langchain_community.llms.fake import FakeListLLM
    except ImportError as e:
        print(f"Dependencies not available: {e}")
        return
    
    handler = VerboseCallbackHandler("CallbackManager-Handler")
    
    # Create callback manager explicitly
    callback_manager = CallbackManager(handlers=[handler])
    
    llm = FakeListLLM(responses=["Manager test response"])
    
    def worker():
        print(f"Worker thread: {threading.current_thread().name}")
        # Use callback manager directly
        result = llm.invoke("test", config={"callbacks": callback_manager})
        return result
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(worker)
        result = future.result()
    
    print(f"Result: {result}")
    print(f"Callbacks fired: {handler.call_count}")


def main():
    print("\n" + "=" * 80)
    print("DEBUGGING: WHY LLM CALLBACKS DON'T FIRE IN SEPARATE THREADS")
    print("=" * 80)
    
    test_llm_in_main_thread()
    test_llm_in_worker_thread_no_callbacks()
    test_llm_in_worker_thread_with_callbacks()
    test_llm_in_threadpool()
    test_langgraph_node_in_separate_thread()
    test_langgraph_node_in_threadpool()
    test_callback_manager_propagation()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
If callbacks fired in main thread but NOT in worker threads, check:

1. Are callbacks being PASSED to the invoke() call via config?
   llm.invoke("prompt", config={"callbacks": [handler]})

2. Are callbacks attached to the LLM at construction AND via config?
   Some LangChain versions need both.

3. Is the callback_manager being propagated through LangGraph nodes?
   LangGraph might not pass the config through to nested LLM calls.

4. Check if you need to use 'inheritable_callbacks' vs 'callbacks':
   config={"callbacks": [handler]}  # For this run only
   config={"inheritable_callbacks": [handler]}  # Propagates to children
    """)


if __name__ == "__main__":
    main()
