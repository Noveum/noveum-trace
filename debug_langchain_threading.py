"""
Debug script to test actual LangChain/LangGraph threading behavior with callbacks.

This tests whether the NoveumTraceCallbackHandler correctly receives and processes
callbacks when LangGraph runs operations in different threads.
"""

import os
import logging
import contextvars
from typing import Any
from uuid import uuid4

# Set up logging to see what's happening
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Check if LangChain is available
try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.messages import HumanMessage, AIMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain not available - skipping real LangChain tests")


class DebugCallbackHandler(BaseCallbackHandler):
    """
    A debug callback handler that tracks which thread callbacks are invoked in.
    """

    def __init__(self):
        super().__init__()
        self.callback_invocations = []
        import threading
        self._main_thread_id = threading.current_thread().ident

    def _record_callback(self, callback_name: str, run_id: Any, parent_run_id: Any = None):
        """Record callback invocation details."""
        import threading
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
            "span_found": span is not None,
            "span_id": getattr(span, "span_id", None) if span else None,
        }
        self.callback_invocations.append(record)
        logger.info(f"Callback: {callback_name} | Thread: {current_thread.name} | "
                   f"Trace: {record['trace_id']} | Main: {is_main_thread}")
        return record

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs):
        self._record_callback("on_llm_start", run_id, parent_run_id)

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        self._record_callback("on_llm_end", run_id, parent_run_id)

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kwargs):
        self._record_callback("on_chat_model_start", run_id, parent_run_id)

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        self._record_callback("on_chain_start", run_id, parent_run_id)

    def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kwargs):
        self._record_callback("on_chain_end", run_id, parent_run_id)

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kwargs):
        self._record_callback("on_tool_start", run_id, parent_run_id)

    def on_tool_end(self, output, *, run_id, parent_run_id=None, **kwargs):
        self._record_callback("on_tool_end", run_id, parent_run_id)

    def print_summary(self):
        """Print a summary of all callback invocations."""
        print("\n" + "=" * 80)
        print("CALLBACK INVOCATION SUMMARY")
        print("=" * 80)

        for i, record in enumerate(self.callback_invocations):
            trace_status = "✅" if record["trace_found"] else "❌"
            thread_status = "Main" if record["is_main_thread"] else "Worker"
            print(f"{i+1}. {record['callback']:25} | {thread_status:8} | "
                  f"Trace: {trace_status} {record['trace_id'] or 'None':36}")

        # Analyze results
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
            print("Affected callbacks:")
            for r in callbacks_without_trace:
                print(f"  - {r['callback']} (Thread: {r['thread_name']})")


def test_with_fake_llm():
    """Test with FakeLLM to avoid API costs but still test callback flow."""
    print("\n" + "=" * 80)
    print("TEST: Callback Handler with FakeLLM")
    print("=" * 80)

    if not LANGCHAIN_AVAILABLE:
        print("LangChain not available - skipping test")
        return

    try:
        from langchain_community.llms.fake import FakeListLLM
        from langchain_core.prompts import PromptTemplate
    except ImportError:
        print("langchain_community not installed - skipping test")
        return

    from noveum_trace import init, get_client
    from noveum_trace.core.context import set_current_trace, get_current_trace
    from noveum_trace.core.trace import Trace

    # Initialize noveum-trace
    init(api_key="test-key-debug", endpoint="http://localhost:4318/v1/traces", debug=True)

    # Create a mock trace
    client = get_client()
    trace = client.start_trace("test-fakellm-trace")
    set_current_trace(trace)
    print(f"Set up trace: {trace.trace_id}")
    print(f"Trace in main thread: {get_current_trace()}")

    # Create debug handler
    debug_handler = DebugCallbackHandler()

    # Create fake LLM
    fake_llm = FakeListLLM(
        responses=["This is a fake response from the LLM."],
        callbacks=[debug_handler]
    )

    # Run a simple chain
    prompt = PromptTemplate.from_template("Tell me about {topic}")
    chain = prompt | fake_llm

    try:
        result = chain.invoke(
            {"topic": "Python"},
            config={"callbacks": [debug_handler]}
        )
        print(f"\nResult: {result}")
    except Exception as e:
        print(f"Error: {e}")

    debug_handler.print_summary()


def test_with_langgraph():
    """Test with LangGraph to see threading behavior."""
    print("\n" + "=" * 80)
    print("TEST: Callback Handler with LangGraph")
    print("=" * 80)

    if not LANGCHAIN_AVAILABLE:
        print("LangChain not available - skipping test")
        return

    try:
        from langgraph.graph import StateGraph, END
        from langchain_community.llms.fake import FakeListLLM
        from typing import TypedDict
    except ImportError as e:
        print(f"LangGraph or dependencies not installed - skipping test: {e}")
        return

    from noveum_trace import init, get_client
    from noveum_trace.core.context import set_current_trace, get_current_trace

    # Initialize noveum-trace
    init(api_key="test-key-debug", endpoint="http://localhost:4318/v1/traces", debug=True)

    # Create a mock trace
    client = get_client()
    trace = client.start_trace("test-langgraph-trace")
    set_current_trace(trace)
    print(f"Set up trace: {trace.trace_id}")

    # Create debug handler
    debug_handler = DebugCallbackHandler()

    # Define state
    class GraphState(TypedDict):
        messages: list[str]
        result: str

    # Create fake LLM
    fake_llm = FakeListLLM(responses=["Graph node response"])

    def node_a(state: GraphState) -> GraphState:
        """First node - calls LLM."""
        logger.info(f"Node A executing, trace: {get_current_trace()}")
        response = fake_llm.invoke("Process this")
        return {"messages": state["messages"] + ["A processed"], "result": response}

    def node_b(state: GraphState) -> GraphState:
        """Second node."""
        logger.info(f"Node B executing, trace: {get_current_trace()}")
        return {"messages": state["messages"] + ["B processed"], "result": state["result"]}

    # Build graph
    workflow = StateGraph(GraphState)
    workflow.add_node("node_a", node_a)
    workflow.add_node("node_b", node_b)
    workflow.set_entry_point("node_a")
    workflow.add_edge("node_a", "node_b")
    workflow.add_edge("node_b", END)

    app = workflow.compile()

    try:
        result = app.invoke(
            {"messages": ["start"], "result": ""},
            config={"callbacks": [debug_handler]}
        )
        print(f"\nResult: {result}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    debug_handler.print_summary()


def test_noveum_handler_directly():
    """Test the actual NoveumTraceCallbackHandler."""
    print("\n" + "=" * 80)
    print("TEST: NoveumTraceCallbackHandler Directly")
    print("=" * 80)

    if not LANGCHAIN_AVAILABLE:
        print("LangChain not available - skipping test")
        return

    try:
        from langchain_community.llms.fake import FakeListLLM
        from langchain_core.prompts import PromptTemplate
    except ImportError:
        print("langchain_community not installed - skipping test")
        return

    from noveum_trace import init, get_client
    from noveum_trace.core.context import set_current_trace, get_current_trace
    from noveum_trace.integrations.langchain import NoveumTraceCallbackHandler

    # Initialize noveum-trace
    init(api_key="test-key-debug", endpoint="http://localhost:4318/v1/traces", debug=True)

    # Create a mock trace
    client = get_client()
    trace = client.start_trace("test-noveum-handler-trace")
    set_current_trace(trace)
    print(f"Set up trace: {trace.trace_id}")

    # Create Noveum handler
    noveum_handler = NoveumTraceCallbackHandler()
    debug_handler = DebugCallbackHandler()

    # Create fake LLM
    fake_llm = FakeListLLM(
        responses=["This is a response from the Noveum test."],
    )

    # Run a simple chain
    prompt = PromptTemplate.from_template("Tell me about {topic}")
    chain = prompt | fake_llm

    try:
        result = chain.invoke(
            {"topic": "tracing"},
            config={"callbacks": [noveum_handler, debug_handler]}
        )
        print(f"\nResult: {result}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    # Show debug handler summary
    debug_handler.print_summary()

    # Show Noveum handler state
    print("\n" + "-" * 80)
    print("NOVEUM HANDLER STATE")
    print("-" * 80)
    print(f"Active runs: {noveum_handler._active_runs()}")
    print(f"Handler: {noveum_handler}")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("LANGCHAIN THREADING DEBUG TESTS")
    print("=" * 80)

    test_with_fake_llm()
    test_with_langgraph()
    test_noveum_handler_directly()

    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
