"""
Debug script for exo_venv environment with langchain-openai and langgraph.
"""

import threading
import os
from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models.fake import FakeListLLM


class VerboseHandler(BaseCallbackHandler):
    """Tracks all callback invocations."""
    
    def __init__(self, name="Handler"):
        super().__init__()
        self.name = name
        self.events = []
        self._lock = threading.Lock()
    
    def _log(self, event, **kwargs):
        with self._lock:
            thread = threading.current_thread().name
            self.events.append({"event": event, "thread": thread})
            print(f"🔔 [{self.name}] {event:30} | Thread: {thread}")
    
    def on_llm_start(self, *args, **kwargs): self._log("on_llm_start")
    def on_llm_end(self, *args, **kwargs): self._log("on_llm_end")
    def on_chat_model_start(self, *args, **kwargs): self._log("on_chat_model_start")
    def on_chain_start(self, *args, **kwargs): self._log("on_chain_start")
    def on_chain_end(self, *args, **kwargs): self._log("on_chain_end")
    
    def summary(self):
        llm = len([e for e in self.events if "llm" in e["event"] or "chat" in e["event"]])
        print(f"\n📊 Total: {len(self.events)} events, LLM: {llm}")
        if llm == 0:
            print("❌ NO LLM CALLBACKS FIRED!")


def test_basic_llm():
    """Test basic LLM callback firing."""
    print("\n" + "=" * 60)
    print("TEST: Basic FakeListLLM")
    print("=" * 60)
    
    handler = VerboseHandler("Basic")
    llm = FakeListLLM(responses=["test response"])
    
    result = llm.invoke("hello", config={"callbacks": [handler]})
    print(f"Result: {result}")
    handler.summary()


def test_langgraph_sync():
    """Test LangGraph sync invoke."""
    print("\n" + "=" * 60)
    print("TEST: LangGraph Sync Invoke")
    print("=" * 60)
    
    from langgraph.graph import StateGraph, END
    
    handler = VerboseHandler("LangGraph-Sync")
    llm = FakeListLLM(responses=["graph response"])
    
    class State(TypedDict):
        result: str
    
    def node_a(state: State, config: RunnableConfig) -> State:
        print(f">>> Node executing in: {threading.current_thread().name}")
        print(f">>> Config: {config is not None}")
        response = llm.invoke("query", config=config)
        return {"result": response}
    
    workflow = StateGraph(State)
    workflow.add_node("node", node_a)
    workflow.set_entry_point("node")
    workflow.add_edge("node", END)
    app = workflow.compile()
    
    result = app.invoke({"result": ""}, config={"callbacks": [handler]})
    print(f"Result: {result}")
    handler.summary()


def test_langgraph_async():
    """Test LangGraph async invoke."""
    print("\n" + "=" * 60)
    print("TEST: LangGraph Async Invoke (ainvoke)")
    print("=" * 60)
    
    import asyncio
    from langgraph.graph import StateGraph, END
    
    handler = VerboseHandler("LangGraph-Async")
    llm = FakeListLLM(responses=["async response"])
    
    class State(TypedDict):
        result: str
    
    async def async_node(state: State, config: RunnableConfig) -> State:
        print(f">>> Async node in: {threading.current_thread().name}")
        response = llm.invoke("query", config=config)
        return {"result": response}
    
    workflow = StateGraph(State)
    workflow.add_node("node", async_node)
    workflow.set_entry_point("node")
    workflow.add_edge("node", END)
    app = workflow.compile()
    
    async def run():
        return await app.ainvoke({"result": ""}, config={"callbacks": [handler]})
    
    result = asyncio.run(run())
    print(f"Result: {result}")
    handler.summary()


def test_langgraph_node_no_config():
    """Test when node doesn't accept/forward config - THIS IS LIKELY YOUR ISSUE."""
    print("\n" + "=" * 60)
    print("TEST: LangGraph Node WITHOUT Config Parameter")
    print("=" * 60)
    
    from langgraph.graph import StateGraph, END
    
    handler = VerboseHandler("No-Config")
    llm = FakeListLLM(responses=["no config response"])
    
    class State(TypedDict):
        result: str
    
    # NODE DOES NOT ACCEPT config PARAMETER!
    def node_without_config(state: State) -> State:
        print(f">>> Node (no config) in: {threading.current_thread().name}")
        # Can't pass config because we don't have it!
        response = llm.invoke("query")  # NO CONFIG PASSED
        return {"result": response}
    
    workflow = StateGraph(State)
    workflow.add_node("node", node_without_config)
    workflow.set_entry_point("node")
    workflow.add_edge("node", END)
    app = workflow.compile()
    
    result = app.invoke({"result": ""}, config={"callbacks": [handler]})
    print(f"Result: {result}")
    handler.summary()


def test_llm_with_callback_in_constructor():
    """Test when LLM has callback in constructor."""
    print("\n" + "=" * 60)
    print("TEST: LLM with Callback in Constructor")
    print("=" * 60)
    
    from langgraph.graph import StateGraph, END
    
    handler = VerboseHandler("Constructor")
    
    # Callback attached at construction time
    llm = FakeListLLM(responses=["constructor response"], callbacks=[handler])
    
    class State(TypedDict):
        result: str
    
    def node(state: State) -> State:
        print(f">>> Node in: {threading.current_thread().name}")
        response = llm.invoke("query")  # No config, but LLM has callbacks
        return {"result": response}
    
    workflow = StateGraph(State)
    workflow.add_node("node", node)
    workflow.set_entry_point("node")
    workflow.add_edge("node", END)
    app = workflow.compile()
    
    result = app.invoke({"result": ""})  # Note: no callbacks in invoke either
    print(f"Result: {result}")
    handler.summary()


def test_threaded_execution():
    """Test execution in explicit ThreadPoolExecutor."""
    print("\n" + "=" * 60)
    print("TEST: Explicit ThreadPoolExecutor")
    print("=" * 60)
    
    from langgraph.graph import StateGraph, END
    
    handler = VerboseHandler("ThreadPool")
    llm = FakeListLLM(responses=["threaded response"])
    
    class State(TypedDict):
        result: str
    
    def run_in_thread():
        """Simulates what might happen with certain executors."""
        print(f">>> Running in: {threading.current_thread().name}")
        
        def node(state: State, config: RunnableConfig) -> State:
            print(f">>> Inner node in: {threading.current_thread().name}")
            return {"result": llm.invoke("q", config=config)}
        
        workflow = StateGraph(State)
        workflow.add_node("node", node)
        workflow.set_entry_point("node")
        workflow.add_edge("node", END)
        app = workflow.compile()
        
        return app.invoke({"result": ""}, config={"callbacks": [handler]})
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_in_thread)
        result = future.result()
    
    print(f"Result: {result}")
    handler.summary()


def main():
    print("=" * 60)
    print("DEBUGGING CALLBACK ISSUES - exo_venv Python 3.12")
    print("=" * 60)
    
    test_basic_llm()
    test_langgraph_sync()
    test_langgraph_async()
    test_langgraph_node_no_config()
    test_llm_with_callback_in_constructor()
    test_threaded_execution()
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("""
If LLM callbacks fired in some tests but not others:

1. Check if your node function accepts `config: RunnableConfig`
2. Check if you're passing config to llm.invoke()
3. Or attach callbacks to LLM at construction time
    """)


if __name__ == "__main__":
    main()
