"""
Debug script for REAL scenarios where callbacks don't propagate.

Common causes:
1. LLM created without callbacks, and config not passed to invoke()
2. Node function doesn't forward RunnableConfig to nested calls
3. Using LLM directly instead of through a chain/runnable
"""

import threading
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.runnables import RunnableConfig
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


class VerboseHandler(BaseCallbackHandler):
    def __init__(self, name="Handler"):
        self.name = name
        self.calls = []
    
    def _log(self, method):
        self.calls.append(method)
        print(f"🔔 [{self.name}] {method} in {threading.current_thread().name}")
    
    def on_llm_start(self, *args, **kwargs): self._log("on_llm_start")
    def on_llm_end(self, *args, **kwargs): self._log("on_llm_end")
    def on_chat_model_start(self, *args, **kwargs): self._log("on_chat_model_start")
    def on_chain_start(self, *args, **kwargs): self._log("on_chain_start")
    def on_chain_end(self, *args, **kwargs): self._log("on_chain_end")


def test_scenario_1_llm_created_in_node():
    """
    SCENARIO 1: LLM is created INSIDE the node function
    
    This is a common pattern where the LLM doesn't receive callbacks
    because it's instantiated without them.
    """
    print("\n" + "=" * 80)
    print("SCENARIO 1: LLM created INSIDE node (callbacks NOT attached)")
    print("=" * 80)
    
    try:
        from langchain_community.llms.fake import FakeListLLM
        from langgraph.graph import StateGraph, END
        from typing import TypedDict
    except ImportError as e:
        print(f"Skip: {e}")
        return
    
    handler = VerboseHandler("Scenario1")
    
    class State(TypedDict):
        result: str
    
    def node_creates_llm(state: State) -> State:
        print(f"Node running in: {threading.current_thread().name}")
        
        # LLM created here - NO callbacks attached!
        llm = FakeListLLM(responses=["Response"])
        
        # Invoke without config - callbacks won't fire
        response = llm.invoke("query")
        return {"result": response}
    
    workflow = StateGraph(State)
    workflow.add_node("node", node_creates_llm)
    workflow.set_entry_point("node")
    workflow.add_edge("node", END)
    app = workflow.compile()
    
    result = app.invoke({"result": ""}, config={"callbacks": [handler]})
    print(f"Result: {result}")
    print(f"❌ LLM callbacks fired: {len([c for c in handler.calls if 'llm' in c])}")
    print("   (Expected 0 because LLM was created without callbacks)")


def test_scenario_2_llm_invoke_without_config():
    """
    SCENARIO 2: LLM exists with callbacks, but invoke() doesn't get config
    
    The node function doesn't forward the RunnableConfig.
    """
    print("\n" + "=" * 80)
    print("SCENARIO 2: LLM.invoke() called WITHOUT passing config")
    print("=" * 80)
    
    try:
        from langchain_community.llms.fake import FakeListLLM
        from langgraph.graph import StateGraph, END
        from typing import TypedDict
    except ImportError as e:
        print(f"Skip: {e}")
        return
    
    handler = VerboseHandler("Scenario2")
    
    # LLM created with callbacks
    llm = FakeListLLM(responses=["Response"], callbacks=[handler])
    
    class State(TypedDict):
        result: str
    
    def node_no_config_forward(state: State) -> State:
        print(f"Node running in: {threading.current_thread().name}")
        
        # LLM invoke WITHOUT config - uses LLM's default callbacks
        response = llm.invoke("query")  # <-- No config passed!
        return {"result": response}
    
    workflow = StateGraph(State)
    workflow.add_node("node", node_no_config_forward)
    workflow.set_entry_point("node")
    workflow.add_edge("node", END)
    app = workflow.compile()
    
    handler.calls.clear()
    result = app.invoke({"result": ""}, config={"callbacks": [handler]})
    print(f"Result: {result}")
    print(f"✅ LLM callbacks fired: {len([c for c in handler.calls if 'llm' in c])}")
    print("   (Works because LLM was created with callbacks)")


def test_scenario_3_proper_config_forwarding():
    """
    SCENARIO 3: Proper pattern - forwarding config to nested calls
    """
    print("\n" + "=" * 80)
    print("SCENARIO 3: PROPER config forwarding to LLM")
    print("=" * 80)
    
    try:
        from langchain_community.llms.fake import FakeListLLM
        from langgraph.graph import StateGraph, END
        from typing import TypedDict
        from langchain_core.runnables import RunnableConfig
    except ImportError as e:
        print(f"Skip: {e}")
        return
    
    handler = VerboseHandler("Scenario3")
    
    # LLM created WITHOUT callbacks
    llm = FakeListLLM(responses=["Response"])
    
    class State(TypedDict):
        result: str
    
    def node_with_config(state: State, config: RunnableConfig) -> State:
        """Node that properly receives and forwards config."""
        print(f"Node running in: {threading.current_thread().name}")
        print(f"Config received: {config}")
        
        # Forward config to LLM invoke - THIS IS THE KEY!
        response = llm.invoke("query", config=config)
        return {"result": response}
    
    workflow = StateGraph(State)
    workflow.add_node("node", node_with_config)
    workflow.set_entry_point("node")
    workflow.add_edge("node", END)
    app = workflow.compile()
    
    result = app.invoke({"result": ""}, config={"callbacks": [handler]})
    print(f"Result: {result}")
    print(f"✅ LLM callbacks fired: {len([c for c in handler.calls if 'llm' in c])}")


def test_scenario_4_async_with_real_threading():
    """
    SCENARIO 4: Async execution where LangGraph might use different threads
    """
    print("\n" + "=" * 80)
    print("SCENARIO 4: Async execution with potential threading")
    print("=" * 80)
    
    try:
        from langchain_community.llms.fake import FakeListLLM
        from langgraph.graph import StateGraph, END
        from typing import TypedDict
        from langchain_core.runnables import RunnableConfig
        import asyncio
    except ImportError as e:
        print(f"Skip: {e}")
        return
    
    handler = VerboseHandler("Scenario4")
    llm = FakeListLLM(responses=["Async Response"])
    
    class State(TypedDict):
        result: str
    
    async def async_node(state: State, config: RunnableConfig) -> State:
        print(f"Async node in: {threading.current_thread().name}")
        
        # Using asyncio.to_thread to simulate async LLM call
        def sync_llm_call():
            print(f"LLM call in: {threading.current_thread().name}")
            return llm.invoke("query", config=config)
        
        response = await asyncio.to_thread(sync_llm_call)
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
    print(f"Callbacks: {handler.calls}")


def test_scenario_5_llm_as_class_attribute():
    """
    SCENARIO 5: LLM is a class attribute - common in real applications
    """
    print("\n" + "=" * 80)
    print("SCENARIO 5: LLM as class attribute (real-world pattern)")
    print("=" * 80)
    
    try:
        from langchain_community.llms.fake import FakeListLLM
        from langgraph.graph import StateGraph, END
        from typing import TypedDict
        from langchain_core.runnables import RunnableConfig
    except ImportError as e:
        print(f"Skip: {e}")
        return
    
    handler = VerboseHandler("Scenario5")
    
    class MyAgent:
        def __init__(self):
            # LLM created at init time - no callbacks known yet
            self.llm = FakeListLLM(responses=["Agent response"])
        
        def process(self, query: str, config: RunnableConfig = None) -> str:
            print(f"Agent.process in: {threading.current_thread().name}")
            
            # Does NOT forward config!
            return self.llm.invoke(query)  # <-- BUG: config not passed
        
        def process_fixed(self, query: str, config: RunnableConfig = None) -> str:
            print(f"Agent.process_fixed in: {threading.current_thread().name}")
            
            # Properly forwards config
            return self.llm.invoke(query, config=config)  # <-- FIXED
    
    agent = MyAgent()
    
    class State(TypedDict):
        result: str
    
    def buggy_node(state: State, config: RunnableConfig) -> State:
        response = agent.process("query", config)  # Bug: doesn't matter, agent ignores it
        return {"result": response}
    
    def fixed_node(state: State, config: RunnableConfig) -> State:
        response = agent.process_fixed("query", config)
        return {"result": response}
    
    # Test buggy version
    print("\n--- Buggy version ---")
    workflow = StateGraph(State)
    workflow.add_node("node", buggy_node)
    workflow.set_entry_point("node")
    workflow.add_edge("node", END)
    app = workflow.compile()
    
    handler.calls.clear()
    result = app.invoke({"result": ""}, config={"callbacks": [handler]})
    llm_calls_buggy = len([c for c in handler.calls if 'llm' in c])
    print(f"❌ LLM callbacks (buggy): {llm_calls_buggy}")
    
    # Test fixed version
    print("\n--- Fixed version ---")
    workflow2 = StateGraph(State)
    workflow2.add_node("node", fixed_node)
    workflow2.set_entry_point("node")
    workflow2.add_edge("node", END)
    app2 = workflow2.compile()
    
    handler.calls.clear()
    result = app2.invoke({"result": ""}, config={"callbacks": [handler]})
    llm_calls_fixed = len([c for c in handler.calls if 'llm' in c])
    print(f"✅ LLM callbacks (fixed): {llm_calls_fixed}")


def main():
    print("\n" + "=" * 80)
    print("REAL-WORLD SCENARIOS WHERE CALLBACKS DON'T FIRE")
    print("=" * 80)
    
    test_scenario_1_llm_created_in_node()
    test_scenario_2_llm_invoke_without_config()
    test_scenario_3_proper_config_forwarding()
    test_scenario_4_async_with_real_threading()
    test_scenario_5_llm_as_class_attribute()
    
    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("""
The callback handler itself is fine. The issue is HOW the LLM is invoked:

1. ❌ LLM created inside node without callbacks
2. ❌ LLM.invoke() called without passing config
3. ❌ Agent/class methods that don't forward config to LLM

SOLUTION: Always forward RunnableConfig to nested invoke() calls:

    def my_node(state: State, config: RunnableConfig) -> State:
        response = llm.invoke("query", config=config)  # Pass config!
        return {"result": response}

OR attach callbacks to LLM at creation time:

    llm = ChatOpenAI(callbacks=[handler])
    """)


if __name__ == "__main__":
    main()
