"""
Find LCEL-Based Agents that Send Non-Dict Inputs

Based on investigation: LangGraph's create_react_agent uses RunnableSequence
and sends list inputs to on_chain_start. LangChain has similar LCEL-based agents:
- create_openai_functions_agent
- create_tool_calling_agent  
- create_react_agent (LangChain version, not LangGraph)

These likely exhibit the same behavior.
"""

import os
from dotenv import load_dotenv

load_dotenv()

import noveum_trace
from noveum_trace.integrations import NoveumTraceCallbackHandler

noveum_trace.init(
    project=os.getenv("NOVEUM_PROJECT", "find-lcel-agents"),
    api_key=os.getenv("NOVEUM_API_KEY"),
    environment="development",
)


def test_langchain_create_react_agent():
    """Test LangChain's create_react_agent (LCEL-based)."""
    print("\n=== TEST: LangChain create_react_agent (LCEL) ===")
    
    try:
        from langchain.agents import create_react_agent, AgentExecutor
        from langchain import hub
        from langchain.tools import Tool
        from langchain_openai import ChatOpenAI
        
        callback_handler = NoveumTraceCallbackHandler()
        
        def tool_func(x: str) -> str:
            return f"Result: {x}"
        
        tool = Tool(name="TestTool", func=tool_func, description="Test tool")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, callbacks=[callback_handler])
        
        # Get ReAct prompt from hub
        prompt = hub.pull("hwchase17/react")
        
        # Create LCEL-based agent
        agent = create_react_agent(llm, [tool], prompt)
        
        # Wrap in AgentExecutor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=[tool],
            callbacks=[callback_handler],
            verbose=True,
        )
        
        result = agent_executor.invoke({"input": "Use TestTool with 'test data'"})
        
        print("✓ Success - completed")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()


def test_langchain_create_openai_functions_agent():
    """Test LangChain's create_openai_functions_agent (LCEL-based)."""
    print("\n=== TEST: LangChain create_openai_functions_agent (LCEL) ===")
    
    try:
        from langchain.agents import create_openai_functions_agent, AgentExecutor
        from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain.tools import Tool
        from langchain_openai import ChatOpenAI
        
        callback_handler = NoveumTraceCallbackHandler()
        
        def tool_func(x: str) -> str:
            return f"Result: {x}"
        
        tool = Tool(name="TestTool", func=tool_func, description="Test tool")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, callbacks=[callback_handler])
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])
        
        # Create LCEL-based OpenAI Functions agent
        agent = create_openai_functions_agent(llm, [tool], prompt)
        
        # Wrap in AgentExecutor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=[tool],
            callbacks=[callback_handler],
            verbose=True,
        )
        
        result = agent_executor.invoke({"input": "Use TestTool with 'test data'"})
        
        print("✓ Success - completed")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()


def test_langchain_create_tool_calling_agent():
    """Test LangChain's create_tool_calling_agent (LCEL-based)."""
    print("\n=== TEST: LangChain create_tool_calling_agent (LCEL) ===")
    
    try:
        from langchain.agents import create_tool_calling_agent, AgentExecutor
        from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain.tools import Tool
        from langchain_openai import ChatOpenAI
        
        callback_handler = NoveumTraceCallbackHandler()
        
        def tool_func(x: str) -> str:
            return f"Result: {x}"
        
        tool = Tool(name="TestTool", func=tool_func, description="Test tool")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, callbacks=[callback_handler])
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])
        
        # Create LCEL-based tool calling agent
        agent = create_tool_calling_agent(llm, [tool], prompt)
        
        # Wrap in AgentExecutor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=[tool],
            callbacks=[callback_handler],
            verbose=True,
        )
        
        result = agent_executor.invoke({"input": "Use TestTool with 'test data'"})
        
        print("✓ Success - completed")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()


def test_langchain_create_structured_chat_agent():
    """Test LangChain's create_structured_chat_agent (LCEL-based)."""
    print("\n=== TEST: LangChain create_structured_chat_agent (LCEL) ===")
    
    try:
        from langchain.agents import create_structured_chat_agent, AgentExecutor
        from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain.tools import Tool
        from langchain_openai import ChatOpenAI
        
        callback_handler = NoveumTraceCallbackHandler()
        
        def tool_func(x: str) -> str:
            return f"Result: {x}"
        
        tool = Tool(name="TestTool", func=tool_func, description="Test tool")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, callbacks=[callback_handler])
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])
        
        # Create LCEL-based structured chat agent
        agent = create_structured_chat_agent(llm, [tool], prompt)
        
        # Wrap in AgentExecutor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=[tool],
            callbacks=[callback_handler],
            verbose=True,
        )
        
        result = agent_executor.invoke({"input": "Use TestTool with 'test data'"})
        
        print("✓ Success - completed")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()


def test_langchain_create_json_chat_agent():
    """Test LangChain's create_json_chat_agent (LCEL-based)."""
    print("\n=== TEST: LangChain create_json_chat_agent (LCEL) ===")
    
    try:
        from langchain.agents import create_json_chat_agent, AgentExecutor
        from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain.tools import Tool
        from langchain_openai import ChatOpenAI
        
        callback_handler = NoveumTraceCallbackHandler()
        
        def tool_func(x: str) -> str:
            return f"Result: {x}"
        
        tool = Tool(name="TestTool", func=tool_func, description="Test tool")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, callbacks=[callback_handler])
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])
        
        # Create LCEL-based JSON chat agent
        agent = create_json_chat_agent(llm, [tool], prompt)
        
        # Wrap in AgentExecutor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=[tool],
            callbacks=[callback_handler],
            verbose=True,
        )
        
        result = agent_executor.invoke({"input": "Use TestTool with 'test data'"})
        
        print("✓ Success - completed")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("=" * 80)
    print("TESTING LCEL-BASED AGENTS FROM LANGCHAIN")
    print("=" * 80)
    print("\nThese agents use RunnableSequence internally, similar to LangGraph.")
    print("They likely send list inputs to on_chain_start, causing the same failure.\n")
    
    tests = [
        test_langchain_create_react_agent,
        test_langchain_create_openai_functions_agent,
        test_langchain_create_tool_calling_agent,
        test_langchain_create_structured_chat_agent,
        test_langchain_create_json_chat_agent,
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"\nTest exception: {e}")
    
    print("\n" + "=" * 80)
    print("CHECK LOGS ABOVE FOR 'inputs:' DEBUG PRINTS")
    print("=" * 80)
    print("\nLook for:")
    print("  - 'inputs: [...]' indicates list input (FAILURE)")
    print("  - 'ERROR.*chain start.*list' indicates the .items() failure")
    
    noveum_trace.flush()


if __name__ == "__main__":
    main()

