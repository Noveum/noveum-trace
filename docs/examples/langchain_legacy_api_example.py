"""
Legacy LangChain API Example with Noveum Trace Integration

This example demonstrates using OLDER LangChain API patterns (pre-LCEL, v0.1/v0.2 style)
that explicitly trigger the callback handlers:
- on_chain_start (via LLMChain, SequentialChain)
- on_tool_start (via Tool class)
- on_agent_start (via initialize_agent, AgentExecutor)

This is useful for:
1. Testing callback handler compatibility with legacy code
2. Debugging inputs parameter handling in callbacks
3. Understanding migration from old to new API

Prerequisites:
    pip install noveum-trace[langchain]
    pip install langchain langchain-openai langchain-community langgraph

Environment Variables:
    NOVEUM_API_KEY: Your Noveum API key
    OPENAI_API_KEY: Your OpenAI API key
"""

import os
import time
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, AgentType, initialize_agent
from langchain.chains import LLMChain, SequentialChain, TransformChain
from langchain.prompts import PromptTemplate
from langchain.tools import StructuredTool, Tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# LangGraph prebuilt (compatibility layer for old agent patterns)
try:
    from langgraph.prebuilt import (
        create_react_agent,
        chat_agent_executor,
        ToolNode,
        tools_condition,
    )
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

import noveum_trace
from noveum_trace.integrations import NoveumTraceCallbackHandler

load_dotenv()

# =============================================================================
# LEGACY TOOLS (Pre-decorator style)
# =============================================================================


def calculator_function(expression: str) -> str:
    """
    Legacy tool function: Calculator.
    This is the old way of defining tools (before @tool decorator).
    """
    time.sleep(0.3)
    
    try:
        # Safety: Only allow basic math operations
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


def web_search_function(query: str) -> str:
    """
    Legacy tool function: Web Search.
    Simulates searching the web.
    """
    time.sleep(0.5)
    
    # Simulated search results
    search_db = {
        "python": "Python is a high-level programming language known for simplicity and readability.",
        "langchain": "LangChain is a framework for developing applications powered by language models.",
        "ai": "Artificial Intelligence is the simulation of human intelligence by machines.",
    }
    
    for key, content in search_db.items():
        if key.lower() in query.lower():
            return f"Search result: {content}"
    
    return "Search result: General information available."


def text_counter_function(text: str) -> str:
    """
    Legacy tool function: Text Counter.
    Counts words and characters in text.
    """
    time.sleep(0.2)
    
    words = text.split()
    return f"Character count: {len(text)}, Word count: {len(words)}"


# =============================================================================
# STRUCTURED TOOL (Old style with explicit schema)
# =============================================================================


class AnalyzerInput(BaseModel):
    """Input schema for the analyzer tool."""
    text: str = Field(description="The text to analyze")
    detail_level: str = Field(default="basic", description="Level of detail: basic or detailed")


def analyzer_function(text: str, detail_level: str = "basic") -> str:
    """
    Structured tool function: Text Analyzer.
    This demonstrates the old StructuredTool pattern.
    """
    time.sleep(0.3)
    
    words = text.split()
    chars = len(text)
    sentences = text.count('.') + text.count('!') + text.count('?')
    
    if detail_level == "detailed":
        return (f"Detailed Analysis:\n"
                f"- Characters: {chars}\n"
                f"- Words: {len(words)}\n"
                f"- Sentences: {sentences}\n"
                f"- Avg word length: {chars / len(words) if words else 0:.2f}")
    else:
        return f"Basic Analysis: {len(words)} words, {chars} characters"


# =============================================================================
# EXAMPLE 1: Legacy LLMChain (triggers on_chain_start)
# =============================================================================


def example_legacy_llm_chain():
    """
    Example 1: Legacy LLMChain pattern.
    This triggers on_chain_start with inputs dict.
    """
    # Initialize Noveum Trace
    noveum_trace.init(
        project=os.getenv("NOVEUM_PROJECT", "legacy-langchain-example"),
        api_key=os.getenv("NOVEUM_API_KEY"),
        environment="development",
    )
    
    # Create callback handler
    callback_handler = NoveumTraceCallbackHandler()
    
    # Create LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        callbacks=[callback_handler]
    )
    
    # Create legacy prompt template
    prompt = PromptTemplate(
        input_variables=["topic", "style"],
        template="Write a {style} sentence about {topic}."
    )
    
    # Create legacy LLMChain
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        callbacks=[callback_handler],
        verbose=True
    )
    
    # Run chain with inputs dict (triggers on_chain_start)
    inputs = {"topic": "artificial intelligence", "style": "technical"}
    
    try:
        result = chain.run(**inputs)
    except Exception as e:
        pass
    
    time.sleep(1)


# =============================================================================
# EXAMPLE 2: SequentialChain (multiple on_chain_start calls)
# =============================================================================


def example_sequential_chain():
    """
    Example 2: SequentialChain pattern.
    This triggers multiple on_chain_start callbacks in sequence.
    """
    # Create callback handler
    callback_handler = NoveumTraceCallbackHandler()
    
    # Create LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        callbacks=[callback_handler]
    )
    
    # First chain: Generate a topic
    prompt1 = PromptTemplate(
        input_variables=["subject"],
        template="Suggest a specific topic related to {subject}. Just return the topic name, nothing else."
    )
    chain1 = LLMChain(
        llm=llm,
        prompt=prompt1,
        output_key="topic",
        callbacks=[callback_handler],
        verbose=True
    )
    
    # Second chain: Write about the topic
    prompt2 = PromptTemplate(
        input_variables=["topic"],
        template="Write one sentence about {topic}."
    )
    chain2 = LLMChain(
        llm=llm,
        prompt=prompt2,
        output_key="description",
        callbacks=[callback_handler],
        verbose=True
    )
    
    # Create sequential chain
    sequential_chain = SequentialChain(
        chains=[chain1, chain2],
        input_variables=["subject"],
        output_variables=["topic", "description"],
        callbacks=[callback_handler],
        verbose=True
    )
    
    # Run sequential chain (triggers multiple on_chain_start)
    inputs = {"subject": "machine learning"}
    
    try:
        result = sequential_chain(inputs)
    except Exception as e:
        pass
    
    time.sleep(1)


# =============================================================================
# EXAMPLE 3: TransformChain (data transformation chain)
# =============================================================================


def example_transform_chain():
    """
    Example 3: TransformChain pattern.
    This shows data transformation with on_chain_start.
    """
    # Create callback handler
    callback_handler = NoveumTraceCallbackHandler()
    
    # Transform function
    def transform_func(inputs: Dict[str, Any]) -> Dict[str, Any]:
        text = inputs["text"]
        return {"transformed_text": text.upper(), "length": len(text)}
    
    # Create transform chain
    transform_chain = TransformChain(
        input_variables=["text"],
        output_variables=["transformed_text", "length"],
        transform=transform_func,
        callbacks=[callback_handler]
    )
    
    # Run transform chain (triggers on_chain_start)
    inputs = {"text": "hello world from legacy api"}
    
    try:
        result = transform_chain(inputs)
    except Exception as e:
        pass
    
    time.sleep(1)


# =============================================================================
# EXAMPLE 4: Legacy Agent with Tools (triggers on_agent_start, on_tool_start)
# =============================================================================


def example_legacy_agent_with_tools():
    """
    Example 4: Legacy initialize_agent pattern.
    This triggers on_agent_start and on_tool_start callbacks.
    """
    # Create callback handler
    callback_handler = NoveumTraceCallbackHandler()
    
    # Create LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        callbacks=[callback_handler]
    )
    
    # Create legacy tools using Tool class (old way)
    tools = [
        Tool(
            name="Calculator",
            func=calculator_function,
            description="Useful for performing mathematical calculations. Input should be a math expression."
        ),
        Tool(
            name="WebSearch",
            func=web_search_function,
            description="Search for information on the web. Input should be a search query."
        ),
        Tool(
            name="TextCounter",
            func=text_counter_function,
            description="Count words and characters in text. Input should be the text to count."
        ),
    ]
    
    # Initialize agent using legacy initialize_agent (old way)
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Old AgentType enum
        callbacks=[callback_handler],
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,
    )
    
    # Run agent with input dict (triggers on_agent_start and on_tool_start)
    query = "Search for information about Python, then count the words in the result."
    
    try:
        # Legacy .run() method with string input
        result = agent.run(query)
    except Exception as e:
        pass
    
    time.sleep(1)


# =============================================================================
# EXAMPLE 5: StructuredTool with explicit schema
# =============================================================================


def example_structured_tool():
    """
    Example 5: StructuredTool with explicit input schema (old pattern).
    This triggers on_tool_start with structured inputs.
    """
    # Create callback handler
    callback_handler = NoveumTraceCallbackHandler()
    
    # Create LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        callbacks=[callback_handler]
    )
    
    # Create StructuredTool (old way with explicit schema)
    structured_tool = StructuredTool.from_function(
        func=analyzer_function,
        name="TextAnalyzer",
        description="Analyze text with configurable detail level",
        args_schema=AnalyzerInput,
    )
    
    # Create tools list
    tools = [
        structured_tool,
        Tool(
            name="Calculator",
            func=calculator_function,
            description="Perform calculations"
        ),
    ]
    
    # Initialize agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        callbacks=[callback_handler],
        verbose=True,
    )
    
    # Run agent
    query = "Analyze this text with detailed level: 'LangChain is amazing for building AI apps'"
    
    try:
        result = agent.run(query)
    except Exception as e:
        pass
    
    time.sleep(1)


# =============================================================================
# EXAMPLE 6: Manual AgentExecutor (most explicit old pattern)
# =============================================================================


def example_manual_agent_executor():
    """
    Example 6: Manual AgentExecutor construction (old explicit pattern).
    This is the most explicit way to create agents in the old API.
    """
    # Create callback handler
    callback_handler = NoveumTraceCallbackHandler()
    
    # Create LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        callbacks=[callback_handler]
    )
    
    # Create tools
    tools = [
        Tool(
            name="Calculator",
            func=calculator_function,
            description="Calculate mathematical expressions"
        ),
        Tool(
            name="WebSearch",
            func=web_search_function,
            description="Search for information"
        ),
    ]
    
    # Use initialize_agent to get agent and then manually work with executor
    # This is the old way before LCEL
    agent_executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        callbacks=[callback_handler],
        verbose=True,
        return_intermediate_steps=True,  # Old pattern for getting steps
    )
    
    # Run with invoke (old pattern that accepts dict input)
    inputs = {"input": "Calculate 25 * 17 and then search for LangChain"}
    
    try:
        result = agent_executor.invoke(inputs)
    except Exception as e:
        pass
    
    time.sleep(1)


# =============================================================================
# EXAMPLE 7: LangGraph Prebuilt create_react_agent (compatibility layer)
# =============================================================================


def example_langgraph_prebuilt_agent():
    """
    Example 7: LangGraph's prebuilt create_react_agent.
    This is the compatibility layer that bridges old LangChain agents and LangGraph.
    It internally uses the old agent pattern but with graph execution.
    """
    if not LANGGRAPH_AVAILABLE:
        return
    
    # Initialize Noveum Trace
    noveum_trace.init(
        project=os.getenv("NOVEUM_PROJECT", "legacy-langchain-example"),
        api_key=os.getenv("NOVEUM_API_KEY"),
        environment="development",
    )
    
    # Create callback handler
    callback_handler = NoveumTraceCallbackHandler()
    
    # Create LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        callbacks=[callback_handler]
    )
    
    # Create tools (using old Tool class)
    tools = [
        Tool(
            name="Calculator",
            func=calculator_function,
            description="Calculate math expressions"
        ),
        Tool(
            name="WebSearch",
            func=web_search_function,
            description="Search for information"
        ),
    ]
    
    # Create agent using LangGraph's prebuilt (compatibility layer)
    agent_executor = create_react_agent(llm, tools)
    
    # Run with messages (LangGraph style but with old agent pattern internally)
    query = "Search for AI information, then calculate 100 * 5"
    
    # LangGraph uses messages but internally triggers old callbacks
    # This should trigger the exception where inputs is not a dict
    result = agent_executor.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"callbacks": [callback_handler]}
    )


# =============================================================================
# EXAMPLE 8: LangGraph ToolNode (direct tool execution)
# =============================================================================


def example_langgraph_tool_node():
    """
    Example 8: LangGraph's ToolNode for direct tool execution.
    This demonstrates how ToolNode handles tool execution.
    """
    if not LANGGRAPH_AVAILABLE:
        return
    
    # Initialize Noveum Trace
    noveum_trace.init(
        project=os.getenv("NOVEUM_PROJECT", "legacy-langchain-example"),
        api_key=os.getenv("NOVEUM_API_KEY"),
        environment="development",
    )
    
    # Create callback handler
    callback_handler = NoveumTraceCallbackHandler()
    
    # Create tools using @tool decorator (modern way that ToolNode expects)
    from langchain_core.tools import tool
    
    @tool
    def search_tool(query: str) -> str:
        """Search for information."""
        return web_search_function(query)
    
    @tool
    def calculator(expression: str) -> str:
        """Calculate mathematical expressions."""
        return calculator_function(expression)
    
    # Create ToolNode
    tools = [search_tool, calculator]
    tool_node = ToolNode(tools)
    
    # Create a state with tool calls (simulating what an agent would produce)
    from langchain_core.messages import AIMessage, ToolCall
    
    tool_calls = [
        ToolCall(
            name="search_tool",
            args={"query": "LangChain"},
            id="call_123",
            type="tool_call"
        )
    ]
    
    ai_message = AIMessage(content="", tool_calls=tool_calls)
    state = {"messages": [ai_message]}
    
    # Invoke ToolNode (this should trigger callbacks)
    from langgraph.graph import StateGraph
    
    # Build minimal graph with ToolNode
    workflow = StateGraph(dict)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("tools")
    workflow.set_finish_point("tools")
    
    graph = workflow.compile()
    
    # Execute with callbacks
    result = graph.invoke(
        state,
        config={"callbacks": [callback_handler]}
    )


# =============================================================================
# EXAMPLE 9: LangGraph chat_agent_executor
# =============================================================================


def example_chat_agent_executor():
    """
    Example 9: LangGraph's chat_agent_executor.
    Alternative to create_react_agent for chat-based agents.
    """
    if not LANGGRAPH_AVAILABLE:
        return
    
    # Initialize Noveum Trace
    noveum_trace.init(
        project=os.getenv("NOVEUM_PROJECT", "legacy-langchain-example"),
        api_key=os.getenv("NOVEUM_API_KEY"),
        environment="development",
    )
    
    # Create callback handler
    callback_handler = NoveumTraceCallbackHandler()
    
    # Create LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        callbacks=[callback_handler]
    )
    
    # Create tools (using old Tool class)
    tools = [
        Tool(
            name="Calculator",
            func=calculator_function,
            description="Calculate math expressions"
        ),
        Tool(
            name="WebSearch",
            func=web_search_function,
            description="Search for information"
        ),
    ]
    
    # Create agent using chat_agent_executor
    agent_executor = chat_agent_executor.create_tool_calling_executor(llm, tools)
    
    # Run with messages
    query = "Search for Python and calculate 50 * 20"
    
    result = agent_executor.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"callbacks": [callback_handler]}
    )


# =============================================================================
# EXAMPLE 10: Custom LangGraph with ToolNode and tools_condition
# =============================================================================


def example_custom_graph_with_toolnode():
    """
    Example 10: Custom LangGraph graph using ToolNode and tools_condition.
    This is the modern way to build agents with LangGraph.
    """
    if not LANGGRAPH_AVAILABLE:
        return
    
    # Initialize Noveum Trace
    noveum_trace.init(
        project=os.getenv("NOVEUM_PROJECT", "legacy-langchain-example"),
        api_key=os.getenv("NOVEUM_API_KEY"),
        environment="development",
    )
    
    # Create callback handler
    callback_handler = NoveumTraceCallbackHandler()
    
    # Create LLM with tool binding
    from langchain_core.tools import tool
    
    @tool
    def search(query: str) -> str:
        """Search for information."""
        return web_search_function(query)
    
    @tool
    def calculate(expression: str) -> str:
        """Calculate expressions."""
        return calculator_function(expression)
    
    tools = [search, calculate]
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        callbacks=[callback_handler]
    )
    llm_with_tools = llm.bind_tools(tools)
    
    # Create ToolNode
    tool_node = ToolNode(tools)
    
    # Define agent node
    def agent_node(state):
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    # Build graph
    from langgraph.graph import StateGraph, END
    from typing import TypedDict, Annotated
    from langgraph.graph.message import add_messages
    
    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]
    
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    
    # Use tools_condition to route
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
    )
    workflow.add_edge("tools", "agent")
    
    graph = workflow.compile()
    
    # Execute
    query = "Search for AI and calculate 100 / 4"
    result = graph.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"callbacks": [callback_handler], "recursion_limit": 10}
    )


# =============================================================================
# MAIN
# =============================================================================


def main():
    """
    Run all LangGraph examples to find which ones fail.
    """
    examples = [
        ("create_react_agent", example_langgraph_prebuilt_agent),
        ("ToolNode", example_langgraph_tool_node),
        ("chat_agent_executor", example_chat_agent_executor),
        ("custom_graph_with_toolnode", example_custom_graph_with_toolnode),
    ]
    
    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            import traceback
            traceback.print_exc()
    
    # Flush traces
    noveum_trace.flush()


if __name__ == "__main__":
    main()

