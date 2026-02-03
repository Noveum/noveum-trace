#!/usr/bin/env python3
"""
Test script to verify tool call attachment to LLM spans.

Tests both LangGraph and LangChain agents with tools to verify that executed
tool calls are properly attached to LLM spans via the llm.executed_tool_calls
attribute.

This demonstrates:
1. LangGraph ReAct Agent - tool execution with LangGraph
2. LangChain AgentExecutor - tool execution with traditional LangChain agents
"""

import os

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

import noveum_trace
from noveum_trace import NoveumTraceCallbackHandler

# Load environment variables
load_dotenv()

# Define some test tools


@tool
def get_temperature(city: str) -> str:
    """Get the current temperature for a city."""
    temps = {
        "new york": "45°F",
        "san francisco": "62°F",
        "london": "55°F",
        "tokyo": "68°F",
    }
    city_lower = city.lower()
    return f"The temperature in {city} is {temps.get(city_lower, '72°F')}"


@tool
def calculate(expression: str) -> float:
    """Evaluate a mathematical expression."""
    try:
        # Safe eval for basic math
        result = eval(expression, {"__builtins__": {}}, {})
        return float(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def get_capital(country: str) -> str:
    """Get the capital city of a country."""
    capitals = {
        "usa": "Washington D.C.",
        "uk": "London",
        "japan": "Tokyo",
        "france": "Paris",
        "germany": "Berlin",
    }
    country_lower = country.lower()
    return capitals.get(country_lower, f"Capital of {country} not found")


def test_langgraph_agent():
    """Test LangGraph ReAct agent with tools."""
    print("\n" + "=" * 80)
    print("LangGraph ReAct Agent with Tools")
    print("=" * 80)

    # Initialize callback
    callback = NoveumTraceCallbackHandler()

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Create tools
    tools = [get_temperature, calculate, get_capital]

    # Create agent
    agent = create_react_agent(llm, tools)

    # Test query
    query = "What's the temperature in Tokyo and what's the capital of Japan?"

    print(f"\nQuery: {query}\n")
    print("Invoking LangGraph agent...")

    # Invoke agent with callback
    result = agent.invoke(
        {"messages": [("user", query)]}, config={"callbacks": [callback]}
    )

    print("\n" + "-" * 80)
    print("LangGraph Agent Response:")
    print("-" * 80)
    for message in result["messages"]:
        if message.content:
            print(f"{message.type}: {message.content}")

    print("\n✅ Test complete!")


def test_langchain_agent_executor():
    """Test LangChain AgentExecutor with tools."""
    print("\n" + "=" * 80)
    print("LangChain AgentExecutor with Tools")
    print("=" * 80)

    # Initialize callback
    callback = NoveumTraceCallbackHandler()

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Create tools
    tools = [get_temperature, calculate, get_capital]

    # Create prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that can check temperatures, do calculations, and provide country information.",
            ),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # Create agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Test query
    query = "What's the temperature in London and what's the capital of UK?"

    print(f"\nQuery: {query}\n")
    print("Invoking LangChain AgentExecutor...")

    # Invoke agent with callback
    result = agent_executor.invoke({"input": query}, config={"callbacks": [callback]})

    print("\n" + "-" * 80)
    print("AgentExecutor Response:")
    print("-" * 80)
    print(f"Output: {result['output']}")

    print("\n✅ Test complete!")


def main():
    """Run all tests."""
    print("=" * 80)
    print("Testing Tool Call Attachment to LLM Spans")
    print("=" * 80)

    # Initialize NoveumTrace
    api_key = os.getenv("NOVEUM_API_KEY")
    if not api_key:
        print("ERROR: NOVEUM_API_KEY not found in environment")
        return

    noveum_trace.init(api_key=api_key, project="test-tool-attachment")

    try:
        # Test 1: LangGraph agent
        test_langgraph_agent()

        # Test 2: LangChain AgentExecutor
        test_langchain_agent_executor()

    finally:
        # Flush traces
        print("\n" + "=" * 80)
        print("Flushing trace data...")
        print("=" * 80)
        noveum_trace.flush()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS COMPLETE!")
        print("=" * 80)
        print("\nCheck the Noveum dashboard for the traces.")
        print("Both LangGraph and LangChain AgentExecutor should show:")
        print("  - 'llm.executed_tool_calls' attribute on LLM spans")
        print("  - Tool execution results attached to the LLM that requested them")


if __name__ == "__main__":
    main()
