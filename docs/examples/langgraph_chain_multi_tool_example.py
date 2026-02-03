"""
LangGraph and LangChain Multi-Tool Examples

This example demonstrates:
1. A LangGraph agent with custom StateGraph using ToolNode (get_weather, calculate)
2. A LangGraph agent using create_react_agent (get_weather, calculate)
3. A LangChain agent with multiple tools (search_database, format_data)
4. All examples use Noveum Trace for observability

The examples show different patterns for tool calling:
- LangGraph: Custom StateGraph with ToolNode (more control)
- LangGraph: create_react_agent (simplest, most common)
- LangChain: Single agent with multiple tools (traditional pattern)
"""

import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import noveum_trace
from noveum_trace.integrations.langchain import NoveumTraceCallbackHandler

load_dotenv()

# =============================================================================
# TOOLS DEFINITION
# =============================================================================


@tool
def get_weather(location: str) -> str:
    """
    Get the current weather for a location.

    Args:
        location: The city or location name

    Returns:
        Weather information as a string
    """
    print(f"[TOOL] get_weather called with location: {location}")
    # Placeholder implementation
    weather_data = f"Sunny, 22°C in {location}"
    print(f"[TOOL] get_weather returning: {weather_data}")
    return weather_data


@tool
def calculate(expression: str) -> str:
    """
    Perform mathematical calculations.

    Args:
        expression: Mathematical expression to evaluate (e.g., "25 * 4")

    Returns:
        Calculation result as a string
    """
    print(f"[TOOL] calculate called with expression: {expression}")
    try:
        result = eval(expression)
        result_str = f"Result: {result}"
        print(f"[TOOL] calculate returning: {result_str}")
        return result_str
    except Exception as e:
        error_str = f"Error: {str(e)}"
        print(f"[TOOL] calculate returning: {error_str}")
        return error_str


@tool
def search_database(query: str) -> str:
    """
    Search a database for information.

    Args:
        query: Search query string

    Returns:
        Search results as a string
    """
    print(f"[TOOL] search_database called with query: {query}")
    # Placeholder implementation
    results = f"Database search results for '{query}': Found 5 relevant records"
    print(f"[TOOL] search_database returning: {results}")
    return results


@tool
def format_data(data: str) -> str:
    """
    Format data into a structured output.

    Args:
        data: Raw data string to format

    Returns:
        Formatted data string
    """
    print(
        f"[TOOL] format_data called with data length: {len(data)} characters")
    # Placeholder implementation
    formatted = f"FORMATTED: {data.upper()}"
    print(f"[TOOL] format_data returning formatted data")
    return formatted


# =============================================================================
# LANGGRAPH AGENT EXAMPLE
# =============================================================================


def example_langgraph_agent():
    """Example: LangGraph agent calling multiple tools."""
    print("\n" + "=" * 80)
    print("LANGGRAPH AGENT - MULTIPLE TOOLS")
    print("=" * 80)

    try:
        from langgraph.graph import END, StateGraph
        from langgraph.graph.message import add_messages
        from typing import Annotated, Literal, TypedDict

        # Initialize Noveum Trace
        noveum_trace.init(
            project=os.getenv("NOVEUM_PROJECT", "test-project"),
            api_key=os.getenv("NOVEUM_API_KEY"),
            environment=os.getenv("NOVEUM_ENVIRONMENT", "dev"),
        )

        # Create callback handler
        handler = NoveumTraceCallbackHandler()
        print("✅ Noveum Trace callback handler created")

        # Define state
        class AgentState(TypedDict):
            messages: Annotated[list, add_messages]
            iteration_count: int
            max_iterations: int

        def agent_node(state: AgentState) -> AgentState:
            """Agent node - LLM with tools bound."""
            print(f"\n  🤖 Agent iteration {state['iteration_count'] + 1}")

            # Create LLM with tools BOUND
            llm = ChatOpenAI(model="gpt-4o-mini",
                             temperature=0, callbacks=[handler])
            llm_with_tools = llm.bind_tools([get_weather, calculate])

            # Invoke LLM - it decides which tools to call
            response = llm_with_tools.invoke(state["messages"])

            state["iteration_count"] += 1
            return {"messages": [response], "iteration_count": state["iteration_count"]}

        def tool_node(state: AgentState) -> AgentState:
            """Tool execution node - executes tool calls."""
            from langchain_core.messages import ToolMessage

            # Try to use ToolNode if available, otherwise fall back to manual execution
            try:
                from langgraph.prebuilt import ToolNode
                tools = [get_weather, calculate]
                tool_executor = ToolNode(tools)

                # Get the last message to show what tools are being executed
                last_message = state["messages"][-1]
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    tool_names = [tc["name"] for tc in last_message.tool_calls]
                    print(f"  🔧 Executing tools: {tool_names}")

                # ToolNode handles all the tool execution and ToolMessage creation
                return tool_executor.invoke(state)
            except (ImportError, AttributeError, TypeError) as e:
                # Fallback: Manual tool execution (for version compatibility)
                # This handles cases where ToolNode can't be imported or instantiated
                # due to version mismatches between langgraph and langchain-core
                # Fallback: Manual tool execution (for version compatibility)
                last_message = state["messages"][-1]
                tool_messages = []

                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    tools_map = {
                        "get_weather": get_weather,
                        "calculate": calculate,
                    }

                    tool_names = [tc["name"] for tc in last_message.tool_calls]
                    print(
                        f"  🔧 Executing tools: {tool_names} (using fallback method)")

                    for tool_call in last_message.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call.get("args", {})

                        if tool_name in tools_map:
                            tool = tools_map[tool_name]
                            result = tool.invoke(tool_args)
                            tool_messages.append(
                                ToolMessage(
                                    content=str(result),
                                    tool_call_id=tool_call.get("id", ""),
                                )
                            )

                return {"messages": tool_messages}

        def should_continue(state: AgentState) -> Literal["tools", "end"]:
            """Decide whether to continue with tool calls or end."""
            last_message = state["messages"][-1]

            if state["iteration_count"] >= state["max_iterations"]:
                print("  ⚠️  Max iterations reached")
                return "end"

            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                tool_names = [tc["name"] for tc in last_message.tool_calls]
                print(f"  🔧 Tool calls requested: {tool_names}")
                return "tools"

            print("  ✅ No more tool calls needed")
            return "end"

        # Build graph
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent", should_continue, {"tools": "tools", "end": END}
        )
        workflow.add_edge("tools", "agent")

        app = workflow.compile()
        print("✅ LangGraph agent created with tools: get_weather, calculate")

        # Query that requires both tools
        query = (
            "What's the weather in Paris? Also calculate 25 * 4. "
            "Make sure to use both the weather tool and the calculator tool."
        )

        print(f"\n📋 Query: {query}")
        print("\n🚀 Executing agent...")

        # Execute agent with Noveum Trace callback handler
        result = app.invoke(
            {
                "messages": [HumanMessage(content=query)],
                "iteration_count": 0,
                "max_iterations": 5,
            },
            config={
                "callbacks": [handler],
            },
        )

        # Extract final answer
        final_messages = [
            msg
            for msg in result["messages"]
            if hasattr(msg, "content") and msg.content
        ]
        if final_messages:
            print(f"\n✅ Agent Result:")
            print(f"   {final_messages[-1].content[:200]}...")

        print("\n📊 Trace Info:")
        print("   • Agent should have called both get_weather and calculate tools")
        print("   • Check Noveum dashboard for tool call spans")

    except ImportError as e:
        error_msg = str(e)
        if "TOOL_MESSAGE_BLOCK_TYPES" in error_msg or "langchain_core" in error_msg:
            print(f"\n⚠️  Version compatibility error: {e}")
            print(
                "   This is due to version mismatch between langgraph and langchain-core")
            print("   The example will use fallback manual tool execution")
            print("   To fix: pip install --upgrade langgraph langchain-core")
        else:
            print(f"\n⚠️  Import error: {e}")
            print("   Make sure langgraph is installed: pip install langgraph")
    except Exception as e:
        print(f"\n❌ Error in LangGraph agent example: {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# LANGGRAPH CREATE_REACT_AGENT EXAMPLE
# =============================================================================


def example_langgraph_react_agent():
    """Example: LangGraph agent using create_react_agent (simplest pattern)."""
    print("\n" + "=" * 80)
    print("LANGGRAPH CREATE_REACT_AGENT - MULTIPLE TOOLS")
    print("=" * 80)

    try:
        # Try to import create_react_agent
        # Note: This may fail if langgraph and langchain-core versions are incompatible
        try:
            from langgraph.prebuilt import create_react_agent
        except (ImportError, AttributeError) as e:
            print(f"\n⚠️  Cannot import create_react_agent: {e}")
            print(
                "   This is likely due to version incompatibility between langgraph and langchain-core")
            print("   Try: pip install --upgrade langgraph langchain-core")
            raise

        # Initialize Noveum Trace
        noveum_trace.init(
            project=os.getenv("NOVEUM_PROJECT", "test-project"),
            api_key=os.getenv("NOVEUM_API_KEY"),
            environment=os.getenv("NOVEUM_ENVIRONMENT", "dev"),
        )

        # Create callback handler
        handler = NoveumTraceCallbackHandler()
        print("✅ Noveum Trace callback handler created")

        # Create LLM
        llm = ChatOpenAI(model="gpt-4o-mini",
                         temperature=0, callbacks=[handler])

        # Create ReAct agent - tools automatically bound, no explicit tool node needed!
        agent = create_react_agent(llm, [get_weather, calculate])
        print("✅ LangGraph ReAct agent created with tools: get_weather, calculate")
        print("   • No explicit tool node needed - create_react_agent handles everything")

        # Query that requires both tools
        query = (
            "What's the weather in Paris? Also calculate 25 * 4. "
            "Make sure to use both the weather tool and the calculator tool."
        )

        print(f"\n📋 Query: {query}")
        print("\n🚀 Executing agent...")

        # Execute agent with Noveum Trace callback handler
        result = agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config={
                "callbacks": [handler],
            },
        )

        # Extract final answer
        final_messages = [
            msg
            for msg in result["messages"]
            if hasattr(msg, "content") and msg.content
        ]
        if final_messages:
            print(f"\n✅ Agent Result:")
            print(f"   {final_messages[-1].content[:200]}...")

        print("\n📊 Trace Info:")
        print("   • Agent should have called both get_weather and calculate tools")
        print("   • create_react_agent handles tool execution internally")
        print("   • Check Noveum dashboard for tool call spans")

    except ImportError as e:
        error_msg = str(e)
        if "TOOL_MESSAGE_BLOCK_TYPES" in error_msg or "langchain_core" in error_msg:
            print(f"\n⚠️  Version compatibility error: {e}")
            print(
                "   This is due to version mismatch between langgraph and langchain-core")
            print("   Try: pip install --upgrade langgraph langchain-core")
        else:
            print(f"\n⚠️  Import error: {e}")
            print("   Make sure langgraph is installed: pip install langgraph")
    except Exception as e:
        print(f"\n❌ Error in LangGraph ReAct agent example: {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# LANGCHAIN AGENT EXAMPLE
# =============================================================================


def example_langchain_agent():
    """Example: LangChain agent with multiple tools using llm.bind_tools() (modern pattern)."""
    print("\n" + "=" * 80)
    print("LANGCHAIN AGENT - MULTIPLE TOOLS (with llm.bind_tools)")
    print("=" * 80)

    try:
        from langchain.agents import AgentExecutor, create_openai_functions_agent
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

        # Initialize Noveum Trace
        noveum_trace.init(
            project=os.getenv("NOVEUM_PROJECT", "test-project"),
            api_key=os.getenv("NOVEUM_API_KEY"),
            environment=os.getenv("NOVEUM_ENVIRONMENT", "dev"),
        )

        # Create callback handler
        handler = NoveumTraceCallbackHandler()
        print("✅ Noveum Trace callback handler created")

        # Create LLM
        llm = ChatOpenAI(
            model="gpt-4o-mini", temperature=0, callbacks=[handler]
        )

        # Create tools
        tools = [search_database, format_data]

        # Create prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant. Use the available tools to answer questions."),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        # Create agent using llm.bind_tools() internally (modern pattern)
        # This uses function calling instead of ReAct text-based tool selection
        agent = create_openai_functions_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            callbacks=[handler],
            verbose=False,
        )

        print("✅ LangChain agent created with multiple tools")
        print("   • Uses llm.bind_tools() internally (modern function calling)")
        print("   • search_database tool")
        print("   • format_data tool")
        print("   • Tools auto-detected in trace (no manual injection needed)")

        query = (
            "Search for information about Python programming, "
            "then format the results in uppercase."
        )

        print(f"\n📋 Query: {query}")
        print("\n🚀 Executing agent...")

        # Execute agent - tools are bound to LLM, so they're auto-detected
        result = agent_executor.invoke(
            {"input": query},
            config={"callbacks": [handler]},
        )

        output = result.get("output", str(result))
        print(f"\n✅ Agent Result:")
        print(f"   {output[:200]}...")

        print("\n📊 Trace Info:")
        print("   • Agent should have called both search_database and format_data tools")
        print("   • Available tools are injected via metadata for trace visibility")
        print(
            "   • Check Noveum dashboard for tool call spans and available_tools attributes")

    except ImportError as e:
        print(f"\n⚠️  Import error: {e}")
        print("   Make sure langchain is installed: pip install langchain")
    except Exception as e:
        print(f"\n❌ Error in LangChain agent example: {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    """Run all examples."""
    print("=" * 80)
    print("LANGGRAPH AND LANGCHAIN - MULTI-TOOL EXAMPLES")
    print("=" * 80)
    print("\nThis file demonstrates:")
    print("  • LangGraph patterns: Custom StateGraph with ToolNode, create_react_agent")
    print("  • LangChain pattern: Single agent with multiple tools")

    # Check if API keys are set
    if not os.getenv("NOVEUM_API_KEY"):
        print("Warning: NOVEUM_API_KEY not set. Using mock mode.")

    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Some examples may fail.")

    # Run LangGraph custom StateGraph example (with ToolNode)
    example_langgraph_agent()

    # Wait a bit between examples
    import time
    time.sleep(2)

    # Run LangGraph create_react_agent example (simplest pattern)
    example_langgraph_react_agent()

    # Wait a bit between examples
    time.sleep(2)

    # Run LangChain agent example
    example_langchain_agent()

    print("\n" + "=" * 80)
    print("✅ ALL EXAMPLES COMPLETED")
    print("=" * 80)
    print("\nCheck your Noveum Trace dashboard to see:")
    print("  • LangGraph custom StateGraph with ToolNode (get_weather, calculate)")
    print("  • LangGraph create_react_agent (get_weather, calculate)")
    print("  • LangChain agent spans with tool calls (search_database, format_data)")
    print("  • Tool call spans nested under agent/chain spans")

    # Flush any pending traces
    noveum_trace.flush()


if __name__ == "__main__":
    main()
